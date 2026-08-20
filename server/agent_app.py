"""The agent — the capture pipeline with a voice and something to say.

Everything up to now has been about *listening*: deciding when speech starts,
what the words are, and when the person has finished. This adds the other half.
On a confirmed turn end the transcript goes to a language model, the reply is
spoken as it is generated, and the whole thing can be interrupted.

    turn end ──► LLM (streaming) ──► sentence splitter ──► TTS ──► browser
        ▲                                                            │
        └──────────────── barge-in cancels all of it ◄───────────────┘

Three things this file exists to get right
------------------------------------------
**Speak the first sentence before the last one is written.** The model streams
tokens; waiting for the full reply before synthesising would add its entire
generation time to the silence. So text is cut at sentence boundaries and each
sentence is synthesised as soon as it is complete. The reply starts about a
second sooner and nobody hears the difference.

**Barge-in must cancel everything, at once.** When the speaker starts again mid
reply, three things are in flight: the language model still generating, the
speech synthesiser still producing audio, and audio already queued in the
browser. Cancelling only the first two leaves the agent talking from the buffer,
which is the most infuriating possible failure. The task is cancelled, the
queue is dropped, and the browser is told to flush what it holds.

**The microphone is never muted.** It would be the easy way to stop the agent
hearing itself, and it would make interruption impossible — you cannot barge in
through a closed microphone. Echo is the browser canceller's job; ours is to
keep listening.

The use cases come from demo-v2, unchanged
------------------------------------------
The three demo personas — CRM Sales (live Zoho writes), Customer Support
(tickets + knowledge base) and Travel Planner (simulated) — are the ``chat``
package lifted across as-is, together with their tool schemas and the
tool-calling loop in ``chat/voice_agent.py``. None of that is re-implemented
here; this file only supplies the two attributes that loop expects of its host
and translates events for the browser.

The interface is demo-v2's too. It predates this pipeline and speaks its own
event vocabulary, so rather than rewrite a working UI, the server produces the
names it already understands (``user_text``, ``assistant_text_delta``,
``tool_call``…). The richer pipeline events are sent alongside, so the stripped
debug page at /debug keeps working against the same socket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import struct
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from chat.agents import AGENTS, DEFAULT_AGENT, agents_public, get_agent

from .config import settings
from .llm_client import LLMClient
from .speech import denoise as D
from .speech.denoise import make_denoiser
from .speech.echo_guard import EchoGuard, LadderConfig
from .speech.fusion import FusionConfig, TurnFusion
from .speech.measure import SessionRecorder
from .speech.session import CaptureSession, SessionConfig, to_float16k
from .speech.speech_gate import GateConfig, SpeechGate
from .speech.stt_parakeet import _norm as _norm_words
from .speech.stt_parakeet import similarity, transcripts_differ
from .speech.vad_ten import TenVAD
from .tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
)
log = logging.getLogger("agent")

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "client" / "index.html"      # the demo-v2 UI, reused as-is
CLIENT_MIN = ROOT / "client" / "agent.html"  # the stripped pipeline view
RECORDINGS = ROOT / "recordings"

app = FastAPI(title="voice pipeline v3 — agent")

OPTS: dict = {
    "stt": True, "denoise": "dpdfnet4", "att_context": (70, 1), "device": "cuda",
    # threshold is the bar to START speech, release_threshold the (lower) bar to
    # KEEP it. See GateConfig for the frame distributions these came from.
    "threshold": 0.60, "release_threshold": 0.35,
    # --echo-mode ladder dials; see server/speech/echo_guard.py.
    "barge_multiple": 8.0, "barge_sustain_ms": 560.0, "barge_needed": 6, "barge_cap": 0.7,
    "mode": "count", "min_frames": 2, "confirm_chunks": 2,
    "in_rate": 24000, "require_word": False,
    "turn": "both", "turn_device": "cuda", "turn_quantize": None,
    "tts": True, "record": True, "agent": DEFAULT_AGENT,
    # off | shadow | live | primary. Shadow was the right default while the two
    # seats were unmeasured — it logs both transcripts and changes nothing, which
    # is how the numbers got collected in the first place. They are collected now
    # (54.0% entity recall against 36.8%), so the default answers from the
    # accurate pass and accepts ~150-450 ms for it. See _await_accurate.
    "accurate": "primary",
    # english | indic — which half of the pipeline this session is serving.
    # It selects the ACCURATE seat only; the live seat stays Nemotron either
    # way, because it is the only streaming model we run and English is what it
    # is good at. Chosen by the user rather than detected: cascade language ID
    # costs 70-200 ms and drops a quarter of the words on mid-sentence switches,
    # which is exactly the audio the indic mode exists to serve.
    "lang_mode": "english",
    # Which Indian language, when lang_mode is indic. A request parameter on
    # every call, not a loaded model, so it can change between turns.
    "lang": "ta-IN",
    # half_duplex | guard | off — see AgentSession._is_echo.
    "echo_mode": "half_duplex",
}

# How long after the last sample can still be audible we keep the microphone
# shut. Covers room reverb and anything left in the browser's buffer.
ECHO_TAIL_SEC = 0.60
# The interface pre-rolls a jitter buffer before it starts a speech burst, so
# audio becomes audible a little after we send it and finishes a little later.
CLIENT_JITTER_SEC = 0.30
TTS_SR = 24000

# A sentence ends at . ? ! — but not inside "Mr." or "4.5". Good enough to cut
# streaming text into speakable pieces, and cheap enough to run per token.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Below this a "sentence" is probably an abbreviation or a stray token; holding
# it back and letting it join the next one avoids synthesising fragments.
_MIN_SENTENCE_CHARS = 12

# How many sentences may be in the synthesiser at once. See on_final for the
# measurements — 4 concurrent gave 3.56x over sequential, and the gain is in the
# batching, so anything above 1 helps. Held at 3 rather than pushed higher because
# every extra in-flight sentence is audio that has to be thrown away if the user
# barges in, and Orpheus shares its GPU with the SNAC decoder.
SPEAK_CONCURRENCY = 3


def split_sentences(buf: str) -> tuple[list[str], str]:
    """Split streamed text into complete sentences plus the unfinished tail."""
    parts = _SENTENCE_END.split(buf)
    if len(parts) == 1:
        return [], buf
    done, tail = parts[:-1], parts[-1]
    out: list[str] = []
    for p in done:
        if out and len(out[-1]) < _MIN_SENTENCE_CHARS:
            out[-1] = f"{out[-1]} {p}".strip()
        else:
            out.append(p.strip())
    if out and len(out[-1]) < _MIN_SENTENCE_CHARS:
        # Too short to speak alone — push it back onto the tail.
        tail = f"{out.pop()} {tail}".strip()
    return [s for s in out if s], tail


class AgentSession:
    """One conversation: listens, thinks, speaks, and can be interrupted."""

    def __init__(self, sock: WebSocket, session: CaptureSession, llm: LLMClient,
                 tts: TTSEngine | None, recorder: SessionRecorder | None,
                 agent_id: str = DEFAULT_AGENT):
        self.sock = sock
        self.session = session
        self.llm = llm
        self.tts = tts
        self.recorder = recorder
        self.history: list[dict] = []
        self._reply: asyncio.Task | None = None
        self._speaking = False
        self.voice = settings.tts_voice
        self.accurate = None          # set by the server once loaded
        self._utt_at_final = None     # the clip the live transcript came from
        # ── echo control ──
        self.echo_mode = OPTS["echo_mode"]
        self._spoken_recent = ""      # what the agent has said lately, for echo matching
        # Used only by echo_mode="ladder". Holds its own estimates, so it is
        # built per session and never shared between calls.
        # Ticks arrive on SessionConfig.tick_ms (100 ms), so the sustain window
        # is expressed in ms here and converted once, rather than leaving callers
        # to reason about tick counts.
        self._ladder_logged = 0.0
        self.echo_guard = EchoGuard(cfg=LadderConfig(
            echo_multiple=float(OPTS["barge_multiple"]),
            # Observations now arrive per audio chunk (80 ms), not per UI tick.
            sustain_window=max(1, round(float(OPTS["barge_sustain_ms"]) / 80.0)),
            sustain_ticks=max(1, int(OPTS["barge_needed"])),
            max_bar_fraction=float(OPTS["barge_cap"]),
        ))
        self._playback_ended_at = 0.0
        self._generating = False      # the model/synthesiser are still producing
        self._audio_until = 0.0       # when the audio we have SENT stops being audible
        self._muted = False           # for logging the transitions

        # The ported tool-calling provider expects a "pipeline" with exactly two
        # attributes: an interrupt Event it polls between tool rounds, and a
        # queue it pushes tool_call / tool_result / end_call events onto. Rather
        # than drag demo-v2's 1500-line pipeline across, this session satisfies
        # that interface directly.
        self._interrupted = asyncio.Event()
        self._text_out_queue: asyncio.Queue = asyncio.Queue()
        self._provider = None
        self.set_agent(agent_id)
        self._drain: asyncio.Task | None = None

    def _note_audio_sent(self, n_bytes: int) -> None:
        """Extend the estimate of when the agent stops being audible.

        Derived from the samples actually sent rather than from the browser
        saying so. Audio plays in real time, so N samples occupy N/24000 seconds
        no matter what — and unlike a playback_ended message, this estimate
        cannot be lost, delayed, or dropped by a reconnect. Bursts sent faster
        than real time queue up, hence the max() against the running end.
        """
        secs = (n_bytes / 2) / TTS_SR
        now = time.monotonic()
        self._audio_until = max(now + CLIENT_JITTER_SEC, self._audio_until) + secs
        self.echo_guard.far_end_audio(secs)

    def agent_is_audible(self) -> bool:
        """Is agent sound actually reaching the room right now?

        Deliberately NOT mic_is_contaminated(). That one also returns True while
        the reply is still being generated, which is correct for keeping the
        microphone shut but wrong for the echo guard: during generation there is
        no echo to measure, so anything the guard learns then is the caller's own
        voice, and a bar built from it fires on the caller.

        This is the fact the guard is supposed to key off — audio on the wire,
        plus the render tail that outlives the last frame.
        """
        return time.monotonic() < self._audio_until + ECHO_TAIL_SEC

    def mic_is_contaminated(self) -> bool:
        """Is the agent's own voice reaching the microphone right now?

        Three independent signals, and the mic stays shut if ANY of them says so:

          * we are still generating a reply,
          * audio we already sent has not finished playing (our own estimate),
          * the browser last told us it was playing very recently.

        The first was the bug: generation finishes well before playback does —
        the synthesiser runs barely faster than real time and the interface
        buffers on top — so releasing the microphone when generation ended
        reopened it in the middle of the agent's own sentence.
        """
        now = time.monotonic()
        return (
            self._generating
            or self._speaking
            or now < self._audio_until + ECHO_TAIL_SEC
            or (now - self._playback_ended_at) < ECHO_TAIL_SEC
        )

    def _is_echo(self, text: str) -> bool:
        """Does this transcript look like the agent hearing itself?

        The reliable discriminator is not loudness or timing — it is that the
        words ARE the agent's own. Loudness fails because a speaker close to a
        microphone is as loud as a person; timing fails because a real
        interruption happens at exactly the same moment an echo does.

        Only consulted while the microphone is contaminated, so a user who
        genuinely repeats the agent back is not silenced the rest of the time.
        """
        if not text.strip() or not self._spoken_recent:
            return False
        # A short fragment of a long reply still scores low against the whole
        # thing, so containment is checked as well as overall similarity.
        a, b = _norm_words(text), _norm_words(self._spoken_recent)
        if not a:
            return False
        overlap = sum(1 for w in a if w in b) / len(a)
        return overlap >= 0.7 or similarity(text, self._spoken_recent) >= 0.6

    def set_agent(self, agent_id: str) -> None:
        """Switch persona. Tools and system prompt come with it."""
        from chat.voice_agent import make_voice_provider

        if agent_id not in AGENTS:
            agent_id = DEFAULT_AGENT
        self.agent_id = agent_id
        self._provider = make_voice_provider(agent_id, self)
        self.history.clear()
        # Each persona carries its own voice (chat/agents.py), and the text path
        # in chat/app.py already honours it. This path did not, so switching to
        # the travel agent kept whatever voice the env file happened to name.
        self.voice = get_agent(agent_id).voice or settings.tts_voice
        log.info("agent persona: %s (voice=%s)", get_agent(agent_id).name, self.voice)

    async def pump_tool_events(self) -> None:
        """Forward the provider's tool events to the browser as they happen.

        They arrive on a queue rather than in the token stream because a tool
        round produces no text — without this the UI would sit silent through a
        CRM lookup with no indication anything was happening.
        """
        while True:
            ev = await self._text_out_queue.get()
            if ev is None:
                return
            await self.send(ev)

    # ── output ──
    async def send(self, obj: dict) -> None:
        """Send one event, tolerating a socket that has already gone.

        Disconnect is a race, not an error: the teardown path interrupts the
        reply, and a caller who closed the tab mid-sentence has no socket left to
        hear about it. Raising there produced a full ASGI traceback per hangup
        that meant nothing. Checking the state rather than catching everything so
        a genuine send failure still surfaces.
        """
        if self.sock.application_state is not WebSocketState.CONNECTED:
            return
        try:
            await self.sock.send_text(json.dumps(obj))
        except (WebSocketDisconnect, RuntimeError):
            # Closed between the check and the write.
            pass

    async def _synth_into(self, text: str, out: asyncio.Queue,
                          sem: asyncio.Semaphore) -> None:
        """Synthesise one sentence into its own queue. Runs concurrently.

        Terminates the queue with ``None`` on every path, including cancellation
        — the sender is blocked on that sentinel, and losing it would hang the
        reply rather than end it.
        """
        try:
            async with sem:
                async for pcm in self.tts.synthesize_stream(text, voice=self.voice):
                    out.put_nowait(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("synthesis failed for %r — skipping this sentence", text[:60])
        finally:
            out.put_nowait(None)

    async def _send_slots(self, slots: asyncio.Queue, first: dict, t0: float) -> None:
        """Send finished audio to the browser, one sentence at a time, in order.

        This is the half that makes concurrency safe. Synthesis runs out of order
        and in parallel; playback must not. Draining one sentence's queue to its
        sentinel before touching the next keeps the audio in the order the words
        were written, no matter which sentence finished generating first.
        """
        while True:
            slot = await slots.get()
            if slot is None:
                return
            while True:
                pcm = await slot.get()
                if pcm is None:
                    break
                await self.sock.send_bytes(struct.pack("<I", len(pcm)) + pcm)
                self._note_audio_sent(len(pcm))
                if first["ms"] is None:
                    first["ms"] = (time.perf_counter() - t0) * 1000.0
                if self.recorder is not None:
                    # The agent's own audio on the mic timebase — this is what
                    # makes the echo question answerable later.
                    self.recorder.ref(to_float16k(pcm, 24000))

    async def _speak(self, text: str) -> None:
        """Synthesise one sentence and stream it to the browser.

        Sequential. Kept for one-off utterances; the reply path uses
        :meth:`_synth_into` plus :meth:`_send_slots` instead, because doing this
        one sentence at a time is what made TTS the bottleneck — see on_final.
        """
        if self.tts is None or not text.strip():
            return
        # Keep a rolling record of the agent's own words so anything the
        # microphone picks up can be checked against them.
        self._spoken_recent = (self._spoken_recent + " " + text)[-600:]
        # self.voice, not the engine default. The engine is shared by every
        # session, so the voice has to travel with the request — and until this
        # argument existed, ``set_voice`` from the interface and the per-persona
        # voices in ``chat/agents.py`` were both silently discarded here.
        async for pcm in self.tts.synthesize_stream(text, voice=self.voice):
            # Cancellation lands here between chunks: the moment the user starts
            # talking this task is cancelled and the remaining audio is never
            # sent, rather than being queued and played over them.
            #
            # Framing: the interface expects [uint32 little-endian byte length]
            # followed by the int16 PCM. Send bare PCM and it reads the first
            # four bytes as a length, decides the frame is malformed, and drops
            # every chunk without a word of complaint — which is precisely how
            # this looked: speech synthesised, sent, and silently discarded.
            await self.sock.send_bytes(struct.pack("<I", len(pcm)) + pcm)
            self._note_audio_sent(len(pcm))
            if self.recorder is not None:
                # The agent's own audio on the mic timebase — this is what makes
                # the echo question answerable later.
                self.recorder.ref(to_float16k(pcm, 24000))

    async def _accurate_pass(self, live_text: str, audio) -> None:
        """Second opinion on the finished clip, running beside the reply.

        Never awaited by the reply path. Whatever it finds, the answer already
        being spoken was started from the live transcript — this either confirms
        it or, once calibrated, corrects it.
        """
        if self.accurate is None or audio is None or not len(audio):
            return
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            budget = self._accurate_timeout(audio)
            text = await asyncio.wait_for(
                loop.run_in_executor(None, self.accurate.transcribe, audio),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            log.info("accurate pass timed out at %.0fms — live transcript stands",
                     self._accurate_timeout(audio) * 1000)
            return
        except Exception:
            log.exception("accurate pass failed — live transcript stands")
            return
        if not text:
            return

        ms = (time.perf_counter() - t0) * 1000.0
        sim = similarity(live_text, text)
        differs = transcripts_differ(live_text, text)
        log.info("accurate[%s] %.0fms sim=%.2f%s\n    live    : %r\n    accurate: %r",
                 OPTS["accurate"], ms, sim, "  DIFFERS" if differs else "", live_text, text)
        await self.send({"type": "accurate_transcript", "text": text,
                         "live": live_text, "similarity": round(sim, 3),
                         "differs": differs, "ms": round(ms),
                         "applied": False})
        if self.recorder is not None:
            self.recorder.mark("accurate", live=live_text, accurate=text,
                               similarity=round(sim, 3), differs=differs, ms=round(ms))
        # In shadow mode that is the whole job: measure, never act. Switching to
        # "live" is a decision for after the numbers exist.

    def _accurate_timeout(self, audio) -> float:
        """Timeout for one accurate pass, scaled to how much audio it must read.

        ``timeout_sec`` is a flat 2 s, which was safe while the utterance buffer
        was capped at 30 s. With the cap gone (SessionConfig.max_utterance_sec) a
        long turn would blow through a fixed budget and fall back to the live
        transcript — quietly reintroducing, for exactly the long turns the cap
        removal was meant to fix, the loss it was meant to prevent.

        Cohere's cost is close to linear in clip length, measured on real clips:

            1s  133ms      5s  418ms      20s  652ms
            2s  157ms      8s  438ms      35s  863ms

        That is ~25 ms per second of audio. The allowance below is 50 ms/s on top
        of the flat budget — roughly double the measured slope, so a slow pass
        still has room while a hung one is still caught.
        """
        base = getattr(self.accurate, "timeout_sec", 2.0)
        if audio is None or not len(audio):
            return base
        return base + (len(audio) / 16000.0) * 0.05

    @staticmethod
    def _both_transcripts(primary: str, live: str, sim: float) -> str:
        """The user message to hand the LLM when the two recognisers disagree.

        Neither seat is reliably right. Measured over 87 Indian names, places and
        mail ids, the accurate seat wins on average (54.0% entity recall against
        36.8%) — which is why it is primary — but "on average" is not "always",
        and the failure is silent. From recordings/agent-20260819-074807, the user
        correcting a teammate's name:

            live     : "No, it's not it's not in the shop. It's Indusha"
            accurate : "No, it's not Indisha, it's Hindustan."     sim 0.50

        The live seat had the name RIGHT and the accurate seat turned it into
        Hindustan, and because primary mode replaced one with the other the
        correct spelling never reached the LLM at all. No amount of threshold
        tuning fixes that: the pass that is wrong is not knowable from the audio.

        So when they disagree, the model is given both and told to reconcile or
        ask. It is the only stage in the pipeline that can weigh "which of these
        is a real name" — and it can also just ask the user, which is what a
        person would do.

        Only on disagreement, at ``transcripts_differ``'s 0.85 word similarity.
        That fires on 50% of the 133 recorded turns, which looked too high until
        the band was inspected — at 0.80-0.85 the disagreements are real and are
        exactly the ones worth arbitrating:

            live "I'm flying from Quambatur"   accurate "Kwayam Battur"
            live "TKT iPhone zero zero one"    accurate "TKT-001"
            live "Coimbatour City"             accurate "Coimbatur City"

        Stripping filler words first only moves 50% to 44%, so the differences are
        substantive rather than "uh" noise. The threshold therefore stays and the
        BLOCK does the filtering instead: it names what counts as worth confirming
        and explicitly tells the model to ignore the block otherwise. Declaring
        everything unconfirmed would buy back the bug the persona prompt already
        warns about — asking "did you say Chennai?" when Chennai was correct.
        """
        return (
            f"{primary}\n\n"
            f"[SAME AUDIO, TWO RECOGNISERS DISAGREE — similarity {sim:.2f}\n"
            f'   A: "{primary}"\n'
            f'   B: "{live}"\n'
            " A is the transcript of record; B is the other recogniser on the same\n"
            " audio. Either may be right.\n"
            " If they differ on a NAME, PLACE, EMAIL or REFERENCE, treat it as\n"
            " unconfirmed: pick the most plausible real one and confirm it in ONE\n"
            " short question before passing it to any tool.\n"
            " If they differ only in filler words or ordinary wording, IGNORE this\n"
            " block and answer normally — confirm nothing.\n"
            " Never read this block aloud or mention that it exists.]"
        )

    async def _await_accurate(self, live_text: str, audio) -> tuple[str, str]:
        """Block for the accurate transcript and answer from THAT. ``primary`` mode.

        Returns ``(text, llm_text)``. ``text`` is the settled transcript — what
        the interface shows and what goes into history. ``llm_text`` is the same
        thing unless the two seats disagreed, in which case it carries both (see
        :meth:`_both_transcripts`). They are kept apart on purpose: the
        disagreement block is guidance for deciding THIS turn, and leaving it in
        history would carry a stale argument about a name into every later turn.

        This inverts the original design, deliberately and with numbers behind it.
        The two-seat pipeline was built so the reply never waits — but it was built
        before we measured what the seats actually produce. On our own audio, over
        87 Indian names, places and mail ids:

            nemotron-driven live transcript      36.8% entity recall
            cohere-transcribe over the clip      54.0%

        The LLM is the thing that acts on names: it writes them to CRM, it looks
        them up, it reads them back for confirmation. Handing it the worse
        transcript to save latency optimises the wrong quantity — the reply
        arrives sooner and is about the wrong customer.

        What it costs, measured on real clips at real turn lengths:

            1s  133ms      5s  418ms      20s  652ms
            2s  157ms      8s  438ms      35s  863ms
            3s  354ms     12s  512ms

        A spoken turn is usually 2-8 s, so this adds roughly 150-450 ms before the
        agent starts speaking. That is comparable to the language model's own
        first-token latency, and it buys a single clean reply instead of the
        alternative — starting on the live transcript and audibly cancelling
        itself when the two disagree.

        Nemotron keeps its seat regardless. It is not decoration: the endpointer
        needs incremental text to know a turn contains words at all
        (``SessionConfig.min_chars``), barge-in in guard mode fires on words, and
        the interface draws the live caption from it. It simply stops being the
        transcript of record.

        Falls back to ``live_text`` on timeout, failure, or a clip too short to
        judge — the pass keeps its "no opinion" contract, so a slow or broken
        accurate model degrades to the old behaviour rather than losing the turn.
        """
        if self.accurate is None or audio is None or not len(audio):
            return live_text, live_text
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            budget = self._accurate_timeout(audio)
            text = await asyncio.wait_for(
                loop.run_in_executor(None, self.accurate.transcribe, audio),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            log.info("accurate[primary] timed out at %.0fms (%.1fs of audio) — "
                     "live transcript stands",
                     self._accurate_timeout(audio) * 1000, len(audio) / 16000.0)
            return live_text, live_text
        except Exception:
            log.exception("accurate[primary] failed — live transcript stands")
            return live_text, live_text

        ms = (time.perf_counter() - t0) * 1000.0
        if not text:
            log.info("accurate[primary] no opinion in %.0fms — live transcript stands", ms)
            return live_text, live_text

        sim = similarity(live_text, text)
        differs = transcripts_differ(live_text, text)
        # Only a real disagreement earns the extra block; see _both_transcripts.
        llm_text = self._both_transcripts(text, live_text, sim) if differs else text
        log.info("accurate[primary] %.0fms sim=%.2f — %s"
                 "\n    live    : %r\n    accurate: %r", ms, sim,
                 "BOTH handed to the LLM to reconcile" if differs
                 else "answering from the accurate pass", live_text, text)
        await self.send({"type": "accurate_transcript", "text": text,
                         "live": live_text, "similarity": round(sim, 3),
                         "differs": differs, "both_sent": differs,
                         "ms": round(ms), "applied": True})
        if self.recorder is not None:
            # BOTH transcripts, always. The corrected value is what the agent
            # acted on, the raw one is the only way to tell a good correction
            # from a confident bad one later.
            self.recorder.mark("accurate", live=live_text, accurate=text,
                               similarity=round(sim, 3), applied=True,
                               both_sent=differs, ms=round(ms))
        return text, llm_text

    async def on_final(self, text: str) -> None:
        """A turn was confirmed. Think, then speak — sentence by sentence."""
        t0 = time.perf_counter()
        # What the LLM is asked to answer. Diverges from ``text`` only when the
        # two recognisers disagreed and both were handed over — history and the
        # interface keep the settled transcript, the model gets the argument.
        llm_text = text
        if OPTS["accurate"] == "primary":
            # Blocks. See _await_accurate for why that is the right trade.
            text, llm_text = await self._await_accurate(text, self._utt_at_final)
            # Only now does the interface learn what was said, because only now
            # is it settled. The live caption (``stt_partial``) has been updating
            # throughout, so the user has not been staring at nothing — this is
            # the line that becomes the permanent chat entry, and it has to match
            # what the agent actually answered.
            if text.strip():
                await self.send({"type": "user_text", "text": text})
        elif self.accurate is not None and self._utt_at_final is not None:
            # Fire and forget, deliberately: the reply must not wait for it.
            asyncio.ensure_future(self._accurate_pass(text, self._utt_at_final))
        self.history.append({"role": "user", "content": text})
        await self.send({"type": "assistant_text_start"})
        self._speaking = True
        self._generating = True

        buf, reply = "", ""
        # Sentences are synthesised CONCURRENTLY and sent in order.
        #
        # Doing one at a time was the bottleneck. Orpheus is a batch-1
        # autoregressive decoder: 99.7% of its iterations produced exactly one
        # token (107,247 of 107,580), so a single request runs at ~83 tok/s while
        # real-time playback needs 82.4 (7 tokens per 85ms frame). Measured over
        # 74 real streams that is 0.89-0.97x realtime — the generator is always
        # slightly BEHIND the speaker, so the browser drains its buffer and every
        # sentence boundary becomes an audible gap.
        #
        # Handing vLLM several sentences at once lets it batch them. Measured on
        # four real sentences:
        #
        #   sequential   20.05s   82.9 tok/s   1.01x realtime
        #   concurrent    5.63s  299.0 tok/s   3.64x realtime   (3.56x faster)
        #
        # Which turns a starving stream into one that buffers ahead. It also stops
        # synthesis smothering the event loop, which was delaying barge-in.
        #
        # Ordering is the risk, and it is handled structurally rather than
        # cleverly: one queue per sentence, and _send_slots drains them strictly
        # in order. Generation order cannot leak into playback order.
        slots: asyncio.Queue = asyncio.Queue()
        sem = asyncio.Semaphore(SPEAK_CONCURRENCY)
        synth: list[asyncio.Task] = []
        first = {"ms": None}
        sender = asyncio.ensure_future(self._send_slots(slots, first, t0))

        def dispatch(sentence: str) -> None:
            if self.tts is None or not sentence.strip():
                return
            # Keep a rolling record of the agent's own words so anything the
            # microphone picks up can be checked against them. Recorded at
            # dispatch rather than at send, so the echo guard is armed before any
            # of this sentence is audible.
            self._spoken_recent = (self._spoken_recent + " " + sentence)[-600:]
            q: asyncio.Queue = asyncio.Queue()
            slots.put_nowait(q)
            synth.append(asyncio.ensure_future(self._synth_into(sentence, q, sem)))

        try:
            self._interrupted.clear()
            # llm_text, not text: the disagreement block (when there is one) has
            # to reach the model, and history[:-1] excludes the clean copy we
            # just appended so the turn is not presented twice.
            async for tok in self._provider(llm_text, self.history[:-1]):
                buf += tok
                reply += tok
                await self.send({"type": "assistant_text_delta", "text": tok})
                sentences, buf = split_sentences(buf)
                for s in sentences:
                    dispatch(s)
            if buf.strip():
                dispatch(buf)
            slots.put_nowait(None)      # no more sentences; let the sender finish
            await sender
        except asyncio.CancelledError:
            log.info("reply cancelled by barge-in after %.0fms", (time.perf_counter() - t0) * 1000)
            raise
        finally:
            self._generating = False
            # Barge-in cancels this task, not its children. Anything still
            # synthesising would carry on producing audio for a reply the user has
            # already talked over, so it goes here — on every exit path, not just
            # the cancelled one.
            for task in synth:
                if not task.done():
                    task.cancel()
            if not sender.done():
                sender.cancel()
            # NOT _speaking: the browser is still playing what we just sent, and
            # mic_is_contaminated() keeps the microphone shut until the audio we
            # sent has had time to finish.
        first_audio = first["ms"]

        if reply.strip():
            self.history.append({"role": "assistant", "content": reply})
        # Keep the context bounded — a voice conversation does not need an hour
        # of history, and every extra turn is latency on the next reply.
        self.history = self.history[-12:]
        await self.send({
            "type": "assistant_text_done", "text": reply.strip(),
            "first_audio_ms": round(first_audio or 0),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        })

    def _ladder_log(self, audible: bool, ev: dict) -> None:
        """Once a second while the agent is audible, say what the guard sees.

        Exists because every failure so far has been invisible: the guard was
        never consulted, or consulted with the wrong signal, and the log showed
        nothing at all either way. Silence is not evidence of correctness.
        """
        if not audible:
            return
        now = time.monotonic()
        if now - self._ladder_logged < 1.0:
            return
        self._ladder_logged = now
        snap = self.echo_guard.snapshot()
        log.info("ladder: mic %.5f vs bar %.5f (echo %.5f win %d floor %.5f warm %.1fs) "
                 "reply_live=%s speaking=%s",
                 float(ev.get("rms") or 0.0), snap["bar"], snap["echo"], snap["window"],
                 snap["floor"], snap["far_end_sec"],
                 self._reply is not None and not self._reply.done(), self._speaking)

    async def interrupt(self, why: str) -> None:
        """Stop talking immediately — model, synthesiser and browser queue.

        The task being finished does NOT mean the agent has stopped talking. A
        reply task completes when the last audio has been *sent*; the browser is
        still holding several seconds of it. Returning early in that state was a
        silent no-op exactly when barge-in was most needed — the ladder fired,
        this method did nothing, and the agent talked calmly over the caller. So
        a finished task skips the cancellation and still drops the queue.
        """
        reply_live = self._reply is not None and not self._reply.done()
        if not reply_live and not self.agent_is_audible():
            return                      # nothing generating, nothing audible
        if reply_live:
            self._interrupted.set()
            self._reply.cancel()
            try:
                await self._reply
            except (asyncio.CancelledError, Exception):
                pass
            self._reply = None
        self._speaking = False
        self._generating = False
        # An interrupt drops the browser's queue, so nothing further is audible.
        self._audio_until = 0.0
        self.echo_guard.reset_burst()
        # The browser is holding audio we already sent. Without this it keeps
        # playing for as long as its buffer lasts, which is exactly the failure
        # barge-in exists to prevent.
        # "interrupt" is what the v2 UI listens for to drop its audio queue.
        await self.send({"type": "interrupt", "reason": why})
        log.info("interrupted (%s)", why)

    # ── the loop ──
    async def run_events(self) -> None:
        """Supervise the event pump so a failure cannot be silent.

        This was a real outage, not a hypothetical. The pump is started with
        ``asyncio.create_task`` and never awaited, so an exception inside it was
        stored on the task object and then discarded by the ``finally`` that
        cancels it on disconnect. The result: the recogniser kept working and
        kept logging ``turn end``, the interface kept showing the caller's words,
        and no reply was ever generated again. Three turns died that way in one
        session with not one line in the log to say why.

        So: log loudly here, and isolate each event below so one bad event costs
        one event rather than the rest of the call.
        """
        try:
            await self._run_events()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("EVENT PUMP DIED — this session can no longer reply")
            try:
                await self.send({"type": "pump_died"})
            except Exception:
                pass

    async def _run_events(self) -> None:
        """Translate pipeline events into what the UI speaks, and drive replies.

        The UI predates this pipeline and has its own vocabulary. Rather than
        rewrite a working interface, its names are produced here — while the
        richer events (turn scores, frame meters, abandoned utterances) are also
        forwarded, so the stripped-down debug page keeps working against the
        same socket.
        """
        last_partial = ""
        async for ev in self.session.events():
            try:
                kind = ev.get("type")

                # Barge-in. While the agent is audible we do NOT interrupt on voice
                # activity alone: the loudest thing in the room is the agent, and
                # acting on it means the agent cuts itself off every sentence. We
                # wait for words instead, and check whose they are.
                if kind == "level" and self.echo_mode == "ladder":
                    # Deliberately NOT gated on self._speaking. That flag is set only
                    # by a browser message, so one dropped message disabled barge-in
                    # for the rest of the reply and no amount of tuning could show it.
                    # agent_is_audible() is derived from bytes we sent ourselves and
                    # cannot go missing, and the guard already ignores everything
                    # outside that window.
                    #
                    # Ladder mode also does not key off the VAD gate: when the browser
                    # ducks the mic the gate stops opening at all, and every
                    # gate-driven mode goes deaf.
                    audible = self.agent_is_audible()
                    self._ladder_log(audible, ev)
                    if self.echo_guard.observe(
                        float(ev.get("rms") or 0.0),
                        agent_audible=audible,
                        caller_speaking=bool(ev.get("in_speech")),
                        vad=ev.get("vad"),
                    ):
                        if self.recorder is not None:
                            self.recorder.mark("ladder_barge_in", **self.echo_guard.snapshot())
                        await self.interrupt("barge-in")
                elif kind in ("start", "resume") and self._speaking:
                    if self.echo_mode == "off":
                        await self.interrupt("barge-in")
                elif kind == "word" and self._speaking and self.echo_mode == "guard":
                    said = (ev.get("text") or "").strip()
                    if said and not self._is_echo(said):
                        await self.interrupt("barge-in")

                if kind == "tick":
                    partial = ev.get("partial") or ""
                    if partial and partial != last_partial:
                        last_partial = partial
                        await self.send({"type": "stt_partial", "text": partial})
                elif kind == "final":
                    last_partial = ""
                    if self.mic_is_contaminated() and self._is_echo(ev.get("text", "")):
                        # The agent transcribed its own voice. Answering it starts a
                        # conversation between the demo and itself.
                        log.info("dropped echo turn: %r", (ev.get("text") or "")[:60])
                        await self.send({"type": "echo_dropped", "text": ev.get("text", "")})
                        if self.recorder is not None:
                            self.recorder.mark("echo_dropped", text=ev.get("text", ""))
                        continue
                    # Grab the clip NOW: the session clears its utterance buffer as
                    # part of finalising, so a moment later there is nothing to give
                    # the second recogniser.
                    self._utt_at_final = self.session.last_utterance()
                    if ev.get("text", "").strip() and OPTS["accurate"] != "primary":
                        # In primary mode this send moves into on_final, AFTER the
                        # accurate pass has run. Showing the live transcript here and
                        # then answering from a different one puts the interface and
                        # the agent visibly out of step — the user reads "climber
                        # two" on screen while the reply talks about Coimbatore, and
                        # cannot tell which one was written to the CRM.
                        await self.send({"type": "user_text", "text": ev["text"]})

                # Forward the raw event too: the debug page reads these, and they
                # cost nothing to a UI that ignores unknown types. "level" is the
                # exception — it is a per-chunk internal stream for the echo guard,
                # 12.5/s of data no interface asks for, and the throttled "tick"
                # already carries the same number for the meters.
                if kind != "level":
                    await self.send(ev)

                if kind == "final" and ev.get("text", "").strip():
                    if self._reply is not None and not self._reply.done():
                        await self.interrupt("new turn")
                    self._reply = asyncio.create_task(self.on_final(ev["text"]))
            except asyncio.CancelledError:
                raise
            except Exception:
                # One malformed or unexpected event must not end the call.
                log.exception('event pump: dropped a %r event', kind)
                continue


@app.on_event("startup")
async def warm() -> None:
    if OPTS["stt"]:
        try:
            from .speech.stt_nemotron import preload

            log.info("preloading recogniser...")
            await asyncio.to_thread(preload, device=OPTS["device"])
        except Exception:
            log.exception("recogniser preload failed")
    if OPTS["turn"] != "off":
        try:
            f = _build_fusion()
            if f is not None:
                log.info("preloading turn models (%s)...", OPTS["turn"])
                await f.start()
        except Exception:
            log.exception("turn preload failed")
    app.state.accurate = None
    if OPTS["accurate"] != "off":
        try:
            a = _build_accurate()
            if a is not None:
                log.info("preloading accurate recogniser (%s)...", a.name)
                await a.start()
                app.state.accurate = a
                log.info("accurate recogniser ready (%s, mode=%s)",
                         a.name, OPTS["accurate"])
        except Exception:
            log.exception("accurate recogniser unavailable — live transcript only")

    app.state.llm = LLMClient()
    await app.state.llm.start()
    ok = await app.state.llm.verify()
    log.info("LLM %s (%s)", "ready" if ok else "UNREACHABLE", settings.llm_model)
    app.state.tts = None
    if OPTS["tts"]:
        try:
            t = TTSEngine()
            await t.start()
            app.state.tts = t
            log.info("TTS ready (%s, voice=%s)", settings.tts_model, settings.tts_voice)
        except Exception:
            log.exception("TTS unavailable — the agent will reply in text only")
    log.info("agent is live")


@app.on_event("shutdown")
async def cool() -> None:
    if getattr(app.state, "llm", None):
        await app.state.llm.stop()
    if getattr(app.state, "tts", None):
        await app.state.tts.stop()


@app.get("/")
async def index():
    return FileResponse(CLIENT, media_type="text/html")


@app.get("/debug")
async def debug_page():
    """The pipeline view: frame meters, turn scores, labels. Same socket."""
    return FileResponse(CLIENT_MIN, media_type="text/html")


@app.get("/recordings/{path:path}")
async def api_recording(path: str):
    """Serve a file from ``recordings/`` so a session can be listened to.

    Exists because the recordings are the evidence. Every claim in this pipeline
    about what the VAD did or what a model heard is settled by playing the audio,
    and copying files off a GPU box by hand is enough friction that people argue
    from transcripts instead.

    Read-only, and resolved against ``RECORDINGS`` before being served: the path
    comes from a URL, so it is checked to still be inside that directory rather
    than trusted, which is what stops ``../../etc/passwd`` from being a valid
    recording name.
    """
    target = (RECORDINGS / path).resolve()
    if not target.is_file() or RECORDINGS.resolve() not in target.parents:
        return JSONResponse({"error": "not found"}, status_code=404)
    kind = "audio/wav" if target.suffix == ".wav" else "application/json"
    return FileResponse(target, media_type=kind, filename=target.name)


@app.get("/api/clips")
async def api_clips():
    """List what is available under ``recordings/`` — newest sessions first."""
    files = sorted(RECORDINGS.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return JSONResponse({"clips": [
        {"url": f"/recordings/{p.relative_to(RECORDINGS).as_posix()}",
         "mb": round(p.stat().st_size / 1e6, 2)}
        for p in files[:40]
    ]})


@app.get("/api/agents")
async def api_agents():
    return JSONResponse({"agents": agents_public(), "default": OPTS["agent"]})


@app.get("/api/health")
async def api_health():
    return JSONResponse({
        "ok": True,
        "llm": settings.llm_model,
        "tts": settings.tts_voice if getattr(app.state, "tts", None) else "off",
        "stt": "nemotron" if OPTS["stt"] else "off",
        "turn": OPTS["turn"],
        "lang_mode": OPTS["lang_mode"],
        "lang": OPTS["lang"] if OPTS["lang_mode"] == "indic" else "en",
        "accurate": getattr(app.state, "accurate", None)
                    and app.state.accurate.name or "off",
    })


@app.post("/api/chat")
async def api_chat(req: Request):
    """Text-only chat, streamed as newline-delimited JSON.

    The UI's typed panel. It runs the same personas and the same tools as the
    voice path — only the audio is missing — so a demo can be given without a
    microphone, and so spelling-sensitive input has somewhere to go.
    """
    from chat.runtime import run_agent

    body = await req.json()
    agent_id = body.get("agent") or DEFAULT_AGENT
    messages = body.get("messages") or []

    async def stream():
        try:
            async for ev in run_agent(agent_id, messages):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            log.exception("text chat failed")
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/params")
async def api_params_set(req: Request):
    """Accept and ignore, honestly.

    The v2 panel expects to write settings back. In v3 they are per-session
    startup arguments, so a write here would appear to work and change nothing.
    Reporting that plainly beats a silent no-op.
    """
    return JSONResponse({"ok": False, "reason": "settings are per-session startup flags in v3"},
                        status_code=200)


@app.get("/api/params")
async def api_params():
    """The v2 UI's settings panel. Read-only here.

    In demo-v2 these were live-editable knobs on a shared pipeline. In v3 the
    speech settings are per-session constructor arguments chosen at startup, so
    editing them mid-call would silently apply to nobody. Reporting them without
    pretending they can be changed is the honest version.
    """
    return JSONResponse({"params": [], "values": {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in OPTS.items()
    }, "editable": False})


@app.get("/api/config")
async def config():
    return JSONResponse({
        **{k: list(v) if isinstance(v, tuple) else v for k, v in OPTS.items()},
        "llm_model": settings.llm_model, "tts_voice": settings.tts_voice,
    })


def _build_fusion():
    if OPTS["turn"] == "off":
        return None
    acoustic = semantic = None
    if OPTS["turn"] in ("s2", "both"):
        try:
            from .speech.turn_smart import SmartTurnV2

            acoustic = SmartTurnV2(device=OPTS["turn_device"])
        except Exception:
            log.exception("S2 unavailable")
    if OPTS["turn"] in ("s3", "both"):
        try:
            from .speech.turn_ten import TenTurn

            semantic = TenTurn(device=OPTS["turn_device"], quantize=OPTS["turn_quantize"])
        except Exception:
            log.exception("S3 unavailable")
    if acoustic is None and semantic is None:
        return None
    return TurnFusion(acoustic, semantic, FusionConfig())


def _build_stt():
    if not OPTS["stt"]:
        return None
    try:
        from .speech.stt_nemotron import NemotronStreamingSTT

        return NemotronStreamingSTT(att_context=tuple(OPTS["att_context"]), device=OPTS["device"])
    except Exception:
        log.exception("recogniser unavailable")
        return None


def _build_accurate():
    """Pick the second-pass recogniser for the language mode this server serves.

    One seat, two occupants, chosen once at startup because both are model
    loads rather than switches. They present the same three methods, so
    ``AgentSession._accurate_pass`` never learns which one it got.

    ``english``  Cohere Transcribe. Best entity recall of everything we could
                 load, measured on our own audio: 54.0% against 36.8% for the
                 model that used to hold this seat. Apache-2.0, and 2.5x faster
                 than the runner-up. See ``stt_cohere`` for the full table.
    ``indic``    Sarvam saaras:v3 in codemix mode. Not a better model than the
                 English one — the only one that can transcribe the audio at
                 all, since nothing else in this stack speaks Tamil.

    ``canary`` and ``parakeet`` stay reachable because they are the measured
    baselines the choice was made against, and a comparison you cannot re-run is
    a comparison you have to take on faith.
    """
    which = OPTS["lang_mode"]
    if which == "indic":
        from .speech.stt_sarvam import SarvamAccurate

        return SarvamAccurate(
            api_key=settings.sarvam_api_key, language=OPTS["lang"],
            model_id=settings.sarvam_model, mode=settings.sarvam_mode,
            timeout_sec=settings.sarvam_timeout_sec,
        )
    if which == "parakeet":
        from .speech.stt_parakeet import ParakeetAccurate

        return ParakeetAccurate(device=OPTS["device"])
    if which == "canary":
        from .speech.stt_canary import CanaryAccurate

        return CanaryAccurate(device=OPTS["device"])

    from .speech.stt_cohere import CohereAccurate

    return CohereAccurate(device=OPTS["device"])


@app.websocket("/ws/voice")
async def ws_voice(sock: WebSocket):
    """The path demo-v2's interface dials.

    Kept as the primary name because the UI is the thing we did not rewrite;
    /ws is registered alongside it for the debug page and the test clients.
    """
    await _voice_session(sock)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await _voice_session(sock)


async def _voice_session(sock: WebSocket):
    await sock.accept()
    name = time.strftime("agent-%Y%m%d-%H%M%S")
    recorder = SessionRecorder(
        RECORDINGS, name,
        # The ref track IS wanted here: the agent speaks, so there is finally
        # something to measure echo leakage against.
        record_ref=True,
        meta={"tool": "agent_app", **{k: list(v) if isinstance(v, tuple) else v
                                      for k, v in OPTS.items()}},
    ) if OPTS["record"] else None

    session = CaptureSession(
        vad=TenVAD(),
        gate=SpeechGate(GateConfig(
            threshold=OPTS["threshold"], mode=OPTS["mode"],
            release_threshold=OPTS["release_threshold"],
            min_frames=OPTS["min_frames"], confirm_chunks=OPTS["confirm_chunks"],
            require_word_for_silence_reset=OPTS["require_word"],
        )),
        stt=_build_stt(), recorder=recorder, denoiser=make_denoiser(OPTS["denoise"]),
        fusion=_build_fusion(), cfg=SessionConfig(in_rate=OPTS["in_rate"],
                                                  require_word=OPTS["require_word"]),
    )
    agent = AgentSession(sock, session, app.state.llm, app.state.tts, recorder,
                         agent_id=OPTS["agent"])
    agent.accurate = app.state.accurate
    pump: asyncio.Task | None = None
    try:
        await session.start()
        await agent.send({
            "type": "ready", "recording": name,
            "denoise": session.denoiser.name,
            "stt": session.stt.name if session.stt else "off",
            "turn": "off" if session.fusion is None else "+".join(
                x.name for x in (session.fusion.acoustic, session.fusion.semantic) if x),
            "tts": settings.tts_voice if app.state.tts else "off",
            "llm": settings.llm_model,
            "accurate": OPTS["accurate"] if app.state.accurate else "off",
            "accurate_model": app.state.accurate.name if app.state.accurate else "off",
            "lang_mode": OPTS["lang_mode"],
            "lang": OPTS["lang"] if OPTS["lang_mode"] == "indic" else "en",
            "echo_mode": OPTS["echo_mode"],
            "agent": agent.agent_id,
            "agents": agents_public(),
        })
        pump = asyncio.create_task(agent.run_events())
        agent._drain = asyncio.create_task(agent.pump_tool_events())

        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                if OPTS["echo_mode"] == "half_duplex" and agent.mic_is_contaminated():
                    # The bluntest fix and the only one that cannot fail: while
                    # the agent is audible the microphone simply does not reach
                    # the pipeline. Costs barge-in; guarantees the agent never
                    # hears itself. Use headphones and --echo-mode guard to get
                    # interruption back.
                    if not agent._muted:
                        agent._muted = True
                        log.info("mic MUTED (agent audible)")
                        await agent.send({"type": "mic_muted", "muted": True})
                    continue
                if agent._muted:
                    agent._muted = False
                    log.info("mic live")
                    await agent.send({"type": "mic_muted", "muted": False})
                await session.feed(data)
            elif (text := msg.get("text")) is not None:
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    continue
                kind = cmd.get("type")
                if kind == "mark":
                    session.mark(str(cmd.get("label", "mark"))[:64], source="ui")
                elif kind in ("text", "text_input"):
                    # Typed input: the same path a spoken turn takes, which is
                    # how spelling-sensitive things (emails, names) get in
                    # without fighting the recogniser.
                    said = str(cmd.get("text", "")).strip()
                    if said:
                        await agent.interrupt("typed")
                        agent._reply = asyncio.create_task(agent.on_final(said))
                elif kind == "set_agent":
                    agent.set_agent(str(cmd.get("agent", DEFAULT_AGENT)))
                    await agent.send({"type": "ready", "agent": agent.agent_id,
                                      "text": get_agent(agent.agent_id).name})
                elif kind == "set_voice":
                    agent.voice = str(cmd.get("voice", settings.tts_voice))
                elif kind in ("playback_started", "playback_ended"):
                    if kind == "playback_ended":
                        agent._playback_ended_at = time.monotonic()
                    # The browser is the only place that knows when audio is
                    # actually audible. Kept for the echo bookkeeping and so
                    # barge-in knows the agent still holds the floor.
                    agent._speaking = (kind == "playback_started")
                elif kind == "interrupt":
                    await agent.interrupt("user")
                elif kind == "bye":
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("agent session failed")
    finally:
        if agent._reply is not None:
            agent._reply.cancel()
        if pump is not None:
            pump.cancel()
        if agent._drain is not None:
            agent._drain.cancel()
        await session.stop()
        if recorder is not None:
            s = recorder.close()
            log.info("session %s — %.1fs, %d chunks", s["name"], s["duration_sec"], s["chunks"])


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="v3 voice agent: listen, think, speak.")
    ap.add_argument("--port", type=int, default=settings.chat_port)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--denoise", default="dpdfnet4", choices=list(D.CHOICES),
                    help="VAD front-end. dpdfnet4 keeps the widest speech/noise "
                         "margin on our recordings; gtcrn is smaller and faster "
                         "but carries room noise over the VAD bar. "
                         "See scripts/bench_denoise.py.")
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--no-tts", action="store_true", help="text-only replies")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--att", default="70,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="frame score needed to START speech")
    ap.add_argument("--release-threshold", type=float, default=0.35,
                    help="lower bar to KEEP speech running (hysteresis). "
                         "Pass -1 to disable and use one bar for both.")
    ap.add_argument("--mode", default="count", choices=["count", "run", "max"])
    ap.add_argument("--agent", default=DEFAULT_AGENT, choices=sorted(AGENTS),
                    help="which demo persona to start on")
    ap.add_argument("--turn", default="both", choices=["off", "s2", "s3", "both"])
    ap.add_argument("--turn-4bit", action="store_true")
    ap.add_argument("--echo-mode", default="half_duplex",
                    choices=["half_duplex", "guard", "ladder", "off"],
                    help="half_duplex: mic ignored while the agent speaks (no echo, no "
                         "barge-in). guard: keep listening but reject the agent's own "
                         "words (needs headphones or a quiet speaker). off: neither.")
    ap.add_argument("--accurate", default="primary",
                    choices=["off", "shadow", "live", "primary"],
                    help="what the second-pass recogniser is for. primary (default): WAIT "
                         "for it and answer from its transcript — best entity accuracy, "
                         "adds ~150-450ms per turn. shadow: log both, never act. live: "
                         "start on the live transcript and restart the reply on "
                         "disagreement. off: live transcript only.")
    ap.add_argument("--lang-mode", default="english",
                    choices=["english", "indic", "canary", "parakeet"],
                    help="which accurate pass to load. english: Cohere Transcribe, best "
                         "measured entity recall (54%% vs 37%%), for Indian/US/British "
                         "English. indic: Sarvam codemix, the only option that "
                         "transcribes Tamil/Telugu/Kannada mixed with English. "
                         "canary/parakeet: the measured baselines, kept so the "
                         "comparison stays runnable. Live seat is Nemotron either way.")
    ap.add_argument("--lang", default="ta-IN",
                    help="BCP-47 language for --lang-mode=indic (ta-IN, te-IN, kn-IN, "
                         "hi-IN, ...; 'unknown' auto-detects, which we do not recommend)")
    ap.add_argument("--require-word", action="store_true")
    ap.add_argument("--certs", default=str(ROOT.parent / "demo-v2" / "certs"))
    args = ap.parse_args()

    OPTS.update(
        stt=not args.no_stt, tts=not args.no_tts, record=not args.no_record,
        denoise=args.denoise, att_context=tuple(int(x) for x in args.att.split(",")),
        device=args.device, threshold=args.threshold, mode=args.mode,
        release_threshold=(None if args.release_threshold < 0
                           else args.release_threshold),
        turn=args.turn, turn_quantize="4bit" if args.turn_4bit else None, agent=args.agent,
        require_word=args.require_word, accurate=args.accurate,
        echo_mode=args.echo_mode, lang_mode=args.lang_mode, lang=args.lang,
    )
    RECORDINGS.mkdir(parents=True, exist_ok=True)

    cert, key = Path(args.certs) / "cert.pem", Path(args.certs) / "key.pem"
    ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)} if cert.is_file() and key.is_file() else {}
    if not ssl:
        log.warning("no certs at %s — the browser will BLOCK the microphone", args.certs)
    log.info("agent on %s://%s:%d (turn=%s tts=%s)", "https" if ssl else "http",
             args.host, args.port, args.turn, "off" if args.no_tts else settings.tts_voice)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
