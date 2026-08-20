"""The Zoho knowledge-base demo — one voice, one job: answer product questions.

A third front end onto the same speech pipeline. ``capture_app`` listens and
shows you the machinery; ``agent_app`` runs the three tool-using personas; this
one does exactly one thing, so it can be pointed at an audience without any
setup: ask a question about a Zoho product out loud, get a spoken answer out of
the product documentation.

    mic ──► CaptureSession (VAD, Nemotron, turn fusion, accurate pass)
              │
              ▼
        correct_query   — a local LLM repairs misheard Zoho product names
              │
              ▼
          Zia RAG       — retrieval + generation, hosted  (~1.5-2.4s)
              │            └─ filler line covers the wait
              ▼
        clean_for_tts   — markdown out, speakable text in
              │
              ▼
             TTS

What is reused, and why that matters
------------------------------------
Everything to the left of the RAG call is ``agent_app``'s, imported rather than
copied: the session construction, the model preloads, barge-in, echo control, the
accurate second pass. The only thing this file replaces is *where the reply comes
from* — ``AgentSession`` reaches its answer through ``self._provider``, so
swapping that one attribute swaps the entire brain and leaves the ears alone.

Copying those 200 lines instead would mean this demo quietly missing the next fix
to the endpointer or the echo guard, which is precisely how a demo becomes the
version that "used to work".

Sentence-at-a-time, not token-at-a-time
---------------------------------------
The provider yields whole cleaned sentences rather than raw tokens. It has to:
the service answers in markdown, and ``**Workflow Rules**`` arrives split across
several tokens, so cleaning a fragment at a time would leave stray asterisks in
the audio. Sentences are the smallest unit that can be cleaned correctly, and
they are also the unit TTS wants, so nothing is lost but the typewriter effect in
the transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import agent_app as A
from .config import settings
from .llm_client import LLMClient
from .speech import denoise as D
from .speech.denoise import make_denoiser
from .speech.measure import SessionRecorder
from .speech.session import CaptureSession, SessionConfig
from .speech.speech_gate import GateConfig, SpeechGate
from .speech.vad_ten import TenVAD
from .tts_engine import TTSEngine
from .zia_rag import (
    COLLECTION,
    FillerPicker,
    ZiaRAG,
    clean_for_tts,
    correct_query,
    isc_token,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
)
log = logging.getLogger("kb")

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "client" / "kb.html"
RECORDINGS = ROOT / "recordings"

app = FastAPI(title="voice pipeline v3 — Zoho knowledge base")

# Speech settings live in agent_app.OPTS, which the shared builders read. This
# demo has no personas and no tools, so those keys are dropped; everything else
# is the pipeline as tuned.
OPTS: dict = {
    # How long to wait for the first answer token before covering the gap with a
    # spoken line. Below ~350ms the filler would talk over an answer that was
    # about to arrive anyway; retrieval has never come back faster than 1.5s.
    "filler_after_ms": 400.0,
    "collection": COLLECTION,
    "correct": True,      # run the Zoho-product-name repair before retrieval
    # Hard ceiling on what the shared LLM may add to this demo. Retrieval answers
    # in ~1.6s on its own infrastructure; the LLM's first token was measured at
    # median 993ms but p90 20.2s, so an unbounded correction would hand this demo
    # the exact latency problem the persona demos have. See correct_query.
    "correct_budget_ms": 600.0,
    "filler": True,
    # Hang up when the caller says goodbye. Off-by-default would mean the demo
    # never ends a call, which is the behaviour the personas already have.
    "end_call": True,
}


# ── the RAG brain ────────────────────────────────────────────────────────────

# ── what the retriever is actually asked ─────────────────────────────────────
# ``agent_app`` hands the provider ``llm_text``, which on a disagreement between
# the two recognisers is the transcript PLUS a ~500-character instruction block
# telling the model how to weigh them. For a tool-using persona that is useful
# context. Here it is poison, three times over:
#
#   * the product-name repair is handed a page of instructions instead of a
#     question, and its length guard (`len(fixed) > 2 * len(text)`) goes slack
#     because `text` is now enormous;
#   * if the repair is skipped — timeout, or the circuit breaker open, which the
#     logs show happening — `query` falls back to this text and the whole block
#     is sent to the retriever AS THE SEARCH QUERY;
#   * the goodbye check reads it too, and it contains the word "confirm" and a
#     lot else that has nothing to do with the caller.
#
# A retriever wants the question and nothing else, so strip it at this boundary
# rather than asking agent_app to stop producing it — the personas still need it.
_DISAGREE_MARK = "[SAME AUDIO, TWO RECOGNISERS DISAGREE"


def transcript_of_record(text: str) -> str:
    """The caller's words, with any recogniser-disagreement block removed."""
    if not text:
        return ""
    i = text.find(_DISAGREE_MARK)
    return (text[:i] if i != -1 else text).strip()



def make_rag_provider(session: "KBSession"):
    """Build ``provider(user_text, history) -> AsyncIterator[str]`` over Zia RAG.

    Signature matches ``chat.voice_agent.make_voice_provider`` exactly, which is
    the whole trick: ``AgentSession.on_final`` cannot tell the difference between
    a persona with tools and this.

    ``history`` from the caller is ignored in favour of ``session.rag_history``.
    The caller's copy contains what was *spoken*, filler lines included, and
    feeding "Sure, let me look that up for you." back into a retriever as
    conversational context is noise at best.
    """

    async def prepare_and_stream(user_text: str):
        """Correct the query, then stream the answer. One generator on purpose.

        The correction and the retrieval are both waiting-for-a-server, and what
        the listener experiences is the sum. Wrapping them together means the
        filler below races the FIRST SPOKEN WORD rather than one internal stage:
        correction alone can take seconds when the shared vLLM is busy (17
        running / 14 queued while this was written), and covering only retrieval
        left that silence uncovered.
        """
        rag: ZiaRAG = session.rag
        query = user_text
        if OPTS["correct"]:
            query, ms = await correct_query(
                session.llm, user_text,
                timeout_sec=OPTS["correct_budget_ms"] / 1000.0,
            )
            if query != user_text:
                await session.send({"type": "query_corrected", "heard": user_text,
                                    "query": query, "ms": round(ms)})
                if session.recorder is not None:
                    session.recorder.mark("query_corrected", heard=user_text,
                                          query=query, ms=round(ms))
        session.last_query = query
        await session.send({"type": "rag_start", "query": query,
                            "collection": OPTS["collection"]})

        # Drain the HTTP stream as fast as it arrives, into a queue, and yield
        # from the queue instead of straight through.
        #
        # Without this the consumer paces the producer: on_final awaits _speak()
        # for each sentence before pulling the next fragment, so a 1.6s retrieval
        # was measured taking 35.3s wall-clock with the connection held open the
        # whole time — TTS speaking at the far end throttling an HTTP response at
        # the near end. That risks a server-side idle timeout on a request that
        # actually finished in under two seconds, and it makes the reported
        # retrieval time meaningless.
        #
        # maxsize=0: an answer is a few KB, so there is nothing to gain by
        # bounding it, and a bound could reintroduce exactly the coupling being
        # removed here.
        q: asyncio.Queue = asyncio.Queue()

        async def drain() -> None:
            try:
                async for frag in rag.stream(query, session.rag_history):
                    q.put_nowait(frag)
            except Exception as exc:      # surfaced below, on the consumer side
                q.put_nowait(exc)
            finally:
                q.put_nowait(None)

        task = asyncio.ensure_future(drain())
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not task.done():
                task.cancel()

    async def provider(raw_text: str, _history: list[dict]):
        # Everything below works from the question alone; see transcript_of_record.
        user_text = transcript_of_record(raw_text)
        rag: ZiaRAG = session.rag
        session.last_query = user_text
        t0 = time.perf_counter()
        # Launched here, in the same scope that reads it below — see wants_to_end
        # for why it runs concurrently rather than in front of retrieval.
        ending = (asyncio.ensure_future(wants_to_end(session.llm, user_text))
                  if OPTS["end_call"] else None)
        stream = prepare_and_stream(user_text).__aiter__()
        buf, answer, spoke_filler = "", [], False

        async def next_fragment():
            try:
                return await stream.__anext__()
            except StopAsyncIteration:
                return None

        pending = asyncio.ensure_future(next_fragment())
        try:
            while True:
                if not spoke_filler and OPTS["filler"]:
                    # Race the first fragment against the patience budget. Whoever
                    # wins, the other is still awaited below — pending is never
                    # dropped, so no fragment is lost to the timeout.
                    try:
                        frag = await asyncio.wait_for(
                            asyncio.shield(pending),
                            timeout=OPTS["filler_after_ms"] / 1000.0,
                        )
                    except asyncio.TimeoutError:
                        line = session.fillers.next()
                        spoke_filler = True
                        await session.send({"type": "filler", "text": line})
                        if session.recorder is not None:
                            session.recorder.mark("filler", text=line)
                        log.info("filler after %.0fms: %r",
                                 (time.perf_counter() - t0) * 1000, line)
                        yield line + " "
                        continue
                else:
                    frag = await pending

                if frag is None:
                    break
                pending = asyncio.ensure_future(next_fragment())
                spoke_filler = True   # past the decision point either way

                if frag == "\x00":
                    # The service put reasoning on the path we were reading. Drop
                    # what we have rather than speak a monologue; see _AnswerPicker.
                    buf, answer = "", []
                    await session.send({"type": "rag_reset"})
                    continue

                buf += frag
                sentences, buf = A.split_sentences(buf)
                for s in sentences:
                    spoken = clean_for_tts(s)
                    if spoken:
                        answer.append(spoken)
                        yield spoken + " "
            if buf.strip():
                spoken = clean_for_tts(buf)
                if spoken:
                    answer.append(spoken)
                    yield spoken
        finally:
            if not pending.done():
                pending.cancel()

        # Read the goodbye check now, after the answer. The browser hangs up once
        # playback drains, so this lands in time without ever being on the clock.
        if ending is not None:
            try:
                if await ending:
                    log.info("end_call: %r", user_text[:60])
                    await session.send({"type": "end_call"})
                    if session.recorder is not None:
                        session.recorder.mark("end_call", text=user_text[:200])
            except Exception:
                log.exception("end-call check failed — staying on the call")

        full = " ".join(answer).strip()
        if full:
            # Only the real question and answer become retrieval context — the
            # CORRECTED question, which is what was actually retrieved against.
            session.rag_history.append({"role": "user", "content": session.last_query})
            session.rag_history.append({"role": "assistant", "content": full})
            session.rag_history[:] = session.rag_history[-6:]
        await session.send({
            "type": "rag_done",
            "ttft_ms": round(rag.last_ttft_ms or 0),
            "total_ms": round(rag.last_total_ms or 0),
            "chars": len(full),
        })
        if session.recorder is not None:
            session.recorder.mark("rag", query=session.last_query,
                                  ttft_ms=round(rag.last_ttft_ms or 0),
                                  total_ms=round(rag.last_total_ms or 0), answer=full[:400])

    return provider


class KBSession(A.AgentSession):
    """An AgentSession whose answers come from retrieval instead of a persona."""

    def set_agent(self, agent_id: str = "kb") -> None:
        """Install the RAG provider. Overrides persona/tool selection entirely.

        Called from ``AgentSession.__init__``, so every attribute this touches has
        to be created here rather than assumed — hence the defaults below.
        """
        self.agent_id = "kb"
        self.rag = ZiaRAG(collection=OPTS["collection"])
        self.rag_history: list[dict] = []
        self.last_query = ""      # the corrected query the answer was retrieved for
        self.fillers = FillerPicker()
        self._provider = make_rag_provider(self)
        self.voice = settings.tts_voice
        log.info("knowledge-base agent ready (collection=%s, voice=%s)",
                 OPTS["collection"], self.voice)


# ── lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    # The shared builders read agent_app.OPTS, so this demo's speech settings are
    # written there. Only the speech half is touched; tools and personas are not
    # used on this path at all.
    A.OPTS.update(tts=True, record=True, accurate="primary")
    A.RECORDINGS.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    if A.OPTS["stt"]:
        from .speech.stt_nemotron import preload as preload_nemotron

        log.info("preloading live recogniser...")
        await loop.run_in_executor(
            None, preload_nemotron, "nvidia/nemotron-speech-streaming-en-0.6b",
            A.OPTS["device"],
        )
    if A.OPTS["turn"] != "off":
        try:
            f = A._build_fusion()
            if f is not None:
                log.info("preloading turn models (%s)...", A.OPTS["turn"])
                await f.start()
        except Exception:
            log.exception("turn preload failed")

    app.state.accurate = None
    try:
        a = A._build_accurate()
        if a is not None:
            log.info("preloading accurate recogniser (%s)...", a.name)
            await a.start()
            app.state.accurate = a
    except Exception:
        log.exception("accurate recogniser unavailable — live transcript only")

    app.state.llm = LLMClient()
    await app.state.llm.start()
    ok = await app.state.llm.verify()
    log.info("correction LLM %s (%s)", "ready" if ok else "UNREACHABLE", settings.llm_model)

    app.state.tts = None
    try:
        t = TTSEngine()
        await t.start()
        app.state.tts = t
    except Exception:
        log.exception("TTS unavailable — answers will be text only")

    # Fail loudly here rather than on the first question: an expired or
    # unreachable ISC endpoint is the one thing that makes this demo do nothing,
    # and finding that out in front of an audience is avoidable.
    try:
        await isc_token()
        log.info("Zia RAG reachable, collection=%s", OPTS["collection"])
    except Exception:
        log.exception("ISC token unavailable — RAG WILL FAIL until this is fixed")

    log.info("knowledge-base demo is live")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if getattr(app.state, "llm", None):
        await app.state.llm.stop()
    if getattr(app.state, "tts", None):
        await app.state.tts.stop()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(CLIENT)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "collection": OPTS["collection"],
        "stt": "nemotron" if A.OPTS["stt"] else "off",
        "accurate": getattr(app.state, "accurate", None) and app.state.accurate.name or "off",
        "tts": settings.tts_voice if getattr(app.state, "tts", None) else "off",
        "correct": OPTS["correct"],
        "correct_budget_ms": OPTS["correct_budget_ms"],
        "filler": OPTS["filler"],
    })


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await _kb_session(sock)


@app.websocket("/ws/voice")
async def ws_voice(sock: WebSocket) -> None:
    await _kb_session(sock)



# ── ending the call ──────────────────────────────────────────────────────────
# agent_app's personas get this from the LLM as a tool call, because they run a
# tool-calling loop. This demo cannot: the answer is generated by a hosted
# retrieval service that has no tools and no knowledge of the call. So the intent
# is read off the caller's own transcript instead, by the same local LLM that
# repairs product names.
#
# Run CONCURRENTLY with correction and retrieval and only read at the end, so it
# costs nothing on the clock. A goodbye is the one turn where the caller has
# already stopped caring how fast the answer is, but every other turn pays for
# anything placed in front of retrieval.
_END_CALL_SYSTEM = """Decide whether the speaker is ending the conversation.

Answer with exactly one word: END or CONTINUE.

END when they are wrapping up and expect nothing further:
  "bye" / "goodbye" / "thanks, that's all" / "no, that's it" / "nothing else"
  "that's all I needed" / "ok thanks bye" / "we're done" / "catch you later"

CONTINUE for everything else, including:
  - any question, even a short one ("and Bigin?" / "what about pricing")
  - plain thanks that is not a farewell ("thanks, and how do I export it?")
  - "okay" / "got it" / "hmm" on their own — acknowledgement, not an ending
  - anything ambiguous. Ending a call by mistake is far worse than missing one.
"""


async def wants_to_end(llm, text: str, *, timeout_sec: float = 1.2) -> bool:
    """True when the caller's turn is a goodbye.

    Fails closed on every error path — timeout, exception, unparseable answer all
    mean CONTINUE. A false END hangs up on a paying customer mid-question, which
    is not a failure mode worth trading anything for.
    """
    if not text or not text.strip() or llm is None:
        return False
    try:
        out = await asyncio.wait_for(
            llm.chat_complete(
                [{"role": "system", "content": _END_CALL_SYSTEM},
                 {"role": "user", "content": text.strip()}],
                temperature=0.0, max_tokens=4,
            ),
            timeout=timeout_sec,
        )
    except (asyncio.TimeoutError, Exception):
        return False
    return (out or "").strip().upper().startswith("END")



async def _kb_session(sock: WebSocket) -> None:
    await sock.accept()
    name = time.strftime("kb-%Y%m%d-%H%M%S")
    recorder = SessionRecorder(
        RECORDINGS, name, record_ref=True,
        meta={"tool": "kb_app", "collection": OPTS["collection"],
              **{k: list(v) if isinstance(v, tuple) else v for k, v in A.OPTS.items()}},
    ) if A.OPTS["record"] else None

    session = CaptureSession(
        vad=TenVAD(),
        gate=SpeechGate(GateConfig(
            threshold=A.OPTS["threshold"], mode=A.OPTS["mode"],
            release_threshold=A.OPTS["release_threshold"],
            min_frames=A.OPTS["min_frames"], confirm_chunks=A.OPTS["confirm_chunks"],
            require_word_for_silence_reset=A.OPTS["require_word"],
        )),
        stt=A._build_stt(), recorder=recorder, denoiser=make_denoiser(A.OPTS["denoise"]),
        fusion=A._build_fusion(),
        cfg=SessionConfig(in_rate=A.OPTS["in_rate"], require_word=A.OPTS["require_word"]),
    )
    agent = KBSession(sock, session, app.state.llm, app.state.tts, recorder)
    agent.accurate = app.state.accurate

    pump: asyncio.Task | None = None
    try:
        await session.start()
        await agent.send({
            "type": "ready", "recording": name,
            "collection": OPTS["collection"],
            "denoise": session.denoiser.name,
            "stt": session.stt.name if session.stt else "off",
            "turn": "off" if session.fusion is None else "+".join(
                x.name for x in (session.fusion.acoustic, session.fusion.semantic) if x),
            "tts": settings.tts_voice if app.state.tts else "off",
            "accurate": app.state.accurate.name if app.state.accurate else "off",
            "echo_mode": agent.echo_mode,
        })
        pump = asyncio.create_task(agent.run_events())
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                # half_duplex deliberately discards the microphone while the
                # agent is audible — no echo, but no barge-in either, because the
                # pipeline never sees the interrupting words. The other two modes
                # keep feeding and let run_events decide.
                if A.OPTS["echo_mode"] == "half_duplex" and agent.mic_is_contaminated():
                    continue
                await session.feed(data)
            elif (txt := msg.get("text")) is not None:
                await _on_text(agent, txt)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("kb session failed")
    finally:
        if pump is not None:
            pump.cancel()
        await agent.interrupt("session closed")
        await session.stop()
        if recorder is not None:
            recorder.close()


async def _on_text(agent: KBSession, txt: str) -> None:
    """Client control messages. Typed questions take the same path as spoken ones."""
    import json

    try:
        m = json.loads(txt)
    except ValueError:
        return
    kind = m.get("type")
    if kind == "playback_started":
        agent._speaking = True
    elif kind in ("playback_done", "playback_ended"):
        agent._speaking = False
        # mic_is_contaminated() reads this to keep the microphone shut for a beat
        # after the last sample is audible. Without it the gate reopens inside the
        # agent's own trailing syllable.
        agent._playback_ended_at = time.monotonic()
    elif kind == "interrupt":
        # The Stop-talking button. Was doing nothing here: the browser dropped its
        # queue while the server carried on generating and synthesising.
        await agent.interrupt("user")
    elif kind == "mark":
        agent.session.mark(str(m.get("label", "mark"))[:64], source="ui")
    elif kind == "ask" and (q := (m.get("text") or "").strip()):
        # Typed question — same provider, so the demo works without a microphone.
        await agent.interrupt("new question")
        # No user_text send here either: on_final emits it once the transcript is
        # settled (accurate=primary). Sending it here too showed the question
        # twice in the interface.
        # No history append here: on_final does it. Doing both puts the question
        # in twice and the model answers a duplicate.
        #
        # And clear the clip, or the accurate pass would re-transcribe whatever
        # audio the LAST spoken turn left behind and answer THAT instead of what
        # was typed — with the typed text replaced on screen by it.
        agent._utt_at_final = None
        agent._reply = asyncio.create_task(agent.on_final(q))


def main() -> None:
    ap = argparse.ArgumentParser(description="Zoho knowledge-base voice demo")
    ap.add_argument("--port", type=int, default=8445)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--voice", default=None, help="Orpheus TTS voice")
    ap.add_argument("--no-correct", action="store_true",
                    help="skip the Zoho-product-name repair before retrieval")
    ap.add_argument("--no-filler", action="store_true",
                    help="stay silent while retrieval runs")
    ap.add_argument("--no-end-call", action="store_true",
                    help="never hang up on a spoken goodbye. The check is a local-LLM "
                         "read of the caller's own turn, run concurrently with "
                         "retrieval, and it fails closed — so this is for demos where "
                         "the call must never end on its own, not a latency switch.")
    ap.add_argument("--filler-after-ms", type=float, default=400.0)
    ap.add_argument("--correct-budget-ms", type=float, default=600.0,
                    help="hard cap on the product-name repair. The shared LLM is "
                         "the only slow thing on this path; this is what stops it "
                         "gating the answer.")
    ap.add_argument("--device", default="cuda")
    # Exposed here because the browser's echo canceller ducks the microphone
    # while the agent is audible — measured at -13 dB on speakers, -3 dB on
    # headphones. Barge-in is looking for speech in that ducked signal, so the
    # bar that is right for ending a turn can be far too high for interrupting
    # one. Lower this if the agent will not let you cut in.
    # The two dials for --echo-mode ladder. Exposed because the right values are
    # a property of the room and the line, so they are found by testing rather
    # than derived: turn the multiple UP if the agent cuts itself off, DOWN if it
    # will not let you interrupt. Every decision logs the resulting bar.
    ap.add_argument("--barge-cap", type=float, default=0.7,
                    help="ladder: hard ceiling on the bar as a fraction of the "
                         "caller's measured voiced level; keeps the bar reachable "
                         "even as residual echo drifts (default 0.7)")
    ap.add_argument("--barge-multiple", type=float, default=8.0,
                    help="ladder: bar = this x the measured echo floor (default 8)")
    ap.add_argument("--barge-sustain-ms", type=float, default=560.0,
                    help="ladder: window length in ms; the bar must be cleared for "
                         "most of it before interrupting (default 500)")
    ap.add_argument("--barge-needed", type=int, default=6,
                    help="ladder: observations within the window that must clear the "
                         "bar — below the window length on purpose, so a dip between "
                         "syllables does not restart the count (default 6 of 7)")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="VAD bar for the speech gate (default 0.60)")
    ap.add_argument("--release-threshold", type=float, default=0.35,
                    help="lower bar to stay in speech once started (default 0.35)")
    ap.add_argument("--denoise", default="dpdfnet4", choices=list(D.CHOICES),
                    help="VAD front-end. dpdfnet4 keeps the widest speech/noise "
                         "margin on our recordings; gtcrn is smaller and faster "
                         "but carries room noise over the VAD bar. "
                         "See scripts/bench_denoise.py.")
    ap.add_argument("--turn", default="both", choices=["off", "s2", "s3", "both"])
    # "off" means "no echo SUPPRESSION", not "no echo handling": the browser's
    # AEC loopback in kb.html cancels the agent's voice from the mic, and this mode
    # is the only one where barge-in fires on a speech onset rather than waiting
    # for words and then text-matching them. Text matching was measured letting
    # degraded echo through as a fake interruption ("Brok okay" matched nothing the
    # agent had said), so with AEC in place onset detection is both faster and
    # safer. Fall back to half_duplex if you are on open speakers with no AEC.
    ap.add_argument("--echo-mode", default="off",
                    choices=["off", "guard", "ladder", "half_duplex"],
                    help="off: barge-in on speech onset, relies on browser AEC "
                         "(default). guard: barge-in on words, echo-checked by "
                         "text. half_duplex: mic ignored while speaking, NO barge-in. ladder: barge-in on mic energy above 3x the echo measured live, sustained ~400ms — see server/speech/echo_guard.py.")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--certs", default=str(ROOT.parent / "demo-v2" / "certs"))
    args = ap.parse_args()

    OPTS.update(collection=args.collection, correct=not args.no_correct,
                filler=not args.no_filler, end_call=not args.no_end_call, filler_after_ms=args.filler_after_ms,
                correct_budget_ms=args.correct_budget_ms)
    A.OPTS.update(device=args.device, denoise=args.denoise, turn=args.turn,
                  echo_mode=args.echo_mode, record=not args.no_record,
                  threshold=args.threshold, release_threshold=args.release_threshold,
                  barge_multiple=args.barge_multiple,
                  barge_sustain_ms=args.barge_sustain_ms,
                  barge_needed=args.barge_needed, barge_cap=args.barge_cap)
    if args.voice:
        settings.tts_voice = args.voice

    cert, key = Path(args.certs) / "cert.pem", Path(args.certs) / "key.pem"
    kw = {}
    if cert.exists() and key.exists():
        # The microphone needs a secure context in the browser, so https is not
        # optional for a page anyone else has to open.
        kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    else:
        log.warning("no certs at %s — serving http, mic will only work on localhost",
                    args.certs)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **kw)


if __name__ == "__main__":
    main()
