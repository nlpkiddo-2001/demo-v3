"""
The live path — microphone in, speech state and text out.

This is the layer that turns the pieces into a pipeline. Audio arrives from the
browser, and this file drives, in order:

    browser pcm ──► to_float16k ──► TenVAD.frames()  ──► SpeechGate.update()
                          │                                      │
                          ├──────────────► SessionRecorder ◄─────┤
                          │
                          └──► STT, but only over the speech ────► words out

What lives here and nowhere else
--------------------------------
**Fix 4 — the re-onset / double-audio bug.** This is the bug the handoff calls
the highest priority, and it is a bug about *bookkeeping*, which is why it
belongs here rather than in the VAD or the gate. The old code re-sent its 250 ms
buffer every time speech restarted after a pause, so audio that had already been
transcribed got sent again and words came out twice.

The fix is to stop thinking in terms of "buffers to send" and track one number
instead: **how far into the stream have we already fed the STT**. Every chunk is
appended to a ring buffer with an absolute sample position, and feeding always
means "send everything from the fed-up-to mark to now". From that single
invariant both cases fall out for free:

  *Cold start* — a genuinely new utterance. We rewind the mark to just before
  the speech began, so the STT gets a lead-in and the first phoneme is not
  clipped.
  *Warm resume* — same utterance, short pause, stream still alive. The mark is
  already sitting where feeding stopped, so we send exactly the audio recorded
  since then. No re-send, no reset, and equally no gap.

Duplicated audio is then not merely unlikely, it is unrepresentable: the mark
only ever moves forward, so no sample can be fed twice.

**Onset lag is paid for out of the ring, not guessed at.** The gate confirms
speech two chunks late on purpose (Fix 2), so at the moment we hear "start", the
first 160 ms of the word is already behind us. The gate reports exactly how far
back that was, so the rewind is ``onset_lag_ms + lead_in_ms`` — measured, not a
fixed 250 ms constant that happened to be roughly right.

What is deliberately NOT here yet
---------------------------------
Turn-taking. Deciding the user is *finished* needs S2 (sounds done) and S3
(means done), and neither exists yet. Until they do, this session ends a turn on
plain silence — :attr:`SessionConfig.endpoint_silence_ms` — which is the control
baseline the handoff asks for, not the final design. It is marked as such below
so nobody mistakes it for the real endpointer.

Also absent: the per-utterance audio buffer for the accurate second ASR pass. It
has no natural end point until a real turn detector exists, so it arrives with
that work rather than being guessed at now.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import gcd
from typing import AsyncIterator

import numpy as np
from scipy.signal import resample_poly

from .denoise import Denoiser, NullDenoiser
from .fusion import EndpointConfig, Endpointer, TurnFusion
from .measure import SessionRecorder, rms_dbfs
from .speech_gate import GateConfig, SpeechGate
from .vad_ten import TenVAD

log = logging.getLogger(__name__)

TARGET_SR = 16000


def to_float16k(pcm_bytes: bytes, in_rate: int) -> np.ndarray:
    """Convert raw int16 PCM at ``in_rate`` to mono float32 at 16 kHz.

    Polyphase resampling, which keeps the high-frequency detail that both the
    VAD and the recogniser rely on — much better than linear interpolation, and
    the reason every model in this stack sees the same signal.
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if in_rate != TARGET_SR:
        g = gcd(in_rate, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, in_rate // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


@dataclass
class STTUpdate:
    """One incremental result from a recogniser after feeding audio."""

    new_words: list[str] = field(default_factory=list)
    partial: str = ""


class STTProvider(ABC):
    """A streaming recogniser. Deliberately minimal so models stay swappable.

    ``accept_audio`` may be GPU-heavy and is always called inside a thread
    executor, so it never blocks the event loop serving the browser.
    """

    name: str = "base"

    async def start(self) -> None:
        """Load the model. Override if needed."""

    async def stop(self) -> None:
        """Release resources. Override if needed."""

    @abstractmethod
    def accept_audio(self, audio16k: np.ndarray) -> STTUpdate:
        """Feed 16 kHz float32 audio; return whatever words that produced."""

    @abstractmethod
    def partial(self) -> str:
        """The running transcript for the current utterance."""

    @abstractmethod
    def reset(self) -> None:
        """Clear per-utterance state so the next turn starts clean."""


@dataclass
class SessionConfig:
    in_rate: int = 24000            # what the browser sends
    lead_in_ms: float = 120.0       # audio before the detected onset, so soft
                                    # word beginnings are not clipped
    # Feed the recogniser for as long as the turn is open, rather than cutting
    # the feed off partway into the pause.
    #
    # The old behaviour stopped feeding at ``trailing_ms`` into a pause, but the
    # endpointer is not even consulted until ``EndpointConfig.stop_ms`` and may
    # wait until ``settle_ms``. So there was always a window — 300 ms at best,
    # 3.3 s on a settle — in which the turn was still open and the recogniser was
    # receiving nothing. Measured on recordings/agent-20260818-064657, a turn
    # ending on settle:
    #
    #   silence  720 ms  feeding stops, partial = "...my name is uh"
    #   silence 2080 ms  partial STILL "...my name is uh"  (frozen, unfed)
    #   settle fires  ->  flush's 240 ms pad yields "...uh Vipp"
    #   accurate pass, same clip, ungated              ->  "...uh Vipin"
    #
    # The decoder was not short of future audio, it was short of *any* audio.
    # That frozen window is also why the live transcript appears to hang at the
    # end of every turn in the UI.
    #
    # Feeding the whole turn also retires a second hazard: while the feed is
    # stopped the "resume" rewind can only move ``_fed_upto`` FORWARD, so audio
    # in the gap was skipped for good. If the feed never stops, there is no gap.
    #
    # Silence is cheap here — Nemotron was measured to return an empty string on
    # room tone, and one 160 ms step costs far less than a lost word.
    feed_whole_turn: bool = True
    trailing_ms: float = 700.0      # how far into a pause to keep feeding when
                                    # feed_whole_turn is False. Kept so the old
                                    # behaviour is still reachable for an A/B.
    ring_ms: float = 2000.0         # rolling raw-audio history the rewind draws on
    tick_ms: float = 100.0          # min gap between UI state events

    # Used only when no turn detector is attached: the pure-VAD control
    # baseline. With fusion present, the endpointer's graduated bar replaces it.
    endpoint_silence_ms: float = 800.0

    # How much of the current utterance to keep for the turn detectors and the
    # accurate second ASR pass. ``None`` means keep all of it, which is the
    # default: the accurate pass has to see the whole turn.
    #
    # The old 30 s cap kept the LAST 30 s and silently discarded the front. On
    # recordings/agent-20260819-075031 a turn ran 39.4 s, so the accurate pass
    # never saw its first 9.4 seconds — the part where the speaker named the
    # sender they were complaining about:
    #
    #   live     : "...receiving some email from indigo airlines frequently,
    #               also from Air India ... and then Misho, Flipkart..."
    #   accurate : "Airlines frequently also like from Air India Airlines and
    #               then from Misho Flipkart..."          sim 0.55
    #
    # Indigo was in the audio and in the live transcript, and vanished before the
    # LLM saw it — so the ticket it opened named the wrong senders. Trimming the
    # front of a clip nobody is watching is exactly the kind of loss that never
    # shows up as an error.
    #
    # Cost of keeping everything: 64 KB per second of turn (16 kHz float32), so a
    # five-minute turn is ~19 MB, and it is freed the moment the turn commits. S2
    # slices its own last 8 s and is unaffected either way.
    max_utterance_sec: float | None = None
    # Purely informational: log once when a turn passes this, so a wedged session
    # growing without limit is visible rather than silent. Does NOT trim.
    long_utterance_warn_sec: float = 120.0
    min_speech_ms: float = 400.0    # never end a turn on a blip of noise

    # Fix 1 — a turn needs WORDS, not just sound. Measured on a real room take:
    # 7 of 11 turns ended with an empty transcript, because breathing and
    # background voices cleared the speech/silence bars on their own. Once an
    # LLM is attached those become the agent talking to nobody.
    min_chars: int = 2

    # Fix 1's other half. If sound was detected but no words ever arrived, the
    # turn can never satisfy min_chars, so the state would wedge and speech_ms
    # would grow forever. After this much silence we quietly drop the utterance
    # instead — no turn, no reply, and a marker in the recording so the rate
    # stays measurable.
    abandon_silence_ms: float = 3000.0

    # Fix 5 — only real words reset the silence clock, so a cough during a pause
    # no longer buys another full wait. Kept switchable: it is newly possible
    # (the recogniser stays silent on non-speech) and deserves an A/B.
    require_word: bool = False
    word_grace_ms: float = 600.0    # a word this recent counts as "still talking"


class CaptureSession:
    """One browser connection's audio pipeline. Not shared between clients."""

    def __init__(
        self,
        vad: TenVAD,
        gate: SpeechGate,
        stt: STTProvider | None = None,
        recorder: SessionRecorder | None = None,
        cfg: SessionConfig | None = None,
        denoiser: "Denoiser | None" = None,
        fusion: "TurnFusion | None" = None,
        endpoint: "EndpointConfig | None" = None,
    ):
        self.vad = vad
        self.gate = gate
        self.stt = stt
        self.recorder = recorder
        self.cfg = cfg or SessionConfig()
        # Cleaned audio decides *when*; raw audio decides *what*. Measured, not
        # assumed: denoising costs the recogniser accuracy on clean speech and
        # costs it badly under heavy noise, while costing the VAD nothing.
        self.denoiser = denoiser or NullDenoiser()
        self.fusion = fusion
        # With no turn detector, the graduated bar has nothing to grade, so the
        # settle point becomes the plain silence timeout and the endpointer
        # reduces exactly to the old control baseline — one code path either way.
        if endpoint is None:
            endpoint = EndpointConfig(
                # With fusion attached, take EndpointConfig's own settle default
                # rather than pinning it here — it is the number that was tuned
                # against real recordings, and a second copy of it in this file
                # is a copy that goes stale.
                #
                # Without fusion, settle IS the plain silence timeout, and
                # stop_ms has to come down with it or EndpointConfig's own
                # settle > stop invariant fails (stop_ms is 1000, the baseline
                # is 800). stop_ms=0 is the honest value here: there are no
                # models to throttle, so there is nothing to wait before
                # consulting. The turn then ends when the bar reaches zero,
                # which is exactly endpoint_silence_ms — the same pure-VAD
                # baseline as before, via one code path.
                **({} if fusion is not None
                   else {"settle_ms": self.cfg.endpoint_silence_ms,
                         "stop_ms": 0.0}),
                min_speech_ms=self.cfg.min_speech_ms,
            )
        self.endpointer = Endpointer(endpoint)

        self._in_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._out_q: asyncio.Queue[dict | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

        # ── the Fix 4 bookkeeping ──
        # A rolling window of recent audio, plus the absolute stream position of
        # its first sample. Absolute positions are what make "already fed up to
        # here" meaningful across a buffer that is constantly being trimmed.
        self._ring = np.zeros(0, dtype=np.float32)
        self._ring_start = 0        # absolute index of _ring[0]
        self._total = 0             # absolute samples seen this session
        self._fed_upto = 0          # absolute index the STT has consumed to
        self._feeding = False

        self._last_tick = 0.0
        # Fix 5 bookkeeping: when the recogniser last produced a real word, and
        # whether this utterance has produced one at all.
        self._last_word_at = -10**9
        self._had_any_word = False
        # The current utterance's audio, from just before the first word to now.
        # The ring only holds 2 s; S2 wants up to 8 s, so this accumulates
        # separately rather than the ring being made huge for one consumer.
        self._utt = np.zeros(0, dtype=np.float32)
        self._last_utt: np.ndarray | None = None   # the clip of the finished turn
        self._warned_long = False   # one long-utterance warning per utterance

    # ── lifecycle ──
    async def start(self) -> None:
        await self.denoiser.start()
        await self.vad.start()
        if self.fusion is not None:
            await self.fusion.start()
        if self.stt is not None:
            await self.stt.start()
        self._running = True
        self._task = asyncio.create_task(self._worker())
        log.info(
            "capture session ready (vad=%s stt=%s denoise=%s turn=%s in_rate=%d)",
            self.vad.name, self.stt.name if self.stt else "off",
            self.denoiser.name,
            "off" if self.fusion is None else
            "+".join(x.name for x in (self.fusion.acoustic, self.fusion.semantic) if x),
            self.cfg.in_rate,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._in_q.put(None)
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        await self.vad.stop()
        await self.denoiser.stop()
        if self.fusion is not None:
            await self.fusion.stop()
        if self.stt is not None:
            await self.stt.stop()

    # ── io ──
    async def feed(self, pcm_bytes: bytes) -> None:
        """Hand one chunk of browser microphone audio to the pipeline."""
        if self._running:
            await self._in_q.put(pcm_bytes)

    async def events(self) -> AsyncIterator[dict]:
        """Yield pipeline events until the session closes."""
        while True:
            item = await self._out_q.get()
            if item is None:
                return
            yield item

    def last_utterance(self) -> np.ndarray | None:
        """Audio of the utterance that just ended, or None.

        Held separately from the live buffer on purpose. Finalising queues the
        turn event and then clears ``_utt``, so anything reading the live buffer
        after receiving that event races the reset and loses — reliably, since
        the consumer is a different task. This copy is taken before the reset
        and survives until the next utterance ends.
        """
        return self._last_utt if (self._last_utt is not None and self._last_utt.size) else None

    def mark(self, label: str, **fields) -> None:
        """Label the current moment in the recording (the UI's noise buttons)."""
        if self.recorder is not None:
            self.recorder.mark(label, **fields)

    # ── ring buffer ──
    def _append(self, audio: np.ndarray) -> None:
        self._ring = np.concatenate([self._ring, audio])
        self._total += audio.size
        keep = int(self.cfg.ring_ms / 1000.0 * TARGET_SR)
        if self._ring.size > keep:
            drop = self._ring.size - keep
            self._ring = self._ring[drop:]
            self._ring_start += drop
            self._fed_upto = max(self._fed_upto, self._ring_start)

    def _slice_from(self, abs_pos: int) -> np.ndarray:
        """Audio from an absolute stream position to now."""
        lo = max(0, min(abs_pos, self._total) - self._ring_start)
        return self._ring[lo:]

    def _rewind_samples(self, onset_lag_ms: float) -> int:
        """How far back to reach for the start of a word, in raw-stream samples.

        Three terms, and the third is easy to forget: the gate watched the
        *denoised* copy, which runs ``denoiser.delay_ms`` behind the raw audio in
        the ring. So a word the gate places 160 ms back actually began 160 ms +
        that delay back in raw time. Leave the term out and every utterance
        loses its first frames — quietly, and only when a denoiser is fitted.
        """
        ms = onset_lag_ms + self.cfg.lead_in_ms + self.denoiser.delay_ms
        return int(ms / 1000.0 * TARGET_SR)

    # ── worker ──
    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                pcm = await self._in_q.get()
                if pcm is None:
                    break
                audio = to_float16k(pcm, self.cfg.in_rate)
                if audio.size == 0:
                    continue

                self._append(audio)

                # 0) Split the stream. The ring — and therefore everything the
                #    recogniser is ever fed — holds the RAW audio. Only the copy
                #    handed to the VAD is cleaned.
                clean = await loop.run_in_executor(None, self.denoiser.process, audio)

                # 1) VAD — always, on every chunk. It is cheap (~0.5 ms) and it
                #    is the only thing that can tell us speech has started.
                frames = await loop.run_in_executor(None, self.vad.frames, clean)

                # Fix 5 — hand the gate the recogniser's opinion, but only once
                # this utterance has actually produced a word. Before that we
                # pass None (meaning "no opinion"), because the recogniser is
                # always a little behind the audio: vetoing the very first chunks
                # of speech would stop any utterance from ever starting.
                asr_had_word = None
                if self.cfg.require_word and self._had_any_word:
                    grace = int(self.cfg.word_grace_ms / 1000.0 * TARGET_SR)
                    asr_had_word = (self._total - self._last_word_at) <= grace
                upd = self.gate.update(frames, asr_had_word=asr_had_word)

                # 2) STT gating. Feed only over speech: an eager recogniser fed
                #    silence invents words, and everything downstream believes it.
                if upd.event == "start":
                    # Cold start — rewind far enough to cover the confirmation
                    # delay plus a lead-in, so the first phoneme survives.
                    if self.stt is not None:
                        self.stt.reset()
                    # onset_lag_ms is measured back from the END of this chunk,
                    # and _total already includes this chunk — so the rewind is
                    # taken from _total directly. Subtracting the chunk length as
                    # well would reach a further 80 ms into the silence before
                    # the word, which is precisely the audio we gate out.
                    rewind = self._rewind_samples(upd.onset_lag_ms)
                    self._fed_upto = max(self._ring_start, self._total - rewind)
                    # Seed the utterance buffer HERE, not later: the STT block
                    # below advances _fed_upto to _total, after which the same
                    # slice would come back empty and the turn detectors would
                    # be handed nothing at all.
                    self._utt = self._slice_from(self._fed_upto).copy()
                    self._feeding = True
                elif upd.event == "resume":
                    # Warm resume — the mark already sits where feeding stopped,
                    # so carrying on from there re-sends nothing and leaves no
                    # gap. But it must not drag the whole pause along with it:
                    # the audio between the trailing cut-off and the new word is
                    # silence, and a long hold would hand seconds of it to an
                    # eager recogniser. So skip forward to just before the word,
                    # unless feeding stopped more recently than that.
                    #
                    # max() is what keeps the Fix 4 guarantee intact — the mark
                    # only ever moves forward, so nothing can be fed twice.
                    rewind = self._rewind_samples(upd.onset_lag_ms)
                    self._fed_upto = max(self._fed_upto, self._total - rewind)
                    self._feeding = True

                # Keep feeding while the turn is open. ``have_speech`` is exactly
                # "a turn is in progress" — it is set on the first confirmed
                # onset and cleared only by _finalize/_abandon — so this feeds
                # right up to the moment the turn is committed and no further.
                keep_feeding = upd.in_speech or (
                    upd.have_speech and (
                        self.cfg.feed_whole_turn
                        or upd.silence_ms <= self.cfg.trailing_ms
                    )
                )
                if self._feeding and not keep_feeding:
                    self._feeding = False

                if self.stt is not None and self._feeding:
                    pending = self._slice_from(self._fed_upto)
                    if pending.size:
                        self._fed_upto = self._total   # advance BEFORE awaiting,
                                                       # so a slow model cannot
                                                       # cause the same audio to
                                                       # be queued twice
                        out = await loop.run_in_executor(
                            None, self.stt.accept_audio, pending
                        )
                        for word in out.new_words:
                            if word:
                                self._last_word_at = self._total
                                self._had_any_word = True
                                await self._out_q.put({"type": "word", "text": word})

                # 2b) accumulate the utterance. Seeded at the cold start from the
                #     same rewind the recogniser got, so the turn detectors judge
                #     exactly the audio that produced the words — including the
                #     pauses, which is where the cadence lives.
                if upd.event != "start" and self._utt.size:
                    self._utt = np.concatenate([self._utt, audio])
                    if self.cfg.max_utterance_sec is not None:
                        cap = int(self.cfg.max_utterance_sec * TARGET_SR)
                        if self._utt.size > cap:
                            self._utt = self._utt[-cap:]
                    elif not self._warned_long:
                        warn = int(self.cfg.long_utterance_warn_sec * TARGET_SR)
                        if self._utt.size > warn:
                            self._warned_long = True
                            log.warning(
                                "utterance past %.0fs and still open (%.1f MB buffered) — "
                                "not trimming, but check the endpointer is not wedged",
                                self.cfg.long_utterance_warn_sec,
                                self._utt.nbytes / 1e6,
                            )

                # 3) record everything — raw audio, frame scores, gate verdict
                if self.recorder is not None:
                    self.recorder.chunk(
                        audio,
                        vad=max(frames.probs) if frames.probs else None,
                        frames=frames.probs,
                        asr_text=self.stt.partial() if self.stt else None,
                        is_speech=upd.is_speech,
                        dbfs_clean=round(rms_dbfs(clean), 2),
                        n_above=upd.n_above,
                        speech_ms=round(upd.speech_ms),
                        silence_ms=round(upd.silence_ms),
                    )

                # 4) tell the browser
                if upd.event:
                    await self._out_q.put({
                        "type": upd.event,
                        "onset_lag_ms": round(upd.onset_lag_ms),
                        "speech_ms": round(upd.speech_ms),
                    })
                rms = float(np.sqrt((audio ** 2).mean())) if audio.size else 0.0
                # Every chunk, unthrottled and deliberately tiny. The UI tick is
                # throttled to tick_ms for the meters' benefit, but barge-in pays
                # that throttle in latency, so the echo guard gets its own stream
                # at the real chunk rate (80 ms).
                await self._out_q.put({
                    "type": "level",
                    "rms": round(rms, 6),
                    "in_speech": upd.in_speech,
                    # The raw VAD score, not the gate's boolean. The echo guard
                    # needs "how speech-like was this" and NOT "did the gate
                    # open" — the gate is what stops working when the browser
                    # ducks the microphone, but the score still separates a
                    # rustle from a voice.
                    "vad": round(max(frames.probs) if frames.probs else 0.0, 3),
                })
                await self._maybe_tick(upd, frames, rms=rms)

                # 5) endpoint. Fix 1 (a turn needs words) and Fix 2 (never end
                #    while sound is arriving) live inside the endpointer now.
                have_text = True
                if self.stt is not None:
                    have_text = len(self.stt.partial().strip()) >= self.cfg.min_chars

                args = dict(
                    speech_ms=upd.speech_ms, silence_ms=upd.silence_ms,
                    have_text=have_text, onset_pending=upd.onset_pending,
                )
                v = self.endpointer.check(conf=None, **args)

                conf = None
                if v.should_query and self.fusion is not None and upd.have_speech:
                    # Only now — after the endpointer has agreed a pause is worth
                    # judging — do we spend a GPU pass on the turn models.
                    fused = await loop.run_in_executor(
                        None,
                        lambda: self.fusion.evaluate(
                            audio16k=self._utt,
                            transcript=self.stt.partial() if self.stt else "",
                        ),
                    )
                    conf = fused.prob
                    await self._out_q.put({
                        "type": "turn_score",
                        "conf": round(conf, 3),
                        "acoustic": round(fused.p_acoustic, 3),
                        "semantic": None if fused.p_semantic is None else round(fused.p_semantic, 3),
                        "labels": {k: round(x, 3) for k, x in fused.labels.items()},
                        "bar": round(v.eff_threshold, 3),
                        "ms": round(fused.ms),
                    })
                    if self.recorder is not None:
                        self.recorder.mark(
                            "turn_score", conf=round(conf, 3),
                            acoustic=round(fused.p_acoustic, 3),
                            semantic=None if fused.p_semantic is None else round(fused.p_semantic, 3),
                            bar=round(v.eff_threshold, 3),
                        )
                elif self.fusion is None:
                    # No turn models: no opinion. The endpointer then ends the
                    # turn at its settle point, which the constructor set to the
                    # plain silence timeout — the control baseline, unchanged.
                    conf = 0.0

                if conf is not None:
                    v = self.endpointer.check(conf=conf, **args)

                if v.end:
                    await self._finalize(reason=v.reason)
                elif (
                    upd.have_speech and not have_text
                    and upd.silence_ms >= self.cfg.abandon_silence_ms
                ):
                    await self._abandon()
        except Exception:
            log.exception("capture session worker failed")
        finally:
            await self._out_q.put(None)

    async def _maybe_tick(self, upd, frames, rms: float = 0.0) -> None:
        """Throttled state event so the UI can draw a live meter."""
        now = time.monotonic()
        if now - self._last_tick < self.cfg.tick_ms / 1000.0:
            return
        self._last_tick = now
        await self._out_q.put({
            "type": "tick",
            # Linear RMS of the RAW mic chunk. The echo guard works in energy,
            # not in VAD confidence, precisely because the browser's canceller
            # can duck the signal far enough that confidence stops separating
            # anything — see server/speech/echo_guard.py.
            "rms": round(rms, 6),
            "frames": [round(p, 3) for p in frames.probs],
            "is_speech": upd.is_speech,
            "in_speech": upd.in_speech,
            "n_above": upd.n_above,
            "speech_ms": round(upd.speech_ms),
            "silence_ms": round(upd.silence_ms),
            "partial": self.stt.partial() if self.stt else "",
        })

    async def _abandon(self) -> None:
        """Drop an utterance that turned out to have no words in it.

        Sound was detected — breathing, a door, someone talking in the next
        room — but the recogniser never produced anything. There is no turn
        here, so nothing is sent downstream. The recording still gets a marker,
        because how often this happens is the false-alarm rate we are trying to
        drive down, and a silent fix that hides its own workload is worse than
        the bug.
        """
        if self.recorder is not None:
            self.recorder.mark("abandoned", speech_ms=round(self.gate.speech_ms))
        log.info("abandoned utterance: %.0fms of sound, no words", self.gate.speech_ms)
        await self._out_q.put({"type": "abandoned", "speech_ms": round(self.gate.speech_ms)})
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        if self.stt is not None:
            self.stt.reset()
        self.gate.reset()
        self.vad.reset()
        self.denoiser.reset()
        self._feeding = False
        self._last_word_at = -10**9
        self._had_any_word = False
        self._utt = np.zeros(0, dtype=np.float32)
        self._warned_long = False
        self.endpointer.reset()

    async def _finalize(self, reason: str = "silence") -> None:
        """End the current turn and reset per-utterance state.

        ``reason`` records WHY — "confidence" (the models were sure), "settle"
        (we ran out of patience) or "hard_cap" (the safety net). Worth keeping:
        a system that mostly ends on settle is one whose turn models are not
        earning their place, and that is invisible without the label.
        """
        # Snapshot the clip before the event goes out and the buffer is cleared.
        self._last_utt = self._utt.copy() if self._utt.size else None
        text = ""
        if self.stt is not None:
            loop = asyncio.get_running_loop()
            # Hand over anything the recogniser has not seen yet, before asking
            # it for a final answer. With feed_whole_turn on there is normally
            # nothing left, but this is the one place it can be guaranteed rather
            # than assumed — and it still matters when trailing_ms is in charge,
            # or when a ring trim moved _fed_upto forward under us. Real room
            # tone from the ring is also better tail context than flush's
            # synthetic zeros, for the reason flush's own docstring gives.
            pending = self._slice_from(self._fed_upto)
            if pending.size:
                self._fed_upto = self._total
                await loop.run_in_executor(None, self.stt.accept_audio, pending)
            # Drain the tail. The recogniser holds back the newest frames
            # because they need audio that has not arrived — at the end of an
            # utterance it has, so flushing recovers the last word or two that
            # would otherwise be dropped from every single turn.
            flush = getattr(self.stt, "flush", None)
            if callable(flush):
                await loop.run_in_executor(None, flush)
            text = self.stt.partial().strip()

        await self._out_q.put({"type": "final", "text": text, "reason": reason})
        if self.recorder is not None:
            self.recorder.mark("turn_end", text=text, reason=reason)
        log.info("turn end (%s): %r", reason, text[:60])
        # Shared with _abandon so the two paths can never drift apart — in
        # particular the Fix 5 flags, which if left set would let the *next*
        # utterance's opening chunks be vetoed before it had said anything.
        self._reset_utterance()
        # Leave _fed_upto where it is: it is an absolute position in a stream
        # that has not restarted, and the next cold start rewinds it anyway.
