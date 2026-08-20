"""S1 — TEN VAD, frame level. Reports what the model saw; decides nothing.

Role in the stack (handoff.md Part 2)
-------------------------------------
S1 answers exactly one question: "is there speech in this audio right now?" It
knows nothing about words and nothing about whether the speaker is finished —
that is S2's and S3's job. This file narrows S1 further still: it is only the
*model wrapper*. It turns audio into per-frame scores and stops there.

Why frames and not one number per chunk
---------------------------------------
The browser streams 80 ms chunks. At TEN's 16 ms hop that is five frames, and
the old implementation collapsed them with ``max()`` — so a single 16 ms click,
pop or cough onset in one frame dragged the whole 80 ms over the line and got
counted as speech. That is handoff Fix 1, and it cannot be fixed by tuning a
threshold, because a loud cough genuinely scores higher than quiet speech.

So this wrapper returns all five scores and deliberately offers no way to
collapse them. There is no ``speech_prob()`` and no ``max()`` anywhere in this
file. The rule for turning frames into a speech/not-speech verdict — how many
frames must clear the line, whether they must be consecutive, how many chunks
confirm an onset — is a policy decision, and it lives one layer up so it can be
changed and measured without touching the model. What this file guarantees is
that the layer above is never handed a number that has already thrown the
evidence away.

Threshold status
----------------
None is applied here. handoff Fix 3 is explicit that the inherited 0.65 was
picked for Silero and is a guess on TEN's different scale, so this file refuses
to bake in any cut-off. It reports probabilities; the analysis pass over a
Phase 0 recording is what sets the number.

``flags`` is TEN's own verdict at whatever ``model_threshold`` the handle was
created with (default 0.5). It is recorded rather than used, so we can check
whether TEN applies internal smoothing/hangover — if ``flag`` ever disagrees
with ``prob >= model_threshold``, it does, and that is worth knowing before we
design our own hysteresis.

Usage
-----
    vad = TenVAD()
    await vad.start()
    out = vad.frames(audio16k)          # -> VADFrames(probs=[...], flags=[...])
    out.count_above(0.5)                # how many frames cleared a line
    out.longest_run_above(0.5)          # longest consecutive run over it
    vad.reset()                         # between utterances (buffer only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000

# TEN's native hop: 256 samples = 16 ms @ 16 kHz. Five of these make up one
# 80 ms browser chunk, and those five are the frames handoff Fix 1 is about.
DEFAULT_HOP = 256


@dataclass
class VADFrames:
    """Per-frame VAD output for one call to :meth:`TenVAD.frames`.

    Pure evidence: the raw numbers the model produced, plus query helpers that
    answer questions *about* those numbers. Nothing here stores a threshold or
    reaches a verdict — every helper takes the threshold as an argument, so this
    object can be re-interrogated at different cut-offs (which is precisely what
    the Fix 3 threshold sweep does over a recording).
    """

    probs: list[float] = field(default_factory=list)
    flags: list[int] = field(default_factory=list)
    hop_ms: float = 1000.0 * DEFAULT_HOP / TARGET_SR

    def __len__(self) -> int:
        return len(self.probs)

    @property
    def duration_ms(self) -> float:
        """Audio covered by these frames (excludes anything left in the buffer)."""
        return len(self.probs) * self.hop_ms

    def count_above(self, threshold: float) -> int:
        """How many frames scored at or above ``threshold``.

        This is the "at least 2 frames" half of handoff Fix 1.
        """
        return sum(1 for p in self.probs if p >= threshold)

    def longest_run_above(self, threshold: float) -> int:
        """Length of the longest consecutive run of frames at or above ``threshold``.

        This is the "or 2 in a row" half of handoff Fix 1. Consecutiveness is the
        stricter test: real speech sustains across neighbouring frames, whereas an
        impulse noise spikes one frame and drops back.
        """
        best = run = 0
        for p in self.probs:
            run = run + 1 if p >= threshold else 0
            best = max(best, run)
        return best


class TenVAD:
    """Frame-level TEN VAD for one audio stream.

    Not thread-safe and not shareable: the underlying handle carries streaming
    state, so give every session its own instance. The session already calls
    into this from a thread executor, one chunk at a time, which satisfies that.

    On state and ``reset()``
    ------------------------
    TEN's C handle keeps history across frames and exposes no reset, so genuinely
    clearing it means destroying and recreating the handle
    (:meth:`reset_model_state`). :meth:`reset` deliberately does *not* do that —
    it only drops our partial-frame buffer.

    That is a real decision, not an oversight. The microphone stream is
    continuous even though utterances are not, and the model's history is how it
    adapts to the room's noise floor. Wiping it at every end-of-turn would throw
    that adaptation away and make the first frames of each utterance the least
    reliable ones — exactly the frames that onset detection depends on.
    """

    name = "ten"

    def __init__(
        self,
        *,
        hop_size: int = DEFAULT_HOP,
        model_threshold: float = 0.5,
        sample_rate: int = TARGET_SR,
    ):
        self.hop_size = int(hop_size)
        self.model_threshold = float(model_threshold)
        self.sample_rate = int(sample_rate)
        self.hop_ms = 1000.0 * self.hop_size / self.sample_rate
        self._vad = None
        # Samples left over when a chunk does not divide evenly into hops. They
        # are carried into the next call so no audio is dropped and no frame
        # boundary drifts. With 1280-sample chunks and a 256 hop this stays
        # empty, but chunk sizes are a client-side detail we should not assume.
        self._tail = np.zeros(0, dtype=np.float32)

    # ── lifecycle ──
    async def start(self) -> None:
        """Load the TEN library and create the streaming handle."""
        from ten_vad import TenVad

        self._vad = TenVad(hop_size=self.hop_size, threshold=self.model_threshold)
        log.info(
            "TEN VAD ready (hop=%d = %.0fms, model_threshold=%.2f)",
            self.hop_size, self.hop_ms, self.model_threshold,
        )

    async def stop(self) -> None:
        """Release the handle. Safe to call more than once."""
        self._vad = None
        self._tail = np.zeros(0, dtype=np.float32)

    # ── inference ──
    def frames(self, audio16k: np.ndarray) -> VADFrames:
        """Score one chunk of 16 kHz mono float32 audio, frame by frame.

        Accepts any chunk length. Whole hops are scored in order; a partial hop
        at the end is buffered for the next call. A chunk shorter than one hop
        therefore returns an empty result — that is correct, not a failure.

        Raises ``RuntimeError`` if :meth:`start` has not been called. Returning
        zeros instead would be worse than useless here: a measurement run would
        quietly fill up with confident-looking silence and we would tune a
        threshold against a VAD that was never actually running.
        """
        if self._vad is None:
            raise RuntimeError("TenVAD.frames() called before start()")

        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._tail, audio]) if self._tail.size else audio

        n_hops = len(buf) // self.hop_size
        n_used = n_hops * self.hop_size
        # Copy the remainder: it is a view into ``buf``, and keeping a view alive
        # would pin the whole chunk in memory for the life of the buffer.
        self._tail = buf[n_used:].copy()
        if n_hops == 0:
            return VADFrames(hop_ms=self.hop_ms)

        # One conversion for the whole chunk rather than per frame. Clipping
        # first matters: without it a sample beyond ±1.0 wraps around on cast and
        # a loud moment reads as the opposite polarity — a spike of pure garbage
        # fed straight into the thing we are trying to measure.
        pcm = np.ascontiguousarray(
            (np.clip(buf[:n_used], -1.0, 1.0) * 32767.0).astype(np.int16)
        )

        probs: list[float] = []
        flags: list[int] = []
        for i in range(0, n_used, self.hop_size):
            # Slicing a C-contiguous int16 array with step 1 stays contiguous,
            # which the binding requires — it hands the raw pointer to C.
            prob, flag = self._vad.process(pcm[i : i + self.hop_size])
            probs.append(float(prob))
            flags.append(int(flag))

        return VADFrames(probs=probs, flags=flags, hop_ms=self.hop_ms)

    # ── state ──
    def reset(self) -> None:
        """Clear the partial-frame buffer. Call between utterances.

        Leaves the model's streaming state intact on purpose — see the class
        docstring for why.
        """
        self._tail = np.zeros(0, dtype=np.float32)

    def reset_model_state(self) -> None:
        """Destroy and recreate the handle, wiping TEN's internal history.

        The only true reset available. Intended for offline work — scoring a
        batch of separate test clips through one instance, where history bleeding
        from the previous clip would contaminate the next — not for the live path.
        """
        if self._vad is None:
            return
        from ten_vad import TenVad

        self._vad = None  # drop first, so the old handle is destroyed before the new one
        self._vad = TenVad(hop_size=self.hop_size, threshold=self.model_threshold)
        self._tail = np.zeros(0, dtype=np.float32)
