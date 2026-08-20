"""S1 policy — turn per-frame VAD scores into "is the user talking, and for how long".

This is the decision half of S1. :mod:`vad_ten` reports what the model saw; this
file decides what it means, and keeps the speech/silence clocks that (per
handoff.md Part 2) "drive everything downstream". Splitting them this way means
the rule below can be changed, A/B'd and measured without going near the model.

The three handoff fixes that live here
--------------------------------------
**Fix 1 — one spike must not count as speech.** The old code collapsed five
16 ms frames with ``max()``, so a single click could carry an entire 80 ms chunk.
:attr:`GateConfig.mode` replaces that with a rule that needs corroboration:

  ``count`` — at least ``min_frames`` of the five clear the threshold (default)
  ``run``   — at least ``min_frames`` *consecutive* frames clear it (stricter:
              real speech sustains across neighbouring frames, an impulse does
              not)
  ``max``   — the old behaviour, kept deliberately so the same recording can be
              scored both ways and the change can be justified with a number
              instead of an argument

**Fix 2 — confirm speech start.** Onset is not declared on the first speech
chunk but after ``confirm_chunks`` of them. The beginning of the word is not
lost, because the session keeps a 250 ms pre-roll; to spend it correctly the
caller needs to know how far back the speech really began, so a confirmed onset
reports :attr:`GateUpdate.onset_lag_ms`. The pending chunks are credited
retroactively to ``speech_ms`` — they were speech, we were just not sure yet.

**Fix 5 — only real words reset the silence clock.** A cough that fools VAD
resets the silence timer and costs another full wait for nothing. When
``require_word_for_silence_reset`` is on and the caller passes ASR's verdict for
the same window, a chunk only counts as speech if VAD *and* ASR agree. Off by
default: Phase 0 changes no behaviour, and this cannot be switched on until
Parakeet is streaming.

What this file does NOT do
--------------------------
It does not end turns. Deciding the user is *finished* is S2 (sounds done) and S3
(means done) fused together — this layer only reports that speech stopped and
how long ago. It also does not decide cold-start vs warm-resume; it reports
``start`` and ``resume`` as distinct events and lets the session, which is the
only layer that knows whether an STT stream is alive, act on the difference
(that is handoff Fix 4).

Usage
-----
    gate = SpeechGate(GateConfig(threshold=0.6, mode="run", min_frames=2))
    upd = gate.update(vad.frames(audio16k))
    if upd.event == "start":     ...   # cold: send pre-roll, open STT stream
    elif upd.event == "resume":  ...   # warm: same utterance, just keep feeding
    if upd.silence_ms > 800:     ...   # ask the turn layer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .vad_ten import VADFrames

log = logging.getLogger(__name__)

MODES = ("count", "run", "max")


@dataclass
class GateConfig:
    """Tuning for the speech decision. Every default here is provisional."""

    # The frame score above which a frame counts as speech — the bar to *start*.
    #
    # 0.60, measured. Frame-score distributions over the 3695 frames of
    # recordings/agent-20260819-064223 (59 s, one speaker, normal room):
    #
    #   non-speech frames  p50 0.25   p99 0.55   max 0.70
    #   speech frames      p50 0.92   p75 0.96   p95 0.98
    #
    # Raising the onset bar 0.50 -> 0.60 drops non-speech frames that clear it
    # from 38 to 7 (2.1% -> 0.4%). The speech frames it costs are recovered by
    # release_threshold below, which is the whole point of splitting the two.
    threshold: float = 0.60

    # Fix 1. See the module docstring for what each mode means.
    mode: str = "count"
    min_frames: int = 2

    # Hysteresis: once speech is running, frames only need to clear this (lower)
    # bar to keep it running, so a quiet syllable mid-word does not chop the
    # utterance in two. TEN applies no smoothing of its own — its flag output is
    # a plain threshold on the probability — so any hysteresis has to be ours.
    #
    # Now measured, on the same recording as ``threshold``. With one bar at 0.50
    # doing both jobs, 5.1% of genuine speech frames (99 of 1925) fell below it,
    # producing 22 mid-utterance "they stopped talking" events in 59 seconds —
    # one every 2.7 s. Each one starts the silence clock, and a clock that
    # reaches EndpointConfig.stop_ms gets the turn evaluated mid-sentence.
    #
    #   release  speech frames lost      non-speech frames held
    #     0.50      99  (5.1%)              38  (2.1%)
    #     0.40      49  (2.5%)             206  (11.6%)
    #     0.35      33  (1.7%)             321  (18.1%)   <- chosen
    #     0.25      13  (0.7%)             923  (52.1%)
    #
    # 0.35 is the knee: it recovers two thirds of the lost speech frames while
    # the non-speech frames it holds are only held *during* speech, where they
    # are the quiet gaps inside a sentence and belong to the utterance anyway.
    # Below 0.30 the hold becomes indiscriminate and pauses stop being detected.
    #
    # ``None`` restores the old single-bar behaviour for an A/B.
    release_threshold: float | None = 0.35

    # Fix 2. Chunks of speech needed before onset is declared. 1 = declare
    # immediately (the old behaviour).
    confirm_chunks: int = 2

    # Fix 5. Requires the caller to pass ASR's word/blank verdict per chunk.
    require_word_for_silence_reset: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {self.min_frames}")
        if self.confirm_chunks < 1:
            raise ValueError(f"confirm_chunks must be >= 1, got {self.confirm_chunks}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.release_threshold is not None:
            if not 0.0 <= self.release_threshold <= 1.0:
                raise ValueError(f"release_threshold must be in [0, 1], got {self.release_threshold}")
            if self.release_threshold > self.threshold:
                raise ValueError(
                    "release_threshold must be <= threshold — a higher release bar "
                    "would make speech harder to sustain than to start, which is "
                    "hysteresis backwards"
                )


@dataclass
class GateUpdate:
    """The gate's read on one chunk. Everything needed to log or act on it.

    The evidence fields (``n_above``, ``best_run``, ``threshold_used``) are here
    so a measurement run can record *why* a verdict came out the way it did, not
    just what it was — that is what makes a threshold sweep possible afterwards.
    """

    is_speech: bool = False       # this chunk cleared the rule
    in_speech: bool = False       # currently inside a confirmed run of speech
    have_speech: bool = False     # confirmed speech has occurred this utterance
    speech_ms: float = 0.0        # confirmed speech accumulated this utterance
    silence_ms: float = 0.0       # trailing silence since speech last stopped
    event: str | None = None      # "start" | "resume" | "pause" | None
    onset_lag_ms: float = 0.0     # on start/resume: how far back speech began
    onset_pending: bool = False   # a candidate onset is mid-confirmation
    n_above: int = 0              # frames over the threshold
    best_run: int = 0             # longest consecutive run over it
    threshold_used: float = 0.0   # onset or release bar, whichever applied
    dt_ms: float = 0.0            # audio this update accounted for


class SpeechGate:
    """Frames in, speech verdict and clocks out. One instance per session."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self._in_speech = False
        self._have_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._pending_chunks = 0    # consecutive speech chunks not yet confirmed
        self._pending_ms = 0.0      # audio those chunks covered

    # ── the Fix 1 rule ──
    def _frame_rule(self, frames: VADFrames, threshold: float) -> tuple[bool, int, int]:
        """Apply the configured rule. Returns (is_speech, n_above, best_run)."""
        n_above = frames.count_above(threshold)
        best_run = frames.longest_run_above(threshold)
        if self.cfg.mode == "max":
            # Equivalent to max(probs) >= threshold, which is exactly "at least
            # one frame cleared it" — written this way so all three modes share
            # one code path and the comparison stays honest.
            is_speech = n_above >= 1
        elif self.cfg.mode == "run":
            is_speech = best_run >= self.cfg.min_frames
        else:  # "count"
            is_speech = n_above >= self.cfg.min_frames
        return is_speech, n_above, best_run

    # ── main entry point ──
    def update(
        self,
        frames: VADFrames,
        *,
        asr_had_word: bool | None = None,
        dt_ms: float | None = None,
    ) -> GateUpdate:
        """Fold one chunk of frame scores into the speech state.

        ``asr_had_word`` is ASR's verdict for this same window: ``True`` a real
        word came out, ``False`` it ran and produced blank, ``None`` unknown or
        not wired up yet. Only consulted when
        ``require_word_for_silence_reset`` is on, and ``None`` always falls back
        to VAD alone — so an ASR outage degrades to today's behaviour rather than
        silently freezing the speech clock.

        ``dt_ms`` defaults to the audio the frames actually cover. A chunk too
        short to fill a hop yields no frames and therefore advances no clock;
        that audio is buffered in the VAD and counted on the call that completes
        the hop, so no time is lost or double-counted.
        """
        if dt_ms is None:
            dt_ms = frames.duration_ms

        # Hysteresis: a lower bar to keep speech going than to start it.
        threshold = self.cfg.threshold
        if self._in_speech and self.cfg.release_threshold is not None:
            threshold = self.cfg.release_threshold

        is_speech, n_above, best_run = self._frame_rule(frames, threshold)

        # Fix 5: VAD alone is not enough to call it speech when ASR is watching
        # and saw no word. Deliberately narrow — it only applies when the caller
        # opted in AND actually supplied a verdict.
        vad_said_speech = is_speech
        if (
            self.cfg.require_word_for_silence_reset
            and asr_had_word is not None
            and is_speech
            and not asr_had_word
        ):
            is_speech = False

        upd = GateUpdate(
            is_speech=is_speech,
            n_above=n_above,
            best_run=best_run,
            threshold_used=threshold,
            dt_ms=dt_ms,
        )

        if dt_ms <= 0.0:
            # No audio was scored — report state, change nothing.
            upd.in_speech = self._in_speech
            upd.have_speech = self._have_speech
            upd.speech_ms = self._speech_ms
            upd.silence_ms = self._silence_ms
            return upd

        if is_speech:
            if self._in_speech:
                # Already confirmed and still going.
                self._speech_ms += dt_ms
                self._silence_ms = 0.0
            else:
                # Fix 2: candidate onset — corroborate before declaring it.
                self._pending_chunks += 1
                self._pending_ms += dt_ms
                if self._pending_chunks >= self.cfg.confirm_chunks: # we need min of confirm chunks to be equaled with the pending chunks [0.1,0.2,...] & [0.9,0.8,...]
                    upd.event = "resume" if self._have_speech else "start"
                    upd.onset_lag_ms = self._pending_ms
                    # The pending chunks were speech all along; credit them now.
                    self._speech_ms += self._pending_ms
                    self._silence_ms = 0.0
                    self._in_speech = True
                    self._have_speech = True
                    self._pending_chunks = 0
                    self._pending_ms = 0.0
                elif self._have_speech:
                    # Unconfirmed mid-utterance: the clock keeps running. If this
                    # turns into a real onset the time is transferred above, so
                    # nothing is lost either way.
                    self._silence_ms += dt_ms
        else:
            if self._in_speech:
                self._in_speech = False
                upd.event = "pause"
            # A broken run of pending chunks is not an onset — start over.
            self._pending_chunks = 0
            self._pending_ms = 0.0
            if self._have_speech:
                self._silence_ms += dt_ms

        upd.in_speech = self._in_speech
        upd.have_speech = self._have_speech
        upd.speech_ms = self._speech_ms
        upd.silence_ms = self._silence_ms
        # Sound is arriving right now, we are simply not sure yet. Whoever ends
        # turns must not do so on this chunk — see SessionConfig docs.
        upd.onset_pending = self._pending_chunks > 0

        if upd.event and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "[gate] %s thr=%.2f above=%d/%d run=%d speech=%.0fms silence=%.0fms lag=%.0fms%s",
                upd.event, threshold, n_above, len(frames), best_run,
                upd.speech_ms, upd.silence_ms, upd.onset_lag_ms,
                "" if vad_said_speech == is_speech else " (asr veto)",
            )
        return upd

    # ── state ──
    def reset(self) -> None:
        """Clear all per-utterance state. Call when a turn is committed."""
        self._in_speech = False
        self._have_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._pending_chunks = 0
        self._pending_ms = 0.0

    @property
    def onset_pending(self) -> bool:
        return self._pending_chunks > 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def have_speech(self) -> bool:
        return self._have_speech

    @property
    def speech_ms(self) -> float:
        return self._speech_ms

    @property
    def silence_ms(self) -> float:
        return self._silence_ms
