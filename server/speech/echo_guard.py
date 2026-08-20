"""Barge-in that measures the echo instead of guessing at it.

The problem the other two modes leave open
------------------------------------------
``echo_mode`` already offers two answers and both are brittle in the same place.
``off`` interrupts on a VAD onset, which fires on room noise the moment the room
is noisy. ``guard`` waits for recognised words, which needs a word to survive the
microphone — and when the browser's canceller ducks the mic, no word does.

Measured on our own recordings, that duck is the whole story. With headphones the
microphone drops 2.8 dB while the agent talks; on speakers it drops 13.4 dB. The
same fixed VAD bar sits in a completely different place relative to the signal in
those two cases, so a threshold tuned in one is wrong in the other, and neither
mode has any way to notice.

The rule
--------
Two facts the server actually knows, combined:

**When can echo exist.** From the instant agent audio goes out on the wire, plus a
render tail long enough to cover jitter buffer, transit, loudspeaker playout and
room decay. That window is not "is a track playing" — the last frame leaving the
server is not the last sound reaching the room, and reopening the mic when
playback "stops" catches the tail of the agent's own voice. :meth:`AgentSession.
mic_is_contaminated` already computes this window from bytes sent, and this
module consumes it rather than re-deriving it.

**Whether this burst is echo.** While the agent is audible, the echo is the
**floor** of the microphone signal, not its average — the caller talking over the
agent shows up as peaks on top of it. So the estimate is a low percentile of a
rolling window rather than a running mean, and the bar is a multiple of that.
Real speech is louder than residual echo and clears the bar; echo does not.
Because the level is measured live, the bar follows this room, this device and
this line without being told about any of them.

An earlier version averaged the level while the VAD gate said "not speech", and
it failed in exactly the case that matters. When the browser ducks the mic the
gate stops opening at all, so every sample — the caller's voice included — got
averaged in as echo. On one session that pushed the estimate to 0.02 and the bar
to 0.06, above almost anything the caller could produce, and barge-in went
silent. Measured on the same audio, the 20th percentile of the window sits at
0.004 while the caller's speech reaches 0.08: a 20x margin instead of none. The
lesson is that the estimator must not depend on the gate, because the gate is
the thing that breaks.

The floor under the bar
-----------------------
Twice the measured echo is the operating rule, but early in a call there is not
yet an echo measurement worth trusting, so the bar is clamped from below by a
floor that descends as the call earns it:

``0.020``  cold — nothing known yet. Deliberately conservative: over-suppress for
           a moment rather than leak echo into the recogniser.
``0.005``  the canceller's adaptive filter has converged (about a second of
           far-end audio), so residual echo is lower and the floor follows.
``0.0015`` this caller's own voiced level has been measured, so the floor is
           scaled to them — a quiet talker and a loud talker stop sharing one
           absolute number. Never goes below this.

Why energy and not the transcript
---------------------------------
``_is_echo`` in ``agent_app`` argues that loudness is the wrong discriminator
because a speaker close to a microphone is as loud as a person. That is true of a
*fixed* loudness threshold and it is why this module does not use one. A bar at
2x the echo measured a moment ago moves when the speaker moves; the objection
does not carry over.

What is NOT settled
-------------------
On our recordings speech during playback lands at about 4.4x the median echo, so
the premise holds. What that data cannot tell us is the false-fire rate, because
barge-in has been broken in both directions and there are correspondingly few
real interruptions in it to score against. This mode logs the measured echo and
the live bar on every decision so that a session's worth of use produces the
evidence the offline analysis was missing. Treat the constants as a starting
point, not a result.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class LadderConfig:
    #: The operating rule: bar = multiple x measured echo.
    #:
    #: 8x rather than the 2x the design was first written around, because the
    #: estimator changed underneath it and only the resulting BAR matters. The
    #: old running mean already had the caller's speech averaged into it and read
    #: ~0.020, so 3x landed the bar at 0.060. The percentile reads the echo floor
    #: at ~0.004, so 8x lands it at 0.032 — a LOWER bar than before, which is why
    #: barge-in works now where 3x-of-the-mean did not. Judge a change to this
    #: number by the bar it produces in the logs, never by the multiple alone.
    echo_multiple: float = 8.0
    #: Floors, descending as the call gives the guard more to work with.
    cold_floor: float = 0.020
    warm_floor: float = 0.005
    calibrated_floor: float = 0.0015
    #: Audible far-end time before the canceller is assumed converged. Measured
    #: as wall-clock time while the echo window is open, NOT as audio handed to
    #: the transport: TTS bursts ahead of real time, so counting bytes sent
    #: declares the filter converged before the sound has been heard.
    warm_after_sec: float = 1.0
    #: No interruption in the first stretch of a reply. The caller has only just
    #: stopped talking and their own voice is still decaying in the microphone;
    #: without this, their trailing syllable reads as a barge-in against an echo
    #: estimate built from that same syllable.
    lockout_sec: float = 1.0
    #: Fraction of the caller's measured voiced level to use as their floor.
    #: Onsets are quieter than sustained vowels, so this is well below 1.
    voiced_fraction: float = 0.15
    #: Hard CEILING on the bar, as a fraction of the caller's measured voiced
    #: level. Without it the bar is free to drift above anything the caller can
    #: physically produce, and barge-in silently stops working.
    #:
    #: This is the failure the ladder shipped with. Residual echo genuinely rises
    #: when the agent speaks louder, so `multiple x echo` legitimately drifts —
    #: measured live it wandered from 0.027 to 0.074 inside one session while the
    #: caller's speech peaked near 0.10. Same interruption, cleared the bar early
    #: in the call and missed it later: the intermittency was the bar moving, not
    #: the speech changing. Capping against the caller's own calibrated level is
    #: what makes the bar reachable BY CONSTRUCTION rather than by luck.
    #: 0.7 and not lower: capping harder keeps bounding the drift but costs
    #: precision fast (0.7 -> 54%, 0.5 -> 36%), because a lower bar turns clean
    #: fast-path fires into marginal ones that need the whole window. 0.7 halves
    #: the worst-case bar — measured p95 0.072 uncapped, 0.036 capped — which is
    #: the drift that caused the misses, for the smallest cost available.
    max_bar_fraction: float = 0.7
    #: A level this many times OVER the bar is treated as decisive and fires
    #: after `fast_ticks` observations instead of the full window.
    #:
    #: The point is that evidence strength and evidence duration are
    #: interchangeable. Measured on real barge-ins the caller's voice peaks
    #: around 11x the bar, which is not an ambiguous signal and should not be
    #: made to wait half a second to prove itself. Marginal levels still serve
    #: the full sentence.
    #: MUST exceed `smooth`. With fast_ticks == smooth a single loud sample
    #: satisfies the fast path on its own: it stays inside the smoothing mean for
    #: `smooth` observations, so "N observations above the ratio" is met without a
    #: second real sample ever arriving. Measured live, 2 of 26 fires were exactly
    #: this — one 80 ms transient cancelling a reply while the CURRENT microphone
    #: reading sat below the bar (`level 0.04079 (mic 0.00205)`). The constructor
    #: enforces the invariant rather than trusting whoever edits these numbers.
    #: 5.0, raised from 2.5 after a false fire in a noisy room. Loud noise bursts
    #: there reach 0.064 while the median REAL barge-in is 0.027 — the room is
    #: louder than the caller, so no absolute threshold separates them. The RATIO
    #: to the measured bar does: the false fire sat at 3.6x, the genuine
    #: interruption in the same session at 5.5x. Corpus-wide this drops 139 fires
    #: to 124 with precision unchanged (59% -> 58%), so it removes marginal ones
    #: rather than real ones.
    fast_ratio: float = 5.0
    fast_ticks: int = 3
    #: Fire when `sustain_ticks` of the last `sustain_window` observations are
    #: over the bar. Observations arrive per audio chunk (80 ms).
    #:
    #: "K of the last M" and NOT "K consecutive". Consecutive was measurably
    #: wrong: speech dips between syllables, every dip reset the counter, and
    #: interruptions that should have landed in ~300 ms took 800-960 ms on real
    #: audio. Tolerating one dip inside the window removes that tail without
    #: widening the window itself. Same rule SpeechGate uses, for the same reason.
    sustain_ticks: int = 6
    sustain_window: int = 7
    #: Echo is read off a rolling window as this percentile — low, because the
    #: floor of the signal is the echo and the peaks are the caller.
    echo_percentile: float = 20.0
    #: Window length in observations (~100 ms each), so 50 is about 5 s.
    echo_window: int = 50
    #: Observations needed before the window is worth reading at all.
    echo_min_samples: int = 10
    #: Observations averaged before comparing against the bar.
    #:
    #: Judging one 80 ms RMS directly does not work: speech oscillates across
    #: any fixed bar at that timescale — syllables, plosives, gaps — so the
    #: over-bar runs measured on real audio were 1-2 observations long and a
    #: "sustained" test could never be satisfied. Loosening the test instead
    #: made isolated noise spikes qualify. Averaging first is what separates
    #: them: continuous speech holds a raised mean, a single spike does not
    #: survive division by `smooth`.
    smooth: int = 2
    #: How fast the caller's voiced level tracks.
    voiced_alpha: float = 0.10
    #: Minimum VAD score, anywhere in the sustain window, before firing.
    #:
    #: Energy alone provably cannot separate this caller's room from their voice:
    #: measured, loud bursts reach 0.064 while their median real interruption is
    #: 0.027 — the room is louder than the person. What does separate them is
    #: speech-likeness. At both false fires the VAD peaked at 0.66-0.68; real
    #: speech sits at 0.95-0.98. Across every recording this veto takes precision
    #: from 57% to 91% while keeping 89% of the genuine fires.
    #:
    #: This is NOT the gate that failed in echo_mode="off". That fired on a gate
    #: ONSET — two frames over 0.6, 32ms, no confirmation. This is a veto on top
    #: of sustained energy, and it reads the score rather than the gate's verdict,
    #: so a ducked microphone weakens it gracefully instead of disabling it.
    vad_floor: float = 0.75


@dataclass
class EchoGuard:
    """Decides whether mic energy during playback is a real interruption.

    One instance per session. Fed one observation per session tick; it keeps its
    own estimates and answers :meth:`observe` with True exactly once per
    interruption, when the evidence has been sustained long enough to act on.
    """

    cfg: LadderConfig = field(default_factory=LadderConfig)

    def __post_init__(self) -> None:
        if self.cfg.fast_ticks <= self.cfg.smooth:
            raise ValueError(
                f"fast_ticks ({self.cfg.fast_ticks}) must exceed smooth "
                f"({self.cfg.smooth}) — otherwise one transient fires barge-in on "
                "its own; see LadderConfig.fast_ticks")
        self._win = deque(maxlen=self.cfg.echo_window)
        self._recent = deque(maxlen=self.cfg.sustain_window)
        self._smooth = deque(maxlen=max(1, self.cfg.smooth))
        self._strong = deque(maxlen=max(1, self.cfg.fast_ticks))
        self._vad = deque(maxlen=max(self.cfg.sustain_window, self.cfg.fast_ticks))

    _win: deque = field(default_factory=lambda: deque(maxlen=50))  # recent levels while audible
    _voiced: float = 0.0        # this caller's voiced level, agent silent
    _far_end_sec: float = 0.0   # AUDIBLE far-end time, drives cold -> warm
    _audible_since: float = 0.0 # start of the current audible run, for the lockout
    _last_obs: float = 0.0      # previous observation, to integrate audible time
    _recent: deque = field(default_factory=lambda: deque(maxlen=4))  # over-bar history
    _smooth: deque = field(default_factory=lambda: deque(maxlen=3))  # envelope window
    _strong: deque = field(default_factory=lambda: deque(maxlen=2))  # fast-path history
    _vad: deque = field(default_factory=lambda: deque(maxlen=7))     # recent VAD scores
    _fired: bool = False        # latch, so one burst fires one interrupt
    _last_log: float = 0.0

    # ── inputs ──
    def far_end_audio(self, secs: float) -> None:
        """Retained for callers that report audio handed to the transport.

        Deliberately does NOT drive the cold->warm step — see `warm_after_sec`.
        Kept so the snapshot can show how far playback is running ahead of sound.
        """

    def reset_burst(self) -> None:
        """Called when the agent stops being audible: nothing to interrupt now."""
        self._recent.clear()
        self._smooth.clear()
        self._strong.clear()
        self._vad.clear()
        self._fired = False
        self._audible_since = 0.0
        self._last_obs = 0.0
        # The window deliberately survives: residual echo is a property of the
        # room and the line, not of this particular reply.

    # ── the bar ──
    @property
    def _echo(self) -> float:
        """Residual echo: a low percentile of the recent audible-window levels.

        Returns 0.0 until the window has enough in it to mean anything, which
        keeps the bar on its floor rather than on a number built from two samples.
        """
        if len(self._win) < self.cfg.echo_min_samples:
            return 0.0
        return float(np.percentile(np.fromiter(self._win, float), self.cfg.echo_percentile))

    @property
    def floor(self) -> float:
        if self._far_end_sec < self.cfg.warm_after_sec:
            return self.cfg.cold_floor
        if self._voiced <= 0.0:
            return self.cfg.warm_floor
        scaled = self._voiced * self.cfg.voiced_fraction
        # Scaled to this caller, but never below the calibrated floor and never
        # above the warm one — calibration may only ever tighten the bar.
        return max(self.cfg.calibrated_floor, min(self.cfg.warm_floor, scaled))

    @property
    def bar(self) -> float:
        want = max(self.cfg.echo_multiple * self._echo, self.floor)
        if self._voiced > 0:
            # Never ask for more than half of what this caller actually produces.
            # The floor still wins if the two ever cross, because leaking echo is
            # worse than a bar that is briefly too low.
            want = max(self.floor, min(want, self._voiced * self.cfg.max_bar_fraction))
        return want

    # ── the decision ──
    def observe(self, rms: float, *, agent_audible: bool, caller_speaking: bool,
                vad: float | None = None) -> bool:
        """Take one tick of microphone level. True means "interrupt now".

        `agent_audible` is the echo window from ``mic_is_contaminated()`` — the
        span where echo can physically be present, not whether a track is
        playing. `caller_speaking` is the VAD gate's opinion, used only to decide
        which estimate this tick feeds; the interruption decision itself does not
        depend on it, which is the point when the gate has stopped working.
        """
        now = time.monotonic()
        if not agent_audible:
            # Clean window: the caller's own level is measurable here, and there
            # is nothing to interrupt.
            if caller_speaking and rms > 0:
                self._voiced = (rms if self._voiced <= 0 else
                                (1 - self.cfg.voiced_alpha) * self._voiced
                                + self.cfg.voiced_alpha * rms)
            self.reset_burst()
            return False

        # Integrate audible wall time — the only honest measure of how long the
        # canceller has had real sound to adapt to.
        if self._audible_since <= 0:
            self._audible_since = now
        elif self._last_obs > 0:
            self._far_end_sec += max(0.0, min(now - self._last_obs, 0.5))
        self._last_obs = now
        run = now - self._audible_since

        # Every audible sample goes in the window, with no reference to the VAD
        # gate. The percentile does the separating; asking the gate which samples
        # are "really" echo is what broke the previous version.
        if rms > 0:
            self._win.append(rms)
        self._smooth.append(rms)
        if vad is not None:
            self._vad.append(float(vad))
        level = float(np.mean(self._smooth)) if self._smooth else rms

        # Measure, THEN decide. While cold there is no echo measurement worth
        # acting on, and inside the lockout the loudest thing in the microphone
        # is the caller's own decaying voice. Either way: observe, never fire.
        if self._far_end_sec < self.cfg.warm_after_sec or run < self.cfg.lockout_sec:
            self._recent.clear()
            self._strong.clear()
            return False

        bar = self.bar
        self._recent.append(level > bar)
        if len(self._recent) > self.cfg.sustain_window:
            # maxlen is set from config at construction, but stay correct if the
            # window was widened after the fact.
            while len(self._recent) > self.cfg.sustain_window:
                self._recent.popleft()

        # Fast path: a decisive margin, held only briefly.
        self._strong.append(level > bar * self.cfg.fast_ratio)
        strong = sum(self._strong) >= self.cfg.fast_ticks

        over = sum(self._recent)
        if over == 0:
            self._fired = False        # burst is over; re-arm
        if self._fired or not (strong or over >= self.cfg.sustain_ticks):
            return False

        # Speech-likeness veto. Loud is not enough — the room can be louder than
        # the caller. Applied last so the logs still show what the energy side
        # decided before this overrode it.
        if self._vad and max(self._vad) < self.cfg.vad_floor:
            log.debug("ladder: energy cleared the bar but vad peaked at %.2f (<%.2f) — not speech",
                      max(self._vad), self.cfg.vad_floor)
            return False

        self._fired = True             # latch until the level drops back under
        log.info("ladder barge-in [%s]: level %.5f (mic %.5f) > bar %.5f "
                 "(echo %.5f x%.1f, floor %.5f, %d/%d over, vad %.2f)",
                 "fast" if strong else "sustained",
                 level, rms, bar, self._echo, self.cfg.echo_multiple, self.floor,
                 over, len(self._recent), max(self._vad) if self._vad else -1.0)
        return True

    def snapshot(self) -> dict:
        """Current state, for the recording and the logs."""
        return {
            "echo": round(self._echo, 6),
            "window": len(self._win),
            "voiced": round(self._voiced, 6),
            "bar": round(self.bar, 6),
            "floor": round(self.floor, 6),
            "cap": round(self._voiced * self.cfg.max_bar_fraction, 6) if self._voiced > 0 else None,
            "far_end_sec": round(self._far_end_sec, 1),
        }
