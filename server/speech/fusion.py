"""Turn fusion and endpointing — two opinions and a clock become one decision.

This is the last piece of the turn layer. S2 says how finished the speaker
*sounds*, S3 says how finished they *read*, S1 says how long the silence has
lasted. Something has to turn that into a single yes/no, and this is it.

Two jobs, kept apart on purpose:

  :class:`TurnFusion`  — combine S2 and S3 into one confidence. Scores only.
  :class:`Endpointer`  — decide, from that confidence and the silence clock,
                         whether the turn is over. Decides only.

Why a logistic pool, and not a veto
------------------------------------
The obvious design lets S3 overrule: if the words read unfinished, hold the
floor. It was tried and it failed badly. In log-odds a hard veto is an
*infinite-weight* vote — it annihilates the acoustic evidence no matter how
certain the acoustics were. Combined with a semantic stage that false-vetoed
around 80% of turns, every endpoint hung for roughly 2.9 extra seconds.

So both opinions are pooled instead:

    logit(p) = bias + w_acoustic · logit(p_S2) + w_semantic · logit(p_S3)

A confident acoustic ending overcomes a *weak* semantic doubt; a *strong*
semantic "unfinished" still pulls the result down, but smoothly, and it can
always be outvoted. Every opinion has finite weight, which is the whole point.

Why the confidence bar falls as the pause grows
------------------------------------------------
A fixed bar has a failure mode: when the models never reach it, the turn hangs
until the hard cap, every time. So the bar *drops* — from ``threshold`` at
``stop_ms`` down to zero at ``settle_ms``. A clearly finished sentence ends
almost immediately because high confidence clears a high bar; an ambiguous one
still ends within ``settle_ms``. It is how a person waits a beat longer when you
sound unfinished, and then answers anyway.

Above it all sits the hard cap from handoff Part 2: after ``hard_cap_ms`` the
turn ends regardless of what any model thinks. A pipeline that can hang forever
on an unsure model is not a pipeline.

Cost control
------------
S3 is a 7B model. It is only consulted when S2 already thinks the speaker might
be done (``p_S2 >= semantic_gate``): below that we are going to keep waiting
whatever the words say, so the expensive call would change nothing. In practice
that skips the majority of pause checks.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .turn import TurnDetector

log = logging.getLogger(__name__)

_EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)    
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:                              
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class FusionConfig:
    """Weights for the opinion pool. All provisional until measured on real turns."""

    w_acoustic: float = 1.0
    w_semantic: float = 1.0
    bias: float = 0.0
    # Below this acoustic confidence we skip S3 entirely. Set to 0 to always ask
    # (useful when measuring the two stages against each other).
    semantic_gate: float = 0.25


@dataclass
class FusedDecision:
    """One fused read, with both inputs kept so a decision can be explained."""

    prob: float = 0.0                 # fused P(turn complete)
    p_acoustic: float = 0.0
    p_semantic: float | None = None   # None when S3 was skipped or absent
    semantic_ran: bool = False
    labels: dict = field(default_factory=dict)   # S3's three-way distribution
    ms: float = 0.0                   # total inference time for this call


class TurnFusion:
    """
    Combines S2 and S3 into a single turn-complete confidence.

    Either detector may be ``None``. With only one present its opinion passes
    through unchanged, which is what makes the stack degrade sensibly: lose S3
    and turn-taking gets less clever, not broken.
    """

    def __init__(
        self,
        acoustic: TurnDetector | None = None,
        semantic: TurnDetector | None = None,
        cfg: FusionConfig | None = None,
    ):
        self.acoustic = acoustic
        self.semantic = semantic
        self.cfg = cfg or FusionConfig()

    async def start(self) -> None:
        if self.acoustic is not None:
            await self.acoustic.start()
        if self.semantic is not None:
            await self.semantic.start()

    async def stop(self) -> None:
        if self.acoustic is not None:
            await self.acoustic.stop()
        if self.semantic is not None:
            await self.semantic.stop()

    def evaluate(self, *, audio16k: np.ndarray, transcript: str) -> FusedDecision:
        """Score the current utterance. Runs inside a thread executor."""
        t0 = time.perf_counter()
        out = FusedDecision()

        if self.acoustic is not None:
            d = self.acoustic.predict(audio16k=audio16k, transcript=transcript)
            out.p_acoustic = self.acoustic.confidence(d.prob)
        elif self.semantic is None:
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out
        else:
            out.p_acoustic = 0.5      # no acoustic opinion — neutral in the pool

        ask_semantic = (
            self.semantic is not None
            and transcript.strip()
            and out.p_acoustic >= self.cfg.semantic_gate
        )
        if ask_semantic:
            d = self.semantic.predict(audio16k=audio16k, transcript=transcript)
            out.p_semantic = self.semantic.confidence(d.prob)
            out.semantic_ran = True
            out.labels = dict(getattr(self.semantic, "last_labels", {}) or {})

        if out.semantic_ran and self.acoustic is not None:
            z = (
                self.cfg.bias
                + self.cfg.w_acoustic * _logit(out.p_acoustic)
                + self.cfg.w_semantic * _logit(out.p_semantic)
            )
            out.prob = _sigmoid(z)
        elif out.semantic_ran:
            out.prob = out.p_semantic
        else:
            # S3 skipped or unavailable — the acoustic opinion stands alone
            # rather than being diluted by a made-up neutral value.
            out.prob = out.p_acoustic

        out.ms = (time.perf_counter() - t0) * 1000.0
        return out


@dataclass
class EndpointConfig:
    """When to call the turn over. Every number here is provisional."""

    # Tuned for the demo's actual failure mode. Cutting someone off mid-sentence
    # is expensive and obvious; waiting an extra second is cheap and nobody
    # notices. So every number here is biased toward patience.
    #
    # The old values (400 / 0.70 / 2000 / 8000) cut real turns in half. Measured
    # on recordings/agent-20260819-052639, where 17 turns produced:
    #   'Uh yes from Chen'    ended at bar=0.70 on conf=0.99  (was "from Chennai")
    #   'Malaysia, Maurit'    ended on settle at 3.8s pause
    #   'It would be under'   ended on settle   (was "put it in underneath")
    # The first is the turn models being confidently wrong 400 ms into a pause;
    # the rest are the settle timer running out while the speaker was still
    # thinking. Both are fixed by giving the speaker more room.
    # Silence before the turn layer is consulted at all.
    #
    # 700 was measured to sit inside the speaker's own rhythm. Bucketing every
    # mid-sentence gap in recordings/agent-20260819-064223 by how long it lasted
    # before speech came back:
    #
    #    160-319 ms  ########  (8)
    #    480-639 ms  ###       (3)
    #    640-799 ms  #######   (7)   <- stop_ms=700 fell in here
    #   1280-1439 ms  #        (1)
    #
    # Seven normal thinking pauses in one minute crossed the line and got the
    # turn evaluated mid-sentence. Once evaluated, a confident acoustic head
    # against a falling bar ends the turn — which is how 'Bro, are you close to
    # me? Okay, the thing is...' was committed as 'Brok okay'.
    #
    # 1000 clears that cluster with room to spare. It costs 300 ms of extra
    # latency on genuinely finished sentences; being interrupted costs far more.
    stop_ms: float = 1000.0
    threshold: float = 0.85         # confidence needed at stop_ms — high, because
                                    # ending early on a wrong guess is the bug
    settle_ms: float = 4000.0       # by this pause an unsure turn ends anyway
    hard_cap_ms: float = 12000.0    # handoff Part 2: a turn ALWAYS ends eventually
    min_speech_ms: float = 400.0    # never end on a blip of noise
    check_interval_ms: float = 200.0  # minimum gap between model queries

    def __post_init__(self) -> None:
        if self.settle_ms <= self.stop_ms:
            raise ValueError("settle_ms must exceed stop_ms — the bar needs room to fall")
        if self.hard_cap_ms < self.settle_ms:
            raise ValueError("hard_cap_ms must be >= settle_ms, or the cap fires first")


@dataclass
class EndpointVerdict:
    end: bool = False
    reason: str = ""              # "confidence" | "settle" | "hard_cap" | ""
    should_query: bool = False    # is it worth running the models this chunk
    eff_threshold: float = 0.0    # the bar at this moment, after the ramp


class Endpointer:
    """Turns a confidence and a silence clock into an end-of-turn decision.

    Deliberately has no models in it. It is pure arithmetic over numbers other
    layers produced, so its behaviour can be tested exhaustively without loading
    a single weight — which matters, because this is the logic that decides
    whether a person gets interrupted.
    """

    def __init__(self, cfg: EndpointConfig | None = None):
        self.cfg = cfg or EndpointConfig()
        self._last_query_ms = -1e9

    def reset(self) -> None:
        self._last_query_ms = -1e9

    def effective_threshold(self, silence_ms: float) -> float:
        """The confidence bar right now — it falls as the pause lengthens."""
        span = self.cfg.settle_ms - self.cfg.stop_ms
        frac = min(max((silence_ms - self.cfg.stop_ms) / span, 0.0), 1.0)
        return self.cfg.threshold * (1.0 - frac)

    def check(
        self,
        *,
        speech_ms: float,
        silence_ms: float,
        have_text: bool,
        conf: float | None = None,
        onset_pending: bool = False,
    ) -> EndpointVerdict:
        """Decide whether this chunk ends the turn.

        Call every chunk. ``conf`` is the fused confidence if it was computed
        this chunk, otherwise ``None`` — the returned ``should_query`` says when
        computing one is worth the GPU time.
        """
        v = EndpointVerdict(eff_threshold=self.effective_threshold(silence_ms))

        # Sound is arriving right now, unconfirmed. Whatever the models said a
        # moment ago, ending here would cut into a word already in progress.
        if onset_pending:
            return v

        if speech_ms < self.cfg.min_speech_ms:
            return v

        # The hard cap outranks everything, including "we have no words". A turn
        # that never ends is worse than one that ends wrongly.
        if silence_ms >= self.cfg.hard_cap_ms:
            v.end, v.reason = True, "hard_cap"
            return v

        if not have_text:
            return v

        if silence_ms < self.cfg.stop_ms:
            return v

        # Worth asking the models? Throttled so a long pause does not run a 7B
        # on every 80 ms chunk.
        if silence_ms - self._last_query_ms >= self.cfg.check_interval_ms:
            v.should_query = True

        if conf is None:
            return v
        self._last_query_ms = silence_ms

        # Settle is checked FIRST, and that ordering is about honesty rather than
        # behaviour. At settle_ms the bar has fallen to zero, so `conf >= bar` is
        # trivially true for any confidence — test it first and every giving-up
        # gets logged as "the model was confident", which is exactly backwards.
        # Ending here means we ran out of patience, and the logs should say so.
        if silence_ms >= self.cfg.settle_ms:
            v.end, v.reason = True, "settle"
        elif conf >= v.eff_threshold:
            v.end, v.reason = True, "confidence"
        return v
