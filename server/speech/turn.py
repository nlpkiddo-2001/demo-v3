"""The turn layer's interface — "have they finished?", asked two different ways.

S1 (the VAD and gate) answers "is sound happening". It cannot answer whether a
person is *done*, because being done is not an acoustic fact about the current
80 ms — it is a judgement about the whole utterance. handoff.md Part 2 splits
that judgement in two, and this module is the contract both halves implement:

  **S2 — does it SOUND finished?** (``modality = "audio"``)
      Tone and rhythm. A falling pitch usually means done; a mid-sentence pause
      usually does not. Knows nothing about words.

  **S3 — does it MEAN finished?** (``modality = "text"``)
      The transcript. "Book a flight to…" sounds finished by tone and is
      obviously not finished by meaning. This is the only part of the turn layer
      that needs the recogniser.

Keeping them apart matters because they fail differently, and a fused score
built from two independent opinions is only worth having if the opinions really
are independent. It also lets each be tested alone — which is exactly what
handoff Part 4 asks for, and what stops a mystery in the fused system being
untraceable later.

Nothing here decides when to end a turn. These produce scores; the session
decides, gated by S1's silence clock and a hard cap, so a turn always ends
eventually even when every model is unsure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TurnDecision:
    """One detector's read on whether the speaker has finished.

    ``prob`` is the model's raw output. ``complete`` is that raw number put
    against the detector's own threshold — useful for testing a detector alone,
    and ignored by the fusion step, which works from calibrated confidence
    instead so that two models on different scales can be compared at all.
    """

    complete: bool
    prob: float = 0.0


class TurnDetector(ABC):
    """A model that judges whether an utterance is finished.

    ``modality`` tells the session what this detector actually consumes:
      - ``"audio"``: the recent utterance audio (S2)
      - ``"text"``:  the running transcript (S3)
      - ``"both"``:  uses whatever it is given

    Both are handed to :meth:`predict` regardless, so a detector can be swapped
    for one of a different modality without the caller changing. Declaring it
    lets the session skip work it knows will be ignored — there is no point
    assembling eight seconds of audio for a detector that only reads text.
    """

    name: str = "base"
    modality: str = "audio"

    async def start(self) -> None:
        """Load the model. Override if needed."""

    async def stop(self) -> None:
        """Release resources. Override if needed."""

    def reset(self) -> None:
        """Clear any per-utterance state. Override if needed."""

    def confidence(self, prob: float) -> float:
        """Map this detector's raw ``prob`` onto a 0..1 turn-complete confidence.

        The identity by default. It exists because raw outputs are not
        comparable across models — one may sit near 0.9 for everything, another
        may spread across the range — and the fusion step needs a number that
        means the same thing whichever detector produced it. A detector whose
        scale is skewed overrides this rather than forcing every threshold
        downstream to be re-tuned per model.
        """
        return prob

    @abstractmethod
    def predict(self, *, audio16k: np.ndarray, transcript: str) -> TurnDecision:
        """Judge the current utterance.

        Called only when S1 reports a candidate pause, so this may be moderately
        expensive — but it runs inside a thread executor and must never block the
        event loop itself.
        """
