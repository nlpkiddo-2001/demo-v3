"""S3 — TEN Turn Detection: does the speaker *mean* to be finished?

The question it answers
-----------------------
"Book a flight to…" sounds finished. The pitch falls, the pause is real, and S2
will happily score it complete. Only the words reveal that it obviously is not.
That is this model's entire job, and it is the one piece of the turn layer that
depends on the recogniser.

It is a Qwen2.5-7B fine-tune that classifies a transcript as **finished**,
**unfinished**, or **wait** ("hold on a second" — the user explicitly asking for
the floor). Only *finished* means the turn is over; the other two both mean keep
listening, for different reasons.

Drive it the way it was trained — this is not a style preference
----------------------------------------------------------------
It is a fine-tune, so it behaves like the classifier it was trained to be only
when driven through its own chat template with an (optionally empty) system
prompt. Writing our own instruction — "reply with one word: finished or
unfinished" — is out of distribution and collapses it toward *unfinished*.
Measured previously at roughly an 80% false-veto rate, which stretched every
endpoint by about 2.9 seconds. So: the model's template, an empty system slot,
and no hand-written task text anywhere.

Why we read logits instead of generating
-----------------------------------------
Generating a word would give a hard label and throw away how sure it was. Fusion
needs a *number*: an "unfinished" at 0.51 and one at 0.99 should not hold the
floor equally hard. So we take one forward pass, read the next-token
distribution, sum the probability mass landing on each label's surface forms, and
renormalise across the three. That yields a calibrated P(finished) in (0, 1) for
the cost of a single forward pass — no sampling, no generation loop.

Cost
----
~15 GB in bfloat16, so it loads only when actually selected. The forward pass is
short (a prompt of a few dozen tokens, one position of output), which is what
makes a 7B model affordable to consult during a pause at all.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .turn import TurnDecision, TurnDetector

log = logging.getLogger(__name__)

MODEL_ID = "TEN-framework/TEN_Turn_Detection"

# The fine-tune emits exactly these. Only "finished" ends a turn: "unfinished"
# is mid-thought, and "wait" is the user explicitly asking us to hold — different
# reasons, same instruction to us.
LABELS = ("finished", "unfinished", "wait")

# Weights are large and shared. The lock also serialises inference, which is
# wanted here: several sessions running 7B forward passes at once would contend
# for the same GPU and make every one of them late.
_CACHE: dict[tuple, tuple] = {}
_LOCK = threading.Lock()


def _first_token_ids(tokenizer, label: str) -> set[int]:
    """Token ids a label could START with, across plausible surface forms.

    The model might emit "finished", " finished" or "Finished", and each
    tokenises differently. Summing over all of them means the label's probability
    does not hinge on guessing which spelling the sampler would have produced.
    """
    ids: set[int] = set()
    for surface in (label, " " + label, label.capitalize(), " " + label.capitalize()):
        toks = tokenizer.encode(surface, add_special_tokens=False)
        if toks:
            ids.add(int(toks[0]))
    return ids


class TenTurn(TurnDetector):
    """Semantic end-of-turn detector. Reads text only; never sees audio."""

    name = "ten_turn"
    modality = "text"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = MODEL_ID,
        system_prompt: str = "",
        max_chars: int = 600,
        quantize: str | None = None,
    ):
        self.device = device
        self.model_id = model_id
        # ``"4bit"`` / ``"8bit"`` trade a little accuracy for a lot of memory.
        # Worth reaching for when the GPU is shared; see the class docstring for
        # the measured cost.
        self.quantize = quantize
        # The model card allows a scenario in the system slot to prime a domain
        # ("You are a booking assistant"). Empty is the trained default and what
        # we use until there is evidence a scenario helps.
        self.system_prompt = system_prompt
        # Long transcripts cost prompt tokens for no benefit: completeness is
        # decided at the end of a sentence, not the start of a paragraph.
        self.max_chars = max_chars
        self._model = None
        self._tok = None
        self._label_ids: dict[str, set[int]] = {}
        self.last_ms = 0.0
        self.last_labels: dict[str, float] = {}

    async def start(self) -> None:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("cuda requested but unavailable — TEN turn on cpu (very slow)")
            self.device = "cpu"

        key = (self.model_id, self.device, self.quantize)
        with _LOCK:
            got = _CACHE.get(key)
            if got is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                log.info("loading TEN turn detection on %s (quantize=%s) ...", self.device, self.quantize or "off")
                t0 = time.time()
                tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                kw = dict(
                    trust_remote_code=True,
                    dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                    device_map=self.device,
                )
                if self.quantize in ("4bit", "8bit"):
                    from transformers import BitsAndBytesConfig

                    kw["quantization_config"] = (
                        BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        if self.quantize == "4bit"
                        else BitsAndBytesConfig(load_in_8bit=True)
                    )
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
                model.eval()
                got = (model, tok)
                _CACHE[key] = got
                log.info("TEN turn detection ready in %.0fs", time.time() - t0)
        self._model, self._tok = got
        self._label_ids = {lab: _first_token_ids(self._tok, lab) for lab in LABELS}

    async def stop(self) -> None:
        self._model = self._tok = None   # weights stay cached

    def predict(self, *, audio16k: np.ndarray = None, transcript: str = "") -> TurnDecision:
        """Score how finished ``transcript`` reads.

        ``audio16k`` is accepted and ignored — this detector is deliberately deaf,
        so that it and S2 fail for genuinely independent reasons.

        An empty transcript returns 0.0. With nothing said there is nothing to
        judge, and "not finished" is the safe direction: the cost of waiting is a
        beat of silence, the cost of ending early is talking over someone.
        """
        import torch

        if self._model is None:
            raise RuntimeError("TenTurn.predict() called before start()")

        text = (transcript or "").strip()
        if not text:
            return TurnDecision(False, 0.0)
        if len(text) > self.max_chars:
            text = text[-self.max_chars:]   # the end decides completeness

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        t0 = time.perf_counter()
        with torch.no_grad(), _LOCK:
            input_ids = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
            # Raw logits at the generation position. Taking the distribution
            # directly — rather than sampling — is what makes the result a
            # calibrated label distribution instead of a top-p filtered one.
            logits = self._model(input_ids=input_ids).logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        vocab = probs.shape[0]
        mass = {}
        for lab in LABELS:
            ids = [i for i in self._label_ids.get(lab, ()) if i < vocab]
            mass[lab] = float(probs[ids].sum().item()) if ids else 0.0

        total = sum(mass.values())
        if total <= 1e-8:
            # The model put no mass on any label — it is not answering the
            # question we think we asked. Hold the floor rather than trust it.
            log.warning("TEN turn: no probability mass on any label; holding")
            self.last_labels = mass
            return TurnDecision(False, 0.0)

        self.last_labels = {k: v / total for k, v in mass.items()}
        p_finished = mass["finished"] / total
        # "complete" compares labels against each other rather than against a
        # fixed cut-off, which is how a three-way classifier is meant to be read.
        # Fusion uses the probability; this flag is for testing the model alone.
        complete = mass["finished"] >= max(mass["unfinished"], mass["wait"])
        return TurnDecision(complete, p_finished)
