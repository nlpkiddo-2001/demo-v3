"""The accurate second pass — Parakeet v3 over the finished utterance.

Why a second recogniser at all
------------------------------
The live model has to answer "what was said" before the sentence is over, which
costs accuracy: it decides each word with only 160 ms of future audio. Once a
turn ends we have the whole utterance sitting in a buffer, and a model allowed
to look at all of it does measurably better.

The catch is that the better answer arrives about a second late, and a second of
silence after every sentence would ruin the conversation. So it never sits in
the critical path (see ``agent_app``): the reply starts on the live transcript
immediately, this runs alongside, and the reply is only restarted if the two
transcripts genuinely disagree. Late and unused is the normal case.

Why this model
--------------
``parakeet-tdt-0.6b-v3`` loses to nothing on accuracy in its size class, loads
through NeMo exactly as the live model does — no second serving stack — and its
one disqualifying trait for live use is irrelevant here. It needs two seconds of
*future* audio to commit a word, which is fatal when you are streaming and free
when the audio has already finished.

The alternative in the original plan was Cohere Transcribe. It would have meant
a separate vLLM server, and it is "eager": fed anything that is not speech it
invents words. We would have been relying entirely on the VAD to never hand it
silence. Parakeet asks less of us.

What it must never do
---------------------
Block. It has a hard timeout and every failure path returns ``None``, meaning
"no opinion" — the live transcript then stands. A pipeline that stalls because
its optional accuracy layer is slow is worse than one without the layer.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

_WORD = re.compile(r"[^a-z0-9' ]+")


def _norm(text: str) -> list[str]:
    """Lowercase, strip punctuation, split — so only real word differences count."""
    return _WORD.sub(" ", (text or "").lower()).split()


def similarity(live: str, accurate: str) -> float:
    """Word-sequence similarity of two transcripts, 0..1, ignoring case and punctuation."""
    a, b = _norm(live), _norm(accurate)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def transcripts_differ(live: str, accurate: str, threshold: float = 0.85) -> bool:
    """Is the accurate transcript worth restarting the reply for?

    Restarting cuts the agent off mid-sentence, so it has to buy something real.
    Casing and punctuation are ignored; below ``threshold`` word similarity the
    two are treated as different sentences.

    THIS THRESHOLD IS NOT YET CALIBRATED, and one number probably cannot do the
    job. Measured on hand-written pairs:

        "okay thanks"                vs "OK thanks"   -> 0.50   (same meaning)
        "my number is four one five"  vs "...is 415"   -> 0.60   (different!)

    A short phrase where one word changes looks like a 50% rewrite, while a long
    one where the only change is the actual content scores higher. Any single
    cut-off gets one of those two cases wrong, which is exactly why the accurate
    pass runs in SHADOW mode by default: it logs both transcripts and swaps
    nothing until there are real pairs from real audio to calibrate against.
    """
    if not _norm(accurate):
        return False          # no opinion from the accurate pass
    if not _norm(live):
        return True           # live heard nothing at all, accurate did
    return similarity(live, accurate) < threshold

TARGET_SR = 16000
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# Below this there is nothing worth a second opinion, and NeMo would rather not
# be handed a fragment.
MIN_SEC = 0.3

_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()


def preload(model_id: str = MODEL_ID, device: str = "cuda") -> None:
    """Load the weights ahead of any request. Blocks; call it from a thread."""
    key = (model_id, device)
    with _LOCK:
        if key in _CACHE:
            return
        import nemo.collections.asr as nemo_asr

        t0 = time.time()
        m = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        m.eval()
        m.to(device)
        _CACHE[key] = m
        log.info("parakeet v3 loaded in %.0fs (%s)", time.time() - t0, device)


class ParakeetAccurate:
    """Offline transcriber for one finished utterance.

    Stateless between calls — there is no stream to keep, only a clip to read —
    so a single instance is safe to share across sessions. The lock serialises
    GPU work so several turns ending at once cannot make each other late.
    """

    name = "parakeet_v3"

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 timeout_sec: float = 1.5):
        self.model_id = model_id
        self.device = device
        # The budget the reply is willing to wait before this result stops being
        # useful. Exceeding it is not an error; it just means the live
        # transcript keeps the floor.
        self.timeout_sec = timeout_sec
        self._model = None
        self.last_ms = 0.0

    async def start(self) -> None:
        import asyncio

        await asyncio.to_thread(preload, self.model_id, self.device)
        self._model = _CACHE[(self.model_id, self.device)]

    async def stop(self) -> None:
        self._model = None      # weights stay in _CACHE for the next session

    def transcribe(self, audio16k: np.ndarray) -> str | None:
        """Transcribe a finished utterance. ``None`` means "no opinion".

        Blocking and GPU-heavy — always call it inside a thread executor.
        """
        if self._model is None:
            return None
        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        if audio.size < int(MIN_SEC * TARGET_SR):
            return None

        t0 = time.perf_counter()
        try:
            import torch

            with torch.no_grad(), _LOCK:
                out = self._model.transcribe([audio], batch_size=1, verbose=False)
        except Exception:
            # Never propagate: this layer is a bonus, and the live transcript is
            # already good enough to answer with.
            log.exception("parakeet pass failed — keeping the live transcript")
            return None
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if not out:
            return None
        first = out[0]
        text = first.text if hasattr(first, "text") else str(first)
        return (text or "").strip() or None
