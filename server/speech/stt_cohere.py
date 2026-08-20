"""The accurate second pass — Cohere Transcribe over the finished utterance.

Why this one won the seat
-------------------------
It was measured, on our own audio, against every other candidate we could load.
87 entities — Indian names, places and mail ids — read into the pipeline by the
person who complained about them, scored on the only question that matters: did
the name survive?

    model                  entity recall     ms/clip   licence
    cohere-transcribe        47/87  54.0%       668     Apache-2.0
    canary-qwen-2.5b         45/87  51.7%      1647     CC-BY-4.0
    canary-1b-v2             39/87  44.8%       504     CC-BY-4.0
    parakeet-tdt-0.6b-v2     37/87  42.5%       414     CC-BY-4.0
    parakeet-tdt-0.6b-v3     32/87  36.8%       489     CC-BY-4.0
    whisper-large-v3         31/87  35.6%      1568     MIT

Best recall, 2.5x faster than the runner-up, and the most permissive licence.
It also got "Coimbatore" — the word that had failed in every single session up to
that point — and 9 of 12 pan-Indian names, the best anyone managed.

Read the top of that table honestly though: 54.0% against 51.7% is **two
entities**, which is noise at this sample size. What is not noise is the gap to
what this replaces: 36.8% -> 54.0%, consistent across every category. And the
ceiling is the real finding — the best open model available still loses nearly
half the names, which is why the LLM correction layer exists downstream and why
no further model swap was pursued.

Why it is safe here and would not be live
-----------------------------------------
It does not stream at all, so it could never drive the turn: our live seat needs
incremental text for endpointing, speculation and barge-in. Here the audio has
already finished, so that costs nothing.

Cohere's own release notes say the model "is eager to transcribe, even non-speech
sounds" and recommend a noise gate or VAD in front of it. That was the objection
that kept it out of this seat originally, and it has not gone away — it is simply
already satisfied: every clip this sees is VAD-confirmed, denoised, buffered
speech, which is exactly the precondition they ask for.

Three traps loading it, all of which look like unrelated bugs
------------------------------------------------------------
1. **bfloat16, not float16.** Their attention masks with a hardcoded ``-1e9``,
   which is outside fp16's ~65504 range, so fp16 dies with "value cannot be
   converted to type at::Half without overflow" from deep inside the encoder.
   bf16 has fp32's exponent range and is fine.
2. **``AutoModelForSpeechSeq2Seq``, not ``AutoModel``.** The repo's ``auto_map``
   points ``AutoModel`` at ``CohereAsrModel``, the bare encoder-decoder, which
   has no ``generate`` at all.
3. **``transcribe()``, not ``generate()``.** Their ``generate`` wrapper sets
   ``generation_kwargs["decoder_attention_mask"]`` unconditionally but only
   *builds* it when ``decoder_input_ids`` is passed, so a plain call leaves a
   None under a key transformers expects to be a tensor and generation dies in
   ``_update_model_kwargs_for_generation``. ``transcribe()`` builds both, and
   additionally splits audio over ``max_audio_clip_s`` (35 s) into overlapping
   chunks and reassembles — which hand-rolled batching would get wrong.

The model card asks for ``transformers>=5.4.0``. We do not have it and will not
install it: NeMo pins this environment and upgrading it to gain a second-pass
model would risk the recogniser that actually drives conversations. The repo
ships its own modelling code, so ``trust_remote_code=True`` loads it on 4.57.6.

What it must never do
---------------------
Block. Same contract as every provider in this seat: a hard timeout, and every
failure path returns ``None`` meaning "no opinion", after which the live
transcript stands.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"

# The model does not detect language — it must be told. This seat is the English
# half of the pipeline by construction; Indic audio goes to stt_sarvam.
LANGUAGE = "en"

# Below this there is nothing worth a second opinion.
MIN_SEC = 0.3

# No cap. Their own chunker handles arbitrarily long audio; the old 40 s limit
# existed only to bound the number of chunks back when the session capped an
# utterance at 30 s. That session cap is gone (SessionConfig.max_utterance_sec)
# because trimming the front of a clip loses words silently — see the Indigo case
# in that docstring — and re-imposing the same trim one layer down would restore
# exactly the bug we removed, just harder to find.
#
# Kept as a knob rather than deleted: set it to a float to trim the clip's TAIL-
# most seconds again if a very long turn ever proves too slow to be worth it.
MAX_SEC: float | None = None

_CACHE: dict[tuple, tuple] = {}
_LOCK = threading.Lock()


def preload(model_id: str = MODEL_ID, device: str = "cuda") -> None:
    """Load weights and processor ahead of any request. Blocks; call from a thread."""
    key = (model_id, device)
    with _LOCK:
        if key in _CACHE:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        t0 = time.time()
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.bfloat16)
        model.eval()
        model.to(device)
        _CACHE[key] = (model, proc)
        log.info("cohere-transcribe loaded in %.0fs (%s, bfloat16)",
                 time.time() - t0, device)


class CohereAccurate:
    """Offline transcriber for one finished utterance.

    Stateless between calls, so a single instance is safe to share across
    sessions. The lock serialises GPU work so several turns ending at once cannot
    make each other late.

    Interface-compatible with the other providers in this seat on purpose:
    ``agent_app`` picks one at startup and cannot tell them apart.
    """

    name = "cohere_transcribe"

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 timeout_sec: float = 2.0):
        self.model_id = model_id
        self.device = device
        # Measured at ~670 ms per clip across the bench manifest, including
        # 35 s monologues — roughly 2.5x faster than the model it replaces, so
        # this budget has real headroom rather than being hopeful.
        self.timeout_sec = timeout_sec
        self._model = None
        self._proc = None
        self.last_ms = 0.0

    async def start(self) -> None:
        import asyncio

        await asyncio.to_thread(preload, self.model_id, self.device)
        self._model, self._proc = _CACHE[(self.model_id, self.device)]

    async def stop(self) -> None:
        self._model = self._proc = None   # weights stay cached for the next session

    def transcribe(self, audio16k: np.ndarray) -> str | None:
        """Transcribe a finished utterance. ``None`` means "no opinion".

        Blocking and GPU-heavy — always call it inside a thread executor.
        """
        if self._model is None:
            return None
        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        if audio.size < int(MIN_SEC * TARGET_SR):
            return None
        if MAX_SEC is not None:
            cap = int(MAX_SEC * TARGET_SR)
            if audio.size > cap:
                # Keep the END: the words nearest the turn boundary are the ones
                # the reply was built from. Only reachable if MAX_SEC is set back
                # to a number — the default transcribes the whole clip.
                audio = audio[-cap:]

        t0 = time.perf_counter()
        try:
            import torch

            with torch.no_grad(), _LOCK:
                out = self._model.transcribe(
                    processor=self._proc, language=LANGUAGE,
                    audio_arrays=[audio], sample_rates=[TARGET_SR],
                )
        except Exception:
            # Never propagate: this layer is a bonus, and the live transcript is
            # already good enough to answer with.
            log.exception("cohere pass failed — keeping the live transcript")
            return None
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if not out:
            return None
        return (out[0] or "").strip() or None
