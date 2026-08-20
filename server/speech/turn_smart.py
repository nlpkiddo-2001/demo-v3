"""S2 — smart-turn-v2: does the speaker *sound* finished?

What it listens for
-------------------
Prosody, not words. A falling pitch and a settled rhythm read as finished; a
level or rising pitch, or a pause in the middle of a phrase, read as
mid-thought. This is the half of the turn judgement that a transcript cannot
give you: "I want to book a flight to…" and "I want to book a flight" contain
almost the same words and sound completely different.

It is deliberately blind to meaning. That is S3's job, and keeping the two blind
to each other is what makes fusing them worth doing at all.

Loading it
----------
``pipecat-ai/smart-turn-v2`` (BSD-2) ships weights and nothing else — its
architecture, ``Wav2Vec2ForEndpointing``, is custom, and the checkpoint carries
no modelling code, so ``AutoModel`` cannot load it. The module below is
reconstructed to match the checkpoint's own state dict:

    pool_attention : Linear(768, 256) → Tanh → Linear(256, 1)   softmax over time
    classifier     : Linear(768, 256) → LayerNorm → GELU → Dropout(0.1)
                     → Linear(256, 64) → GELU → Linear(64, 1)   → sigmoid

If a future checkpoint changes shape, the load will report missing or unexpected
keys rather than quietly producing a model with random weights in the head — so
:meth:`start` treats that as fatal instead of a warning. A turn detector that
silently scores noise would be worse than none at all, because everything
downstream would still believe it.

What to feed it
---------------
The last few seconds of the current utterance, ending at the pause being judged.
The model was trained on windows up to 8 s and the tail is what carries the
cadence, so longer audio is trimmed from the front, not the back.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .turn import TurnDecision, TurnDetector

log = logging.getLogger(__name__)

MODEL_ID = "pipecat-ai/smart-turn-v2"
SR = 16000
MAX_SEC = 8          # the window the checkpoint was trained on
MIN_SEC = 0.4        # below this there is no cadence to read

# Weights are shared between sessions; the model is stateless per call, so only
# the load needs guarding.
_CACHE: dict[tuple, tuple] = {}
_CACHE_LOCK = threading.Lock()


def _build(device: str):
    """Reconstruct the checkpoint's architecture and load its weights into it."""
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

    class Wav2Vec2ForEndpointing(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model(config)
            # Attention pooling: the model learns *which moments* of the window
            # decide the question, rather than averaging the whole clip — which
            # matters here because the answer lives almost entirely in the last
            # syllable.
            self.pool_attention = nn.Sequential(
                nn.Linear(config.hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, input_values, attention_mask=None):
            hidden = self.wav2vec2(
                input_values, attention_mask=attention_mask
            ).last_hidden_state
            weights = torch.softmax(self.pool_attention(hidden), dim=1)   # [B,T,1]
            pooled = (hidden * weights).sum(dim=1)                        # [B,H]
            return torch.sigmoid(self.classifier(pooled))                 # [B,1]

    cfg = Wav2Vec2Config.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForEndpointing(cfg)
    weights = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    # Anything unaccounted for means our reconstruction has drifted from the
    # checkpoint, and the head would be running on its random initialisation.
    real_missing = [k for k in missing if "masked_spec_embed" not in k]
    if real_missing or unexpected:
        raise RuntimeError(
            f"{MODEL_ID} does not match the expected architecture — "
            f"missing={real_missing[:6]} unexpected={list(unexpected)[:6]}. "
            "Refusing to load: the classifier head would be untrained."
        )
    model.eval().to(device)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    return model, extractor


class SmartTurnV2(TurnDetector):
    """Acoustic end-of-turn detector. One instance per session; weights shared."""

    name = "smart_turn"
    modality = "audio"

    def __init__(self, device: str = "cuda", threshold: float = 0.5):
        # GPU by default, and that is not a preference. Measured: 9 ms on a GPU
        # against 2325 ms on CPU — a quarter of a second is a usable pause, two
        # and a half seconds is not, so on CPU this detector cannot run at all
        # in the live path. It falls back with a warning rather than silently.
        self.device = device
        self.threshold = threshold
        self._model = None
        self._extractor = None
        self.last_ms = 0.0        # inference time of the most recent call

    async def start(self) -> None:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("cuda requested but unavailable — smart-turn-v2 on cpu")
            self.device = "cpu"

        key = (MODEL_ID, self.device)
        with _CACHE_LOCK:
            got = _CACHE.get(key)
            if got is None:
                t0 = time.time()
                got = _build(self.device)
                _CACHE[key] = got
                log.info("smart-turn-v2 loaded in %.1fs (%s)", time.time() - t0, self.device)
        self._model, self._extractor = got

    async def stop(self) -> None:
        self._model = self._extractor = None   # weights stay in _CACHE

    def predict(self, *, audio16k: np.ndarray, transcript: str = "") -> TurnDecision:
        """Score how finished the tail of ``audio16k`` sounds.

        ``transcript`` is accepted and ignored — this detector is deliberately
        deaf to words, and taking the argument anyway keeps it swappable with S3.

        Too little audio returns ``prob=0.0`` (not finished). That is the safe
        direction: claiming "done" on a fragment would cut people off, whereas
        claiming "not done" costs at most one more cycle of waiting.
        """
        import torch

        if self._model is None:
            raise RuntimeError("SmartTurnV2.predict() called before start()")

        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        if audio.size < int(MIN_SEC * SR):
            return TurnDecision(False, 0.0)
        if audio.size > MAX_SEC * SR:
            # Keep the END. The cadence that answers the question is in the last
            # syllable; the beginning of a long utterance says nothing about
            # whether it has finished.
            audio = audio[-MAX_SEC * SR:]

        t0 = time.perf_counter()
        inputs = self._extractor(
            audio, sampling_rate=SR, return_tensors="pt",
            padding="max_length", max_length=MAX_SEC * SR,
            truncation=True, do_normalize=True,
        )
        with torch.no_grad():
            prob = float(
                self._model(inputs.input_values.to(self.device)).squeeze().item()
            )
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return TurnDecision(prob > self.threshold, prob)
