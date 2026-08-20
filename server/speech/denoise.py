"""Noise suppression — clean audio for the decisions, raw audio for the words.

The problem it solves
---------------------
Nothing upstream of us removes noise. The browser is configured with
``noiseSuppression: false``, and there is no denoiser on the server, so every
fan, keyboard tap and cough reaches the VAD exactly as the microphone heard it.
That is the direct cause of two of the handoff's fixes: one spike counting as
speech (Fix 1), and a stray noise resetting the silence clock (Fix 5).

Where it is allowed to apply — and where it is not
--------------------------------------------------
Denoised audio goes to the **VAD and turn detectors**. Raw audio goes to the
**recogniser**. That split is deliberate and is the opposite of the obvious
design, so it is worth stating why.

Front-end enhancement reliably *hurts* modern ASR. The artefacts it introduces —
spectral smearing, unnatural formant transitions — do not match what the
recogniser was trained on, and models of this generation already learned noise
robustness from noisy training data. A 2025 study measuring exactly this found
enhancement degraded word error rate across all noise conditions and every model
tested, including a Parakeet of the same family as ours.

Meanwhile the VAD has no such problem. It is asking "is this speech", not "which
words" — cleaning its input carries no transcription risk at all, because the
transcriber never sees it. So each side gets the signal that suits it, and the
question of which is better for ASR is left to measurement rather than to this
docstring's opinion (see ``scripts/bench_denoise.py``).

The models
----------
Both come through sherpa-onnx, already a dependency, so neither adds a heavy
package. Both are natively 16 kHz (no resampling detour, unlike DeepFilterNet's
48 kHz) and both run on CPU by default on purpose: the GPU is busy with the
recogniser, and a few milliseconds is not worth contending for it.

``gtcrn`` — 523 KB, ~4 ms per 80 ms chunk. The original choice.

``dpdfnet4`` — 11 MB, ~13 ms per 80 ms chunk. The current default, and the
change is worth explaining because it is not the obvious one.

Enhancement quality is not what this pipeline needs from a denoiser. What it
needs is for the VAD to keep telling speech and non-speech apart. Those come
apart badly. GTCRN is a speech *enhancement* model: fed room noise it does not
output silence, it outputs its best reconstruction of the speech it assumes is
buried in there, and that reconstruction is speech-shaped by construction. The
VAD then — correctly — scores it as speech-like. Measured on our own recordings
(``scripts/bench_denoise.py``), GTCRN lifts the VAD score of room noise from
0.40 to 0.76 while lifting real speech only 0.86 → 0.95. The bar is 0.60, so it
carries noise across it and cuts the margin that separates the two from 0.46 to
0.19. In a noisy room that is what fires barge-in on nothing.

DPDFNet-4 lifts speech the same amount (0.86 → 0.95) while leaving noise at
0.46, for a margin of 0.49 — wider than no denoiser at all. Note that the
*larger* DPDFNet variants are worse here, not better: dpdfnet8 scores highest on
enhancement quality and worst on this (noise at 0.80), because more capacity
means a more convincing reconstruction of speech that was never there. Pick this
one on the separation number, not on the model card.

The delay
---------
The streaming denoiser holds one frame internally, so its output runs behind its
input — 16 ms for GTCRN, 10 ms for DPDFNet. Rather than leak that into every caller,
:meth:`Denoiser.process` returns exactly as many samples as it was given, and
the stream is simply delayed by :attr:`delay_ms`. Everything downstream then
shares one consistent timeline, and the delay is a single documented number
instead of a drift nobody accounted for.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_DEFAULT_MODEL = {"gtcrn": "gtcrn_simple.onnx", "dpdfnet": "dpdfnet4.onnx"}


class Denoiser(ABC):
    """Removes noise from a 16 kHz float32 stream.

    Contract, and the whole point of this base class: ``process`` returns
    **exactly as many samples as it is given**. Implementations with internal
    latency absorb it into a fixed delay rather than returning short buffers, so
    callers never have to reconcile two different sample counts.

    One instance per session — implementations carry streaming state. The
    session calls this from a thread executor, one chunk at a time.
    """

    name: str = "base"
    delay_ms: float = 0.0

    async def start(self) -> None:
        """Load the model. Override if needed."""

    async def stop(self) -> None:
        """Release resources. Override if needed."""

    @abstractmethod
    def process(self, audio16k: np.ndarray) -> np.ndarray:
        """Denoise a chunk; return the same number of samples."""

    def reset(self) -> None:
        """Clear streaming state between utterances. Override if needed."""


class NullDenoiser(Denoiser):
    """Passes audio through untouched.

    Not a placeholder — it is how the denoiser gets evaluated. Running the same

    recording through this and through GTCRN is what turns "denoising helps"
    into a number, and it is also the honest default until that number exists.
    """

    name = "none"

    def process(self, audio16k: np.ndarray) -> np.ndarray:
        return np.asarray(audio16k, dtype=np.float32).reshape(-1)


class _SherpaStreamingDenoiser(Denoiser):
    """Shared plumbing for sherpa-onnx's streaming speech denoisers.

    Every model sherpa exposes here has the same shape — build a config, hand it
    one ``.onnx``, feed it chunks — and differs only in which config field the
    path goes into. Subclasses supply that field and a default filename; the
    frame accounting, the delay bookkeeping and the one-for-one sample contract
    all live here, so a new model is a four-line subclass rather than a copy.
    """

    #: sherpa config attribute this model's path belongs to, e.g. ``"gtcrn"``.
    config_key: str = ""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        provider: str = "cpu",
        num_threads: int = 1,
    ):
        self.model_path = Path(model_path) if model_path is not None else (
            MODEL_DIR / _DEFAULT_MODEL[self.config_key])
        self.provider = provider
        self.num_threads = num_threads
        self._den = None
        self._out = np.zeros(0, dtype=np.float32)   # denoised audio not yet returned

    async def start(self) -> None:
        import sherpa_onnx

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"{self.name} model not found at {self.model_path}. Fetch it with:\n"
                f"  {self.fetch_hint()}"
            )

        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig()
        setattr(getattr(cfg.model, self.config_key), "model", str(self.model_path))
        cfg.model.num_threads = self.num_threads
        cfg.model.provider = self.provider
        if not cfg.validate():
            raise RuntimeError(f"sherpa-onnx rejected the denoiser config for {self.model_path}")

        self._den = sherpa_onnx.OnlineSpeechDenoiser(cfg)
        if self._den.sample_rate != TARGET_SR:
            raise RuntimeError(
                f"denoiser expects {self._den.sample_rate} Hz, pipeline is {TARGET_SR} Hz"
            )
        # The model holds one frame back; that frame IS the delay we absorb.
        self.delay_ms = 1000.0 * self._den.frame_shift_in_samples / TARGET_SR
        self.reset()
        log.info(
            "%s denoiser ready (%s, %d thread(s), frame=%d samples, delay=%.0fms)",
            self.name, self.provider, self.num_threads,
            self._den.frame_shift_in_samples, self.delay_ms,
        )

    def fetch_hint(self) -> str:
        return (f"curl -L -o {self.model_path} https://github.com/k2-fsa/sherpa-onnx"
                f"/releases/download/speech-enhancement-models/{self.model_path.name}")

    def process(self, audio16k: np.ndarray) -> np.ndarray:
        """Denoise one chunk, returning the same number of samples.

        The model consumes whole frames and withholds one, so the first call
        comes back short. We pad that once, at the very start of the stream where
        the missing audio is silence anyway, and from then on the buffer runs one
        frame ahead and every call is one-for-one.
        """
        if self._den is None:
            raise RuntimeError(f"{type(self).__name__}.process() called before start()")

        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio

        result = self._den.run(audio, TARGET_SR)
        self._out = np.concatenate([self._out, np.asarray(result.samples, dtype=np.float32)])

        n = audio.size
        if self._out.size < n:
            # Only reachable while priming, and only for the model's one frame.
            out = np.concatenate([np.zeros(n - self._out.size, dtype=np.float32), self._out])
            self._out = np.zeros(0, dtype=np.float32)
            return out

        out, self._out = self._out[:n], self._out[n:].copy()
        return out

    def reset(self) -> None:
        """Clear the model's streaming state and our delay buffer together.

        Both or neither: resetting the model while keeping buffered output would
        splice audio from the previous utterance onto the front of the next one.
        """
        if self._den is not None:
            self._den.reset()
        self._out = np.zeros(0, dtype=np.float32)

    async def stop(self) -> None:
        self._den = None
        self._out = np.zeros(0, dtype=np.float32)


class GtcrnDenoiser(_SherpaStreamingDenoiser):
    """GTCRN speech enhancement, streaming, via sherpa-onnx.

    Kept because it is the baseline every measurement in ``bench_denoise.py`` is
    reported against, and because it is 20x smaller and 3x faster than the
    default. On a quiet mic it is the cheaper choice; see the module docstring
    for why it is no longer the default on a noisy one.
    """

    name = "gtcrn"
    config_key = "gtcrn"

    def fetch_hint(self) -> str:
        return ("huggingface-cli download csukuangfj/speech-enhancement-models "
                "gtcrn_simple.onnx --local-dir models/")


class DpdfnetDenoiser(_SherpaStreamingDenoiser):
    """DPDFNet speech enhancement, streaming, via sherpa-onnx.

    Four sizes ship upstream — baseline, 2, 4, 8, ascending in both cost and
    enhancement quality. ``dpdfnet4`` is the default because it is the one that
    keeps the VAD's speech/noise margin widest, which is not the same ranking as
    enhancement quality and is emphatically not the largest model. Pass a
    different ``model_path`` to try the others; ``bench_denoise.py`` scores them.
    """

    name = "dpdfnet"
    config_key = "dpdfnet"

    def __init__(self, model_path=None, **kwargs):
        super().__init__(model_path, **kwargs)
        # Report the variant, not the family: the session logs and the recording
        # header both carry this string, and "dpdfnet" alone would leave a run
        # ambiguous between four sets of weights.
        self.name = self.model_path.stem


#: Every denoiser the CLI will accept. The dpdfnet sizes are spelled out rather
#: than hidden behind a size argument so that a run is reproducible from its log
#: line alone — "denoise=dpdfnet4" says exactly which weights were loaded.
CHOICES = ("none", "gtcrn", "dpdfnet_baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8")


def make_denoiser(kind: str = "dpdfnet4", **kwargs) -> Denoiser:
    """Build a denoiser by name. ``none`` gives the pass-through."""
    kind = (kind or "none").lower()
    if kind in ("none", "off", "null"):
        return NullDenoiser()
    if kind == "gtcrn":
        return GtcrnDenoiser(**kwargs)
    if kind.startswith("dpdfnet"):
        kwargs.setdefault("model_path", MODEL_DIR / f"{kind}.onnx")
        return DpdfnetDenoiser(**kwargs)
    raise ValueError(f"unknown denoiser {kind!r} (expected one of {', '.join(CHOICES)})")
