"""The accurate second pass — Canary-Qwen-2.5B over the finished utterance.

Why replace Parakeet here
-------------------------
Parakeet v3 was chosen for this seat on its size-class accuracy, and it holds up
better than we expected. Measured on AppTek Call-Center Dialogues — 2026,
*conversational* English, accent-labelled, which is the distribution we actually
serve rather than LibriSpeech:

    model                en_IN    en_US    en_GB    overall
    Phi-4 Multimodal      9.4      6.2     11.0      10.0
    canary-qwen-2.5b      9.5      5.8     10.1       9.2
    parakeet-tdt-0.6b-v3  9.7      5.5     10.3       9.2
    qwen3-asr-1.7b       10.3      5.0      9.2       8.3
    whisper-large-v3     11.9      9.9     16.6      15.0

Read that table honestly: on Indian English this model buys **0.2 WER** over the
one it replaces. That is inside the noise, and nobody should expect to hear the
difference. What the table really says is that Parakeet was already at the open
-source ceiling for en_IN, and that no model swap was ever going to fix names and
places — those are a rare-proper-noun problem, and a recogniser that gets 9.5% of
words wrong can still get 100% of unfamiliar surnames wrong without moving WER.

So this is chosen on the tie-breakers, not the headline:

  * It leads the Open ASR Leaderboard at 5.63% mean WER — the best English
    number available to us at all.
  * It has the *narrowest* accent spread of the strong models: +3.7 points from
    en_US to en_IN, against Parakeet's +4.2. Uneven quality across our three
    accent groups is the complaint that started this, and spread is the number
    that describes it.
  * It loads through NeMo, exactly as Parakeet and the live model do. That was
    the constraint that ruled Cohere Transcribe out of this seat and it has not
    softened: one serving stack, or it does not go in.

Why it is safe here and would not be live
-----------------------------------------
This is a SALM — a FastConformer encoder bolted onto Qwen3-1.7B — so it is an
LLM-decoder recogniser, the family that produced the Qwen3-ASR repetition
spirals. Three things keep that in its box, and all three are load-bearing:

  1. **ASR mode, not LLM mode.** A plain ``generate`` on the audio prompt is
     transcription. ``llm.disable_adapter()`` is the *other* mode — text-only
     reasoning over a transcript — and is exactly what we must never call here.
  2. **A hard token ceiling.** ``max_new_tokens`` bounds a repetition spiral to
     a bad transcript instead of a hang. The model card measures 138.1 chars/min
     of hallucination on MUSAN non-speech, so this is not hypothetical.
  3. **It is never fed silence.** The gate hands this seat VAD-confirmed,
     denoised, buffered speech. That is the same protection Parakeet relies on,
     and it is why an eager model is tolerable here and would not be live.

Nothing about it suits the live seat: 418 RTFx against Parakeet's 3333, and no
streaming at all. In this seat that costs roughly 12 ms on a five-second clip,
which is free.

Limits worth knowing
--------------------
English only — the encoder saw other languages in pretraining but the model was
trained on English data and NVIDIA does not expect it to hold up elsewhere. That
is fine, because this is the English half of the pipeline by construction; Indic
audio goes to :mod:`stt_sarvam` instead. Trained to 40 s of audio and a 1024
-token total sequence, both comfortably above ``SessionConfig.max_utterance_sec``.

It also wants a **file path**, not an array — NeMo's SALM does not take audio in
memory. So each clip goes through a temporary WAV. On tmpfs that is well under a
millisecond for a normal turn, and it is bounded by the same timeout as
everything else here.

What it must never do
---------------------
Block. Same contract as the pass it replaces: a hard timeout, and every failure
path returns ``None`` meaning "no opinion", after which the live transcript
stands. An optional accuracy layer that can stall the pipeline is worse than no
accuracy layer.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import time
import typing
import wave

import numpy as np

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _py310_concatenate_shim():
    """Let NeMo's speechlm2 collection import on Python 3.10.

    Not optional and not cosmetic: without this the import fails outright on
    this box. ``nemo.collections.speechlm2.models`` eagerly pulls in an
    unrelated duplex-TTS module whose codec annotates

        window_fn: Callable[Concatenate[int, ...], Tensor]

    and ``Concatenate[X, ...]`` — Ellipsis in the tail position — is a Python
    3.11 relaxation of PEP 612. On 3.10 ``typing`` rejects it, so importing SALM
    raises a TypeError from a file that has nothing to do with ASR. The venv
    here is 3.10.13 and no newer interpreter is installed.

    So we substitute a throwaway ParamSpec for the Ellipsis, which makes the
    annotation legal and is harmless because nothing ever reads it — it is a
    type hint on a default argument in a model we do not load.

    Scoped to the import and reverted immediately. A speech server that
    permanently monkeypatches ``typing`` for every other library in the process
    is a debugging problem waiting to happen; a patch that exists for the
    duration of one import is a compatibility note.

    Delete this the day the interpreter moves to 3.11+.
    """
    original = typing.Concatenate
    placeholder = typing.ParamSpec("_NemoShimP")

    @typing._SpecialForm
    def _concatenate(self, parameters):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        if parameters and parameters[-1] is Ellipsis:
            parameters = parameters[:-1] + (placeholder,)
        return original[parameters]

    typing.Concatenate = _concatenate
    try:
        yield
    finally:
        typing.Concatenate = original

TARGET_SR = 16000
MODEL_ID = "nvidia/canary-qwen-2.5b"

# Below this there is nothing worth a second opinion, and the encoder would
# rather not be handed a fragment.
MIN_SEC = 0.3

# The model was trained to 40 s. Our own utterance cap is 30 s, so this only
# ever fires if that config is raised — but silently degrading at 41 s is the
# kind of thing nobody finds for a month, so clamp rather than trust.
MAX_SEC = 40.0

# A spoken turn is a sentence or two. This is loose enough never to truncate a
# real answer and tight enough that a repetition spiral ends as a bad transcript
# rather than a hang the timeout has to clean up.
MAX_NEW_TOKENS = 128

# The instruction the model card documents for transcription. The audio locator
# tag is substituted per-model, hence the format at call time rather than here.
ASR_PROMPT = "Transcribe the following: {audio_tag}"

_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()


def preload(model_id: str = MODEL_ID, device: str = "cuda") -> None:
    """Load the weights ahead of any request. Blocks; call it from a thread."""
    key = (model_id, device)
    with _LOCK:
        if key in _CACHE:
            return
        with _py310_concatenate_shim():
            # The documented path is ``from nemo.collections.speechlm2.models
            # import SALM``. We reach past that package's __init__ deliberately:
            # it imports every model in the collection, and only SALM is wanted
            # here. Fewer megatron/TTS imports to go wrong, and faster.
            from nemo.collections.speechlm2.models.salm import SALM

        t0 = time.time()
        m = SALM.from_pretrained(model_id)
        m.eval()
        m.to(device)
        _CACHE[key] = m
        log.info("canary-qwen-2.5b loaded in %.0fs (%s)", time.time() - t0, device)


def _write_wav(audio: np.ndarray, path: str) -> None:
    """Dump float32 mono to a 16 kHz 16-bit WAV, because SALM wants a path.

    Clipped before the cast: float audio from the ring can sit slightly outside
    [-1, 1] after denoising, and letting that wrap round in int16 turns a loud
    word into white noise rather than a clipped one.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm.tobytes())


class CanaryAccurate:
    """Offline transcriber for one finished utterance.

    Stateless between calls — there is no stream to keep, only a clip to read —
    so a single instance is safe to share across sessions. The lock serialises
    GPU work so several turns ending at once cannot make each other late.

    Interface-compatible with :class:`stt_parakeet.ParakeetAccurate` on purpose:
    ``agent_app`` picks one of them at startup and cannot tell them apart.
    """

    name = "canary_qwen_2.5b"

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 timeout_sec: float = 2.5):
        self.model_id = model_id
        self.device = device
        # The budget the reply is willing to wait before this result stops being
        # useful. Exceeding it is not an error; it just means the live
        # transcript keeps the floor.
        #
        # Larger than Parakeet's 1.5 s, and measured rather than guessed. This
        # model generates its transcript token by token, so unlike a transducer
        # its cost tracks the LENGTH of what was said, not just the audio:
        #
        #     clip      3s    5s    8s   12s   17s   ~40s
        #     parakeet  40ms  39ms  37ms  41ms  50ms  ~1.5s
        #     canary   127ms  82ms 111ms 246ms 512ms  ~1.4s
        #
        # Parakeet is flat and free; this one is free for a normal turn and gets
        # expensive for a monologue. At the 30 s ``max_utterance_sec`` cap it
        # lands near 1.5 s, so leaving the old budget in place would have
        # silently timed out exactly the long turns where a second opinion is
        # worth most. Costs nothing to raise: the pass is not in the reply path.
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
        cap = int(MAX_SEC * TARGET_SR)
        if audio.size > cap:
            # Keep the END of an over-long clip. If the cap ever fires, the
            # words nearest the turn boundary are the ones the reply was built
            # from, so they are the ones worth checking.
            audio = audio[-cap:]

        t0 = time.perf_counter()
        path = None
        try:
            import torch

            fd, path = tempfile.mkstemp(prefix="canary-", suffix=".wav")
            os.close(fd)
            _write_wav(audio, path)

            prompt = ASR_PROMPT.format(audio_tag=self._model.audio_locator_tag)
            with torch.no_grad(), _LOCK:
                # No disable_adapter() here, deliberately — that is LLM mode.
                # This call is the transcription path.
                ids = self._model.generate(
                    prompts=[[{"role": "user", "content": prompt, "audio": [path]}]],
                    max_new_tokens=MAX_NEW_TOKENS,
                )
        except Exception:
            # Never propagate: this layer is a bonus, and the live transcript is
            # already good enough to answer with.
            log.exception("canary pass failed — keeping the live transcript")
            return None
        finally:
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if ids is None or not len(ids):
            return None
        text = self._model.tokenizer.ids_to_text(ids[0].cpu())
        return (text or "").strip() or None
