"""Live STT — Nemotron cache-aware streaming, the fast model that drives the turn.

Why this model and not Parakeet
-------------------------------
handoff.md Part 5 asks for a fast live recogniser with one non-negotiable
property: it must stay quiet on non-speech, so it never invents words that the
turn logic and the LLM then believe. We measured both candidates rather than
trusting model cards:

  * ``parakeet-tdt-0.6b-v3`` streams only by *buffered chunking* — its own
    example uses two seconds of right context, so no word can be emitted until
    two seconds of future audio exist. That cannot drive speculation or
    interruption detection, whatever its accuracy.
  * ``nemotron-speech-streaming-en-0.6b`` is genuinely cache-aware: real
    incremental encoder state, selectable latency down to 80 ms, and on our
    tests it returned an empty string for digital silence, quiet room tone and
    louder noise, while transcribing speech correctly.

So Nemotron takes the live seat. Parakeet keeps the accurate-second-pass seat
the handoff gives Cohere, where two seconds of latency costs nothing.

The one trap
------------
The model loads defaulting to ``[70, 13]`` — its *slowest* mode, 1120 ms. Left
alone it behaves like a batch recogniser and the whole pipeline feels broken for
reasons that never appear in a log. :attr:`NemotronConfig.att_context` is
therefore explicit and required-by-default.

  ``[70, 0]`` →   80 ms      ``[70, 6]``  →  560 ms
  ``[70, 1]`` →  160 ms      ``[70, 13]`` → 1120 ms   (WER improves as latency rises)

How the streaming actually works
--------------------------------
``conformer_stream_step`` consumes **mel features, not audio**, and needs the
pre-encode cache stitched onto the front of every chunk, a different chunk size
on the very first step, and ``drop_extra_pre_encoded`` set on every step but the
first. Getting any of that subtly wrong misaligns the audio and quietly degrades
the transcript rather than failing.

So this class does not reimplement that maths. It drives NeMo's own
``CacheAwareStreamingAudioBuffer``, and adds the one thing that class lacks: it
is written to replay a complete file, whereas we are fed audio as a person
speaks. The difference is handled by only ever consuming a step when a *whole*
chunk of features has arrived, and leaving the remainder in the buffer for the
next call.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .session import STTProvider, STTUpdate

log = logging.getLogger(__name__)

TARGET_SR = 16000
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"

# Latency modes the model was trained for, as {att_context: chunk_ms}. Anything
# else is not a knob — the encoder was trained with these context sizes.
ATT_CONTEXT_MS = {(70, 0): 80, (70, 1): 160, (70, 6): 560, (70, 13): 1120}

# The encoder's own step. Every mode above is a whole number of these, which is
# why the smallest is 80 ms and not something finer. Used by ``flush`` as the
# margin frame when padding the tail.
FRAME_MS = 80

# Weights are large and slow to load, so sessions share them. The streaming
# *state* (caches, hypotheses, feature buffer) is per-instance and never shared.
_MODEL_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()


def preload(model_id: str = DEFAULT_MODEL, device: str = "cuda") -> None:
    """Load the weights into the shared cache ahead of any session.

    Worth doing at server startup rather than on the first connection: the load
    takes about 16 seconds, and if it happens inside a WebSocket handler the
    event loop stalls for that long — the browser sees the page hang and may
    time the connection out before it ever gets a reply. Call this from a
    thread, since it blocks.
    """
    key = (model_id, device)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            t0 = time.time()
            _MODEL_CACHE[key] = NemotronStreamingSTT._load(model_id, device)
            log.info("nemotron preloaded in %.0fs (%s)", time.time() - t0, device)


class NemotronStreamingSTT(STTProvider):
    """Cache-aware streaming recogniser for one utterance stream.

    One instance per session: the encoder caches and decoder hypotheses are this
    conversation's state. The underlying weights are shared, guarded by a lock,
    because ``conformer_stream_step`` keeps no state on the model itself — every
    piece of state is passed in and handed back.
    """

    name = "nemotron"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        att_context: tuple[int, int] = (70, 1),
        device: str = "cuda",
        max_utterance_sec: float = 60.0,
        online_normalization: bool = True,
    ):
        if tuple(att_context) not in ATT_CONTEXT_MS:
            raise ValueError(
                f"att_context {att_context} is not one of the trained modes "
                f"{sorted(ATT_CONTEXT_MS)}"
            )
        self.model_id = model_id
        self.att_context = tuple(att_context)
        self.device = device
        self.max_utterance_sec = max_utterance_sec
        self.online_normalization = online_normalization
        self.chunk_ms = ATT_CONTEXT_MS[self.att_context]

        self._model = None
        self._buf = None            # NeMo's CacheAwareStreamingAudioBuffer
        self._lock = threading.Lock()
        self._stream_id = -1        # -1 until the first append creates the stream

        # per-utterance streaming state
        self._cache_ch = self._cache_t = self._cache_len = None
        self._hyps = None
        self._pred_out = None
        self._partial = ""
        self._steps = 0
        self._raw = np.zeros(0, dtype=np.float32)   # utterance audio, for continuous featurising
        self._frames_done = 0                       # mel frames already appended

    # ── lifecycle ──
    async def start(self) -> None:
        key = (self.model_id, self.device)
        with _CACHE_LOCK:
            model = _MODEL_CACHE.get(key)
            if model is None:
                t0 = time.time()
                model = self._load(self.model_id, self.device)
                _MODEL_CACHE[key] = model
                log.info("nemotron loaded in %.0fs (%s)", time.time() - t0, self.device)
        self._model = model

        # Explicit, every time. The mode is a property of the encoder object,
        # which is shared — so a session that assumed the default would silently
        # change latency for everyone else.
        self._model.encoder.set_default_att_context_size(list(self.att_context))
        self._model.encoder.setup_streaming_params()

        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        self._buf = CacheAwareStreamingAudioBuffer(
            model=self._model, online_normalization=self.online_normalization
        )

        # Samples per streaming step: shift is measured in mel frames, so it
        # converts through the preprocessor's hop, not through any constant.
        cfg = self._model.encoder.streaming_cfg
        shift = cfg.shift_size[1] if isinstance(cfg.shift_size, list) else cfg.shift_size
        self._hop = int(self._model.cfg.preprocessor.window_stride * TARGET_SR)
        self._step_samples = int(shift) * self._hop
        # Left context handed to the preprocessor so the frames we keep are
        # computed with real audio either side, never a padded edge.
        self._ctx_samples = int(0.5 * TARGET_SR)
        # Frames too new to be computed against real audio; see _append().
        win = int(self._model.cfg.preprocessor.window_size * TARGET_SR)
        self._holdback = int(np.ceil((win / 2) / self._hop))
        self.reset()
        log.info(
            "nemotron streaming ready: att_context=%s -> %dms/step (%d samples), device=%s",
            self.att_context, self.chunk_ms, self._step_samples, self.device,
        )

    @staticmethod
    def _load(model_id: str, device: str):
        import os

        import nemo.collections.asr as nemo_asr
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_id)
        nemo_files = [f for f in os.listdir(path) if f.endswith(".nemo")]
        if not nemo_files:
            raise FileNotFoundError(f"no .nemo checkpoint in {path}")
        model = nemo_asr.models.ASRModel.restore_from(
            os.path.join(path, nemo_files[0]), map_location=device
        )
        model.eval()
        return model

    async def stop(self) -> None:
        self._buf = None
        self._model = None   # weights stay alive in _MODEL_CACHE for the next session

    # ── inference ──
    def accept_audio(self, audio16k: np.ndarray) -> STTUpdate:
        """Feed speech audio; return any words that completed.

        Any chunk length is accepted. The audio is featurised continuously (see
        :meth:`_append`) and the model steps forward whenever a whole chunk of
        features has arrived, so short chunks simply produce no words yet.
        """
        if self._model is None or self._buf is None:
            raise RuntimeError("NemotronStreamingSTT.accept_audio() called before start()")

        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        before = self._partial
        with self._lock:
            self._append(audio)
            self._drain()

        new_words = _new_words(before, self._partial)
        return STTUpdate(new_words, self._partial)

    def _append(self, audio: np.ndarray, final: bool = False) -> None:
        """Featurise the utterance so far and append only the frames that are new.

        Why not simply featurise each arriving chunk? Because the mel front end
        pads the edges of whatever signal it is given, so featurising 160 ms at a
        time inserts a padded boundary every 160 ms. We measured the damage: WER
        went from 5.9% at 1120 ms steps (few boundaries) to 52% at 160 ms steps
        (many). Same audio, same model — the boundaries alone.

        So the preprocessor is always handed a *continuous* run of audio ending
        at the present moment, and we take only the frames past the ones already
        appended. Those frames then have real audio either side of them, exactly
        as they would in one-shot transcription.

        The window is bounded rather than the whole utterance: half a second of
        left context is twenty times the 25 ms analysis window, which is ample
        for the frames we keep, and it makes the cost per call constant instead
        of growing with the length of the turn.

        This relies on ``online_normalization=True``, which moves normalisation
        into the per-chunk step. With global normalisation the statistics would
        shift every time the audio grew, silently altering frames we had already
        committed to the buffer.
        """
        import torch

        if audio.size:
            self._raw = np.concatenate([self._raw, audio]) if self._raw.size else audio
        if self._raw.size == 0:
            return

        # The window MUST start on a hop boundary. Mel frame j of a window
        # beginning at sample S lands on absolute frame j + S/hop only when S is
        # a multiple of the hop; start anywhere else and every feature is offset
        # by a fraction of a frame, which reads as plausible audio and wrecks the
        # transcript without erroring.
        want = self._ctx_samples + audio.size + 2 * self._hop
        start = max(0, (self._raw.size - want) // self._hop) * self._hop
        # Never start past a frame we still owe the buffer, or it would be skipped.
        start = min(start, self._frames_done * self._hop)
        window = self._raw[start:]

        feats, _ = self._buf.preprocess_audio(window)
        first_abs = start // self._hop

        # Hold back the newest frames. A frame centred at k*hop needs audio out
        # to k*hop + window/2, so the last few frames of any window are computed
        # against padding rather than the speech that has not arrived yet. Emit
        # them now and they are wrong; wait one call and they are exact. We
        # verified this against one-shot featurisation: holding back
        # ceil((window/2)/hop) frames takes max|difference| to 0.000000, while
        # holding back none leaves a quarter of all frames corrupted — which is
        # what turned a 6% WER into 88%.
        hold = 0 if final else self._holdback
        lo = self._frames_done - first_abs
        hi = feats.size(-1) - hold
        if hi <= lo:
            return

        new = feats[:, :, lo:hi]
        with torch.inference_mode():
            self._buf.append_processed_signal(new, stream_id=self._stream_id)
        # The first append creates stream 0, but NeMo returns -1 for it — that
        # branch never assigns the id. Passing that -1 back would make the next
        # append add a *second batch row* rather than extend the stream, and the
        # encoder cache (batch 1) would then mismatch the input (batch 2). So the
        # id is pinned to 0 here rather than taken from the return value.
        self._stream_id = 0
        self._frames_done = first_abs + hi

    def _drain(self, final: bool = False) -> None:
        """Run every streaming step whose features have fully arrived."""
        import torch

        cfg = self._model.encoder.streaming_cfg
        guard = 0
        while True:
            available = self._buf.buffer.size(-1) - self._buf.buffer_idx
            chunk_size = cfg.chunk_size
            if isinstance(chunk_size, list):
                chunk_size = chunk_size[0] if self._buf.buffer_idx == 0 else chunk_size[1]
            if not final and available < chunk_size:
                break
            if available <= 0:
                break

            try:
                chunk, lengths = next(iter(self._buf))
            except StopIteration:
                break

            # Mirrors NeMo's own loop: the first step has no preceding audio to
            # drop, every later one does.
            drop = 0 if self._steps == 0 else cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                (
                    self._pred_out,
                    texts,
                    self._cache_ch,
                    self._cache_t,
                    self._cache_len,
                    self._hyps,
                ) = self._model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=lengths,
                    cache_last_channel=self._cache_ch,
                    cache_last_time=self._cache_t,
                    cache_last_channel_len=self._cache_len,
                    keep_all_outputs=final,
                    previous_hypotheses=self._hyps,
                    previous_pred_out=self._pred_out,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            self._steps += 1
            if texts:
                text = texts[0]
                self._partial = (text.text if hasattr(text, "text") else str(text)).strip()

            guard += 1
            if guard > 512:   # a step must always advance buffer_idx; if it ever
                log.error("nemotron drain did not terminate — aborting this batch")
                break

    def flush(self) -> STTUpdate:
        """Force out any tail the model is holding. Call at end of utterance.

        The padding is the whole point, and it used to be missing.
        ``keep_all_outputs=True`` gives permission to emit the held-back frames,
        but permission is not enough: this encoder decides a word using
        ``chunk_ms`` of *future* audio, and at the end of an utterance that audio
        does not exist. With nothing appended, the final partial chunk never
        reaches full size, so the last frames are never stepped and the last word
        is simply lost.

        Measured on a real turn from ``recordings/agent-20260818-064657``, where a
        speaker says "the ticket name is Vipin":

            appended nothing (the old behaviour) ->  "...the ticket name is Vipp"
            appended 160 ms of silence           ->  "...the ticket name is Vippin"

        The word was in the audio the whole time — the accurate pass read it back
        correctly from the same buffer, which is what proved the recogniser rather
        than the VAD was cutting it off. One chunk of silence is what lets the
        streaming model see an end and commit.

        Real silence, not zero samples: this is what the end of speech genuinely
        looks like to the model, and it is exactly what NeMo's own offline replay
        of a finished file provides by padding the tail. A margin frame is added
        on top because the boundary lands mid-frame as often as not, and silence
        is free here — no future audio is coming, so nothing is delayed by it.
        """
        if self._model is None or self._buf is None or self._buf.buffer is None:
            return STTUpdate([], self._partial)
        before = self._partial
        pad = np.zeros(
            int((self.chunk_ms + FRAME_MS) / 1000.0 * TARGET_SR), dtype=np.float32
        )
        with self._lock:
            # final=True releases the held-back frames — at the true end of the
            # utterance there is no future audio to wait for, so padding is right.
            self._append(pad, final=True)
            self._drain(final=True)
        return STTUpdate(_new_words(before, self._partial), self._partial)

    def partial(self) -> str:
        return self._partial

    # ── state ──
    def reset(self) -> None:
        """Clear all per-utterance state so the next turn starts clean."""
        if self._model is None:
            return
        with self._lock:
            self._cache_ch, self._cache_t, self._cache_len = (
                self._model.encoder.get_initial_cache_state(batch_size=1)
            )
            self._hyps = None
            self._pred_out = None
            self._partial = ""
            self._steps = 0
            self._raw = np.zeros(0, dtype=np.float32)
            self._frames_done = 0
            self._stream_id = -1
            if self._buf is not None:
                # Drops the feature buffer entirely, which also bounds memory:
                # it grows for the length of one utterance and no further.
                self._buf.reset_buffer()


def _new_words(before: str, after: str) -> list[str]:
    """Words in ``after`` that were not already in ``before``.

    The recogniser hands back the whole running transcript each step, but the
    pipeline wants increments. A plain prefix diff is not enough: a streaming
    decoder revises what it already said, so the tail can change. When it does,
    fall back to reporting the whole new transcript's tail rather than pretending
    nothing happened.
    """
    if after == before:
        return []
    b, a = before.split(), after.split()
    i = 0
    while i < len(b) and i < len(a) and b[i] == a[i]:
        i += 1
    return a[i:]
