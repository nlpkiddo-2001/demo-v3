"""The accurate second pass for Indian languages — Sarvam over the finished clip.

Why this seat exists at all
---------------------------
Neither recogniser in the English pipeline can transcribe Tamil, Telugu or
Kannada. Not badly — *at all*. ``nemotron-3.5-asr-streaming`` covers 40 locales
including hi-IN but no Dravidian language; ``parakeet-tdt-0.6b-v3`` covers 25
European ones. So for the Indian-languages mode this is not an accuracy upgrade,
it is the difference between a transcript and nothing.

Why not an open model
---------------------
IndicConformer 600M is the best open-source Indic recogniser by a wide margin,
and on the Voice of India benchmark (536 h, 15 languages) it is genuinely close:

    language     IndicConformer   Sarvam
    Hindi              8.2          5.0
    Tamil             19.9         14.2
    Kannada           21.4         16.3
    Telugu            23.7         18.2
    Malayalam         26.0         18.9

It is still the wrong model for us, and not because of those numbers. It
supports 22 Indian languages and **no English**, so the sentence this mode
exists to serve —

    "Zoho Cliq epadi use pannanum"

— hits a tokenizer with no token space for "Zoho Cliq". The product name comes
back as ``[UNK]``, or gets mapped onto the nearest Tamil phonemes and written in
Tamil script. A mode that fails hardest on our own product name is worse than no
mode. Monolingual recognisers take a 30–50% relative WER hit on code-switched
speech generally; this is that effect landing on the exact tokens the agent
routes on.

Sarvam is chosen because ``mode=codemix`` is a documented parameter — native
script plus English, in one pass, no language boundary to detect. It also led 13
of 15 languages on Voice of India. The alternative with the same property is
ElevenLabs Scribe v2, whose March 2026 release fixed English being transliterated
into Indic script for exactly Hindi/Telugu/Kannada.

What this costs
---------------
Audio leaves the machine. The English half of the pipeline is fully on-prem and
this half is not, and that is a real trade rather than a detail — it is the price
of the only thing that transcribes code-mixed Tamil well today. If it ever stops
being acceptable, the fallback is fine-tuning IndicConformer on code-mixed data,
which is weeks of work rather than an afternoon.

It is also a network call, not a GPU call. That is why the default timeout here
is larger than the local passes': a round trip cannot be made to take 1.5 s by
wishing. It stays outside the critical path either way — the reply has already
started on the live transcript.

Language selection is explicit, not detected
--------------------------------------------
``language_code`` comes from the mode the user picked, not from automatic
language ID. Cascade LID adds 70–200 ms before routing even begins and drops
25–40% of words on rapid mid-sentence switches, which is precisely the shape of
the audio here. A dropdown removes both problems for free. ``unknown`` is
accepted for auto-detection and is deliberately not the default.

What it must never do
---------------------
Block. Same contract as the local passes: a hard timeout, and every failure path
returns ``None`` meaning "no opinion". A missing API key is one of those paths —
it logs once and then behaves exactly like a session with no accurate pass at
all, rather than raising on every turn.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000
API_URL = "https://api.sarvam.ai/speech-to-text"

# saaras:v3 is the default and the only one that accepts ``mode``, which is the
# whole reason we are here — v4 drops it, so switching model means losing
# codemix. Set SARVAM_MODE=transcribe if you ever move to v4.
MODEL_ID = "saaras:v3"
MODE = "codemix"

# Below this there is nothing worth a second opinion, and it is not worth a
# round trip either.
MIN_SEC = 0.3

# The REST endpoint is the synchronous one, documented for short clips; longer
# audio is what the batch API is for. Our utterances are turns, so this only
# guards against a stuck buffer.
MAX_SEC = 30.0

_LOCK = threading.Lock()


def _wav_bytes(audio: np.ndarray) -> bytes:
    """Encode float32 mono as a 16 kHz 16-bit WAV in memory.

    Clipped before the cast: float audio from the ring can sit slightly outside
    [-1, 1] after denoising, and letting that wrap round in int16 turns a loud
    word into white noise rather than a clipped one.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class SarvamAccurate:
    """Offline transcriber for one finished utterance, over HTTP.

    Stateless between calls, so a single instance is safe to share across
    sessions. Unlike the local passes the lock is not protecting a GPU — it
    bounds concurrent outbound requests, so a burst of simultaneous turn-ends
    cannot open an unbounded number of connections to someone else's API.

    Interface-compatible with :class:`stt_parakeet.ParakeetAccurate` on purpose:
    ``agent_app`` picks one of them at startup and cannot tell them apart.
    """

    name = "sarvam_saaras_v3"

    def __init__(self, api_key: str = "", language: str = "ta-IN",
                 model_id: str = MODEL_ID, mode: str = MODE,
                 timeout_sec: float = 3.0):
        self.api_key = api_key
        self.language = language
        self.model_id = model_id
        self.mode = mode
        # Larger than the local passes on purpose — see the module docstring.
        # Still a hard ceiling: past this the live transcript keeps the floor.
        self.timeout_sec = timeout_sec
        self._client = None
        self.last_ms = 0.0
        self._warned = False

    async def start(self) -> None:
        import httpx

        if not self.api_key:
            # Not an error. The pipeline is designed to run without an accurate
            # pass, so say so once and then be a no-op rather than raising on
            # every turn for the rest of the session.
            log.warning("sarvam: no API key set (SARVAM_API_KEY) — "
                        "indic accurate pass disabled, live transcript only")
            return
        # Leave headroom under the outer budget so a slow response comes back as
        # our own clean timeout rather than the caller cancelling us mid-request.
        self._client = httpx.Client(timeout=max(0.5, self.timeout_sec - 0.25))
        log.info("sarvam accurate pass ready (model=%s mode=%s lang=%s)",
                 self.model_id, self.mode, self.language)

    async def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def set_language(self, language: str) -> None:
        """Point the next call at a different Indian language.

        Cheap by design — the language is a request parameter, not a loaded
        model, so the mode dropdown can change it between turns without any
        reload. That is the main practical argument for one code-mix-native
        recogniser over a router into per-language monolingual models.
        """
        self.language = language

    def transcribe(self, audio16k: np.ndarray) -> str | None:
        """Transcribe a finished utterance. ``None`` means "no opinion".

        Blocking on network IO — always call it inside a thread executor.
        """
        if self._client is None:
            return None
        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        if audio.size < int(MIN_SEC * TARGET_SR):
            return None
        cap = int(MAX_SEC * TARGET_SR)
        if audio.size > cap:
            # Keep the END of an over-long clip: the words nearest the turn
            # boundary are the ones the reply was built from.
            audio = audio[-cap:]

        t0 = time.perf_counter()
        try:
            with _LOCK:
                r = self._client.post(
                    API_URL,
                    headers={"api-subscription-key": self.api_key},
                    files={"file": ("utterance.wav", _wav_bytes(audio), "audio/wav")},
                    data={"model": self.model_id,
                          "language_code": self.language,
                          "mode": self.mode},
                )
            r.raise_for_status()
            body = r.json()
        except Exception:
            # Never propagate: this layer is a bonus, and the live transcript is
            # already good enough to answer with. Logged at warning rather than
            # exception because a flaky network is expected, not exceptional,
            # and a stack trace per turn would drown the session log.
            if not self._warned:
                log.exception("sarvam pass failed — keeping the live transcript "
                              "(further failures logged briefly)")
                self._warned = True
            else:
                log.warning("sarvam pass failed — keeping the live transcript")
            return None
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        text = (body or {}).get("transcript")
        return (text or "").strip() or None
