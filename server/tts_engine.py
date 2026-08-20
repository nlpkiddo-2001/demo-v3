"""Text-to-Speech (TTS) engine using Orpheus + SNAC audio codec.

What does this file do?
------------------------
It takes text (the AI's reply) and turns it into spoken audio that the user hears.

Two-step process:
  Step 1 — Orpheus LLM generates "audio tokens" (special numbers that represent sound)
            via a vLLM API call. Think of these like a secret code for audio.
  Step 2 — SNAC (a neural audio codec) decodes those tokens into real PCM audio
            (the same format as a WAV file) at 24 000 samples per second.

Why two steps? Orpheus is a language model fine-tuned to generate audio codes instead
of words. SNAC is the decoder that turns those codes into actual sound waves.

Token format (how Orpheus encodes audio):
  - 7 tokens per audio "frame" (~2048 audio samples, ~85 ms of audio)
  - Each token looks like: "<custom_token_12345>"
  - The number inside is decoded: int(12345) - 10 - ((position % 7) * 4096)
  - Every 7 valid tokens → 1 SNAC decode call → one audio chunk yielded

Example scenario:
  AI reply: "Your call is scheduled for Thursday."
  → Orpheus generates ~200 custom tokens (streamed)
  → Every 7 tokens → decoded into ~2048 audio samples (~85 ms of speech)
  → Audio chunks are streamed to the browser as they're decoded
  → User starts hearing audio within ~300 ms
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx
import numpy as np
import torch

from .config import settings

log = logging.getLogger(__name__)

# ── SNAC codec (loaded once and reused for all TTS requests) ──────────────
# Loading the model takes a few seconds and ~1 GB of GPU memory.
# We load it once at startup and reuse it for every sentence.
_snac_model = None
_snac_lock = asyncio.Lock()  # prevents two coroutines from loading it simultaneously


async def _get_snac():
    """Load the SNAC model on the configured GPU (lazy — only on first use)."""
    global _snac_model
    if _snac_model is not None:
        return _snac_model
    async with _snac_lock:
        if _snac_model is not None:
            return _snac_model  # another coroutine already loaded it while we waited
        from snac import SNAC

        device = settings.snac_device
        log.info("Loading SNAC model on %s", device)
        _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        log.info("SNAC model loaded")
        return _snac_model


def _turn_token_into_id(token_string: str, index: int) -> int | None:
    """Extract the SNAC code ID from a token string like '<custom_token_12345>'.

    The formula reverses the encoding Orpheus used:
      raw_number  = 12345
      codebook_id = raw_number - 10 - ((position_in_group % 7) * 4096)

    Returns None if the string doesn't match the expected token format.
    """
    token_string = token_string.strip()
    last_start = token_string.rfind("<custom_token_")
    if last_start == -1:
        return None
    last_token = token_string[last_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]  # extract "12345" from "<custom_token_12345>"
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


def _convert_to_audio(
    multiframe: list[int], snac_model, device: str
) -> bytes | None:
    """Decode a batch of SNAC token IDs into raw 16-bit PCM audio bytes.

    SNAC uses 3 "codebooks" (like 3 layers of audio detail).
    Every 7 tokens cover 1 frame:
      token[0]         → codebook 0 (coarse — basic pitch/tone)
      token[1], [4]    → codebook 1 (medium detail)
      token[2],[3],[5],[6] → codebook 2 (fine detail — timbre, texture)

    This mirrors how neural audio codecs like EnCodec work: multiple layers,
    coarse to fine.

    Returns:
      Raw bytes (int16 PCM at 24 kHz), ready to send to the browser.
      Returns None if there aren't enough tokens yet.
    """
    if len(multiframe) < 7:
        return None  # need at least 7 tokens to decode one frame

    # Only decode complete groups of 7.
    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    # Separate the 7 tokens per frame into their 3 codebook channels.
    codes_0 = torch.tensor([], device=device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=device, dtype=torch.int32)

    for j in range(num_frames):
        i = 7 * j
        codes_0 = torch.cat(
            [codes_0, torch.tensor([frame[i]], device=device, dtype=torch.int32)]
        )
        c1 = torch.tensor(
            [frame[i + 1], frame[i + 4]], device=device, dtype=torch.int32
        )
        codes_1 = torch.cat([codes_1, c1])
        c2 = torch.tensor(
            [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]],
            device=device,
            dtype=torch.int32,
        )
        codes_2 = torch.cat([codes_2, c2])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    # Safety check: SNAC codebooks only accept values 0–4096.
    # Out-of-range values mean a corrupted or out-of-order token — skip this batch.
    for c in codes:
        if torch.any(c < 0) or torch.any(c > 4096):
            return None

    # Run the decoder (no gradients needed — pure inference).
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)

    # The model outputs a full frame but the first 2048 samples are "warmup"
    # (overlap from the previous frame). We only keep samples 2048–4096.
    audio_slice = audio_hat[:, :, 2048:4096]
    audio_np = audio_slice.detach().cpu().numpy()

    # Convert float32 → int16 PCM. CLIP to [-1, 1] first: SNAC can emit samples
    # slightly outside that range, and casting an overflowed value straight to
    # int16 wraps around (e.g. 1.2*32767 → wraps negative) which is heard as loud
    # crackle/noise. Clipping removes that. A tiny 0.98 headroom avoids hard-clip fuzz.
    audio_np = np.clip(audio_np, -1.0, 1.0) * 0.98
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


class TTSEngine:
    """Orchestrates Orpheus token generation + SNAC decoding.

    Usage:
      engine = TTSEngine()
      await engine.start()
      async for audio_chunk in engine.synthesize_stream("Hello there!"):
          # send audio_chunk to the browser (raw 16-bit PCM bytes)
    """

    def __init__(self):
        self.api_url = settings.tts_api_url
        self.model = settings.tts_model
        self.voice = settings.tts_voice  # e.g. "tara"
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create HTTP session and pre-load the SNAC decoder.

        Pre-loading SNAC at startup avoids a delay on the first TTS request.
        """
        # Read timeout kept tight (30s): a healthy sentence synthesizes in a few
        # seconds, so anything longer means the TTS server stalled — fail fast and
        # let the pipeline move on instead of freezing the whole turn.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        # Load SNAC now so the first sentence doesn't pay the startup cost.
        await _get_snac()

    async def stop(self) -> None:
        """Close the HTTP session cleanly."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _format_prompt(self, text: str, voice: str | None = None) -> str:
        """Wrap text in the special tokens Orpheus expects.

        Orpheus was fine-tuned with a fixed prompt structure that tells it:
          1. Which voice to use (e.g. "tara:")
          2. Where the text starts and ends
          3. A special start-of-audio marker at the end

        Example output:
          "<custom_token_3><|begin_of_text|>tara: Hello there!<|eot_id|>
           <custom_token_4><custom_token_5><custom_token_1>"

        ``voice`` overrides the engine default for this one utterance. It is a
        parameter rather than engine state because ONE engine is shared by every
        connected session: mutating ``self.voice`` to satisfy one caller changes
        the voice mid-sentence for everybody else. Per-call is the only version
        that is correct with two people on the demo at once, and it costs
        nothing — the voice is just a prefix token, not a loaded model.
        """
        return (
            f"<custom_token_3>"
            f"<|begin_of_text|>"
            f"{voice or self.voice}: {text}"
            f"<|eot_id|>"
            f"<custom_token_4><custom_token_5><custom_token_1>"
        )

    async def synthesize_stream(self, text: str,
                                voice: str | None = None) -> AsyncIterator[bytes]:
        """Convert text to audio, yielding 16-bit PCM chunks as they're decoded.

        This is a real-time streaming pipeline:
          1. Send text to vLLM running Orpheus → get token stream
          2. Collect tokens; every 7 valid ones → decode via SNAC → yield audio

        The "warmup" period (first 28 tokens, 4 frames) is skipped before we
        start yielding audio — this lets the SNAC decoder "warm up" so the
        first audio chunk sounds clean.

        ``voice`` picks the speaker for this utterance only, leaving the shared
        engine untouched — see :meth:`_format_prompt`.

        Example:
          async for pcm in engine.synthesize_stream("See you tomorrow."):
              websocket.send_bytes(pcm)
        """
        if not self._client:
            await self.start()

        snac_model = await _get_snac()
        device = settings.snac_device

        # Wrap the text in the Orpheus prompt format.
        prompt = self._format_prompt(text, voice)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "max_tokens": settings.tts_max_tokens,
            "temperature": settings.tts_temperature,
            "top_p": settings.tts_top_p,
            "repetition_penalty": settings.tts_repetition_penalty,
            "stop_token_ids": [128258],  # token ID that signals end of audio
        }

        buffer: list[int] = []   # accumulates decoded token IDs
        count = 0                # number of valid tokens seen so far
        chunks_yielded = 0
        tokens_skipped = 0

        try:
            async with self._client.stream(
                "POST", self.api_url, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break  # all tokens received
                    try:
                        chunk = json.loads(data_str)
                        # Each SSE chunk contains one token text like "<custom_token_12345>"
                        token_text = (
                            chunk.get("choices", [{}])[0]
                            .get("text", "")
                        )
                    except json.JSONDecodeError:
                        continue

                    if not token_text:
                        continue

                    # Decode the token string → numeric SNAC code ID.
                    token_id = _turn_token_into_id(token_text, count)
                    if count < 5 or (token_id is not None and token_id <= 0 and count < 10):
                        log.debug(
                            "TTS token[%d]: text=%r → id=%s",
                            count if token_id and token_id > 0 else tokens_skipped,
                            token_text[:60], token_id,
                        )
                    if token_id is None:
                        # Not a custom audio token — skip (e.g. text output, not audio).
                        tokens_skipped += 1
                        continue
                    if token_id <= 0:
                        # Invalid or out-of-range code — discard.
                        tokens_skipped += 1
                        continue

                    buffer.append(token_id)
                    count += 1

                    # Once we have 28+ tokens (4 warmup frames) AND a new complete
                    # frame of 7 tokens, decode the latest 28-token window into audio.
                    # "28 tokens warmup" lets SNAC produce clean audio from the start.
                    if count % 7 == 0 and count > 27:
                        audio_bytes = _convert_to_audio(
                            buffer[-28:], snac_model, device
                        )
                        if audio_bytes:
                            chunks_yielded += 1
                            yield audio_bytes  # send to browser immediately

        except httpx.HTTPStatusError as e:
            log.error("TTS HTTP error: %s", e)
        except Exception:
            log.exception("TTS streaming error")
        finally:
            log.info(
                "TTS stream done: %d valid tokens, %d skipped, %d audio chunks yielded",
                count, tokens_skipped, chunks_yielded,
            )

