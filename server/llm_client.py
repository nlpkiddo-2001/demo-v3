"""Language Model (LLM) client — sends user text and streams back AI replies.

What does this file do?
------------------------
When the user finishes speaking, their words are sent here.
This file sends those words to a large language model (an AI "brain") and
streams back the reply text, piece by piece, as fast as the model can generate it.

We use GLM-4.7-Flash, a fast chat model, served via an OpenAI-compatible API
running on an internal GPU machine (crm-l40s-2).

"Streaming" means we don't wait for the whole reply before passing it forward —
the first words come back in milliseconds, so TTS can start speaking immediately.

Example scenario:
  User says: "Who handles the Acme account?"
  → llm.chat_stream(...) sends the question to GLM-4.7-Flash
  → chunks arrive: "The" → " Acme" → " account" → " is handled by Sarah Chen."
  → each chunk is immediately forwarded to the TTS engine to be spoken
  → total delay from question to first word of speech: ~300 ms
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx
import json

from .config import settings

log = logging.getLogger(__name__)


class LLMClient:
    """Async HTTP client for streaming chat completions via GLM-4.7-Flash.

    The server uses OpenAI-compatible API format, so this client works with
    any vLLM-served model just by changing the base URL and model name in config.
    """

    def __init__(self):
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self._client: httpx.AsyncClient | None = None  # created lazily on first use

    async def start(self) -> None:
        """Create the HTTP client session. Called once at pipeline startup."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),  # 60 s total, 10 s to connect
        )

    async def stop(self) -> None:
        """Close the HTTP client session. Called at pipeline shutdown."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat_stream(
        self,
        user_text: str,
        conversation_history: list[dict] | None = None,
        system_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Send the user's message and stream back the AI's reply, word by word.

        Args:
          user_text           : What the user just said.
          conversation_history: Previous turns (so the AI remembers context).
          system_override     : Use this system prompt instead of the default.
                                Used by the outreach pipeline to inject deal info.

        Yields:
          Text delta strings as they arrive (e.g. "The", " deal", " is", …).

        Example:
          async for chunk in llm.chat_stream("What is the deal status?"):
              print(chunk, end="", flush=True)
          # prints: "The deal is currently in Proposal Sent stage."
        """
        if not self._client:
            await self.start()

        # Build the message list in OpenAI chat format:
        # [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        system_prompt = system_override if system_override else settings.system_prompt
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)  # add past turns for memory
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,                      # get chunks as they're generated
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "chat_template_kwargs": {"enable_thinking": False},  # GLM-specific: no chain-of-thought
        }

        try:
            chunks_yielded = 0
            # SSE (Server-Sent Events) streaming: each line starts with "data: "
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue  # skip blank lines or non-data lines
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break  # server signals end of stream
                    try:
                        chunk = json.loads(data_str)
                        # Navigate the OpenAI response structure to get the text piece.
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks_yielded += 1
                            yield content  # hand off to TTS immediately
                    except json.JSONDecodeError:
                        continue  # skip malformed lines silently
            if chunks_yielded == 0:
                log.warning("LLM stream completed with 0 content chunks for: %s", user_text[:60])
        except httpx.HTTPStatusError as e:
            log.error("LLM HTTP error: %s", e)
            raise
        except Exception:
            log.exception("LLM streaming error")
            raise

    async def verify(self) -> bool:
        """Quick health check — returns True if the LLM server is reachable.

        Used by the /health endpoint to report service status.
        """
        if not self._client:
            await self.start()
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            log.exception("LLM health check failed")
            return False

    async def chat_complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Send a message list and wait for the full reply (non-streaming).

        Used for offline tasks that need the complete answer before proceeding:
          - Entity extraction: "What new contacts or blockers were mentioned?"
          - Post-call summary: "Write a 2-sentence summary of this call."

        Args:
          messages    : Full message list (system + user turns).
          max_tokens  : Maximum reply length.
          temperature : Lower = more focused/predictable output (0.2 is conservative).
          json_mode   : If True, the model is forced to reply with valid JSON only.
                        Useful for structured extraction tasks.

        Returns:
          The assistant's full reply as a single string.
        """
        if not self._client:
            await self.start()

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            # Force the model to output only valid JSON (no extra prose).
            payload["response_format"] = {"type": "json_object"}

        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            log.warning("Unexpected LLM completion shape: %s", data)
            return ""
