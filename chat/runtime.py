"""The agent runtime — a native tool-calling loop over the OpenAI-compatible LLM.

Flow for one user turn:
  1. Send the conversation + the agent's tool schemas to the LLM.
  2. If the model asks to call tools, run them, append the results, and loop.
  3. When the model returns a plain answer, stream it to the client word-by-word.

Output is a stream of small event dicts (consumed by app.py and turned into SSE):
  {"type": "status",     "text": "..."}          # e.g. "Thinking…"
  {"type": "tool_call",  "id","name","args"}     # a tool is about to run
  {"type": "tool_result","id","name","result"}   # tool finished
  {"type": "token",      "text": "..."}          # a piece of the final answer
  {"type": "done"}
  {"type": "error",      "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

from server.config import settings

from .agents import get_agent
from .tools import run_tool, schemas_for

log = logging.getLogger("chat.runtime")

MAX_TOOL_ROUNDS = 6          # safety cap on tool-calling iterations per turn

# Generous for a spoken reply, which is 1-2 sentences. It stays generous because
# ``enable_thinking`` is False in the payload below: with thinking off, the voice
# turns measured after the Indian-name correction rules were added to the prompt
# complete in 12-18 completion tokens.
#
# Worth knowing what this guards against. With thinking ON the same turns burn
# 336-1021 tokens deliberating, and the failure is silent and total — hit the
# ceiling mid-thought and ``content`` comes back empty, so the agent says nothing
# at all rather than something short. So if anyone ever re-enables reasoning
# here, this number is the first thing that has to move.
FINAL_MAX_TOKENS = 900


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}


async def _chat_once(client: httpx.AsyncClient, messages: list[dict], tools: list[dict]) -> dict:
    """One non-streaming completion with tools enabled. Returns the message dict."""
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.4,
        "max_tokens": FINAL_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = await client.post("/chat/completions", json=payload, headers=_headers())
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


def _clean_history(messages: list[dict]) -> list[dict]:
    """Keep only role/content user & assistant turns from the client history."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out


async def run_agent(agent_id: str, history: list[dict]) -> AsyncIterator[dict]:
    """Run one assistant turn for the given agent. Yields event dicts."""
    agent = get_agent(agent_id)
    tools = schemas_for(agent.tools)

    messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]
    messages.extend(_clean_history(history))

    base_url = settings.llm_base_url.rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(90.0, connect=10.0)) as client:
        final_text = ""
        try:
            for round_no in range(MAX_TOOL_ROUNDS):
                yield {"type": "status", "text": "Thinking…" if round_no == 0 else "Working…"}
                msg = await _chat_once(client, messages, tools)
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    final_text = msg.get("content") or ""
                    break

                # Record the assistant's tool-call message verbatim (required by the API
                # so the following tool results line up with their call ids).
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError:
                        args = {}
                    call_id = tc.get("id") or name

                    yield {"type": "tool_call", "id": call_id, "name": name, "args": args}
                    result = await run_tool(name, args)
                    yield {"type": "tool_result", "id": call_id, "name": name, "result": result}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result, default=str)[:6000],
                    })
            else:
                # Exhausted tool rounds without a final answer — make one last plain call.
                msg = await _chat_once(client, messages, tools)
                final_text = msg.get("content") or "I've gathered the information but ran out of steps to summarize it."

            if not final_text.strip():
                final_text = "Done."

            # Stream the final answer word-by-word for a live typing feel.
            for chunk in _chunk_stream(final_text):
                yield {"type": "token", "text": chunk}
                await asyncio.sleep(0.012)

            yield {"type": "done"}

        except httpx.HTTPStatusError as e:
            log.error("LLM HTTP error: %s — %s", e, getattr(e.response, "text", "")[:300])
            yield {"type": "error", "message": f"LLM error: {e.response.status_code}"}
        except Exception as e:  # noqa: BLE001
            log.exception("Agent run failed")
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


def _chunk_stream(text: str, size: int = 3):
    """Yield the text in small word-aware chunks (keeps words intact)."""
    words = text.split(" ")
    buf: list[str] = []
    for i, w in enumerate(words):
        buf.append(w)
        if len(buf) >= size:
            yield (" ".join(buf) + (" " if i < len(words) - 1 else ""))
            buf = []
    if buf:
        yield " ".join(buf)
