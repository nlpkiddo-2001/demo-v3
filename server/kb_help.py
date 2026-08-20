"""Zoho CRM Zia RAG help-bot — kb_help mode.

What does this file do?
------------------------
When the pipeline is in "kb_help" mode, user questions are answered by calling
Zoho's hosted Zia RAG API, which performs retrieval and generation in a single
round-trip using the ``ZohoCRM_SME_collection`` document store.

The API streams the answer back as Server-Sent Events (SSE).  We parse the SSE
stream and yield text tokens so TTS can start speaking before the full answer
is ready — same interface as the previous web-crawl implementation.

SSE event flow (simplified):
  event: response_start
  event: system          (retrieval started / success)
  event: content_item_start  (path: response.data[0].output[0].content)
  event: content_item_delta  data: {"value": "The", "path": "response.data[0].output[0].content"}
  event: content_item_delta  data: {"value": " customer", ...}
  ...
  event: content_item_complete
  event: response_end

We only act on ``content_item_delta`` events whose ``path`` is
``response.data[0].output[0].content`` — those carry the answer tokens.

Filtering (search_constraints):
  The request can optionally filter the document collection by ``ticket_id``
  and ``query_status``.  Configure the lists in settings (or .env) — if either
  list is empty the corresponding operation is omitted.  When no operations
  remain, the entire ``search_constraints`` block is omitted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

import httpx
import requests

from .config import settings

log = logging.getLogger(__name__)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(colour: str, msg: str) -> str:
    """Wrap *msg* in the given ANSI colour code."""
    return f"{colour}{msg}{_RESET}"


# ── Dynamic ISC token ────────────────────────────────────────────────────────────────
# A fresh token is fetched from the token service on every RAG call.

_ISC_TOKEN_URL = "http://zcrmdit-a6k-1.csez.zohocorpin.com:7374/generate"


def generate_request_id():
    url = "http://zcrmdit-a6k-1.csez.zohocorpin.com:7374/generate"

    response = requests.get(url)
    response.raise_for_status()

    # Strip quotes and trailing %
    token = response.text.strip().strip('"').rstrip('%')

    return token


# ── Constants ────────────────────────────────────────────────────────────────

# Friendly fallback spoken when the API returns nothing useful.
OUT_OF_SCOPE_REPLY = (
    "I'm sorry, I can only answer questions that are covered by the official "
    "Zoho CRM knowledge base. Could you rephrase your question "
    "or ask about a Zoho CRM feature?"
)

# The SSE ``path`` field that carries LLM-generated answer tokens.
_ANSWER_PATH = "response.data[0].output[0].content"


# ── Zia RAG client ────────────────────────────────────────────────────────────


class KBHelpBot:
    """Thin async wrapper around the Zoho Zia RAG streaming API.

    ``ensure_index()`` is a no-op — the remote API manages its own document
    index.  ``stream_answer()`` calls the RAG endpoint and yields text tokens
    from the SSE stream.
    """

    async def ensure_index(self) -> None:
        """No-op: the Zia RAG API manages its own document index."""
        return

    # ── Internal: build request payload ─────────────────────────────────

    @staticmethod
    def _build_payload(question: str) -> dict:
        """Assemble the JSON body for the RAG API request.

        search_constraints is included only when at least one filter list is
        non-empty (avoids sending an empty ``operations`` array).
        """
        operations: list[dict] = []
        if settings.zia_rag_ticket_ids:
            operations.append(
                {
                    "name": "ticket_id",
                    "value": list(settings.zia_rag_ticket_ids),
                    "operator": "contains_any",
                }
            )
        if settings.zia_rag_query_statuses:
            operations.append(
                {
                    "name": "query_status",
                    "value": list(settings.zia_rag_query_statuses),
                    "operator": "contains_any",
                }
            )

        payload: dict = {
            "id": str(uuid.uuid4()),
            "collection": settings.zia_rag_collection,
            "search_config": {
                "do_hybrid_search": False,
                "do_re_ranking": False,
                "search_threshold": 0,
                "use_common_space": False,
                "top_k": settings.zia_rag_top_k,
                "max_reconstruct_depth": 10,
            },
            "query": question,
            "generate_response": True,
            "stream": True,
            "streaming_config": {"send_as_tokens": True},
        }

        if operations:
            required: list[str] = []
            if settings.zia_rag_ticket_ids:
                required.append("ticket_id")
            if settings.zia_rag_query_statuses:
                required.append("query_status")
            payload["search_constraints"] = {
                "required_properties": required,
                "filters": {
                    "condition": "and",
                    "operations": operations,
                },
            }

        return payload

    # ── Internal: SSE stream parser ──────────────────────────────────────

    @staticmethod
    async def _iter_answer_tokens(resp: httpx.Response) -> AsyncIterator[str]:
        """Process the SSE stream: log system/lifecycle events, yield answer tokens.

        Handles every ``data:`` line:
        - ``system`` events → logged in colour (yellow=started, green=success, red=other)
        - ``content_item_delta`` on the answer path → token yielded to caller
        - Everything else is silently skipped at DEBUG level.
        """
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                log.debug(_c(_CYAN, "[RAG] non-JSON SSE data: %s"), raw[:120])
                continue

            event_type: str = evt.get("event", "")

            if event_type == "system":
                stage = evt.get("stage", "unknown").upper()
                status = evt.get("status", "")
                time_ms = evt.get("time_taken_ms")
                time_str = f" ({time_ms}ms)" if time_ms is not None else ""
                if status == "started":
                    log.info(_c(_YELLOW, "[RAG] [%s] started ..."), stage)
                elif status == "success":
                    log.info(_c(_GREEN, "[RAG] [%s] success%s"), stage, time_str)
                else:
                    log.warning(_c(_RED, "[RAG] [%s] %s%s"), stage, status, time_str)

            elif event_type == "content_item_delta" and evt.get("path") == _ANSWER_PATH:
                token: str = evt.get("value", "")
                if token:
                    yield token

            else:
                log.debug("[RAG] SSE event=%r path=%r (skipped)", event_type, evt.get("path"))

    # ── Public: stream answer ────────────────────────────────────────────

    async def stream_answer(self, question: str, llm) -> AsyncIterator[str]:  # noqa: ARG002
        """Call the Zia RAG API and yield answer tokens as they arrive.

        ``llm`` is accepted for interface compatibility but is not used —
        the RAG API performs its own generation.

        Falls back to ``OUT_OF_SCOPE_REPLY`` on HTTP error, network failure,
        or when the API returns no answer tokens.
        """
        question = (question or "").strip()
        if not question:
            yield OUT_OF_SCOPE_REPLY
            return

        payload = self._build_payload(question)
        isc_token: str = settings.zia_rag_auth  # safe default
        try:
            isc_token = f"SystemAuth {generate_request_id()}"
            log.info(_c(_CYAN, "[RAG] ISC token acquired dynamically"))
        except Exception:
            log.warning(_c(_YELLOW, "[RAG] ISC token unavailable — using config auth"))
        headers = {
            "Authorization": isc_token,
            "content-type": "application/json",
            "remote-service-name": "ZohoCRM",
        }

        log.info(
            _c(_CYAN, "[RAG] Querying collection=%r  question=%r"),
            settings.zia_rag_collection,
            question[:80],
        )
        log.info(_c(_CYAN, f"[RAG] Headers: {headers}"))
        yielded_any = False
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                verify=settings.zia_rag_verify_ssl,
            ) as client:
                async with client.stream(
                    "POST",
                    settings.zia_rag_url,
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        log.error(
                            _c(_RED, "[RAG] HTTP %s FAILED for %r — body: %s"),
                            resp.status_code,
                            question[:80],
                            body[:200],
                        )
                        yield OUT_OF_SCOPE_REPLY
                        return

                    log.info(_c(_GREEN, "[RAG] HTTP %s OK — stream started"), resp.status_code)
                    response_parts: list[str] = []
                    async for token in self._iter_answer_tokens(resp):
                        yielded_any = True
                        log.debug(_c(_CYAN, "[RAG] token: %r"), token)
                        response_parts.append(token)
                        yield token
                    if response_parts:
                        full_response = "".join(response_parts)
                        log.info(
                            _c(_GREEN, "[RAG] Stream complete for %r — full response:\n%s"),
                            question[:80],
                            full_response,
                        )
                    else:
                        log.warning(_c(_YELLOW, "[RAG] Stream finished but no tokens received for %r"), question[:80])

        except httpx.ConnectError:
            log.error(_c(_RED, "[RAG] Connection refused at %s"), settings.zia_rag_url)
        except httpx.TimeoutException:
            log.error(_c(_RED, "[RAG] Request timed out for %r"), question[:80])
        except Exception:
            log.exception(_c(_RED, "[RAG] Unexpected error for %r"), question[:80])

        if not yielded_any:
            log.warning(_c(_YELLOW, "[RAG] No answer tokens — returning out-of-scope reply"))
            yield OUT_OF_SCOPE_REPLY

    # ── Public: non-streaming answer (convenience) ───────────────────────

    async def answer(self, question: str, llm) -> tuple[str, list[dict]]:
        """Collect the full answer as a single string.

        Returns ``(reply_text, [])`` — source documents are not exposed by
        this API in a form suitable for the old article-list interface.
        """
        parts: list[str] = []
        async for token in self.stream_answer(question, llm):
            parts.append(token)
        reply = "".join(parts).strip()
        return reply, []


# ── Module-level singleton ────────────────────────────────────────────────────

# One KBHelpBot instance shared across all WebSocket sessions.
_singleton: KBHelpBot | None = None


def get_kb_helpbot() -> KBHelpBot:
    """Return the shared KBHelpBot instance (creates it on first call)."""
    global _singleton
    if _singleton is None:
        _singleton = KBHelpBot()
    return _singleton
