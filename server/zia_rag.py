"""Zia RAG over the Zoho product docs — the knowledge-base demo's answer source.

This is the third way this stack can produce a reply. The other two go to a
language model: ``chat/voice_agent.py`` runs a persona with tools, and
``llm_client`` streams raw completions. Here retrieval and generation both happen
inside Zoho's hosted Zia RAG service, and all this file does is drive it, decide
which tokens are the answer, and make those tokens speakable.

    query ──► correct_query (local LLM) ──► Zia RAG (SSE) ──► clean_for_tts ──► TTS

Why thinking is off, and why that is the whole latency story
------------------------------------------------------------
The API takes ``enable_thinking``/``do_reasoning``. With them on, measured on
this collection over four real queries:

    query                        thinking ON   thinking OFF
    submit an expense report        19.79s         2.37s
    create a workflow rule         18.18s         2.27s
    what is SalesInbox              7.60s         1.71s
    send mass emails               14.22s         1.67s

Nine times slower, and *worse*: on two of the four, the thinking run refused
("the provided documents do not provide instructions...") while the same query
without thinking returned the real steps. So it is off, and the 6-9s stall it was
blamed for is simply gone.

It also moves where the answer lives, which is the trap in this API:

    thinking ON   output[0].content = the chain-of-thought  (842 tokens)
                  output[1].content = the answer            (16 tokens)
    thinking OFF  output[0].content = the answer

Read ``output[0]`` with thinking on and the agent recites the model's private
reasoning out loud — "Scan the Documents for Keywords... Context 0: no mention of
recall...". :func:`_answer_paths` therefore tracks the highest ``output[N]`` index
it has seen and yields only from that, so the right thing happens either way
rather than depending on a config flag staying put.

Why the answer has to be cleaned before it is spoken
----------------------------------------------------
The service answers in markdown, because its other callers render it:

    To create a workflow rule, follow these steps:

    1.  **Log in to the Zoho Developer Console** and click **CRM for Verticals**.
        Go to Setup > Automation > Workflow Rules.

Fed to TTS that is "one dot log in to the Zoho Developer Console star star". The
asterisks, the list numbering and the ``>`` breadcrumbs all have to go, and the
breadcrumb has to become something a person would say — "Setup, then Automation,
then Workflow Rules". :func:`clean_for_tts` does that, and it runs per sentence
on the streamed text rather than at the end, so the first sentence can still be
spoken while the rest is still arriving.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import AsyncIterator

import httpx

log = logging.getLogger(__name__)

# ── endpoints ────────────────────────────────────────────────────────────────
ISC_URL = "http://zcrmdit-a6k-1.csez.zohocorpin.com:7374/generate"
RAG_URL = "https://crmzia.kites.localzoho.com/rag/api/v1/chat"
COLLECTION = "ZohoCRM_Email_Bot"
MODEL = "crm-di-glm47b_30b_it"

# The ISC token is minted per request by the generator service and carries its own
# issue time. We cache it briefly rather than per-call: a fetch is ~50ms, and
# spending that on every turn is latency for nothing. Any 401 clears the cache
# and retries once, so a short-lived token cannot wedge the demo.
_TOKEN_TTL_SEC = 600.0
_token: tuple[str, float] | None = None
_token_lock = asyncio.Lock()


async def isc_token(force: bool = False) -> str:
    """Fetch (or reuse) a SystemAuth token for the RAG service."""
    global _token
    async with _token_lock:
        if not force and _token is not None and time.monotonic() < _token[1]:
            return _token[0]
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(ISC_URL)
            r.raise_for_status()
            body = r.json()
        # The service returns a bare JSON string. Accept an object too, so a
        # future wrapper shape does not break the demo silently.
        tok = body if isinstance(body, str) else (
            body.get("token") or body.get("auth") or body.get("authorization")
        )
        if not isinstance(tok, str) or not tok:
            raise RuntimeError(f"ISC token endpoint returned no token: {body!r}")
        _token = (tok, time.monotonic() + _TOKEN_TTL_SEC)
        log.info("ISC token acquired (%s...)", tok[:24])
        return tok


# ── answer-token selection ───────────────────────────────────────────────────
_OUTPUT_PATH = re.compile(r"^response\.data\[\d+\]\.output\[(\d+)\]\.content$")


class _AnswerPicker:
    """Decides which streamed deltas are the answer.

    The service emits deltas for reasoning thoughts, enriched queries and one or
    more output items, all on the same SSE stream and distinguished only by
    ``path``. The answer is always the LAST ``output[N]`` item; with thinking off
    that is ``output[0]``, with thinking on it is ``output[1]`` and ``output[0]``
    is the chain-of-thought.

    So rather than trusting a configured flag, this watches the index. If a higher
    one appears mid-stream, everything emitted so far was reasoning — the caller
    is told to discard it and start again from the new item.
    """

    def __init__(self) -> None:
        self.index: int | None = None

    def take(self, path: str) -> tuple[bool, bool]:
        """Return ``(is_answer, discard_what_came_before)`` for one delta."""
        m = _OUTPUT_PATH.match(path or "")
        if m is None:
            return False, False            # reasoning.thoughts, .queries, etc.
        i = int(m.group(1))
        if self.index is None:
            self.index = i
            return True, False
        if i > self.index:
            log.info("answer moved to output[%d] — discarding output[%d] "
                     "(that was reasoning, not the answer)", i, self.index)
            self.index = i
            return True, True
        return i == self.index, False


# ── making the answer speakable ──────────────────────────────────────────────
# Breadcrumbs read as gibberish through TTS. "Setup > Automation > Workflow
# Rules" has to become words before it reaches the synthesiser.
_BREADCRUMB = re.compile(r"\s*>\s*")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1", re.S)
_STRAY_EMPH = re.compile(r"[*_`#]+")
_LIST_NUM = re.compile(r"^\s*\d+[.)]\s+", re.M)
_LIST_BULLET = re.compile(r"^\s*[-•*]\s+", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_ANCHOR = re.compile(r"#\S+?#")          # the docs carry "#heading#" markers
_WS = re.compile(r"[ \t]{2,}")


def clean_for_tts(text: str) -> str:
    """Strip markdown and doc artefacts so a sentence can be spoken.

    Deliberately lossy — it is producing speech, not a document. Numbering is
    dropped rather than read out ("one dot log in to..." is not what a person
    says), and the breadcrumb separator becomes "then", which is how anyone
    reading a menu path aloud would say it.
    """
    if not text:
        return ""
    t = _LINK.sub(r"\1", text)
    t = _ANCHOR.sub(" ", t)
    t = _BOLD_ITALIC.sub(r"\2", t)       # **bold** -> bold, keeping the words
    t = _LIST_NUM.sub("", t)
    t = _LIST_BULLET.sub("", t)
    t = _BREADCRUMB.sub(", then ", t)
    t = _STRAY_EMPH.sub("", t)           # anything the pairs above missed
    t = t.replace("\n", " ")
    t = _WS.sub(" ", t)
    return t.strip()


# ── the request ──────────────────────────────────────────────────────────────
def _payload(query: str, history: list[dict] | None) -> dict:
    return {
        "id": str(uuid.uuid4()).upper(),
        "collection": COLLECTION,
        "generate_response": True,
        "search_config": {
            "do_hybrid_search": True,
            "do_re_ranking": True,
            "search_threshold": 0.2,
            "max_reconstruct_depth": 1,
            "alpha": 0.5,
        },
        "llm_config": {
            "model": MODEL,
            "top_p": 1,
            "top_k": 40,
            "temperature": 0.2,
            # Off. See the module docstring — this one flag is 9x the latency and
            # a worse answer, and it is what buries the answer under output[1].
            "enable_thinking": False,
        },
        "query": query,
        "generation_config": {
            "do_reasoning": False,
            "query_enrichment": True,
            "email_template_required": True,
            "reply_type": "chat",
            "strict": False,
            # Lets the service answer conversational turns ("thanks", "are you
            # there?") itself instead of forcing every utterance through
            # retrieval. Without it those turns come back as a failed lookup and
            # the agent reads out an apology for having no documentation on
            # "thank you".
            "do_chitchat": True,
        },
        "stream": True,
        "streaming_config": {"send_as_tokens": True, "stream_context_first": True},
        "chat_history": history or [],
    }


class ZiaRAG:
    """Streaming client for the Zia RAG chat endpoint."""

    name = "zia-rag"

    def __init__(self, *, collection: str = COLLECTION, timeout_sec: float = 60.0):
        self.collection = collection
        self.timeout_sec = timeout_sec
        self.last_ttft_ms: float | None = None
        self.last_total_ms: float | None = None

    async def stream(
        self, query: str, history: list[dict] | None = None, *, _retry: bool = True
    ) -> AsyncIterator[str]:
        """Yield answer text as it arrives, already cleaned for speech.

        Yields raw *fragments*, not sentences — the caller splits sentences,
        because it is the one that knows what a speakable unit is.
        """
        tok = await isc_token()
        headers = {
            "Authorization": f"SystemAuth {tok}",
            "Content-Type": "application/json",
            "remote-service-name": "ZohoCRM",
        }
        body = _payload(query, history)
        body["collection"] = self.collection

        t0 = time.perf_counter()
        picker = _AnswerPicker()
        self.last_ttft_ms = None
        # verify=False: an internal host with an internal chain. Stated here
        # rather than left implicit, because a silent TLS bypass is worth seeing.
        async with httpx.AsyncClient(timeout=self.timeout_sec, verify=False) as c:
            async with c.stream("POST", RAG_URL, headers=headers, json=body) as r:
                if r.status_code == 401 and _retry:
                    await r.aread()
                    log.info("RAG returned 401 — refreshing ISC token and retrying")
                    await isc_token(force=True)
                    async for frag in self.stream(query, history, _retry=False):
                        yield frag
                    return
                if r.status_code >= 400:
                    detail = (await r.aread())[:300]
                    raise RuntimeError(f"RAG HTTP {r.status_code}: {detail!r}")

                event = None
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                        continue
                    if not line.startswith("data:") or event != "content_item_delta":
                        continue
                    try:
                        d = json.loads(line.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        continue
                    is_answer, discard = picker.take(d.get("path", ""))
                    if discard:
                        # What we streamed was reasoning. Tell the caller to drop
                        # it; better a visible reset than a spoken monologue.
                        yield "\x00"
                        self.last_ttft_ms = None
                    if not is_answer:
                        continue
                    val = d.get("value")
                    if not val:
                        continue
                    if self.last_ttft_ms is None:
                        self.last_ttft_ms = (time.perf_counter() - t0) * 1000.0
                    yield val
        self.last_total_ms = (time.perf_counter() - t0) * 1000.0
        log.info("RAG done: ttft=%s total=%.0fms",
                 f"{self.last_ttft_ms:.0f}ms" if self.last_ttft_ms else "no answer",
                 self.last_total_ms)


# ── the correction layer ─────────────────────────────────────────────────────
# Same idea as the Indian-name correction in chat/agents.py, pointed at Zoho
# product names. The recogniser is English-trained and has never seen this
# vocabulary, so "SalesInbox" comes back as "sales in box" and "Bigin" as
# "begin" — and unlike a person's name, a wrong product name silently retrieves
# from the wrong part of the docs.
_CORRECTION_SYSTEM = """You repair speech-recognition errors in questions about Zoho products.

Return ONLY the corrected question. No explanation, no quotes, no preamble.

Fix misheard Zoho product and feature names to their real spelling. The recogniser
is trained on US English and has never seen this vocabulary. Real examples:
  "zoho male" / "zoho mel"        -> Zoho Mail
  "sales in box" / "sales inbox"  -> SalesInbox
  "begin" / "big in" (as product) -> Bigin
  "zoho desk top"                 -> Zoho Desk
  "creator" / "create or"         -> Zoho Creator
  "catalyst" / "cattle list"      -> Catalyst
  "zoho books" / "zoho book"      -> Zoho Books
  "work drive" / "workdrive"      -> WorkDrive
  "cliq" / "click" (as product)   -> Cliq
  "zia" / "ziya" / "sia"          -> Zia
  "crm" / "see r m" / "sea rm"    -> CRM
  "work flow rule"                -> workflow rule
  "block you print" / "blueprint" -> blueprint
Also fix obvious CRM-domain words: "potentials", "leads", "deals", "modules",
"custom view", "assignment rule", "scoring rule", "webform", "webhook".

Rules:
- Change ONLY what is clearly a misheard product, feature or CRM term.
- If nothing is clearly wrong, return the question EXACTLY as given.
- Never answer the question. Never add or remove intent. Never add punctuation
  beyond what fixing a word requires.
- Do NOT fix grammar, tense, agreement or filler words. "What are the informations
  do we have" stays exactly as it is — the retriever does not care and the caller
  did not ask to be corrected.
- Do NOT re-capitalise or re-punctuate a sentence that has no product name in it.
- Keep it the same length and meaning. This is a spelling repair, not a rewrite."""


class _CorrectionBreaker:
    """Stops paying for a correction pass that is not coming back.

    The correction runs on a shared vLLM. When that box is busy — 13 running and
    16 queued while this was written, with a four-token request taking 11s — every
    turn spends its whole budget waiting and then proceeds uncorrected anyway.
    Paying that on turn after turn is pure latency for nothing.

    So after ``trip_after`` consecutive timeouts the pass is skipped outright for
    ``cool_off_sec``, then one turn is allowed through to test the water. A busy
    server therefore costs the budget a few times, not on every question.
    """

    def __init__(self, trip_after: int = 3, cool_off_sec: float = 120.0):
        self.trip_after = trip_after
        self.cool_off_sec = cool_off_sec
        self._misses = 0
        self._open_until = 0.0

    def should_skip(self) -> bool:
        return time.monotonic() < self._open_until

    def record(self, timed_out: bool) -> None:
        if not timed_out:
            self._misses = 0
            self._open_until = 0.0
            return
        self._misses += 1
        if self._misses >= self.trip_after:
            self._open_until = time.monotonic() + self.cool_off_sec
            self._misses = 0
            log.warning(
                "query correction disabled for %.0fs — the LLM missed its budget "
                "%d times running. Retrieval is unaffected; product names will go "
                "to the retriever as heard.",
                self.cool_off_sec, self.trip_after,
            )


BREAKER = _CorrectionBreaker()


async def correct_query(
    llm, text: str, *, timeout_sec: float = 0.6, breaker: "_CorrectionBreaker | None" = None
) -> tuple[str, float]:
    """Repair misheard Zoho product names before the query goes to retrieval.

    Returns ``(query, ms)``. On timeout, failure, or a suspicious answer the
    original text is returned unchanged — a correction pass that can lose the
    question is worse than no correction pass, and this one sits directly in
    front of the only thing the demo does.

    Why the budget is 600ms and not "however long it takes"
    ------------------------------------------------------
    This call is the one part of the knowledge-base path that touches the shared
    language model, and that model is the reason the persona demos wait 5+ seconds
    for audio — measured at the first token: median 993ms, p90 20.2s, worst 49.4s.
    Retrieval itself answers in ~1.6s on its own infrastructure, so an unbounded
    correction can turn a 1.6s demo into a 20s one for a spelling fix.

    600ms is deliberate: enough for the call to land on an idle server (a short
    completion runs in roughly 400-700ms), and small enough that a busy one costs
    a third of the retrieval time rather than ten times it. Combined with
    :class:`_CorrectionBreaker`, the LLM's worst-case contribution to this demo is
    600ms on a few turns and zero thereafter.

    A truly concurrent version — retrieve on the raw query, then abandon and
    re-retrieve if the correction wins — was considered and rejected. It saves
    only this 600ms in the congested case, spends a wasted retrieval in the
    healthy one, and puts a cancel-and-restart race in the one code path the demo
    cannot afford to have a bug in.
    """
    t0 = time.perf_counter()
    if not text or not text.strip() or llm is None:
        return text, 0.0
    breaker = breaker if breaker is not None else BREAKER
    if breaker.should_skip():
        return text, 0.0
    try:
        out = await asyncio.wait_for(
            llm.chat_complete(
                [{"role": "system", "content": _CORRECTION_SYSTEM},
                 {"role": "user", "content": text.strip()}],
                temperature=0.0, max_tokens=160,
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        breaker.record(timed_out=True)
        log.info("query correction missed its %.0fms budget — retrieving as heard",
                 timeout_sec * 1000)
        return text, (time.perf_counter() - t0) * 1000.0
    except Exception:
        breaker.record(timed_out=True)
        log.exception("query correction failed — original stands")
        return text, (time.perf_counter() - t0) * 1000.0

    breaker.record(timed_out=False)
    ms = (time.perf_counter() - t0) * 1000.0
    fixed = (out or "").strip().strip('"').strip()
    # Guard against the model answering instead of correcting. A repair is about
    # as long as its input; anything much longer is a reply, and speaking a reply
    # into the retriever would be a very confusing bug to chase.
    if not fixed or len(fixed) > max(80, len(text) * 2):
        log.info("query correction rejected (%d chars from %d) — original stands",
                 len(fixed), len(text))
        return text, ms
    if fixed != text.strip():
        log.info("query corrected in %.0fms\n    heard : %r\n    fixed : %r",
                 ms, text.strip(), fixed)
    return fixed, ms


# ── the waiting-room layer ───────────────────────────────────────────────────
# Retrieval plus generation is ~1.7-2.4s with thinking off. That is short enough
# to be worth covering with one spoken line and too long to leave as silence: on
# a phone call two seconds of nothing reads as a dropped connection.
#
# One line only, chosen so it lasts about as long as the wait. Two would overrun
# the answer and have to be talked over.
FILLERS = [
    "Sure, let me look that up for you.",
    "Good question — checking the docs now.",
    "One moment, I'm pulling that up.",
    "Let me find that for you.",
    "Right, let me check the documentation.",
    "Sure thing, just a second.",
    "Let me dig that out for you.",
    "Okay, looking into that now.",
    "Give me one second, I'm checking.",
    "Let me pull up the details on that.",
    "Sure, I'll find that out for you.",
    "Just a moment while I look that up.",
]


class FillerPicker:
    """Hands out filler lines without repeating itself.

    A fixed pool read in order is worse than silence — a demo audience hears the
    same sentence twice and the illusion is done. This keeps the last few and
    never reuses them, so a short session never repeats at all.
    """

    def __init__(self, lines: list[str] | None = None, avoid_last: int = 4):
        self._lines = list(lines or FILLERS)
        self._avoid = max(0, min(avoid_last, len(self._lines) - 1))
        self._recent: list[str] = []
        # Deterministic per-process shuffle would repeat across sessions, and
        # Random() seeded from the clock is exactly what we want here: a fresh
        # order every run, no cross-session pattern for anyone to notice.
        import random

        self._rng = random.Random()

    def next(self) -> str:
        pool = [x for x in self._lines if x not in self._recent] or list(self._lines)
        pick = self._rng.choice(pool)
        self._recent.append(pick)
        if len(self._recent) > self._avoid:
            self._recent.pop(0)
        return pick
