"""Tool-calling response provider for the VOICE path.

The realtime VoicePipeline normally streams the LLM straight to TTS. This module
gives it a smarter provider: for each spoken user turn it runs the same native
tool-calling loop the text agent uses (crm_search, crm_update_record, flight
search, ticket creation, …), pushes tool_call / tool_result events to the UI so
the on-screen chips update live, and then yields the final answer text for TTS.

Wire it up with:
    pipeline._response_provider = make_voice_provider(agent_id, pipeline)
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from server.config import settings

from .agents import get_agent, voice_prompt
from .runtime import MAX_TOOL_ROUNDS, _chat_once, _clean_history
from .tools import run_tool, schemas_for

log = logging.getLogger("chat.voice_agent")

# Emoji / symbol ranges to strip before speaking.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "←-⇿⌀-⏿⬀-⯿️]"
)


def clean_for_tts(text: str) -> str:
    """Strip characters that confuse the speech model (markdown, bullets, emojis, '!').

    Keeps hyphens inside words (e-mail, follow-up) but removes bullet/standalone
    dashes, asterisks, backticks, hashes, underscores, and converts '!' to '.'.
    """
    if not text:
        return text
    text = re.sub(r"\*\*?([^*\n]+)\*\*?", r"\1", text)   # **bold** / *italic* → inner
    text = re.sub(r"`([^`]+)`", r"\1", text)              # `code` → inner
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)          # bullet markers at line start
    text = re.sub(r"\s+[-–—]\s+", ", ", text)             # " - " dash → comma pause
    text = text.replace("*", "").replace("`", "").replace("#", "")
    text = text.replace("_", " ").replace("~", "")
    text = text.replace("!", ".")                          # exclamation confuses TTS
    text = re.sub(r"\n+", ". ", text)                     # newlines → sentence breaks
    text = re.sub(r"\.\s*\.+", ".", text)                 # collapse repeated periods
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def make_speculative_prewarm(agent_id: str):
    """Build a speculative LLM prewarm: ``prewarm(user_text, history) -> (text|None, ms)``.

    Runs a SINGLE LLM completion for a provisional (not-yet-confirmed) utterance.
    Crucially it NEVER executes tools: if the model wants to call a tool we return
    ``None`` so the real turn handles it — speculating a ``crm_update_record`` for an
    utterance the user then continued/cancelled would be a real side-effect bug.
    So speculation only accelerates plain conversational answers (greetings, quick
    replies, chit-chat) — exactly the turns where the latency is most visible.

    Returns the cleaned spoken text (ready for TTS) plus the LLM round-trip ms, or
    ``(None, ms)`` when the utterance needs tools / errors out.
    """
    agent = get_agent(agent_id)
    tools = schemas_for(agent.tools)
    system = voice_prompt(agent_id)
    base_url = settings.llm_base_url.rstrip("/")

    async def prewarm(user_text: str, history: list[dict]):
        messages = [{"role": "system", "content": system}]
        messages.extend(_clean_history(history))
        messages.append({"role": "user", "content": user_text})
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=base_url, timeout=httpx.Timeout(30.0, connect=10.0)
            ) as client:
                msg = await _chat_once(client, messages, tools)
        except Exception:
            log.debug("Speculative prewarm failed", exc_info=True)
            return None, (time.monotonic() - t0) * 1000.0
        ms = (time.monotonic() - t0) * 1000.0
        if msg.get("tool_calls"):
            # Needs tools — do not run them speculatively; let the real turn do it.
            return None, ms
        text = clean_for_tts(msg.get("content") or "")
        return (text or None), ms

    return prewarm


def make_voice_provider(agent_id: str, pipeline):
    """Build an async provider(user_text, history) -> AsyncIterator[str] for the pipeline."""
    agent = get_agent(agent_id)
    tools = schemas_for(agent.tools)
    system = voice_prompt(agent_id)
    base_url = settings.llm_base_url.rstrip("/")

    # Anti-repeat guard: a small model (GLM-Flash) tends to re-emit its previous
    # answer verbatim when a user turn gives it nothing new to do ("okay", "stop").
    # We detect a near-duplicate reply and speak a short varied line instead.
    last_spoken = {"text": "", "i": 0}
    _FALLBACKS = [
        "Sure — is there anything else I can help with?",
        "All set. Anything else you need?",
        "Got it. Let me know if there's anything else.",
        "No problem. What else can I do for you?",
    ]

    def _word_set(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

    def _is_repeat(a: str, b: str) -> bool:
        sa, sb = _word_set(a), _word_set(b)
        if len(sa) < 3 or len(sb) < 3:
            return False
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        return inter / union >= 0.6  # Jaccard overlap → essentially the same answer

    async def provider(user_text: str, history: list[dict]):
        messages = [{"role": "system", "content": system}]
        messages.extend(_clean_history(history))
        messages.append({"role": "user", "content": user_text})

        final_text = ""
        try:
            async with httpx.AsyncClient(
                base_url=base_url, timeout=httpx.Timeout(90.0, connect=10.0)
            ) as client:
                for _ in range(MAX_TOOL_ROUNDS):
                    if pipeline._interrupted.is_set():
                        return
                    msg = await _chat_once(client, messages, tools)
                    tool_calls = msg.get("tool_calls") or []

                    if not tool_calls:
                        final_text = msg.get("content") or ""
                        break

                    messages.append({
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        "tool_calls": tool_calls,
                    })
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        raw = fn.get("arguments") or "{}"
                        try:
                            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                        except json.JSONDecodeError:
                            args = {}
                        call_id = tc.get("id") or name

                        # Surface the tool activity to the UI (voice orbs / chips).
                        await pipeline._text_out_queue.put(
                            {"type": "tool_call", "id": call_id, "name": name, "args": args}
                        )
                        if name == "request_typed_input":
                            # Fallback aid only. Show an OPTIONAL text box but keep the
                            # mic LIVE — the user can type or just say it (voice-first).
                            # (Never pauses the mic, so it can't dead-loop when the user
                            # speaks instead of typing.)
                            field = args.get("field") or "information"
                            prompt = args.get("prompt") or f"Optional: type your {field}"
                            await pipeline._text_out_queue.put(
                                {"type": "request_text", "field": field, "prompt": prompt}
                            )
                            result = {
                                "status": "text_box_shown", "field": field,
                                "message": (f"A text box for '{field}' is on screen and the mic stays "
                                            "LIVE. Ask the user ONCE to type it in (best for emails and "
                                            "phone numbers); their typed value arrives as the next "
                                            "message — use it directly. Do NOT repeat this prompt or "
                                            "ask again while they are still typing."),
                            }
                        elif name == "end_call":
                            # Tell the client to hang up after the farewell finishes.
                            await pipeline._text_out_queue.put({"type": "end_call"})
                            result = {"status": "ending_call",
                                      "message": ("The call will hang up after your spoken "
                                                  "farewell. Say one short warm goodbye now.")}
                        else:
                            result = await run_tool(name, args)
                        await pipeline._text_out_queue.put(
                            {"type": "tool_result", "id": call_id, "name": name, "result": result}
                        )
                        log.info("[VOICE TOOL] %s(%s) -> %s", name, args,
                                 json.dumps(result, default=str)[:160])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps(result, default=str)[:6000],
                        })
                else:
                    # Exhausted tool rounds — ask for a final spoken summary.
                    msg = await _chat_once(client, messages, tools)
                    final_text = msg.get("content") or "Sorry, I couldn't finish that just now."
        except Exception:
            log.exception("Voice tool provider failed")
            final_text = "Sorry, I ran into an error while doing that."

        if not final_text.strip():
            final_text = "Done."

        # Strip markdown/emojis/'!' etc. so the speech model isn't confused.
        final_text = clean_for_tts(final_text)

        # Anti-repeat: if this reply is essentially the model's previous reply,
        # speak a short varied line instead of parroting it back. Compare against
        # the model's *raw* last answer (stored below) so consecutive repeats are
        # all caught, not just the first.
        if _is_repeat(final_text, last_spoken["text"]):
            spoken = _FALLBACKS[last_spoken["i"] % len(_FALLBACKS)]
            last_spoken["i"] += 1
        else:
            spoken = final_text
        last_spoken["text"] = final_text   # always remember what the model produced
        final_text = spoken

        # Yield word-by-word so the pipeline can split into sentences for TTS.
        words = final_text.split(" ")
        for i, w in enumerate(words):
            if pipeline._interrupted.is_set():
                return
            yield w + (" " if i < len(words) - 1 else "")

    return provider
