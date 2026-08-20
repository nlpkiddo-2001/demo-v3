"""Settings for the parts that talk to the outside world — the LLM and the TTS.

Deliberately small. The speech stack takes its configuration as constructor
arguments (a `GateConfig`, an `EndpointConfig`, a device string), because those
are tuning decisions that belong next to the code being tuned and want to be set
per-experiment from the command line. What is left over is genuinely
*environmental* — where the language model lives, which voice to speak with, what
key to use — and that belongs in a file rather than a flag.

Values come from ``demo-v3/.env`` if present, otherwise the process environment,
otherwise the defaults here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Read a .env file into the environment without adding a dependency.

    Existing environment variables win, so a command line or systemd unit can
    always override the file.
    """
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        # Trailing comments are common in hand-edited env files.
        if " #" in val:
            val = val.split(" #", 1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(ROOT / ".env")


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _b(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses short and conversational "
    "(2-3 sentences max). Do not use markdown, bullet points, asterisks, "
    "numbered lists, or special formatting. Speak naturally as in a real "
    "conversation. If asked for a list, describe items conversationally. "
    "Whenever you need the user's name, email address, or any spelling-sensitive "
    "detail, ask them to type it rather than say it."
)


@dataclass
class Settings:
    # ── language model (OpenAI-compatible endpoint) ──
    llm_base_url: str = field(default_factory=lambda: _s("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: _s("LLM_MODEL", "zai-org/GLM-4.7-Flash"))
    llm_api_key: str = field(default_factory=lambda: _s("LLM_API_KEY", "EMPTY"))
    # Short on purpose: this is a voice agent. A long answer is a long silence
    # while it is spoken, and people interrupt rather than listen to a paragraph.
    llm_max_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 256))
    llm_temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.3))
    system_prompt: str = field(default_factory=lambda: _s("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))

    # ── speech output (Orpheus served by vLLM) ──
    tts_api_url: str = field(default_factory=lambda: _s("TTS_API_URL", "http://localhost:5100/v1/completions"))
    tts_model: str = field(default_factory=lambda: _s("TTS_MODEL", "canopylabs/orpheus-tts-0.1-finetune-prod"))
    # Fallback only — each persona names its own voice in chat/agents.py and
    # that wins. This is what an agent with no voice set would speak with.
    tts_voice: str = field(default_factory=lambda: _s("TTS_VOICE", "jess"))
    tts_temperature: float = field(default_factory=lambda: _f("TTS_TEMPERATURE", 0.2))
    tts_top_p: float = field(default_factory=lambda: _f("TTS_TOP_P", 0.9))
    tts_repetition_penalty: float = field(default_factory=lambda: _f("TTS_REPETITION_PENALTY", 1.1))
    tts_max_tokens: int = field(default_factory=lambda: _i("TTS_MAX_TOKENS", 1200))
    # SNAC decodes Orpheus's tokens into audio. Kept off the speech GPU by
    # default so a long reply cannot contend with the recogniser mid-sentence.
    snac_device: str = field(default_factory=lambda: _s("SNAC_DEVICE", "cuda:1"))

    # ── agent server ──
    chat_port: int = field(default_factory=lambda: _i("CHAT_PORT", 8444))
    ssl_certfile: str = field(default_factory=lambda: _s("SSL_CERTFILE", "../demo-v2/certs/cert.pem"))
    ssl_keyfile: str = field(default_factory=lambda: _s("SSL_KEYFILE", "../demo-v2/certs/key.pem"))
    # Read by the ported voice-agent provider; our own turn layer does the real
    # work, so this stays off until speculation is wired to the fusion score.
    speculative_turn: bool = field(default_factory=lambda: _b("SPECULATIVE_TURN", False))

    # ── Sarvam (the accurate pass for the Indian-languages mode) ──
    # The only part of the speech stack that is not on-prem, and the only one
    # that can transcribe code-mixed Tamil/Telugu/Kannada. Without a key the
    # pass disables itself and the live transcript stands — see stt_sarvam.
    sarvam_api_key: str = field(default_factory=lambda: _s("SARVAM_API_KEY", ""))
    sarvam_model: str = field(default_factory=lambda: _s("SARVAM_MODEL", "saaras:v3"))
    # codemix = native script plus English in one pass. Only saaras:v3 accepts
    # this parameter; on v4 it must be "transcribe".
    sarvam_mode: str = field(default_factory=lambda: _s("SARVAM_MODE", "codemix"))
    sarvam_timeout_sec: float = field(default_factory=lambda: _f("SARVAM_TIMEOUT_SEC", 3.0))

    # ── Zoho CRM (the sales use case writes live records) ──
    zoho_accounts_host: str = field(default_factory=lambda: _s("ZOHO_ACCOUNTS_HOST", "accounts.zoho.com"))
    zoho_api_base_url: str = field(default_factory=lambda: _s("ZOHO_API_BASE_URL", "https://www.zohoapis.com/crm/v8"))
    zoho_client_id: str = field(default_factory=lambda: _s("ZOHO_CLIENT_ID", ""))
    zoho_client_secret: str = field(default_factory=lambda: _s("ZOHO_CLIENT_SECRET", ""))
    zoho_refresh_token: str = field(default_factory=lambda: _s("ZOHO_REFRESH_TOKEN", ""))
    zoho_access_token: str = field(default_factory=lambda: _s("ZOHO_ACCESS_TOKEN", ""))

    # ── Zia RAG (the support agent's knowledge base) ──
    zia_rag_url: str = field(default_factory=lambda: _s(
        "ZIA_RAG_URL", "http://crm-rtx6000-1.csez.zohocorpin.com:7579/rag/api/v1/chat"))
    zia_rag_auth: str = field(default_factory=lambda: _s("ZIA_RAG_AUTH", ""))
    zia_rag_collection: str = field(default_factory=lambda: _s("ZIA_RAG_COLLECTION", "ZohoCRM_Email_Bot"))
    zia_rag_top_k: int = field(default_factory=lambda: _i("ZIA_RAG_TOP_K", 20))
    zia_rag_ticket_ids: list = field(default_factory=list)
    zia_rag_query_statuses: list = field(default_factory=list)
    zia_rag_verify_ssl: bool = field(default_factory=lambda: _b("ZIA_RAG_VERIFY_SSL", False))


settings = Settings()
