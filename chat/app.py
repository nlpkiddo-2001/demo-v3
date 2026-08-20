"""FastAPI server for the text chatbot demo.

Endpoints:
  GET  /             → the OpenWebUI-style chat page (chat_client/index.html)
  GET  /api/agents   → list of available agents (id, name, icon, starters…)
  GET  /api/health   → LLM + Zoho CRM connectivity/config status
  POST /api/chat     → run one assistant turn; streams NDJSON events

Run:
  python -m chat.app          (reads HOST/CHAT_PORT from env; defaults 0.0.0.0:8600)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from server.config import (
    get_param_values,
    param_metadata,
    set_param,
    settings,
)

from . import db
from .agents import DEFAULT_AGENT, agents_public, get_agent, voice_prompt
from .runtime import run_agent
from .voice_agent import make_speculative_prewarm, make_voice_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("chat.app")

CLIENT_DIR = Path(__file__).resolve().parent.parent / "chat_client"

app = FastAPI(title="Multi-Agent Chat Demo")


@app.on_event("startup")
async def _startup() -> None:
    # Ensure the ticket DB exists and has a couple of demo tickets to look up.
    db.seed_if_empty()
    log.info("Ticket DB ready: %s", db.stats())
    # Preload STT/VAD/turn models in the background so the *first* voice call
    # connects instantly. Lazy-loading them on the first WebSocket cost ~15-20s
    # and used to drop the socket; warming here (off the request path) avoids
    # both the wait and the drop. HTTP is already serveable while this runs.
    asyncio.create_task(_warmup_speech())


async def _warmup_speech() -> None:
    try:
        from server.config import settings
        from server.speech import build_speech_input

        sess = build_speech_input(settings)
        await sess.connect()
        await sess.disconnect()
        log.info("Speech models warmed — first voice call will be instant")
    except Exception:
        log.exception("Speech warmup failed — models will lazy-load on first call")


@app.get("/")
async def index() -> FileResponse:
    # Which client to serve is config-driven so a second process (e.g. the 8761
    # TEN-VAD + Smart-Turn demo) can ship its own UI without touching the 8600 one.
    name = getattr(settings, "voice_client_html", "index.html") or "index.html"
    target = (CLIENT_DIR / name).resolve()
    # Guard against path escapes; fall back to the default client if missing.
    if CLIENT_DIR.resolve() not in target.parents or not target.is_file():
        target = CLIENT_DIR / "index.html"
    return FileResponse(target)


@app.get("/api/agents")
async def api_agents() -> JSONResponse:
    return JSONResponse({"agents": agents_public()})


# The 8 Orpheus voices with a short tone description, shown in the voice picker.
VOICE_CATALOG = [
    {"id": "tara", "name": "Tara", "gender": "female", "desc": "Warm & clear"},
    {"id": "leah", "name": "Leah", "gender": "female", "desc": "Professional & precise"},
    {"id": "jess", "name": "Jess", "gender": "female", "desc": "Bright & energetic"},
    {"id": "zoe",  "name": "Zoe",  "gender": "female", "desc": "Calm & steady"},
    {"id": "mia",  "name": "Mia",  "gender": "female", "desc": "Smooth & engaging"},
    {"id": "leo",  "name": "Leo",  "gender": "male",   "desc": "Deep & authoritative"},
    {"id": "dan",  "name": "Dan",  "gender": "male",   "desc": "Rugged & direct"},
    {"id": "zac",  "name": "Zac",  "gender": "male",   "desc": "Strong & commanding"},
]

_sample_engine = None
_sample_lock = asyncio.Lock()


@app.get("/api/voices")
async def api_voices() -> JSONResponse:
    return JSONResponse({"voices": VOICE_CATALOG})


@app.get("/api/voice_sample")
async def api_voice_sample(voice: str = "tara") -> Response:
    """Synthesize a short sample in the given voice and return it as a WAV clip."""
    voice = "".join(c for c in voice if c.isalnum() or c == "_")[:20] or "tara"
    name = next((v["name"] for v in VOICE_CATALOG if v["id"] == voice), voice.capitalize())
    global _sample_engine
    async with _sample_lock:  # serialize: one shared engine, avoid voice races
        from server.tts_engine import TTSEngine
        if _sample_engine is None:
            _sample_engine = TTSEngine()
            await _sample_engine.start()
        _sample_engine.voice = voice
        pcm = bytearray()
        async for chunk in _sample_engine.synthesize_stream(
            f"Hi, I'm {name}. This is how I sound as your assistant."
        ):
            pcm += chunk
    return Response(content=_pcm_to_wav(bytes(pcm)), media_type="audio/wav")


def _pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM in a minimal WAV header."""
    n = len(pcm)
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16
    ) + b"data" + struct.pack("<I", n)
    return header + pcm


@app.get("/api/health")
async def api_health() -> JSONResponse:
    # LLM reachability
    llm_ok = False
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get(
                f"{settings.llm_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            llm_ok = r.status_code == 200
    except Exception:
        llm_ok = False
    return JSONResponse({
        "llm": {"reachable": llm_ok, "model": settings.llm_model},
        "db": db.stats(),
    })


@app.get("/api/params")
async def api_params() -> JSONResponse:
    """Return tunable-param metadata + current values for the browser Tune panel.

    ``params`` describes each slider (group, range, step, scope); ``values`` is
    the current value of each one. LLM/TTS params (scope "live") apply on the
    next turn; STT/VAD params (scope "session") apply after a hard refresh.
    """
    return JSONResponse({"params": param_metadata(), "values": get_param_values()})


@app.post("/api/params")
async def api_params_update(request: Request) -> JSONResponse:
    """Apply one or more param changes sent as a flat ``{name: value}`` object.

    Unknown names or incompatible values are ignored. Values are clamped to each
    param's range (or snapped to its allowed choices). Returns the full set of
    current values so the client can re-sync.
    """
    body = await request.json()
    updates = body or {}
    updated: dict = {}
    for name, value in updates.items():
        try:
            updated[name] = set_param(name, value)
        except (KeyError, ValueError, TypeError):
            log.warning("Ignoring invalid param update: %r=%r", name, value)
    if updated:
        log.info("Params updated via UI: %s", updated)
    return JSONResponse({"ok": True, "updated": updated, "values": get_param_values()})


@app.post("/api/chat")
async def api_chat(request: Request) -> StreamingResponse:
    body = await request.json()
    agent_id = body.get("agent") or DEFAULT_AGENT
    history = body.get("messages") or []
    agent = get_agent(agent_id)

    async def event_stream():
        # Send a small header event so the client knows which agent answered.
        yield json.dumps({"type": "start", "agent": agent.id, "agent_name": agent.name}) + "\n"
        async for event in run_agent(agent.id, history):
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Voice: end-to-end speech agent (STT → LLM → TTS) ─────────────────────
# Reuses the proven realtime pipeline from server/. The browser streams mic
# audio in and receives synthesized speech back over the same WebSocket, with
# JSON events (stt_partial / assistant_text_* / interrupt) driving the UI orbs.

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    from server.pipeline import VoicePipeline  # imported lazily (pulls STT/TTS deps)

    await ws.accept()
    log.info("Voice client connected")
    pipeline = VoicePipeline()
    try:
        await pipeline.start()
    except Exception:
        log.exception("Voice pipeline failed to start")
        await ws.close(code=1011, reason="pipeline init failed")
        return

    # Default persona + live tool-calling; client switches with {"type":"set_agent"}.
    pipeline._outreach_system_prompt = voice_prompt(DEFAULT_AGENT)
    pipeline._response_provider = make_voice_provider(DEFAULT_AGENT, pipeline)
    if settings.speculative_turn:
        pipeline._speculative_provider = make_speculative_prewarm(DEFAULT_AGENT)
    pipeline.tts.voice = get_agent(DEFAULT_AGENT).voice
    await ws.send_json({"type": "ready"})

    lock = asyncio.Lock()

    async def send_audio() -> None:
        try:
            async for chunk in pipeline.audio_output_stream():
                async with lock:
                    await ws.send_bytes(struct.pack("<I", len(chunk)) + chunk)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            log.exception("voice audio send error")

    async def send_text() -> None:
        try:
            async for msg in pipeline.text_output_stream():
                async with lock:
                    await ws.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            log.exception("voice text send error")

    async def recv() -> None:
        try:
            while True:
                data = await ws.receive()
                if data.get("bytes"):
                    await pipeline.feed_audio(data["bytes"])
                elif data.get("text"):
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        continue
                    t = msg.get("type", "")
                    if t == "playback_started":
                        pipeline.echo.set_client_playing(True)
                    elif t == "playback_ended":
                        pipeline.echo.set_client_playing(False)
                    elif t == "text_input":
                        text = (msg.get("text") or "").strip()
                        if text:
                            await pipeline.inject_text(text)
                    elif t == "set_agent":
                        aid = msg.get("agent") or "crm"
                        pipeline._outreach_system_prompt = voice_prompt(aid)
                        pipeline._response_provider = make_voice_provider(aid, pipeline)
                        if settings.speculative_turn:
                            pipeline._speculative_provider = make_speculative_prewarm(aid)
                        # Apply this agent's default voice (client may override via set_voice).
                        pipeline.tts.voice = get_agent(aid).voice
                        async with lock:
                            await ws.send_json({"type": "agent_ack", "agent": aid,
                                                "voice": get_agent(aid).voice})
                    elif t == "set_voice":
                        voice = (msg.get("voice") or "tara").strip()
                        if voice.replace("_", "").isalnum() and 1 <= len(voice) <= 20:
                            pipeline.tts.voice = voice
        except (WebSocketDisconnect, RuntimeError):
            log.info("Voice client disconnected")
        except Exception:
            log.exception("voice recv error")

    tasks = [
        asyncio.create_task(pipeline.stt_receive_loop()),
        asyncio.create_task(pipeline.llm_tts_loop()),
        asyncio.create_task(send_audio()),
        asyncio.create_task(send_text()),
        asyncio.create_task(recv()),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for tk in tasks:
            tk.cancel()
        for tk in tasks:
            try:
                await tk
            except asyncio.CancelledError:
                pass
        await pipeline.stop()
    log.info("Voice session ended")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("CHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("CHAT_PORT", "8600"))

    # Use TLS if certs exist — browsers require a secure context (HTTPS) for
    # microphone access from any host other than localhost. Falls back to HTTP.
    ssl_kwargs = {}
    cert, key = Path(settings.ssl_certfile), Path(settings.ssl_keyfile)
    if cert.exists() and key.exists():
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"
    else:
        scheme = "http"
    log.info("Starting chat + voice demo on %s://%s:%s", scheme, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info", **ssl_kwargs)
