"""Capture server — get real audio, from a real room, with labels attached.

Every number we have so far came from audiobook recordings and noise I generated
on this machine. That was enough to build against; it is not enough to tune
against. This server exists to close that gap and nothing else: it is not the
agent, and it has no language model or speech output in it.

What it does
------------
Serves one page, accepts microphone audio over a WebSocket, runs it through the
real pipeline (denoise → VAD → gate → recogniser), streams the state back so you
can watch it decide, and writes every session to disk as audio plus a per-chunk
log.

Why the labels matter more than the transcript
----------------------------------------------
The page has buttons for cough, clap, whisper, typing and silence. Pressing one
writes a marker into the recording at that instant. That is what turns a
recording into an *experiment*: afterwards ``bench_vad.py`` can ask "what did the
VAD score while he was coughing" and answer it exactly, instead of me guessing
from a waveform. Without labels we get a recording; with them we get data.

Running it
----------
    python -m server.capture_app                 # https://<host>:8443

HTTPS is required — browsers only grant microphone access on a secure origin.
The self-signed certificate from demo-v2 is reused, so expect a browser warning
and click through it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .speech.denoise import make_denoiser
from .speech.fusion import FusionConfig, TurnFusion
from .speech.measure import SessionRecorder
from .speech.session import CaptureSession, SessionConfig
from .speech.speech_gate import GateConfig, SpeechGate
from .speech.vad_ten import TenVAD

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
)
log = logging.getLogger("capture")

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "client" / "capture.html"
RECORDINGS = ROOT / "recordings"

app = FastAPI(title="voice pipeline v3 — capture")

# Set from the command line before uvicorn starts.
OPTS: dict = {
    "stt": True,
    "denoise": "gtcrn",
    "att_context": (70, 1),
    "device": "cuda",
    # Bar to START speech, and the lower bar to KEEP it. See GateConfig for the
    # frame distributions these were measured from.
    "threshold": 0.60,
    "release_threshold": 0.35,
    "mode": "count",
    "min_frames": 2,
    "confirm_chunks": 2,
    "in_rate": 24000,
    "require_word": False,
    "turn": "both",          # off | s2 | s3 | both
    "turn_device": "cuda",
    "turn_quantize": None,   # None | "4bit"
}


@app.on_event("startup")
async def warm_models() -> None:
    """Load the recogniser before anyone connects.

    Everything else in the pipeline starts in milliseconds; only the recogniser
    is slow. Loading it here, in a worker thread, means the first person to click
    "start mic" gets a session immediately instead of a stalled page — which is
    exactly what happened the first time this server was tested.
    """
    if OPTS["stt"]:
        try:
            from .speech.stt_nemotron import preload

            log.info("preloading recogniser (~15s)...")
            await asyncio.to_thread(preload, device=OPTS["device"])
        except Exception:
            log.exception("preload failed — sessions will run VAD only")
    # The turn models are loaded here for the same reason as the recogniser:
    # S3 is 15 GB and takes ~12 s. Inside a WebSocket handler that stalls the
    # event loop and the browser gives up before it ever gets a reply.
    if OPTS["turn"] != "off":
        try:
            f = _build_fusion()
            if f is not None:
                log.info("preloading turn models (%s)...", OPTS["turn"])
                # Blocking work, awaited directly: during startup nothing else
                # needs the event loop, and uvicorn does not accept connections
                # until this returns — which is the point.
                await f.start()
                log.info("turn models ready")
        except Exception:
            log.exception("turn preload failed — sessions will fall back")

    # Announced on every path, including --no-stt. Startup scripts wait for this
    # line, and a mode that never printed it would look like a hang.
    log.info("capture page is live")


@app.get("/")
async def index():
    return FileResponse(CLIENT, media_type="text/html")


@app.get("/api/config")
async def config():
    """What the pipeline is actually running, so the page can display it."""
    return JSONResponse({k: list(v) if isinstance(v, tuple) else v for k, v in OPTS.items()})


def _build_fusion():
    """Assemble the turn layer named by --turn, or None for the plain baseline.

    Any part that fails to load is dropped rather than taken as fatal: the
    pipeline is designed to degrade — S3 missing means less clever turn-taking,
    both missing means the silence timeout, and none of that should stop a
    capture session from recording.
    """
    want = OPTS["turn"]
    if want == "off":
        return None
    acoustic = semantic = None
    if want in ("s2", "both"):
        try:
            from .speech.turn_smart import SmartTurnV2

            acoustic = SmartTurnV2(device=OPTS["turn_device"])
        except Exception:
            log.exception("S2 unavailable")
    if want in ("s3", "both"):
        try:
            from .speech.turn_ten import TenTurn

            semantic = TenTurn(device=OPTS["turn_device"], quantize=OPTS["turn_quantize"])
        except Exception:
            log.exception("S3 unavailable")
    if acoustic is None and semantic is None:
        return None
    return TurnFusion(acoustic, semantic, FusionConfig())


def _build_stt():
    """Load the recogniser, or return None if it is switched off / unavailable.

    A missing recogniser must not stop a capture session. The VAD half is the
    part that needs your room most, and it is worth recording even if the GPU is
    busy or the model will not load.
    """
    if not OPTS["stt"]:
        return None
    try:
        from .speech.stt_nemotron import NemotronStreamingSTT

        return NemotronStreamingSTT(
            att_context=tuple(OPTS["att_context"]), device=OPTS["device"]
        )
    except Exception:
        log.exception("recogniser unavailable — continuing with VAD only")
        return None


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """One capture session.

    Binary frames are microphone audio (int16 PCM at ``in_rate``). Text frames
    are control messages: ``{"type":"mark","label":"cough"}`` and
    ``{"type":"bye"}``.
    """
    await sock.accept()
    name = time.strftime("capture-%Y%m%d-%H%M%S")
    recorder = SessionRecorder(
        RECORDINGS, name,
        record_ref=False,          # no agent audio in this tool, so no ref track
        meta={"tool": "capture_app", **{k: list(v) if isinstance(v, tuple) else v
                                        for k, v in OPTS.items()}},
    )
    session = CaptureSession(
        vad=TenVAD(),
        gate=SpeechGate(GateConfig(
            threshold=OPTS["threshold"], mode=OPTS["mode"],
            release_threshold=OPTS["release_threshold"],
            min_frames=OPTS["min_frames"], confirm_chunks=OPTS["confirm_chunks"],
            require_word_for_silence_reset=OPTS["require_word"],
        )),
        stt=_build_stt(),
        recorder=recorder,
        denoiser=make_denoiser(OPTS["denoise"]),
        fusion=_build_fusion(),
        cfg=SessionConfig(in_rate=OPTS["in_rate"], require_word=OPTS["require_word"]),
    )

    pump: asyncio.Task | None = None
    try:
        await session.start()
        await sock.send_text(json.dumps({
            "type": "ready", "recording": name,
            "denoise": session.denoiser.name,
            "stt": session.stt.name if session.stt else "off",
            "turn": "off" if session.fusion is None else "+".join(
                x.name for x in (session.fusion.acoustic, session.fusion.semantic) if x),
        }))

        async def to_browser():
            async for ev in session.events():
                await sock.send_text(json.dumps(ev))

        pump = asyncio.create_task(to_browser())

        while True:
            msg = await sock.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await session.feed(data)
            elif (text := msg.get("text")) is not None:
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "mark":
                    label = str(cmd.get("label", "mark"))[:64]
                    session.mark(label, source="ui")
                    log.info("mark: %s", label)
                elif cmd.get("type") == "bye":
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("capture session failed")
    finally:
        if pump is not None:
            pump.cancel()
        await session.stop()
        summary = recorder.close()
        if summary["chunks"] == 0:
            # A connection that sent no audio is not a take. Leaving an empty wav
            # + jsonl behind would make the recordings directory lie about how
            # many sessions are actually worth analysing.
            for f in (summary["jsonl"], summary["mic_wav"], summary["ref_wav"]):
                if f:
                    Path(f).unlink(missing_ok=True)
            log.info("session %s discarded (no audio)", summary["name"])
        else:
            log.info(
                "session done: %s — %.1fs, %d chunks, %d marks",
                summary["name"], summary["duration_sec"], summary["chunks"], summary["marks"],
            )


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Capture real audio through the v3 pipeline.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CAPTURE_PORT", "8443")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--denoise", default="gtcrn", choices=["gtcrn", "none"])
    ap.add_argument("--no-stt", action="store_true", help="VAD only, no recogniser")
    ap.add_argument("--att", default="70,1", help="recogniser latency mode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="frame score needed to START speech")
    ap.add_argument("--release-threshold", type=float, default=0.35,
                    help="lower bar to KEEP speech running (hysteresis). "
                         "Pass -1 to disable and use one bar for both.")
    ap.add_argument("--mode", default="count", choices=["count", "run", "max"])
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--confirm-chunks", type=int, default=2)
    ap.add_argument("--turn", default="both", choices=["off", "s2", "s3", "both"],
                    help="turn layer: s2=tone, s3=meaning, both=fused, off=plain silence timeout")
    ap.add_argument("--turn-device", default="cuda")
    ap.add_argument("--turn-4bit", action="store_true",
                    help="load S3 in 4-bit (5.6GB instead of 15GB, ~44ms instead of 18ms)")
    ap.add_argument("--require-word", action="store_true",
                    help="Fix 5: only real words reset the silence clock. Off by default — "
                         "on one real take it cost 11%% of transcribed words.")
    ap.add_argument("--certs", default=str(ROOT.parent / "demo-v2" / "certs"))
    args = ap.parse_args()

    OPTS.update(
        stt=not args.no_stt,
        denoise=args.denoise,
        att_context=tuple(int(x) for x in args.att.split(",")),
        device=args.device,
        threshold=args.threshold,
        release_threshold=(None if args.release_threshold < 0
                           else args.release_threshold),
        mode=args.mode,
        min_frames=args.min_frames,
        confirm_chunks=args.confirm_chunks,
        require_word=args.require_word,
        turn=args.turn,
        turn_device=args.turn_device,
        turn_quantize="4bit" if args.turn_4bit else None,
    )
    RECORDINGS.mkdir(parents=True, exist_ok=True)

    cert, key = Path(args.certs) / "cert.pem", Path(args.certs) / "key.pem"
    ssl = {}
    if cert.is_file() and key.is_file():
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    else:
        # Worth saying plainly: without TLS the page loads but the microphone
        # button will not work, and the browser will not explain why.
        log.warning("no certs at %s — serving HTTP; the browser will BLOCK the mic", args.certs)

    log.info("capture on %s://%s:%d  (vad thr=%.2f/%s denoise=%s stt=%s turn=%s)",
             "https" if ssl else "http", args.host, args.port,
             args.threshold, args.mode, args.denoise,
             "off" if args.no_stt else args.att, args.turn)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
