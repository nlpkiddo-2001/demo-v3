#!/usr/bin/env bash
# Start (or restart) the v3 capture server and follow its log.
#
#   ./run.sh                     # defaults: gtcrn denoise, nemotron @160ms, thr 0.5
#   ./run.sh --denoise none      # A/B the denoiser
#   ./run.sh --no-stt            # VAD only, no recogniser (starts instantly)
#   ./run.sh --require-word      # Fix 5 on (costs ~11% of words — measure before trusting)
#   ./run.sh --threshold 0.65 --mode max     # feel the old v2 behaviour
#   ./run.sh stop                # just stop it
#
# Any flag not handled here is passed straight through to capture_app.
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PY="/workspace/speech_demo/demo-v2/.venv/bin/python"
PORT="${CAPTURE_PORT:-8443}"
GPU="${CAPTURE_GPU:-2}"
LOG="capture.log"
PIDFILE=".capture.pid"

stop_server() {
  # Match on the pidfile rather than a process-name pattern: pgrep -f would also
  # match this very script (its own command line contains the pattern), and
  # killing yourself mid-restart is a confusing way to fail.
  # Belt and braces: the pidfile if it is valid, otherwise whatever is actually
  # running with our port on its command line.
  if [ ! -s "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    pgrep -f "server.capture_app --port $PORT" | head -1 > "$PIDFILE" 2>/dev/null || true
  fi
  if [ -s "$PIDFILE" ]; then
    p=$(cat "$PIDFILE")
    if kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null && echo "  stopped old server (PID $p)"
      for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.3; done
      kill -9 "$p" 2>/dev/null
    fi
    rm -f "$PIDFILE"
  fi
}

if [ "${1:-}" = "stop" ]; then stop_server; echo "  done"; exit 0; fi

stop_server

# Certificates: the browser will not hand over a microphone without HTTPS, and
# it fails silently if you try. Reuse demo-v2's self-signed pair.
CERTS="${CAPTURE_CERTS:-../demo-v2/certs}"
[ -f "$CERTS/cert.pem" ] || echo "  ! no cert at $CERTS — the mic button will NOT work"

echo "=== v3 capture on :${PORT} (GPU ${GPU}) ==="
: > "$LOG"
# setsid puts the server in its own session, so it survives this script being
# killed, the terminal closing, or a timeout wrapper — all of which have already
# taken it down once.
setsid nohup env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  "$PY" -m server.capture_app --port "$PORT" --certs "$CERTS" "$@" \
  >> "$LOG" 2>&1 < /dev/null &
# NOT $! — setsid forks, so $! is the wrapper, which exits immediately and
# leaves a pidfile pointing at nothing. Every later "stop" then silently does
# nothing and the next start fails on "address already in use". Resolve the real
# server process instead, once it exists.
sleep 2
pgrep -f "server.capture_app --port $PORT" | head -1 > "$PIDFILE"

# The recogniser takes ~15s to load. Wait for it rather than telling you it is
# ready when it is not — clicking "start mic" too early is the one way to get a
# session with no transcript and no explanation.
echo -n "  loading models "
for i in $(seq 1 90); do
  grep -q "capture page is live" "$LOG" 2>/dev/null && { echo " -> READY"; break; }
  grep -q "Traceback\|Address already in use" "$LOG" 2>/dev/null && { echo " -> FAILED"; tail -20 "$LOG"; exit 1; }
  echo -n "."; sleep 1
  [ "$i" = 90 ] && echo " (still warming — see $LOG)"
done

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "  open:  https://${IP:-<server-ip>}:${PORT}"
echo "         (self-signed cert — click through the browser warning)"
echo "  recordings -> ./recordings/    log -> ./$LOG"
echo "  Ctrl+C stops following the log; the server keeps running."
echo "  stop it with:  ./run.sh stop"
echo "======================================================================"
tail -f "$LOG"
