#!/usr/bin/env bash
# Start (or restart) the v3 voice agent — listen, think, speak.
#
#   ./run_agent.sh                 # full: VAD + STT + turn fusion + LLM + TTS
#   ./run_agent.sh --no-tts        # text replies only (no Orpheus needed)
#   ./run_agent.sh --turn off      # plain silence timeout, for A/B
#   ./run_agent.sh stop
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
PY="/workspace/speech_demo/demo-v2/.venv/bin/python"
PORT="${CHAT_PORT:-8444}"
LOG="agent.log"; PIDFILE=".agent.pid"

# GPU ordering matters. The process sees only what is listed here, renumbered
# from 0 — so this maps cuda:0 -> the speech GPU (recogniser, VAD, turn models)
# and cuda:1 -> the GPU hosting Orpheus, which is where SNAC should decode.
# Exposing a single GPU here is what made SNAC fail with "invalid device ordinal".
SPEECH_GPU="${SPEECH_GPU:-2}"
TTS_GPU="${TTS_GPU:-1}"
export CUDA_VISIBLE_DEVICES="${SPEECH_GPU},${TTS_GPU}"
export SNAC_DEVICE="${SNAC_DEVICE:-cuda:1}"

# Anchored to the interpreter path on purpose. A bare "server.agent_app" pattern
# also matches any shell whose command line merely mentions it — including the
# one running this script — and killing that mid-restart is a baffling failure.
PAT="^[^ ]*python[0-9.]* -m server\.agent_app --port ${PORT}"

stop_server() {
  if [ ! -s "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    pgrep -f "$PAT" | head -1 > "$PIDFILE" 2>/dev/null || true
  fi
  if [ -s "$PIDFILE" ]; then
    p=$(cat "$PIDFILE")
    if kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null && echo "  stopped agent (PID $p)"
      for _ in $(seq 1 20); do kill -0 "$p" 2>/dev/null || break; sleep 0.3; done
      kill -9 "$p" 2>/dev/null
    fi
    rm -f "$PIDFILE"
  fi
}

if [ "${1:-}" = "stop" ]; then stop_server; echo "  done"; exit 0; fi
stop_server

# Orpheus is a separate vLLM server, shared with demo-v2. Without it the agent
# still works — it just replies in text.
if ! curl -s -o /dev/null --max-time 4 http://localhost:5100/v1/models; then
  echo "  ! TTS not running on :5100 — replies will be text only."
  echo "    start it: CUDA_VISIBLE_DEVICES=${TTS_GPU} $PY -m vllm.entrypoints.openai.api_server \\"
  echo "        --model canopylabs/orpheus-tts-0.1-finetune-prod --port 5100 \\"
  echo "        --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.30 &"
fi

echo "=== v3 agent on :${PORT} (speech GPU ${SPEECH_GPU}, tts GPU ${TTS_GPU}) ==="
: > "$LOG"
setsid nohup env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" SNAC_DEVICE="$SNAC_DEVICE" \
  "$PY" -m server.agent_app --port "$PORT" --certs ../demo-v2/certs "$@" \
  >> "$LOG" 2>&1 < /dev/null &
sleep 2
pgrep -f "$PAT" | head -1 > "$PIDFILE"

echo -n "  loading "
for i in $(seq 1 90); do
  grep -q "agent is live" "$LOG" 2>/dev/null && { echo " -> READY"; break; }
  grep -q "Address already in use" "$LOG" 2>/dev/null && { echo " -> PORT BUSY"; exit 1; }
  echo -n "."; sleep 1
done
grep -E "LLM |TTS " "$LOG" | tail -2

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "  open:  https://${IP:-<server-ip>}:${PORT}"
echo "  stop:  ./run_agent.sh stop        log: ./$LOG"
echo "============================================================"
