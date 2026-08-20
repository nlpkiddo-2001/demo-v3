#!/usr/bin/env bash
# Start (or restart) the Zoho knowledge-base voice demo.
#
#   ./run_kb.sh                    # full: VAD + STT + turn fusion + RAG + TTS
#   ./run_kb.sh --no-filler        # stay silent while retrieval runs
#   ./run_kb.sh --no-correct       # skip the Zoho-product-name repair
#   ./run_kb.sh --turn off         # plain silence timeout, for A/B
#   ./run_kb.sh stop
#
# Runs on its own port so it can sit alongside the other two demos:
#   8443 capture (the pipeline view)   8444 agent (CRM / support / travel)
#   8445 this one
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
PY="/workspace/speech_demo/demo-v2/.venv/bin/python"
PORT="${KB_PORT:-8445}"
LOG="kb.log"; PIDFILE=".kb.pid"

# Same GPU mapping rationale as run_agent.sh: the process sees only what is
# listed, renumbered from 0, so cuda:0 is the speech GPU and cuda:1 is the one
# hosting Orpheus, which is where SNAC has to decode.
#
# Defaults differ from run_agent.sh on purpose. That demo takes GPU 2, and at the
# time of writing GPU 2 had 8.3 GB free while GPU 1 had 33 GB — so running both
# at once needs this one somewhere else. Override with SPEECH_GPU= if the layout
# changes.
#
# The two MUST be different devices. Setting both to the same GPU makes
# CUDA_VISIBLE_DEVICES="1,1", which collapses to one visible device and then
# SNAC's cuda:1 is an invalid ordinal — the exact failure run_agent.sh warns
# about, and it fails at startup with a torch stack trace rather than anything
# that mentions GPUs.
SPEECH_GPU="${SPEECH_GPU:-1}"
TTS_GPU="${TTS_GPU:-3}"
if [ "$SPEECH_GPU" = "$TTS_GPU" ]; then
  echo "  ! SPEECH_GPU and TTS_GPU must differ (both are ${SPEECH_GPU}) — SNAC needs cuda:1"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${SPEECH_GPU},${TTS_GPU}"
export SNAC_DEVICE="${SNAC_DEVICE:-cuda:1}"

PAT="^[^ ]*python[0-9.]* -m server\.kb_app --port ${PORT}"

stop_server() {
  if [ ! -s "$PIDFILE" ] || ! kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    pgrep -f "$PAT" | head -1 > "$PIDFILE" 2>/dev/null || true
  fi
  if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    PID="$(cat "$PIDFILE")"
    echo "  stopping $PID"
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 25); do kill -0 "$PID" 2>/dev/null || break; sleep 0.4; done
    kill -0 "$PID" 2>/dev/null && { echo "  force"; kill -9 "$PID" 2>/dev/null || true; }
  fi
  rm -f "$PIDFILE"
}

if [ "${1:-}" = "stop" ]; then stop_server; echo "stopped"; exit 0; fi

stop_server

if ! curl -s -o /dev/null --max-time 4 http://localhost:5100/v1/models; then
  echo "  ! TTS not running on :5100 — answers will be text only."
  echo "    vllm serve canopylabs/orpheus-tts-0.1-finetune-prod --port 5100"
fi
if ! curl -s -o /dev/null --max-time 6 http://zcrmdit-a6k-1.csez.zohocorpin.com:7374/generate; then
  echo "  ! ISC token endpoint unreachable — RAG WILL FAIL."
fi

echo "=== Zoho KB demo on :${PORT} (speech GPU ${SPEECH_GPU}, tts GPU ${TTS_GPU}) ==="
nohup "$PY" -m server.kb_app --port "$PORT" --certs ../demo-v2/certs "$@" \
  >"$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Models take ~40s to load. Wait for the log to say so rather than guessing, and
# bail early on the two failures that are obvious in the log.
for _ in $(seq 1 90); do
  grep -q "knowledge-base demo is live" "$LOG" 2>/dev/null && break
  grep -q "Address already in use" "$LOG" 2>/dev/null && { echo " -> PORT BUSY"; exit 1; }
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo " -> DIED, tail of $LOG:"; tail -20 "$LOG"; exit 1; }
  sleep 1
done

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "  open:  https://${IP:-<server-ip>}:${PORT}"
echo "  log:   $LOG"
