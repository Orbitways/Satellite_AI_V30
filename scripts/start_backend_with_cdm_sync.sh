#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
: "${UVICORN_APP:=api.main:app}"

python -m cdm_sync_daemon &
SYNC_PID=$!

uvicorn "$UVICORN_APP" --host "$HOST" --port "$PORT" &
API_PID=$!

cleanup() {
  trap - INT TERM EXIT
  kill "$SYNC_PID" "$API_PID" 2>/dev/null || true
  wait "$SYNC_PID" "$API_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Stop both processes if either process exits unexpectedly.
set +e
wait -n "$SYNC_PID" "$API_PID"
STATUS=$?
set -e
cleanup
exit "$STATUS"
