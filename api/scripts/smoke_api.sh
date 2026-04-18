#!/usr/bin/env bash
set -euo pipefail

# One-command smoke test for local FastAPI endpoints.
# Usage:
#   ./scripts/smoke_api.sh
#   ./scripts/smoke_api.sh 2026-04-16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$API_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DATE_ARG="${1:-2026-04-16}"

RUN_DIR="${RUN_DIR:-/tmp/coffee_api_smoke_$(date +%Y%m%d_%H%M%S)}"
SERVER_LOG="${SERVER_LOG:-/tmp/coffee_api.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-/tmp/coffee_api.pid}"
BASE_URL="http://$HOST:$PORT"

mkdir -p "$RUN_DIR"

started_by_script=0
server_pid=""

is_healthy() {
  curl -fsS "$BASE_URL/health" > /dev/null 2>&1
}

start_server() {
  echo "[smoke] Starting API server..."
  (
    cd "$API_DIR"
    nohup "$PYTHON_BIN" -m uvicorn backend.main:app --host "$HOST" --port "$PORT" > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID_FILE"
  )
  server_pid="$(cat "$SERVER_PID_FILE")"
  started_by_script=1

  for _ in $(seq 1 30); do
    if is_healthy; then
      echo "[smoke] Server is healthy at $BASE_URL"
      return 0
    fi
    sleep 1
  done

  echo "[smoke] ERROR: Server did not become healthy in time."
  echo "[smoke] Last server log lines:"
  tail -n 100 "$SERVER_LOG" || true
  exit 1
}

stop_server_if_started() {
  if [[ "$started_by_script" -eq 1 && -n "$server_pid" ]]; then
    echo "[smoke] Stopping server PID $server_pid"
    kill "$server_pid" >/dev/null 2>&1 || true
  fi
}

trap stop_server_if_started EXIT

if is_healthy; then
  echo "[smoke] Reusing existing running server at $BASE_URL"
else
  start_server
fi

fetch() {
  local name="$1"
  local url="$2"
  curl -sS -D "$RUN_DIR/${name}.headers.txt" "$url" > "$RUN_DIR/${name}.json"
}

echo "[smoke] Saving responses to $RUN_DIR"
fetch root "$BASE_URL/"
fetch health "$BASE_URL/health"
fetch predict_1 "$BASE_URL/predict?date=$DATE_ARG"
fetch predict_2 "$BASE_URL/predict?date=$DATE_ARG"

# Validate output shape/content with Python for portability.
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

root = json.loads((run_dir / "root.json").read_text())
health = json.loads((run_dir / "health.json").read_text())
p1 = json.loads((run_dir / "predict_1.json").read_text())
p2 = json.loads((run_dir / "predict_2.json").read_text())

checks = []

def check(label, ok, detail=""):
    checks.append((label, ok, detail))

check("root message", isinstance(root.get("message"), str) and len(root["message"]) > 0)
check("health status ok", health.get("status") == "ok", str(health.get("status")))
check("predict has projections", isinstance(p1.get("projections"), list) and len(p1["projections"]) > 0)
check("latest_price positive", isinstance(p1.get("latest_price"), (int, float)) and p1["latest_price"] > 0)

horizons = sorted([row.get("horizon_weeks") for row in p1.get("projections", []) if isinstance(row, dict)])
check("expected horizons", horizons == [4, 12, 26, 52], str(horizons))

projected_prices = [row.get("projected_price") for row in p1.get("projections", []) if isinstance(row, dict)]
check("projected prices positive", len(projected_prices) > 0 and all(isinstance(v, (int, float)) and v > 0 for v in projected_prices))

# Cache behavior note: Friday may force recompute by default. We still record values.
check("cache field present call 1", isinstance(p1.get("cache_hit"), bool), str(p1.get("cache_hit")))
check("cache field present call 2", isinstance(p2.get("cache_hit"), bool), str(p2.get("cache_hit")))

all_ok = True
for label, ok, detail in checks:
    state = "PASS" if ok else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"[{state}] {label}{suffix}")
    all_ok = all_ok and ok

print(f"[info] cache_hit #1: {p1.get('cache_hit')}")
print(f"[info] cache_hit #2: {p2.get('cache_hit')}")
print(f"[info] refresh_reason #1: {p1.get('refresh_reason')}")
print(f"[info] refresh_reason #2: {p2.get('refresh_reason')}")

if not all_ok:
    sys.exit(1)
PY

echo "[smoke] Completed successfully. Artifacts: $RUN_DIR"
