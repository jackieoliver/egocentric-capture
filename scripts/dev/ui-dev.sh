#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing 'uv'. Install from https://docs.astral.sh/uv/ and re-run."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Missing 'npm' (Node.js). Install Node.js and re-run."
  exit 1
fi

export HAPTICA_DEVICE="${HAPTICA_DEVICE:-gopro}"
export HAPTICA_LIVEVIEW_SOURCE="${HAPTICA_LIVEVIEW_SOURCE:-gopro_udp}"

echo "Starting GoPro daemon..."
uv run gopro daemon start

echo "Starting UI backend (port 8001)..."
uv run uvicorn server.main:app --app-dir apps/recorder-ui --host 127.0.0.1 --port 8001 --reload &
BACKEND_PID=$!

cleanup() {
  if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID"
  fi
}
trap cleanup EXIT

echo "Waiting for backend..."
for _ in {1..40}; do
  if curl -sS http://127.0.0.1:8001/api/config >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

echo "Backend config snapshot:"
curl -sS http://127.0.0.1:8001/api/config | python -m json.tool || true

echo "Starting Vite dev server..."
cd apps/recorder-ui
if [ ! -d node_modules ]; then
  npm ci
fi
npm run dev
