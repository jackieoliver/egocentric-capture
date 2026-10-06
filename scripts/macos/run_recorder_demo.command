#!/bin/zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export HAPTICA_HOME="${HAPTICA_HOME:-$HOME/Library/Application Support/HapticaRecorder}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing 'uv'. Install from https://docs.astral.sh/uv/ and re-run."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Missing 'npm' (Node.js). Install Node.js and re-run."
  exit 1
fi

echo "Syncing Python environment..."
uv sync

if [ ! -f "apps/recorder-ui/dist/index.html" ]; then
  echo "Building Recorder UI (Vite)..."
  (cd "apps/recorder-ui" && npm ci && npm run build)
fi

echo "Starting Recorder UI demo (built UI, mock mode)..."
echo "For UI development, use docs/runbook.md instead of this script."
exec uv run transcriptions ui --device dummy --mock
