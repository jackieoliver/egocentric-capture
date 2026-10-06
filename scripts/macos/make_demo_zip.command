#!/bin/zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing 'uv'. Install from https://docs.astral.sh/uv/ and re-run."
  exit 1
fi

DIST_DIR="$REPO_ROOT/dist"
mkdir -p "$DIST_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
STAGE="$DIST_DIR/HapticaRecorderDemo_$STAMP"
ZIP_PATH="$DIST_DIR/HapticaRecorderDemo_$STAMP.zip"

if [ ! -f "apps/recorder-ui/dist/index.html" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "Missing 'npm' (Node.js) to build the UI. Install Node.js or build dist/ before running this."
    exit 1
  fi
  echo "Building Recorder UI (Vite)..."
  (cd "apps/recorder-ui" && npm ci && npm run build)
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "Staging demo folder..."
rsync -a \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "node_modules/" \
  --exclude "state/" \
  --exclude "archive/" \
  --exclude "transcriptions/" \
  --exclude "apps/recorder-ui/node_modules/" \
  --exclude "apps/recorder-ui/test-results/" \
  --exclude "apps/recorder-ui/playwright-report/" \
  --exclude "gopro/tmp/" \
  --exclude "*.DS_Store" \
  "./" "$STAGE/"

echo "Creating zip: $ZIP_PATH"
rm -f "$ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$ZIP_PATH"

echo "Done."
open -R "$ZIP_PATH" || true

