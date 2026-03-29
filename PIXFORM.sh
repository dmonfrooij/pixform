#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x "venv/bin/python" ]]; then
  echo "[ERROR] PIXFORM is not installed yet. Run install_mac.sh first."
  exit 1
fi

PROFILE="${1:-auto}" # auto | nvidia | mac | cpu
case "$PROFILE" in
  nvidia|cuda)
    export PIXFORM_DEVICE="cuda"
    ;;
  mac|mps)
    export PIXFORM_DEVICE="mps"
    ;;
  cpu)
    export PIXFORM_DEVICE="cpu"
    ;;
  auto|*)
    if [[ -f ".pixform_device" ]]; then
      export PIXFORM_DEVICE="$(cat .pixform_device)"
    else
      export PIXFORM_DEVICE="auto"
    fi
    ;;
esac

echo "Starting PIXFORM (PIXFORM_DEVICE=${PIXFORM_DEVICE})..."
"venv/bin/python" "backend/app.py" &
APP_PID=$!

sleep 2
if command -v open >/dev/null 2>&1; then
  open "http://localhost:8000"
fi

wait "$APP_PID"

