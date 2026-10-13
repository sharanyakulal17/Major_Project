#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "[*] Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$SCRIPT_DIR/../venv/bin/activate" ]; then
    echo "[*] Activating virtual environment..."
    source "$SCRIPT_DIR/../venv/bin/activate"
fi

cd "$SCRIPT_DIR/new_backend/projectssssssss"

PORT="${PORT:-5001}"
echo "[*] Launching AURA HEAL (new_backend + new_frontend) on http://localhost:${PORT} ..."
echo "[*] Press Ctrl+C to stop the server."

python app.py
