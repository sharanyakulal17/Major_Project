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
else
    echo "[!] Using system Python..."
fi

cd "$SCRIPT_DIR/backend"

PORT="${PORT:-5001}"
echo "[*] Launching Self-Healing Backend Server on http://localhost:${PORT} ..."
echo "[*] Press Ctrl+C to stop the server."

python app.py
