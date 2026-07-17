#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

VENV="$DIR/.venv"
PYTHON="${VENV}/bin/python3"

# ensure venv
if [ ! -f "$PYTHON" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$VENV"
fi

# ensure deps
"$PYTHON" -m pip install -q ollama httpx 2>/dev/null || {
    echo "Installing dependencies..."
    "$PYTHON" -m pip install -r requirements.txt
}

# check ollama
if ! command -v ollama &>/dev/null; then
    echo "WARNING: ollama not found in PATH. Install it from https://ollama.ai"
fi

echo "Starting AxiomAlien REPL..."
exec "$PYTHON" "$DIR/axiom_alien.py"
