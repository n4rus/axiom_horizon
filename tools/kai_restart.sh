#!/usr/bin/env bash
# Restart Kai services after a reboot or fresh shell.
# Run: bash tools/kai_restart.sh
#
# Idempotent: kills any existing instances first, then relaunches.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/tmp"
mkdir -p "$LOG_DIR"

echo "== kai_restart: $(date) =="

# 1) Stock ollama (11434) — usually already running via systemd, only restart if not
if ! curl -sf --max-time 3 http://127.0.0.1:11434/api/version > /dev/null; then
    echo "[1/4] starting stock ollama (11434)..."
    pkill -f "/usr/local/bin/ollama serve" 2>/dev/null || true
    sleep 1
    setsid /usr/local/bin/ollama serve > "$LOG_DIR/ollama_stock.log" 2>&1 < /dev/null &
    disown
    sleep 3
else
    echo "[1/4] stock ollama already up (11434)"
fi

# 2) Native VFE ollama (11435)
NATIVE_OLLAMA="/tmp/kai_ollama/ollama"
if [[ ! -x "$NATIVE_OLLAMA" ]]; then
    echo "WARN: native ollama binary missing at $NATIVE_OLLAMA"
    echo "      (expected if /tmp was wiped by reboot) — rebuilding..."
    mkdir -p /tmp/kai_ollama/lib/ollama
    cd "$ROOT/sources/ollama" && go build -o /tmp/kai_ollama/ollama . || { echo "go build FAILED"; exit 1; }
    cp -a "$ROOT/sources/llama.cpp/build/bin/llama-server" /tmp/kai_ollama/lib/ollama/
    cp -a "$ROOT/sources/llama.cpp/build/bin/"*.so* /tmp/kai_ollama/lib/ollama/ 2>/dev/null
    echo "      rebuilt ok: $(ls /tmp/kai_ollama/ollama)"
fi
pkill -f "/tmp/kai_ollama/ollama serve" 2>/dev/null || true
sleep 1
echo "[2/4] starting native VFE ollama (11435)..."
setsid env OLLAMA_HOST=127.0.0.1:11435 \
         OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
         "$NATIVE_OLLAMA" serve \
         > "$LOG_DIR/kai_ollama.log" 2>&1 < /dev/null &
disown
sleep 4
if ! curl -sf --max-time 3 http://127.0.0.1:11435/api/tags > /dev/null; then
    echo "  WARN: native ollama not responding on 11435 yet (may need more time)"
fi

# 3) Kai bridge (8765)
pkill -f "kai_bridge.py" 2>/dev/null || true
sleep 1
echo "[3/4] starting kai bridge (8765)..."
cd "$ROOT"
setsid python3 kai_bridge.py --port 8765 \
    > "$LOG_DIR/kai_bridge.log" 2>&1 < /dev/null &
disown
sleep 4

# 4) Opencode REPL (the `ollama launch opencode` UI backend)
#    Only start if you actually want the REPL open now; otherwise skip.
if pgrep -f "ollama launch opencode" > /dev/null; then
    echo "[4/4] opencode already running"
else
    echo "[4/4] opencode not running. Launch with: ollama launch opencode"
fi

echo
echo "== status =="
curl -sf --max-time 3 http://127.0.0.1:11434/api/version > /dev/null && echo "  11434 (stock ollama): UP" || echo "  11434: DOWN"
curl -sf --max-time 3 http://127.0.0.1:11435/api/tags > /dev/null && echo "  11435 (VFE ollama):   UP" || echo "  11435: DOWN"
curl -sf --max-time 3 http://127.0.0.1:8765/health > /dev/null && echo "  8765  (kai bridge):   UP" || echo "  8765:  DOWN"
echo
echo "Done. wiki extract (PID $(pgrep -f wiki_extract.py || echo none)) can be restarted with:"
echo "  bash tools/wiki_extract_resume.sh"