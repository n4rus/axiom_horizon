#!/usr/bin/env bash
# Launch a persistent AutoAGI opencode session on a free-tier DeepSeek model.
# Usage: bash launch_autoagi.sh
#   — spawns: opencode --agent AutoAGI  (DeepSeek V4 Flash Free via OpenRouter)
#   — AutoAGI owns the verification/coding loop while the primary agent steers.
set -euo pipefail
cd "$(dirname "$0")"

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
echo "[launch_autoagi] working dir: $WORKDIR"

# Ensure bridge is up (AutoAGI expects :8765)
if ! curl -s -m 5 http://127.0.0.1:8765/health >/dev/null 2>&1; then
  echo "[launch_autoagi] bridge down — starting..."
  nohup python3 "$WORKDIR/kai_bridge.py" --port 8765 > "$WORKDIR/.kai_backups/bridge.log" 2>&1 &
  sleep 8
fi
curl -s -m 5 http://127.0.0.1:8765/health && echo " [bridge ok]" || { echo "[launch_autoagi] bridge FAILED"; exit 1; }

# Launch opencode with the AutoAGI agent directly.
# opencode reads .opencode/opencode.jsonc + opencode.json automatically;
# the "AutoAGI" agent (defined in opencode.json) is bound to the free-tier model.
exec opencode --agent AutoAGI
