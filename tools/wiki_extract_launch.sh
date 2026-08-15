#!/usr/bin/env bash
# Launch the resumable wiki extraction in a way that CANNOT be killed by
# session/tool timeouts. Uses setsid (new session) + full fd redirection.
# Idempotent: refuses to start a second copy while one is running.
#
# Usage: bash tools/wiki_extract_launch.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUMP="/home/l/Desktop/AxiomTree/axiom_horizon/enwiki-20260601-pages-articles-multistream.xml.bz2"
OUT="$ROOT/wiki_filtered/articles.txt"
STATE="$ROOT/wiki_filtered/.extract_state.json"
LOG="/tmp/wiki_extract_resumable.log"

mkdir -p "$ROOT/wiki_filtered"

# Idempotency guard
if pgrep -f wiki_extract_resumable.py > /dev/null 2>&1; then
    echo "extraction already running: $(pgrep -f wiki_extract_resumable.py | head -1)"
    exit 0
fi

# If a state checkpoint exists, say so (resume will skip those pages)
if [[ -f "$STATE" ]]; then
    echo "resume state present: $(cat "$STATE")"
fi

# Launch fully detached: new session, all fds -> files, parent exits.
setsid bash -c "
    exec python3 -u '$ROOT/tools/wiki_extract_resumable.py' '$DUMP' -o '$OUT' --state '$STATE' > '$LOG' 2>&1
" < /dev/null > /dev/null 2>&1 &

sleep 2
if pgrep -f wiki_extract_resumable.py > /dev/null 2>&1; then
    echo "launched: PID $(pgrep -f wiki_extract_resumable.py | head -1)"
    echo "log: $LOG"
else
    echo "FAILED to start. Log:"
    tail -20 "$LOG" 2>/dev/null || true
    exit 1
fi