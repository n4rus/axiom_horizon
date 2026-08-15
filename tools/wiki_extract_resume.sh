#!/usr/bin/env bash
# Resume wiki extraction from where it left off (or start fresh).
# The extractor streams the full bz2 once, so there's no true resume — but
# we can skip processing the output we already have by splitting + absorbing
# the existing articles.txt even if extraction is still running (it appends).
#
# Run: bash tools/wiki_extract_resume.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="/home/l/Desktop/AxiomTree/axiom_horizon/enwiki-20260601-pages-articles-multistream.xml.bz2"
OUT="$ROOT/wiki_filtered/articles.txt"
LOG="/tmp/wiki_extract.log"

if pgrep -f wiki_extract.py > /dev/null; then
    echo "wiki_extract is already running (PID $(pgrep -f wiki_extract.py | head -1))"
    echo "Tail: tail -f $LOG"
    exit 0
fi

# If we have a partial articles.txt, we can run ingest on it now without
# waiting for full extraction to finish.
if [[ -f "$OUT" ]]; then
    size_mb=$(du -m "$OUT" | cut -f1)
    echo "Existing articles.txt: ${size_mb} MB"
    echo "You can run: bash tools/wiki_ingest.sh  (it filters + absorbs)"
    echo
fi

echo "Starting fresh extraction..."
cd "$ROOT"
mkdir -p "$ROOT/wiki_filtered"
setsid python3 -u tools/wiki_extract.py "$SRC" -o "$OUT" \
    > "$LOG" 2>&1 < /dev/null &
disown
sleep 2
echo "Launched (PID $(pgrep -f wiki_extract.py || echo unknown)). Tail: tail -f $LOG"