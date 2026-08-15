#!/usr/bin/env bash
# Ingest the filtered Wikipedia dump into the Kai doc memory.
#
# Steps:
#   1. (skip — already done by wiki_extract.py) Produce wiki_filtered/articles.txt
#   2. Split into one .txt file per article under wiki_filtered/split/
#      (filter: keep articles with len >= 1500 chars, top N by length)
#   3. Run kai_absorb_docs.py --root wiki_filtered/split/ in background
#
# This script is idempotent: kai_absorb_docs.py --resume skips already-ingested.
#
# Run: bash tools/wiki_ingest.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/wiki_filtered/articles.txt"
SPLIT="$ROOT/wiki_filtered/split"
LOG="/tmp/wiki_ingest.log"
LOCK="$ROOT/.wiki_ingest.lock"

if [[ -f "$LOCK" ]]; then
    echo "wiki_ingest already running (lock present): $LOCK"
    exit 1
fi
touch "$LOCK"
trap "rm -f $LOCK" EXIT

# Defaults — can override via env
MIN_CHARS="${MIN_CHARS:-2000}"
TOP_N="${TOP_N:-50000}"     # cap at 50k substantive articles

echo "[wiki_ingest] $(date) starting"
echo "  source:    $SRC"
echo "  split dir: $SPLIT"
echo "  min_chars: $MIN_CHARS"
echo "  top_n:     $TOP_N"

# Step 2: split, with min-length + top-N filters
echo "[wiki_ingest] splitting + filtering articles..."
mkdir -p "$SPLIT"

# Read articles.txt, filter by length, sort by length desc, take top N, split
python3 - <<PYEOF
import os, re, sys, time

MIN_CHARS = int("${MIN_CHARS}")
TOP_N = int("${TOP_N}")
SRC = "${SRC}"
OUT = "${SPLIT}"

# First pass: collect (title, body) for articles passing min_chars
arts = []
with open(SRC, "r", encoding="utf-8") as f:
    title = None; buf = []
    for line in f:
        if line.strip() == "<<<END>>>":
            if title is not None:
                body = "\n".join(buf).strip()
                if len(body) >= MIN_CHARS:
                    arts.append((title, body))
            title = None; buf = []
            continue
        if line.strip() == "---" and title is None:
            continue
        if title is None:
            title = line.rstrip("\n"); buf = []
        else:
            buf.append(line.rstrip("\n"))

print(f"[wiki_split] {len(arts)} articles pass min_chars>={MIN_CHARS}", flush=True)
# Sort by length descending, take top N
arts.sort(key=lambda x: -len(x[1]))
if len(arts) > TOP_N:
    arts = arts[:TOP_N]
    print(f"[wiki_split] capped at top {TOP_N} by length", flush=True)

# Write files
def safe(t, i):
    s = re.sub(r"[^\\w\\-. ]+", "_", t).strip().replace(" ", "_")
    return (s[:120] or f"article_{i}")

t0 = time.time()
seen = set()
for i, (title, body) in enumerate(arts):
    ch = title[0].lower() if title and title[0].isalpha() else "0"
    sub = os.path.join(OUT, ch)
    os.makedirs(sub, exist_ok=True)
    base = safe(title, i)
    n = base; k = 2
    while n in seen or os.path.exists(os.path.join(sub, n + ".txt")):
        n = f"{base}_{k}"; k += 1
    seen.add(n)
    with open(os.path.join(sub, n + ".txt"), "w", encoding="utf-8") as f:
        f.write(title + "\n\n" + body)
    if (i+1) % 5000 == 0:
        print(f"[wiki_split] {i+1}/{len(arts)} files in {time.time()-t0:.0f}s", flush=True)

print(f"[wiki_split] DONE: {len(arts)} files in {OUT}", flush=True)
PYEOF

# Step 3: absorb
echo "[wiki_ingest] starting absorb (this will take hours)..."
cd "$ROOT"
python3 -u tools/kai_absorb_docs.py --root "$SPLIT" --resume 2>&1 | tee -a "$LOG"

echo "[wiki_ingest] $(date) done"
echo "  log: $LOG"