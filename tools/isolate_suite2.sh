#!/usr/bin/env bash
# tools/isolate_suite2.sh — run fuse + recall arms under --isolate (the
# darwin-chat arm already completed: iso_darwin_chat delta -0.11).
set -u
LOG=/tmp/isolate_suite2.log
cd /home/l/Desktop/AxiomTree/axiom_horizon || exit 1
echo "$(date -Is) isolate suite2 start" >> "$LOG"

echo "$(date -Is) arm 1/2: fuse 12b" >> "$LOG"
python3 tools/kai_physics_bench.py --fuse --queries 14 --isolate --label iso_fuse_12b >> "$LOG" 2>&1
echo "$(date -Is) arm 1 done" >> "$LOG"

echo "$(date -Is) arm 2/2: recall grounding" >> "$LOG"
python3 tools/kai_physics_bench.py --recall --queries 14 --isolate --label iso_recall_grounding >> "$LOG" 2>&1
echo "$(date -Is) arm 2 done, suite complete" >> "$LOG"