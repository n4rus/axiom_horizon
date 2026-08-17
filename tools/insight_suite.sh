#!/usr/bin/env bash
# tools/insight_suite.sh — re-run all three physics arms under the 3-dim
# isolated ruler (complete + ground + insight). The insight dimension
# rewards correct, non-obvious information beyond boilerplate — the physics
# stack's actual claim. 2-dim ruler verdict was: all arms negative.
set -u
LOG=/tmp/insight_suite.log
cd /home/l/Desktop/AxiomTree/axiom_horizon || exit 1
echo "$(date -Is) insight suite start" >> "$LOG"

echo "$(date -Is) arm 1/3: darwin-chat" >> "$LOG"
python3 tools/kai_physics_bench.py --isolate --insight --queries 14 --label ins_darwin_chat >> "$LOG" 2>&1
echo "$(date -Is) arm 1 done" >> "$LOG"

echo "$(date -Is) arm 2/3: fuse 12b" >> "$LOG"
python3 tools/kai_physics_bench.py --fuse --isolate --insight --queries 14 --label ins_fuse_12b >> "$LOG" 2>&1
echo "$(date -Is) arm 2 done" >> "$LOG"

echo "$(date -Is) arm 3/3: recall grounding" >> "$LOG"
python3 tools/kai_physics_bench.py --recall --isolate --insight --queries 14 --label ins_recall_grounding >> "$LOG" 2>&1
echo "$(date -Is) arm 3 done, insight suite complete" >> "$LOG"