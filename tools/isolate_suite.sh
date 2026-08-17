#!/usr/bin/env bash
# tools/isolate_suite.sh — run all three physics arms under --isolate judging
# for the honest ranking. Each arm: 14 queries. darwin arm uses the chat
# path (physics_answer), fuse uses --fuse, recall uses --recall.
set -u
LOG=/tmp/isolate_suite.log
cd /home/l/Desktop/AxiomTree/axiom_horizon || exit 1
echo "$(date -Is) isolate suite start" >> "$LOG"

echo "$(date -Is) arm 1/3: darwin-chat" >> "$LOG"
python3 tools/kai_physics_bench.py --queries 14 --isolate --label iso_darwin_chat >> "$LOG" 2>&1
echo "$(date -Is) arm 1 done" >> "$LOG"

echo "$(date -Is) arm 2/3: fuse 12b" >> "$LOG"
python3 tools/kai_physics_bench.py --fuse --queries 14 --isolate --label iso_fuse_12b >> "$LOG" 2>&1
echo "$(date -Is) arm 2 done" >> "$LOG"

echo "$(date -Is) arm 3/3: recall grounding" >> "$LOG"
python3 tools/kai_physics_bench.py --recall --queries 14 --isolate --label iso_recall_grounding >> "$LOG" 2>&1
echo "$(date -Is) arm 3 done, suite complete" >> "$LOG"