#!/usr/bin/env bash
# tools/bench_after_sessions.sh — wait for the session-history absorption to
# finish (embed bandwidth free), then run the L5 closed-loop recall bench
# (the falsifiable AGI primitive: does test-time self-recall beat raw?).
# Mirrors darwin_after_absorb.sh. Detached via setsid.
set -u
LOG=/tmp/kai_bench_after_sessions.log
cd /home/l/Desktop/AxiomTree/axiom_horizon || exit 1
echo "$(date -Is) watcher start: waiting for sessions absorption DONE" >> "$LOG"

for i in $(seq 1 360); do            # poll every 5 min, up to 30h
    if ! pgrep -f "kai_absorb_docs.py" > /dev/null; then
        echo "$(date -Is) absorption finished (pid gone), grace 120s" >> "$LOG"
        sleep 120
        break
    fi
    sleep 300
done

if pgrep -f "kai_absorb_docs.py" > /dev/null; then
    echo "$(date -Is) STILL absorbing after 30h — aborting bench" >> "$LOG"
    exit 2
fi

echo "$(date -Is) starting recall bench (L5 falsifiable primitive)" >> "$LOG"
python3 tools/kai_physics_bench.py --recall --queries 14 --label recall_after_sessions \
    >> "$LOG" 2>&1
echo "$(date -Is) bench complete, see .kai_physics_bench.jsonl" >> "$LOG"
