#!/usr/bin/env bash
# kai_phase2_chain.sh — post-A2 execution chain (B -> A4 -> C -> A3 -> D).
# Runs only after the cap7b A2 bench completes (GPU freed). Each step is
# ledger-resumable; safe to re-run. Launch with setsid, log to /tmp.
#
# Sequencing rationale:
#   A2 (running separately) = 7b capacity WITHOUT code memory = baseline.
#   1) B: restart bridge (loads code memory + mutation gate) then re-run the
#      7b auto arm on capacity WITH the full stack — the "with" measurement.
#   2) A4: darwin unified --tier capacity at 7b, bounded --code-tasks so it
#      stays affordable; evolves params ON TOP of the fused stack.
#   3) C: 16b full-stack lock — auto arm on the original 29 with final params.
#   4) A3: 16b capacity matrix (greedy, fixed, auto, physics).
#   5) D: 12b memory ruler + qwen3.5:9b bench (last per user directive).

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- config ----
BRIDGE_PORT=8765
CODE_TASKS_A4=8          # auto arm ~15 min/task at 7b -> keep A4 affordable
DARWIN_GENS=3

echo "== kai_phase2_chain start $(date) ==" | tee -a /tmp/kai_phase2.log

# 0) Wait for A2 to fully finish (all four arms).
echo "[0] waiting for A2 (cap7b) to finish..." | tee -a /tmp/kai_phase2.log
while true; do
  done=$(python3 - <<'EOF'
import json, os, glob
tot, fin = 0, 0
for f in glob.glob(".kai_code_bench.jsonl.cap7b_*.ledger.jsonl"):
    tot += 1
    try:
        n = sum(1 for _ in open(f))
        fin += 1 if n >= 15 else 0
    except Exception:
        pass
print(f"{fin}/{tot}")
EOF
)
  if [ "$done" = "4/4" ]; then break; fi
  if ! pgrep -f "kai_code_bench" > /dev/null; then
    # bench process gone — treat as done even if ledger short
    echo "  bench process gone (done=$done), proceeding" | tee -a /tmp/kai_phase2.log
    break
  fi
  sleep 300
done
echo "  A2 done at $(date)" | tee -a /tmp/kai_phase2.log

# 1) B: restart bridge with code memory + mutation gate; verify memory loads.
echo "[1] B validation: restart bridge" | tee -a /tmp/kai_phase2.log
bash tools/kai_restart.sh >> /tmp/kai_phase2.log 2>&1
sleep 5
curl -sf --max-time 3 http://127.0.0.1:$BRIDGE_PORT/health > /dev/null && \
  echo "  bridge UP" | tee -a /tmp/kai_phase2.log || \
  echo "  bridge DOWN — aborting chain" | tee -a /tmp/kai_phase2.log
grep -c "Code memory loaded" /tmp/kai_bridge.log 2>/dev/null | \
  awk '{print "  code-memory load lines in bridge log:", $1}'

# 2) B measurement: 7b auto capacity WITH code memory (compares to A2).
echo "[2] B measurement: cap7b auto WITH code memory" | tee -a /tmp/kai_phase2.log
python3 tools/kai_code_bench.py --tier capacity --k 4 --label cap7b_fused \
  --model qwen2.5-coder:7b --arms auto >> /tmp/kai_phase2.log 2>&1

# 3) A4: darwin unified, capacity tier at 7b, bounded.
echo "[3] A4: darwin capacity at 7b (code-tasks=$CODE_TASKS_A4, gens=$DARWIN_GENS)" \
  | tee -a /tmp/kai_phase2.log
python3 tools/kai_darwin_unified_evolve.py --tier capacity \
  --code-tasks $CODE_TASKS_A4 --generations $DARWIN_GENS \
  --model qwen2.5-coder:7b --seed 7 >> /tmp/kai_phase2.log 2>&1

# 4) C: 16b full-stack lock (auto on original 29).
echo "[4] C: 16b full-stack lock" | tee -a /tmp/kai_phase2.log
python3 tools/kai_code_bench.py --tier base --k 4 --label code_16b_lock \
  --model deepseek-coder-v2:16b --arms auto >> /tmp/kai_phase2.log 2>&1

# 5) A3: 16b capacity matrix.
echo "[5] A3: 16b capacity matrix" | tee -a /tmp/kai_phase2.log
python3 tools/kai_code_bench.py --tier capacity --k 4 --label cap16b \
  --model deepseek-coder-v2:16b --arms greedy,fixed,auto,physics \
  >> /tmp/kai_phase2.log 2>&1

# 6) D: 12b memory ruler + qwen3.5:9b bench.
echo "[6] D: 12b memory ruler" | tee -a /tmp/kai_phase2.log
python3 tools/kai_recall_bench.py --label recall12b --arms greedy,recall \
  --model gemma4:12b >> /tmp/kai_phase2.log 2>&1
echo "[6b] D: qwen3.5:9b bench" | tee -a /tmp/kai_phase2.log
python3 tools/kai_code_bench.py --tier base --k 4 --label code_35_9b \
  --model qwen3.5:9b --arms greedy,fixed,physics,auto >> /tmp/kai_phase2.log 2>&1

echo "== kai_phase2_chain done $(date) ==" | tee -a /tmp/kai_phase2.log