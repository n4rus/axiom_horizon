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

# helper: assert a ledger reached N lines before continuing
expect_ledger() {
  local glob="$1" n="$2"
  for i in $(seq 1 120); do
    local total=0 lines=0
    for f in $glob; do
      [ -f "$f" ] && lines=$((lines + $(wc -l < "$f")))
    done
    if [ "$lines" -ge "$n" ]; then return 0; fi
    # bail if the producing process is gone AND ledger is short
    if ! pgrep -f "${3:-kai_code_bench}" > /dev/null && [ "$lines" -lt "$n" ]; then
      echo "  EXPECT_FAIL: $glob has $lines/$n lines and process gone" | tee -a /tmp/kai_phase2.log
      return 1
    fi
    sleep 30
  done
  echo "  EXPECT_TIMEOUT: $glob never reached $n lines" | tee -a /tmp/kai_phase2.log
  return 1
}

# 0) Wait for A2 to fully finish (all four arms). Only the FOUR original
# cap7b ledgers count — cap7b_fused_* (B's run) and other labels must not
# inflate the count.
echo "[0] waiting for A2 (cap7b) to finish..." | tee -a /tmp/kai_phase2.log
while true; do
  done=$(python3 - <<'EOF'
import json, os, glob
arms = ["greedy", "fixed", "auto", "physics"]
fin, missing = 0, []
for a in arms:
    f = f".kai_code_bench.jsonl.cap7b_{a}.ledger.jsonl"
    if not os.path.exists(f):
        missing.append(a); continue
    n = sum(1 for _ in open(f))
    fin += 1 if n >= 15 else 0
print(f"{fin}/4")
EOF
)
  if [ "$done" = "4/4" ]; then break; fi
  if ! pgrep -f "kai_code_bench.py.*cap7b" > /dev/null; then
    # A2 bench process gone — treat as done even if ledger short
    echo "  A2 bench process gone (done=$done), proceeding" | tee -a /tmp/kai_phase2.log
    break
  fi
  sleep 300
done
echo "  A2 done at $(date)" | tee -a /tmp/kai_phase2.log

# 1) B: restart bridge with code memory + mutation gate; verify memory loads.
echo "[1] B validation: restart bridge" | tee -a /tmp/kai_phase2.log
# Safety: wait for any straggler bench (e.g. an orphaned C run) to finish so
# the bridge restart doesn't sever its in-flight requests.
echo "[1a] waiting for straggler bench processes..." | tee -a /tmp/kai_phase2.log
while pgrep -f "kai_code_bench.py" > /dev/null; do sleep 60; done
bash tools/kai_restart.sh >> /tmp/kai_phase2.log 2>&1
# The bridge takes a while to boot (loads 82k+ memory entries + embeddings).
# Wait up to 120s for health instead of racing it.
echo "[1b] waiting for bridge health..." | tee -a /tmp/kai_phase2.log
for i in $(seq 1 24); do
  if curl -sf --max-time 3 http://127.0.0.1:$BRIDGE_PORT/health > /dev/null 2>&1; then
    echo "  bridge UP after ${i}x5s" | tee -a /tmp/kai_phase2.log
    break
  fi
  sleep 5
done
if ! curl -sf --max-time 3 http://127.0.0.1:$BRIDGE_PORT/health > /dev/null 2>&1; then
  echo "  bridge DOWN after 120s — aborting chain" | tee -a /tmp/kai_phase2.log
  exit 1
fi
sleep 10  # let the bridge finish loading code memory + embeddings
grep -c "Code memory loaded" /tmp/kai_bridge.log 2>/dev/null | \
  awk '{print "  code-memory load lines in bridge log:", $1}'

# 2) B measurement: 7b auto capacity WITH code memory (compares to A2).
echo "[2] B measurement: cap7b auto WITH code memory" | tee -a /tmp/kai_phase2.log
python3 tools/kai_code_bench.py --tier capacity --k 4 --label cap7b_fused \
  --model qwen2.5-coder:7b --arms auto >> /tmp/kai_phase2.log 2>&1
expect_ledger ".kai_code_bench.jsonl.cap7b_fused_auto.ledger.jsonl" 15 kai_code_bench || exit 1

# 3) A4: darwin unified, capacity tier at 7b, bounded.
echo "[3] A4: darwin capacity at 7b (code-tasks=$CODE_TASKS_A4, gens=$DARWIN_GENS)" \
  | tee -a /tmp/kai_phase2.log
python3 tools/kai_darwin_unified_evolve.py --tier capacity \
  --code-tasks $CODE_TASKS_A4 --generations $DARWIN_GENS \
  --model qwen2.5-coder:7b --seed 7 >> /tmp/kai_phase2.log 2>&1
expect_ledger ".kai_code_bench.jsonl.darwin_capacity_g*_code.ledger.jsonl" 1 kai_darwin || true

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