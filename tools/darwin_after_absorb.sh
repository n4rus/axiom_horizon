#!/usr/bin/env bash
# darwin_after_absorb.sh — run the full darwin corpus loop once absorption
# finishes (embed bandwidth free). Watches /tmp/wiki_absorb.log for "DONE".
set -uo pipefail

PROJ=/home/l/Desktop/AxiomTree/axiom_horizon
LOG=/tmp/kai_darwin_after_absorb.log

echo "$(date -Is) watcher start: waiting for wiki absorption DONE" >> "$LOG"
# Poll every 5 min until the absorb log shows DONE (max 20h).
for i in $(seq 1 240); do
  if grep -q "DONE" /tmp/wiki_absorb.log 2>/dev/null; then
    break
  fi
  sleep 300
done

# Extra grace: let the bridge's embed queue drain.
sleep 600

echo "$(date -Is) absorption done, starting darwin corpus loop" >> "$LOG"
cd "$PROJ" || exit 1

# Full closed loop: evolve -> publish (.kai_physics_params.json, gated) ->
# bridge live-reload -> (next: bench).
if python3 tools/kai_darwin_corpus.py --generations 8 --population 8 --samples 6 >> "$LOG" 2>&1; then
  echo "$(date -Is) corpus evolve OK (published or gated-refused)" >> "$LOG"
else
  echo "$(date -Is) corpus evolve rc=$? " >> "$LOG"
fi

# Re-run the physics bench to falsify: physics vs raw with the published params.
if python3 tools/kai_physics_bench.py --queries 8 --label darwin_after_absorb >> "$LOG" 2>&1; then
  echo "$(date -Is) bench complete, see .kai_physics_bench.jsonl" >> "$LOG"
else
  echo "$(date -Is) bench rc=$? (judge model may be busy)" >> "$LOG"
fi

echo "$(date -Is) darwin after-absorb complete" >> "$LOG"
