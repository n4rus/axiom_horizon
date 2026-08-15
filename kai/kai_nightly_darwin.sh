#!/usr/bin/env bash
# Nightly autonomous Darwin evolution — retargeted to the ACTIVE rust tree.
#
# The previous version evolved the stale kai/ tree (no gating, no episodes,
# old CLI flags) and ran its binary from kai/target/release. All continuous
# work (fitness-gated promotion, safety gate, episodic replay, multi-prior
# VFE, 237 tests) lives in rust/kai-fusion. This script:
#   1. replays new chat-memory episodes into the archive (persistent memory)
#   2. runs the gated self-play source evolution (safety + fitness floor bind)
#   3. test-gates: any failure reverts, no commit
#   4. commits only the archive/summary state on green
# Memory-guarded: refuses to run without >5GiB headroom (breaks the OOM loop).
set -uo pipefail

PROJ=/home/l/Desktop/AxiomTree/axiom_horizon
KAI=$PROJ/rust/target/release/kai
LOG=/tmp/kai_darwin_nightly.log
LOCK=/tmp/kai_darwin_nightly.lock
MIN_AVAILABLE_MB=$((5 * 1024))   # require >5GiB free before touching cargo
EVOLVE_TIMEOUT_S=1800

# Guard 1: single instance
if [ -f "$LOCK" ]; then
  pid=$(cat "$LOCK" 2>/dev/null)
  if kill -0 "$pid" 2>/dev/null; then
    echo "$(date -Is) SKIP: already running (pid $pid)" >> "$LOG"; exit 0
  fi
  rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# Guard 2: memory headroom (break the OOM cycle)
avail_mb=$(free -m | awk '/^Mem:/{print $NF}')
if [ "${avail_mb:-0}" -lt "$MIN_AVAILABLE_MB" ]; then
  echo "$(date -Is) DELAY: only ${avail_mb}MB avail (need >${MIN_AVAILABLE_MB}MB)" >> "$LOG"
  exit 0
fi

cd "$PROJ" || { echo "$(date -Is) ERROR: cd $PROJ failed" >> "$LOG"; exit 1; }

echo "$(date -Is) evolve start (avail ${avail_mb}MB, rust tree)" >> "$LOG"

# Build first (catch compile errors before evolution)
if ! cargo build --release --manifest-path rust/Cargo.toml >> "$LOG" 2>&1; then
  echo "$(date -Is) ERROR: cargo build failed, aborting" >> "$LOG"
  exit 1
fi

# Step 1: episodic replay — fold the bridge's remembered conversations into
# the darwin archive so this night's mutation seeds are grounded in what
# actually worked. Glob matches .kai_chat_memory*.json shards.
"$KAI" darwin replay --path ".kai_chat_memory*.json" --max 64 >> "$LOG" 2>&1
echo "$(date -Is) replay done" >> "$LOG"

# Step 2: gated self-play evolution (source evolution in the ACTIVE tree).
# The fitness floor fix makes the gate bind (convergence gen 0 -> gen N);
# the safety gate refuses invariant-breaking candidates.
timeout "$EVOLVE_TIMEOUT_S" "$KAI" darwin self-play >> "$LOG" 2>&1
echo "$(date -Is) self-play done rc=$?" >> "$LOG"

# Step 3: test gate — the trainer evolves candidates in-memory and persists
# only the archive; it never writes source files, so a failed gate leaves
# nothing to revert (and we MUST NOT git-checkout the rust tree — that would
# wipe the day's working changes). Gate = the tree still compiles+passes.
if ! cargo test --release --manifest-path rust/Cargo.toml >> "$LOG" 2>&1; then
  echo "$(date -Is) tests FAILED - archive not advanced, no commit" >> "$LOG"
  exit 1
fi
echo "$(date -Is) tests PASS" >> "$LOG"

# Step 4: commit only the evolutionary state (darwin archive + episode
# buffer). Source files are intentionally NOT staged — self-play persists
# candidates/parameters/episodes in the archive JSON only.
if git status --porcelain .axiom_state/darwin_archive.json 2>/dev/null | grep -q .; then
  git add .axiom_state/darwin_archive.json
  git commit -q -m "kai darwin: nightly evolution pass $(date -Is)" \
    && echo "$(date -Is) committed archive" >> "$LOG"
fi
echo "$(date -Is) darwin nightly complete" >> "$LOG"