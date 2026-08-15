#!/usr/bin/env bash
# Kai "persistent hands": daily self-audit loop.
# Reads the post-relaunch checklist + plan, reports open items, and only
# delegates heavy execution when there is safe memory headroom (avoids OOM).
set -uo pipefail

HZ=/home/l/Desktop/AxiomTree/axiom_horizon
LOG=/tmp/kai_self_audit.log
LOCK=/tmp/kai_self_audit.lock
MIN_AVAIL_MB=$((6*1024))
TODO=$HZ/TODO_POST_RELAUNCH.md

# single-instance guard
if [ -f "$LOCK" ]; then
  pid=$(cat "$LOCK" 2>/dev/null)
  if kill -0 "$pid" 2>/dev/null; then exit 0; fi
  rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

avail=$(free -m | awk '/^Mem:/{print $NF}')
echo "=== $(date -Is) self-audit (avail ${avail}MB) ===" >> "$LOG"

# 1) Reflect: which checklist items look open? Approximate from markdown.
if [ -f "$TODO" ]; then
  echo "-- open checklist items --" >> "$LOG"
  grep -nE '^[0-9]+\. ' "$TODO" >> "$LOG" 2>/dev/null
fi

# 2) Surface current memory + bridge state
echo "mem avail: ${avail}MB" >> "$LOG"
echo "bridge: $(systemctl --user is-active kai-bridge.service 2>/dev/null)" >> "$LOG"

# 3) Delegate heavy work ONLY if memory is safe (conservative hands-first).
if [ "$avail" -ge "$MIN_AVAIL_MB" ]; then
  echo "exec: memory safe (${avail}MB >= ${MIN_AVAIL_MB}MB) -> ready to delegate" >> "$LOG"
else
  echo "exec: DEFER (avail ${avail}MB < ${MIN_AVAIL_MB}MB)" >> "$LOG"
fi
echo "--- done ---" >> "$LOG"