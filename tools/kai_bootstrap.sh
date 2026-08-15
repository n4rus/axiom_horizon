#!/usr/bin/env bash
# kai_bootstrap.sh — LAYER 0 self-heal guardian.
#
# Ensures the whole Kai stack is alive and self-consistent. Call it at boot
# and periodically/cron. Non-destructive: never overwrites live .kai state
# unless a restore is explicitly asked for.
#
#   kai_bootstrap report
#   kai_bootstrap boot           (heal anything missing)
#   kai_bootstrap restore <tar>  (refuse if live .kai state exists)
set -euo pipefail

REPO="${HOME}/Desktop/AxiomTree/axiom_horizon"
KAI_HOME="$HOME/.kai"
OLBIN="$KAI_HOME/ollama/ollama"
OLLAMA_HOST=127.0.0.1:11435
BRIDGE_URL="http://127.0.0.1:8765/health"
ENSURE="$HOME/.config/opencode/kai_ollama_ensure.sh"
REQUIRED=(".kai_darwin_state.json" ".kai_chat_memory.json" ".kai_corpus_attractor.json")

log() { echo "[kai-boot $(date -Is)] $*"; }

check_ollama() { curl -s --max-time 8 "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; }
check_bridge() { curl -s --max-time 8 "$BRIDGE_URL" >/dev/null 2>&1; }

nstates() { find "$REPO" -maxdepth 1 -name '.kai_*.json' | wc -l; }

cmd_report() {
  local ob ox
  [ -x "$OLBIN" ] && ob=present || ob=MISSING
  check_ollama && ox=UP || ox=DOWN
  local bx; check_bridge && bx=UP || bx=DOWN
  printf '  ollama bin : %s  (%s)\n' "$ob" "$([ -x "$OLBIN" ] && du -h "$OLBIN" | awk '{print $1}')"
  printf '  ollama svc : %s  :%s\n' "$ox" "$OLLAMA_HOST"
  printf '  bridge     : %s :8765\n' "$bx"
  printf '  live state : %s files in %s\n' "$(nstates)" "$REPO"
}

cmd_boot() {
  log "layer-0 heal starting"
  if [ ! -x "$OLBIN" ]; then
    log "ollama binary MISSING -> rebuild"
    [ -x "$ENSURE" ] && "$ENSURE" || log "ensure script missing at $ENSURE"
  else
    log "ollama binary present"
  fi
  if ! check_ollama; then
    log "ollama not responding -> restart unit"
    systemctl --user restart kai-vfe-ollama.service >/dev/null 2>&1 || log "unit restart failed"
  else
    log "ollama up"
  fi
  if ! check_bridge; then
    log "bridge not responding -> restart unit"
    systemctl --user restart kai-bridge.service >/dev/null 2>&1 || log "bridge restart failed"
  else
    log "bridge up"
  fi
  for s in "${REQUIRED[@]}"; do
    [ -f "$REPO/$s" ] || log "WARN: $s missing — run: kai_bundle restore"
  done
  log "layer-1 heal done"
}

cmd_restore() {
  local src="$1"
  [ -f "$src" ] || { log "bundle not found: $src"; exit 1; }
  local n; n=$(nstates)
  if [ "$n" -gt 0 ]; then
    log "REFUSING restore: $n live state files present (non-destructive). Use kai_bundle --force."
    exit 1
  fi
  log "restoring full state from $src"
  "$REPO/tools/kai_bundle.sh" restore "$src"
}

case "${1:-}" in
  report)  cmd_report ;;
  boot|ensure) cmd_boot ;;
  restore) cmd_restore "${2:-}" ;;
  *) echo "usage: kai_bootstrap {report|boot|restore <tar>}"; exit 1 ;;
esac