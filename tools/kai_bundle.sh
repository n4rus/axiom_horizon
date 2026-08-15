#!/usr/bin/env bash
# kai_bundle.sh — Layer 0 portable: single portable snapshot/restore of Kai's
# entire live state (attractor, memory shards, darwin, chat, corpus, ingest).
#
#   kai_bundle save [out]             -> .kai_backups/kai-state-<ts>.tar[.zst|.gz]
#   kai_bundle list                   -> list saved bundles
#   kai_bundle restore <file>         -> refuse if live .kai_*.json exist
#   kai_bundle restore <file> --force -> overwrite existing state
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BK="$REPO/.kai_backups"
KAI_HOME="$HOME/.kai"
ZST=$(command -v zstd >/dev/null && echo 1 || echo 0)
EXT=$([ "$ZST" = 1 ] && echo zst || echo gz)

# L0 lived state — explicit, survival-focused. Excludes the multi-GB corpus
# shards (.kai_wiki_memory.* = ~1.2GB, .kai_corpus_attractor.json = 273MB)
# which are recoverable by re-ingest; includes the gapless phen trace +
# darwin archive + bridge/chat memory + bridge source + /tmp tau state.
find_state() {
  {
    # top-level lived JSON (chat memory, bench ledger, etc.)
    find "$REPO" -maxdepth 1 -name '.kai_*.json' \
        ! -name '.kai_wiki_memory.*' ! -name '.kai_corpus*' -type f 2>/dev/null
    # phen trace + darwin archive (deepest lived state)
    [ -f "$REPO/.axiom_state/phen_trace.jsonl" ] && echo "$REPO/.axiom_state/phen_trace.jsonl"
    [ -f "$REPO/.axiom_state/darwin_archive.json" ] && echo "$REPO/.axiom_state/darwin_archive.json"
    [ -f "$REPO/.axiom_state/collision_state.json" ] && echo "$REPO/.axiom_state/collision_state.json"
    [ -f "$REPO/.axiom_state/domain_priors.json" ] && echo "$REPO/.axiom_state/domain_priors.json"
    # bridge + physics source (the only /tmp-independent state)
    [ -f "$REPO/kai_bridge.py" ] && echo "$REPO/kai_bridge.py"
    [ -f "$REPO/phen_continuity.py" ] && echo "$REPO/phen_continuity.py"
  } | sort -u
}

cmd_save() {
  local out="${1:-}"
  [ -z "$out" ] && out="$BK/kai-state-$(date +%Y%m%d-%H%M%S).tar.$EXT"
  mkdir -p "$BK"
  local files n
  mapfile -t files < <(find_state)
  n=${#files[@]}
  echo "[bundle] saving $n state files -> $out  (ollama: $([ -x "$KAI_HOME/ollama/ollama" ] && echo present || echo MISSING))"
  local rel=()
  for x in "${files[@]}"; do rel+=("${x#"$REPO"/}"); done
  if [ "$ZST" = 1 ]; then
    tar --use-compress-program=zstd -cf "$out" -C "$REPO" "${rel[@]}"
  else
    tar -czf "$out" -C "$REPO" "${rel[@]}"
  fi
  # sidecar manifest (sha256 + bytes + counts)
  local mf="${out%.$EXT}.manifest"
  { echo "# kai-state $(date -Is) host=$(basename "$HOSTNAME")"
    echo "# files=$n pack=$(du -h "$out" | awk '{print $1}') ollama=$([ -x "$HOME/.kai/ollama/ollama" ] && echo present || echo MISSING)"
    for x in "${files[@]}"; do printf '%s  %s  %s\n' "$(sha256sum "$x" | cut -d' ' -f1)" "$(stat -c%s "$x")" "${x#$REPO/}"; done
  } > "$mf"
  echo "[bundle] ok  size=$(du -h "$out" | awk '{print $1}')  manifest=$mf"
}

cmd_list() {
  ls -la "$BK"/kai-state-* 2>/dev/null || echo "no bundles in $BK"
}

cmd_restore() {
  local src="$1" force="${2:-}"
  [ -f "$src" ] || { echo "backup not found: $src" >&2; exit 1; }
  local n; n=$(find_state | wc -l)
  if [ "$n" -gt 0 ] && [ "$force" != "--force" ]; then
    echo "REFUSE: $n live .kai_*.json present. Use --force to overwrite." >&2; exit 1
  fi
  echo "[bundle] restoring from $src ..."
  local tmp; tmp=$(mktemp -d "/tmp/kai_restore.XXXXXX")
  if [[ "$src" == *.zst ]]; then
    tar --use-compress-program=zstd -xf "$src" -C "$tmp"
  else
    tar -xzf "$src" -C "$tmp"
  fi
  if [ "$n" -gt 0 ]; then rm -f .kai_*.json && echo "[bundle] removed $n stale live files"; fi
  find "$tmp" -name '.kai_*.json' -type f -exec mv '{}' "$REPO"/ \;
  echo "[bundle] restored: $(find_state | wc -l) .kai_*.json back in $REPO"
  rm -rf "$tmp"
}

case "${1:-}" in
  save)    cmd_save "${2:-}";;
  list)    cmd_list;;
  restore) cmd_restore "$2" "$3";;
  *) echo "usage: kai_bundle {save [out] | list | restore <tar> [--force]}"; exit 1;;
esac