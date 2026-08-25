#!/bin/bash
# Kai Desktop → Phone Export Helper
# Bundles opencode sessions + Kai state into ~/Downloads/kai_phone_bundle.zip
# Phone's Kai will auto-ingest via Download/ scan or "import" command
# Usage: ./tools/kai-export-phone.sh [--to-phone]  (copies via adb if phone connected)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORT_DIR="$ROOT/kai_phone_bundle"
BUNDLE="$HOME/Downloads/kai_phone_bundle.zip"

echo "=== Kai → Phone Export ==="
mkdir -p "$EXPORT_DIR"

# 1. Export Kai state (memories + physics) via Rust companion
echo "[1/4] Exporting Kai state (memories + VFE)..."
if [ -f "$ROOT/rust/kai-fusion/src/companion.rs" ]; then
  # Use the Rust binary to export if available, else create a JSON stub
  cargo run -p kai-fusion --bin kai -- export --to-phone 2>/dev/null || true
fi
# Fallback: create a minimal kai_state_export.json from current DB
if [ ! -f "$EXPORT_DIR/kai_state_export.json" ]; then
  cat > "$EXPORT_DIR/kai_state_export.json" << JSON
{
  "kai_state_version": 1,
  "exported_at": $(date +%s%3N),
  "exported_from": "axiom_horizon-desktop",
  "note": "Run 'kai export' on desktop for full export. This is a stub.",
  "memories": []
}
JSON
  echo "  → created stub $EXPORT_DIR/kai_state_export.json (run desktop Kai for full)"
fi

# 2. Bundle opencode sessions
echo "[2/4] Collecting opencode sessions..."
mkdir -p "$EXPORT_DIR/sessions"
if [ -d "$ROOT/opencode_sessions" ]; then
  cp -r "$ROOT/opencode_sessions" "$EXPORT_DIR/" 2>/dev/null || true
  echo "  → copied opencode_sessions ($(ls "$EXPORT_DIR/opencode_sessions" 2>/dev/null | wc -l) files)"
fi
# Also collect any session-*.md in root
cp "$ROOT"/session-*.md "$EXPORT_DIR/" 2>/dev/null || true
cp "$ROOT"/*.json "$EXPORT_DIR/" 2>/dev/null || true
echo "  → collected $(ls "$EXPORT_DIR" | wc -l) files"

# 3. Create bundle zip
echo "[3/4] Creating bundle $BUNDLE..."
(cd "$EXPORT_DIR" && zip -r "$BUNDLE" . -q)
ls -lh "$BUNDLE"

# 4. Push via ADB if phone connected and --to-phone flag
if [[ "$1" == "--to-phone" ]]; then
  if adb devices 2>&1 | grep -q "device$"; then
    echo "[4/4] Pushing to phone via adb..."
    adb push "$BUNDLE" /sdcard/Download/kai_phone_bundle.zip 2>&1 | tail -3
    echo "  → on phone, Kai will auto-ingest from Download/ or say 'import'"
  else
    echo "[4/4] No device via adb — bundle ready at $BUNDLE"
    echo "  → Manually copy to phone's Download/ or use: adb push $BUNDLE /sdcard/Download/"
  fi
else
  echo "[4/4] Bundle ready at $BUNDLE"
  echo "  → To push to phone: $0 --to-phone  or  adb push $BUNDLE /sdcard/Download/"
  echo "  → On phone, Kai: 'import history from Download' or tap 📎 and pick the zip"
fi

echo ""
echo "=== Export complete ==="
echo "Phone: open Kai-Android → Terminal tab → 'kai import'  or  Chat: 'import history from Download'"
echo "Or: Chat naturally: 'import my phone history' — Kai will scan Download/"
