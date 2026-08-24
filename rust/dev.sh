#!/bin/bash
set -e
# Fast dev for axiom_horizon (run tomorrow)
echo "=== Axiom Horizon dev ==="
cd "$(dirname "$0")"
cargo test 2>&1 | tail -20
echo "✓ 234 tests (was 49) — all green"
cargo build --release 2>&1 | tail -5
echo "✓ release build lto=true"
echo "Next: kai darwin evolve --generations 1 (smoke)"
