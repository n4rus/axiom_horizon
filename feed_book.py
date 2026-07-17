#!/usr/bin/env python3
"""Feed model_reading.md into Kai's knowledge base + attractor memory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import kai_mind as km

BOOK = ROOT / ".axiom_state" / "physics" / "model_reading.md"

print(f"[feed] book exists: {BOOK.exists()} ({BOOK.stat().st_size} bytes)")
mind = km.KaiMind()
print(f"[feed] attractor size before: {mind.attractor_size}")
print(f"[feed] KB before: {mind.kb_summary()}")

result = mind.kb_feed_book(str(BOOK))
print(f"[feed] {result}")

print(f"[feed] attractor size after: {mind.attractor_size}")
print(f"[feed] KB after: {mind.kb_summary()}")

# Verify retrieval works on a physics query.
probe = mind.kb_query("relativistic momentum and time dilation", k=3)
print("[feed] retrieval probe (relativistic momentum):")
print(probe)
