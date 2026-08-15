#!/usr/bin/env python3
"""wiki_guardian.py — guard against the giant-file failure mode.

Why: opencode's git snapshot chokes when the watched tree contains multi-
hundred-MB files, which caused repeated session crashes. This tool:

  1. SCAN — report any file > MAX_BYTES in the workspace (excluding .git,
     gitignored paths and known-large-but-safe artifacts).
  2. CHECK-SHARDS — verify every .kai_doc_memory.*.json / .kai_wiki_memory.*.json
     is valid JSON and below the per-shard cap (doc memory crash-safety).
  3. SPLIT — if wiki_filtered/articles.txt exceeds CHUNK_TARGET and the chunks
     are missing/stale, re-run the chunk splitter to keep chunks bounded.

Usage:
  python3 tools/wiki_guardian.py scan
  python3 tools/wiki_guardian.py check-shards
  python3 tools/wiki_guardian.py split [--force]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 150 * 1024 * 1024       # alert threshold: 150MB
SHARD_CAP = 130 * 1024 * 1024       # per-shard cap: 130MB
CHUNK_TARGET = 80 * 1024 * 1024     # wiki chunk target: 80MB

# paths that are always safe to see at any size (gitignored + disposable)
ALWAYS_SAFE = {
    "wiki_filtered", "opencode_sessions", ".axiom_state",
    "rust", "sources", "enwiki-20260601-pages-articles-multistream.xml.bz2",
}
SAFE_SUFFIXES = (".gguf", ".bin", ".bz2", ".tar.gz", ".db", ".sqlite")


def scan():
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune heavy dirs early (they're gitignored or known-safe)
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "node_modules", "build", "target",
                                    ".venv", "llvm-project", "_deps"}]
        for fn in filenames:
            if fn.endswith(SAFE_SUFFIXES):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            top = rel.split(os.sep)[0]
            if top in ALWAYS_SAFE and top != "wiki_filtered":
                continue
            try:
                sz = os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                continue
            if sz > MAX_BYTES:
                hits.append((sz, rel))
    hits.sort(reverse=True)
    if not hits:
        print("[guardian] OK: no files over %.0fMB" % (MAX_BYTES / 1e6))
        return 0
    for sz, rel in hits:
        print(f"[guardian] LARGE {sz/1e6:.0f}MB  {rel}")
    return 1


def check_shards():
    bad = 0
    for pattern in (".kai_doc_memory.*.json", ".kai_wiki_memory.*.json",
                    ".kai_chat_memory.*.json"):
        for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
            sz = os.path.getsize(p)
            try:
                with open(p) as f:
                    d = json.load(f)
                n = len(d.get("entries", {}))
                status = "OK"
            except Exception as e:
                bad += 1
                status = f"CORRUPT: {e}"
            over = " OVER-CAP!" if sz > SHARD_CAP else ""
            print(f"[guardian] {status}{over} {os.path.basename(p)} "
                  f"{sz/1e6:.1f}MB {n} entries")
    if bad:
        print(f"[guardian] {bad} shard(s) corrupt")
        return 1
    print("[guardian] all shards valid")
    return 0


def split(force=False):
    art = os.path.join(ROOT, "wiki_filtered", "articles.txt")
    chunks = sorted(glob.glob(os.path.join(ROOT, "wiki_filtered", "chunks",
                                           "articles.*.txt")))
    if not os.path.exists(art):
        print("[guardian] no articles.txt; nothing to split")
        return 0
    art_sz = os.path.getsize(art)
    if chunks:
        max_chunk = max(os.path.getsize(c) for c in chunks)
        if not force and max_chunk <= CHUNK_TARGET:
            print(f"[guardian] chunks OK (max {max_chunk/1e6:.0f}MB, "
                  f"articles.txt {art_sz/1e6:.0f}MB)")
            return 0
    print(f"[guardian] splitting {art_sz/1e6:.0f}MB articles.txt "
          f"({len(chunks)} existing chunks)")
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "tools", "wiki_split.py"),
                        art, "-o", os.path.join(ROOT, "wiki_filtered", "chunks")])
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "check-shards", "split"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "scan":
        sys.exit(scan())
    if args.cmd == "check-shards":
        sys.exit(check_shards())
    if args.cmd == "split":
        sys.exit(split(args.force))


if __name__ == "__main__":
    main()
