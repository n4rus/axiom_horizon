#!/usr/bin/env python3
"""wiki_absorb_chunks.py — embed the wiki chunk files (articles.XXXX.txt) into
Kai's memory at ARTICLE boundaries, into a SEPARATE shard namespace.

Each chunk file contains many articles of the form:
    <title>
    ---
    <clean text>
    <<<END>>>

Entries are stored keyed __doc__/wiki/<title>#<i> in .kai_wiki_memory.*.json
shards (own namespace). Titles are the keys, so --resume is idempotent.
The wiki shards are written atomically via persist_shards (crash-safe) and
never touch the .kai_doc_memory.*.json corpus shards.

Usage:
  python3 tools/wiki_absorb_chunks.py [--chunks wiki_filtered/chunks] [--resume]
"""
import argparse
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kai_absorb_docs as kad  # noqa: E402
from kai_absorb_docs import (  # noqa: E402
    embed_batch, chunk_text, load_state, persist_shards, BATCH_SIZE,
    compact_embedding,
)

# Wiki articles are long; use larger chunks than the generic corpus so the
# chunk count / storage stays manageable (~6000 chars < nomic 8192-token ctx).
# The time wall is the GPU (~2100 tok/s); overlap overhead is what larger
# chunks minimize.
kad.CHUNK_SIZE = 6000
kad.CHUNK_OVERLAP = 300

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_MEMORY_PATH = os.path.join(ROOT, ".kai_wiki_memory.json")

PERSIST_EVERY = 3  # persist every 3 chunk files (crash loses <= 3 files)


def slugify(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", title.strip()).strip("_").lower()
    return s[:120] or "untitled"


def norm_prefix(body: str, n: int = 200) -> str:
    """Normalized content fingerprint for duplicate-topic detection.

    Collapses all whitespace and takes the first n chars. Two articles that
    share this prefix are treated as the same topic: wikipedia re-emits the
    same content under many titles (redirects, subpages, 'List of X' variants),
    and the resumable extractor's checkpoint reset re-matched pages already
    absorbed pre-reset. ~30% of the post-reset tail was content already under
    memory — each wasted an embed batch. This prefix check is O(1) per article
    and runs BEFORE any embedding.
    """
    return " ".join(body.split())[:n]


def iter_articles(fp):
    """Yield (title, body) for each article in a chunk file."""
    title = None
    body = []
    for line in open(fp, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if title is None:
            title = line
        elif line == "---":
            continue
        elif line == "<<<END>>>":
            yield title, "\n".join(body).strip()
            title = None
            body = []
        else:
            body.append(line)
    if title is not None:
        yield title, "\n".join(body).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="wiki_filtered/chunks")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Own namespace: load ONLY wiki entries (bounded memory; the doc corpus
    # shards are never read nor rewritten by this tool).
    wiki_entries, _ = load_state(WIKI_MEMORY_PATH)
    if args.resume:
        print(f"[wiki_absorb] resuming: {len(wiki_entries)} existing wiki entries",
              file=sys.stderr)

    # Content-dedup set: normalized prefix of every article already embedded.
    # Existing entries store text[:500]; a fresh entry's prefix is added below
    # as soon as it is embedded. ~30% of post-reset articles were content
    # already absorbed under a different title — each used to waste a full
    # embed batch. This set makes resume skip them in O(1).
    seen_prefixes = set()
    for v in wiki_entries.values():
        if isinstance(v, dict) and v.get("text"):
            seen_prefixes.add(norm_prefix(v["text"], 200))

    files = sorted(glob.glob(os.path.join(args.chunks, "articles.*.txt")))
    print(f"[wiki_absorb] {len(files)} chunk files in {args.chunks} "
          f"({len(seen_prefixes)} content fingerprints loaded)",
          file=sys.stderr)

    absorbed = 0
    chunks_total = 0
    skipped = 0
    t0 = time.time()
    batch_texts, batch_keys, batch_meta = [], [], []

    def flush():
        nonlocal batch_texts, batch_keys, batch_meta
        if not batch_keys:
            return
        embs = embed_batch(batch_texts)
        for key, text, meta, emb in zip(batch_keys, batch_texts, batch_meta, embs):
            if not emb:
                continue
            wiki_entries[key] = {"embedding": compact_embedding(emb),
                                 "text": text[:500],
                                 "kind": "doc", "ts": int(time.time()), **meta}
            if key.endswith("#0"):
                seen_prefixes.add(norm_prefix(text, 200))
        batch_texts, batch_keys, batch_meta = [], [], []

    for fi, fp in enumerate(files):
        t_file = time.time()
        for title, body in iter_articles(fp):
            if not body:
                continue
            prefix = f"__doc__/wiki/{slugify(title)}"
            if args.resume and prefix + "#0" in wiki_entries:
                skipped += 1
                continue
            fp_norm = norm_prefix(body, 200)
            if fp_norm in seen_prefixes:
                skipped += 1
                continue
            chunks = chunk_text(body)
            if not chunks:
                continue
            for ci, c in enumerate(chunks):
                batch_texts.append(c)
                batch_keys.append(f"{prefix}#{ci}")
                batch_meta.append({"path": f"wiki/{title[:60]}", "chunk": ci})
            chunks_total += len(chunks)
            absorbed += 1
            if len(batch_keys) >= BATCH_SIZE:
                flush()
            if args.limit and absorbed >= args.limit:
                break
        if (fi + 1) % 1 == 0:
            rate = absorbed / max(time.time() - t0, 0.1)
            print(f"[wiki_absorb] file {fi+1}/{len(files)}, {absorbed} articles "
                  f"({rate:.2f} art/s, last file {time.time()-t_file:.1f}s, "
                  f"{len(wiki_entries)} wiki entries, {skipped} skipped)",
                  file=sys.stderr, flush=True)
        if (fi + 1) % PERSIST_EVERY == 0:
            flush()
            persist_shards(WIKI_MEMORY_PATH, wiki_entries)
        if args.limit and absorbed >= args.limit:
            break

    flush()
    persist_shards(WIKI_MEMORY_PATH, wiki_entries)
    dt = time.time() - t0
    print(f"[wiki_absorb] DONE: {absorbed} articles, {chunks_total} chunks -> "
          f"{len(wiki_entries)} wiki entries in {dt:.0f}s (skipped {skipped})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
