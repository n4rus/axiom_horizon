#!/usr/bin/env python3
"""kai_absorb_docs.py — bulk-absorb all .md/.txt files under a root dir into
Kai's persistent attractor memory.

Pipeline:
  1. Walk <root> collecting .md/.txt files (with noise exclusions).
  2. Chunk each file into ~1500-char overlapping segments.
  3. Batch-embed chunks via stock Ollama nomic-embed-text (/api/embed).
  4. Store entries keyed __doc__/<relpath>#<i> into .kai_doc_memory.json
     (persisted incrementally so a crash loses little).

Usage:
  python3 tools/kai_absorb_docs.py --root /home/l/Desktop/AxiomTree [--resume]
"""

import argparse
import array
import json
import os
import sys
import time
import urllib.request

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def compact_embedding(emb):
    """Store embeddings as a compact float32 array in memory (~3KB vs ~25KB
    for a 768-element Python float list). 8x RAM savings for large corpora —
    essential on this 15GB box (218k wiki chunks ≈ 5.6GB as lists)."""
    return array.array("f", emb)


def embed_to_json(obj):
    """json.dump default= handler: array.array -> plain list."""
    if isinstance(obj, array.array):
        return list(obj)
    raise TypeError(f"not serializable: {type(obj)}")


def compact_emb_list(emb):
    """Compact a plain-list embedding (from JSON) into array.array('f').

    load_state() must do this or a 200k-entry corpus comes back as Python
    lists (~21KB/entry -> ~5GB) and the process gets OOM-killed on this 15GB
    box alongside the bridge + opencode.
    """
    try:
        return array.array("f", emb)
    except (TypeError, ValueError):
        return emb

# Keys/paths we never want to absorb
EXCLUDED_DIRS = {".git", "node_modules", "build", "target", "_deps", "__pycache__",
                 "llvm-project", ".venv", "dist", "unsloth_compiled_cache",
                 ".git", ".venv", "wiki_filtered"}
EXCLUDED_NAMES = {"wordlist.txt", "shakespeare.txt", "halliday_text.txt",
                  ".kai_doc_memory.json", ".kai_corpus_attractor.json"}
MAX_FILE_BYTES = 1_000_000          # skip pathological files
CHUNK_SIZE = 1500                    # chars per embedding
CHUNK_OVERLAP = 200
BATCH_SIZE = 256                      # embedding batch (larger amortizes per-request overhead)
PERSIST_EVERY = 100                  # files between incremental writes

DOC_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".kai_doc_memory.json")


def embed_batch(texts):
    """Embed a batch of texts -> list of 768-dim lists. Returns [] on error."""
    if not texts:
        return []
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(EMBED_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.loads(resp.read())
            embs = data.get("embeddings", [])
            return [e[:EMBED_DIM] for e in embs]
    except Exception as e:
        print(f"[absorb] WARN embed failed: {e}", file=sys.stderr)
        return []


def chunk_text(text):
    """Split text into overlapping ~CHUNK_SIZE-char chunks, whitespace-aware."""
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            # back off to nearest whitespace to avoid splitting mid-word
            ws = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if ws != -1:
                end = ws
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def collect_files(root):
    """Walk root, return list of (abspath, relpath)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for fn in filenames:
            if not (fn.endswith(".md") or fn.endswith(".txt")):
                continue
            if fn in EXCLUDED_NAMES:
                continue
            fp = os.path.join(dirpath, fn)
            if os.path.getsize(fp) > MAX_FILE_BYTES:
                continue
            rel = os.path.relpath(fp, root)
            found.append((fp, rel))
    found.sort(key=lambda x: x[1])
    return found


def persist_shards(shard_base: str, entries: dict, per_shard: int = 7500):
    """Atomic, crash-safe shard writer for an arbitrary shard base path.

    The base must look like ".kai_xxx.json"; actual files are written as
    "<base minus .json>.<i>.json". Phases:
      1) write every new shard to a .tmp + fsync
      2) os.replace each staged shard into place (atomic per file)
      3) delete stale leftover shard files not in the new set
    A crash mid-persist can never lose the previous good state.
    """
    import glob as _glob
    base = shard_base.replace(".json", "")
    shard_prefix = f"{base}."
    keys = list(entries.keys())
    N = max(1, (len(keys) + per_shard - 1) // per_shard)
    per = (len(keys) + N - 1) // N

    staged = []
    for i in range(N):
        part_keys = keys[i * per:(i + 1) * per]
        if not part_keys:
            continue
        shard_path = f"{shard_prefix}{i}.json"
        tmp = shard_path + ".tmp"
        shard = {"version": 1, "total_entries": len(part_keys),
                 "embed_dim": EMBED_DIM, "kind": "doc_memory",
                 "entries": {k: entries[k] for k in part_keys}}
        with open(tmp, "w") as f:
            # compact separators: indent=1 on 768-float embeddings bloats
            # shards ~4x (one float per line). Compact keeps files small.
            json.dump(shard, f, separators=(",", ":"), default=embed_to_json)
            f.flush()
            os.fsync(f.fileno())
        staged.append((shard_path, tmp))
    for shard_path, tmp in staged:
        os.replace(tmp, shard_path)
    new_names = {sp for sp, _ in staged}
    for old in _glob.glob(shard_prefix + "*.json"):
        if old not in new_names:
            try:
                os.remove(old)
            except OSError:
                pass
    for tmp in _glob.glob(shard_prefix + "*.tmp"):
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_state(path):
    """Load doc memory from shards (.kai_doc_memory.<i>.json) if present,
    otherwise the legacy single file. Returns (entries, total_count)."""
    entries = {}
    base = path.replace(".json", "")
    import glob
    shards = sorted(glob.glob(f"{base}.*.json"))
    if shards:
        for sp in shards:
            try:
                with open(sp, "r") as f:
                    data = json.load(f)
                for k, v in data.get("entries", {}).items():
                    if isinstance(v, dict) and isinstance(v.get("embedding"), list):
                        v["embedding"] = compact_emb_list(v["embedding"])
                    entries[k] = v
            except Exception as e:
                print(f"[absorb] WARN: load shard {sp}: {e}", file=sys.stderr, flush=True)
        return entries, len(entries)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        entries_ = data.get("entries", {})
        for k, v in entries_.items():
            if isinstance(v, dict) and isinstance(v.get("embedding"), list):
                v["embedding"] = compact_emb_list(v["embedding"])
        return entries_, data.get("total_entries", len(entries_))
    return {}, 0


def persist(path, entries):
    """Doc-memory atomic persist (delegates to persist_shards).

    Writes the full new shard set to temp files + fsync FIRST, then swaps
    each into place with os.replace (atomic per file), and ONLY THEN removes
    stale leftover shard files. A crash mid-persist can therefore never lose
    the previous good state.
    """
    try:
        persist_shards(path, entries)
    except Exception as e:
        print(f"[absorb] WARN: persist failed: {e}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/l/Desktop/AxiomTree",
                    help="Root directory to scan for .md/.txt files")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-absorbed entries (by key)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Absorb at most N files (0 = all)")
    args = ap.parse_args()

    doc_path = DOC_MEMORY_PATH
    entries, _ = load_state(doc_path)
    if args.resume and entries:
        print(f"[absorb] resuming: {len(entries)} existing doc entries",
              file=sys.stderr)

    files = collect_files(args.root)
    print(f"[absorb] scanning {args.root}: {len(files)} candidate files",
          file=sys.stderr)
    if args.limit:
        files = files[:args.limit]

    absorbed = 0
    chunks_total = 0
    skipped = 0
    reabsorbed = 0
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
            entries[key] = {"embedding": compact_embedding(emb), "text": text[:500],
                            "kind": "doc", "ts": int(time.time()), **meta}
        batch_texts, batch_keys, batch_meta = [], [], []

    for i, (fp, rel) in enumerate(files):
        prefix = f"__doc__/{rel}"
        try:
            cur_mtime = os.stat(fp).st_mtime
        except OSError:
            cur_mtime = 0
        # Resume logic: skip unchanged files; re-absorb files whose mtime changed
        if args.resume:
            stored = None
            first_key = f"{prefix}#0"
            if first_key in entries:
                stored = entries[first_key].get("mtime")
            if first_key in entries and stored == cur_mtime:
                skipped += 1
                continue
            if first_key in entries:
                # File changed — drop old chunks for this prefix
                stale = [k for k in list(entries) if k.startswith(prefix + "#")]
                for k in stale:
                    del entries[k]
                reabsorbed += 1
        t_file = time.time()
        try:
            with open(fp, "r", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"[absorb] skip {rel}: {e}", file=sys.stderr, flush=True)
            continue
        chunks = chunk_text(content)
        if not chunks:
            continue
        for ci, c in enumerate(chunks):
            batch_texts.append(c)
            batch_keys.append(f"{prefix}#{ci}")
            batch_meta.append({"path": rel, "chunk": ci, "mtime": cur_mtime})
        if len(batch_keys) >= BATCH_SIZE:
            flush()
        absorbed += 1
        chunks_total += len(chunks)
        if (i + 1) % 10 == 0:
            rate = absorbed / max(time.time() - t0, 0.1)
            print(f"[absorb] {i+1}/{len(files)} files, {len(entries)} entries "
                  f"({rate:.1f} files/s, last file {time.time()-t_file:.1f}s)", file=sys.stderr, flush=True)
        if (i + 1) % PERSIST_EVERY == 0:
            flush()
            persist(doc_path, entries)

    flush()
    persist(doc_path, entries)
    dt = time.time() - t0
    print(f"[absorb] DONE: {absorbed} files, {chunks_total} chunks -> "
          f"{len(entries)} doc entries in {dt:.0f}s "
          f"(skipped {skipped}, re-absorbed {reabsorbed})", file=sys.stderr)


if __name__ == "__main__":
    main()
