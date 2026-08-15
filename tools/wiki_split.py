#!/usr/bin/env python3
"""Split wiki_filtered/articles.txt into per-article chunk files.

Input format (produced by wiki_extract_resumable.py):
    <title>
    ---
    <clean text>
    <<<END>>>

Splits at article boundaries (<<<END>>>) so no article is ever cut in half,
aiming for CHUNK_BYTES per chunk. Output: wiki_filtered/articles.XXXX.txt
Resumable via --state (skips already-written chunks).
"""
import argparse
import os
import sys

CHUNK_BYTES = 80 * 1024 * 1024  # 80MB target per chunk


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to articles.txt")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    ap.add_argument("--state", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir or os.path.dirname(args.input) or "."
    os.makedirs(outdir, exist_ok=True)
    state_path = args.state or os.path.join(outdir, ".split_state.json")

    done = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            import json
            done = json.load(f).get("chunks_done", 0)
            f.close()

    chunk_idx = done
    buf = []
    buf_size = 0
    n_articles = 0
    total_articles = 0

    def flush():
        nonlocal buf, buf_size, n_articles, chunk_idx, total_articles
        if not buf:
            return
        out_path = os.path.join(outdir, f"articles.{chunk_idx:04d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(buf))
        total_articles += n_articles
        print(f"[split] {out_path}: {n_articles} articles, {buf_size/1e6:.1f}MB "
              f"(cumulative {total_articles})", file=sys.stderr, flush=True)
        # persist state after each chunk
        import json
        with open(state_path, "w") as f:
            json.dump({"chunks_done": chunk_idx + 1, "articles": total_articles}, f)
        chunk_idx += 1
        buf = []
        buf_size = 0
        n_articles = 0

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        # collect one article (title + --- + text + <<<END>>>) at a time
        article = []
        a_size = 0
        for line in f:
            article.append(line)
            a_size += len(line.encode("utf-8", errors="replace"))
            if line.startswith("<<<END>>>"):
                # complete article
                if chunk_idx >= done:
                    buf.extend(article)
                    buf_size += a_size
                    n_articles += 1
                    if buf_size >= args.chunk_bytes:
                        flush()
                article = []
                a_size = 0
    if article:
        # trailing partial (no END marker) — append to current chunk as-is
        if chunk_idx >= done:
            buf.extend(article)
            buf_size += a_size
            n_articles += 1
    flush()  # final partial chunk

    import json
    with open(state_path, "w") as f:
        json.dump({"chunks_done": chunk_idx, "articles": total_articles,
                   "complete": True}, f)
    print(f"[split] DONE: {chunk_idx} chunks, {total_articles} articles total",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
