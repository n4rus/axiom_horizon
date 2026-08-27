#!/usr/bin/env python3
"""
kai_wiki_ingest.py — progressive Wikipedia ingestion (Phase C, passive ~3d)
Streams enwiki-*-pages-articles-multistream.xml.bz2, extracts article text,
chunks, embeds via Ollama nomic-embed-text, appends to .kai_wiki_memory.*.json
shards (same format as phase3_embed). Resume-safe: tracks offset in
.kai_wiki_progress.json. Rate: ~0.5-1 embed/sec (ollama bound), ~40k/day.
"""
import bz2, json, re, sys, os, time, urllib.request, urllib.error
from pathlib import Path

WIKI_BZ2 = Path("/home/l/Desktop/AxiomTree/axiom_horizon/enwiki-20260601-pages-articles-multistream.xml.bz2")
PROGRESS = Path("/home/l/Desktop/AxiomTree/axiom_horizon/.kai_wiki_progress.json")
SHARD_PREFIX = "/home/l/Desktop/AxiomTree/axiom_horizon/.kai_wiki_memory"
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
SHARD_SIZE = 500  # entries per shard file

def ollama_embed(text: str):
    data = json.dumps({"model": EMBED_MODEL, "prompt": text[:8192]}).encode()
    req = urllib.request.Request(EMBED_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["embedding"]
    except Exception as e:
        print(f"  embed fail: {e}", file=sys.stderr)
        return None

def chunk_text(text: str, n=4):
    # naive paragraph chunking
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    if not paras:
        paras = [text[i:i+700] for i in range(0, len(text), 700)]
    # take up to n chunks per article
    return paras[:n]

def load_progress():
    if PROGRESS.exists():
        try: return json.loads(PROGRESS.read_text())
        except: return {"offset": 0, "articles": 0, "embedded": 0}
    return {"offset": 0, "articles": 0, "embedded": 0}

def save_progress(p):
    PROGRESS.write_text(json.dumps(p))

def next_shard_idx():
    i=0
    while Path(f"{SHARD_PREFIX}.{i}.json").exists():
        # check if shard is full
        try:
            d=json.loads(Path(f"{SHARD_PREFIX}.{i}.json").read_text())
            if d.get("total_entries",0) < SHARD_SIZE:
                return i
        except: pass
        i+=1
    return i

def append_to_shard(key, emb):
    idx = next_shard_idx()
    path = Path(f"{SHARD_PREFIX}.{idx}.json")
    if path.exists():
        data=json.loads(path.read_text())
    else:
        data={"version":1,"total_entries":0,"embed_dim": len(emb), "kind": "doc_memory", "entries": {}}
    data["entries"][key] = {"embedding": emb}
    data["total_entries"] = len(data["entries"])
    path.write_text(json.dumps(data))
    return idx

def main(limit=0):
    prog = load_progress()
    off = prog.get("offset",0)
    embedded = prog.get("embedded",0)
    articles = prog.get("articles",0)
    print(f"[wiki] resume offset={off} articles={articles} embedded={embedded}", file=sys.stderr)
    # stream bz2
    title_re = re.compile(r"<title>(.*?)</title>")
    text_re = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
    buf=""
    # incremental read
    with bz2.open(WIKI_BZ2, "rt", errors="replace") as f:
        # skip to offset (approx line count) - we track articles instead for resume
        n=0
        page=""
        in_page=False
        for line in f:
            if "<page>" in line:
                page=line; in_page=True; continue
            if not in_page: continue
            page+=line
            if "</page>" in line:
                in_page=False
                n+=1
                if n <= articles:
                    page=""; continue
                m_title=title_re.search(page)
                m_text=text_re.search(page)
                if not m_title or not m_text:
                    page=""; continue
                title=m_title.group(1)
                # skip non-article namespaces (File:, Template:, etc.)
                if ":" in title:
                    if any(title.startswith(p+":") for p in ["File","Template","Category","Wikipedia","Help","Portal","Draft","TimedText","Module"]):
                        page=""; continue
                raw=m_text.group(1)
                # strip wiki markup roughly
                raw=re.sub(r"<[^>]+>", "", raw)
                raw=re.sub(r"\{\{[^}]+\}\}", "", raw)
                raw=re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", raw)
                if len(raw) < 200:
                    page=""; continue
                for ci, chunk in enumerate(chunk_text(raw)):
                    key=f"__doc__/wiki/{title}#{ci}"
                    emb=ollama_embed(chunk)
                    if emb:
                        idx=append_to_shard(key, emb)
                        embedded+=1
                        print(f"[wiki] {title}#{ci} -> shard {idx} ({embedded} total)", file=sys.stderr)
                        prog["embedded"]=embedded
                        time.sleep(0.1)
                articles=n
                prog["articles"]=articles
                prog["offset"]=n
                if n % 10==0:
                    save_progress(prog)
                page=""
                if limit and embedded >= limit:
                    save_progress(prog)
                    print(f"[wiki] limit {limit} reached", file=sys.stderr)
                    return
                # throttle to avoid VRAM pressure: 1 embed per ~1s already
        save_progress(prog)
    print(f"[wiki] done articles={articles} embedded={embedded}", file=sys.stderr)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max embeddings this run (0=unlimited)")
    args=ap.parse_args()
    main(limit=args.limit)
