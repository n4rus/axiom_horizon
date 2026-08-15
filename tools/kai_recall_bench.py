#!/usr/bin/env python3
"""
kai_recall_bench.py — measure Kai's retained-memory recall quality.

Queries a fixed topical benchmark set against the bridge /v1/recall endpoint
and reports hit-rate (top-1 similarity threshold) + mean top-1 similarity.
Run BEFORE a memory ingest for a baseline, then AGAIN after to measure the
improvement. Results append to a JSONL history at .kai_recall_bench.jsonl.

Usage:  python3 tools/kai_recall_bench.py [--port 8765] [--top_k 3]
        [--thr 0.50] [--pair <label>]
"""
import argparse, json, time, urllib.request, urllib.parse

RECORD = ".kai_recall_bench.jsonl"

# Fixed, domain-spanning probe set. Targets facts batch-1 wiki should hold.
QUERIES = [
    # physics
    "quantum entanglement measurement",
    "black hole event horizon",
    "entropy second law of thermodynamics",
    "special relativity time dilation",
    # history
    "french revolution causes",
    "world war two timeline",
    "roman empire fall",
    "industrial revolution steam power",
    # geography / nature
    "amazon rainforest biodiversity",
    "sahara desert location",
    "great barrier reef",
    # tech / math
    "neural network attention mechanism",
    "prime number distribution",
    "cryptographic hash function",
    # chemistry / materials
    "titanium alloy properties",
    "solar photovoltaic electricity",
    # project-specific — ONLY answerable now that full opencode history is in
    "kai vfe tau time dilation controller",
    "darwin self improvement generation",
    "wiki batch one ingestion shards",
    "bridge soul system personality memory",
    "persistent ollama systemd unit",
]


def recall(query: str, top_k: int, port: int) -> dict:
    url = f"http://127.0.0.1:{port}/v1/recall?q=" \
          f"{urllib.parse.quote(query)}&top_k={top_k}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.50, help="top-1 sim hit thr")
    ap.add_argument("--pair", default="", help="label for paired delta logging")
    args = ap.parse_args()

    hits, sims, misses = 0, [], []
    for q in QUERIES:
        try:
            d = recall(q, args.top_k, args.port)
            rec = d.get("recalled", [])
            top = rec[0]["similarity"] if rec else 0.0
            sims.append(top)
            if top >= args.thr:
                hits += 1
            else:
                misses.append([q, round(top, 3)])
        except Exception as e:
            print(f"  [ER] {q}: {e}")
            misses.append([q, 0.0]); sims.append(0.0)

    n = len(QUERIES)
    hit_rate = hits / n
    mean_top = sum(sims) / n
    p50 = sorted(sims)[n // 2] if sims else 0.0
    res = {
        "t": time.time(), "pair": args.pair,
        "n": n, "thr": args.thr,
        "hit_rate": round(hit_rate, 4), "mean_top1_sim": round(mean_top, 4),
        "median_top1_sim": round(p50, 4),
        "misses": misses, "top1": [round(s, 3) for s in sims],
    }
    with open(BRIDGE if False else RECORD, "a") as f:
        f.write(json.dumps(res) + "\n")

    print(f"\n==== kai recall bench  pair='{args.pair}'  n={n}  thr={args.thr:.2f} ====")
    print(f"  hit_rate        {hit_rate:.3f}  ({hits}/{n} top-1 above {args.thr})")
    print(f"  mean top1 sim   {mean_top:.3f}")
    print(f"  median top1 sim {p50:.3f}")
    if misses:
        print("  misses:")
        for q, s in misses:
            print(f"    [{s:.2f}] {q}")
    print(f"  -> appended to {RECORD}\n")


if __name__ == "__main__":
    main()