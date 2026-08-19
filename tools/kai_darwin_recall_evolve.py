#!/usr/bin/env python3
"""kai darwin recall-evolve — recursive self-improvement of the RECALL/FUSE
path (LAYER 1 + corpus injection), grounded in the MEMORY ruler.

The code bench showed escalation works but never exercises auto-recall
(code answers are never thin). The recall bench (tools/kai_recall_bench.py)
grades the fused-memory path directly: 14 factual tasks whose gold answers
are in the wiki corpus (exact-match graded). Probe run at 3b: greedy (raw
parametric memory) 0.429 vs recall (bridge fused memory) 0.929 — the fuse
path is the proprietary edge and THIS is its ruler.

Loop (closes evolve -> params file -> live bridge -> memory ruler):
  1. Mutate the recall knobs the bridge reads per-request (mtime-gated,
     no restart): recall_top_k (how many corpus hits injected),
     recall_ctx_chars (per-hit context budget), recall_sim_gate (min sim to
     inject), recall_thin_len (thin-answer auto self-recall threshold).
  2. Fitness = recall-arm pass rate on the 14-task ruler.
  3. Delta-gen gate + mutation-rate adaptation (mirror darwin.rs and the
     bench-evolve loop). Archive in .axiom_state/darwin_recall_archive.json.
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from kai_recall_bench import grade_arm, TASKS, RECORD  # noqa: E402
from kai_code_bench import BRIDGE_PORT  # noqa: E402

PARAMS_PATH = os.path.join(ROOT, ".kai_physics_params.json")
ARCHIVE_PATH = os.path.join(ROOT, ".axiom_state", "darwin_recall_archive.json")

# Evolvable knobs: name -> (default, lo, hi). Ints via _INT_KNOBS.
KNOBS = {
    "recall_top_k": (3, 1, 8),
    "recall_ctx_chars": (300, 50, 800),
    "recall_sim_gate": (0.45, 0.20, 0.90),
    "recall_thin_len": (40, 5, 150),
}
_INT_KNOBS = {"recall_top_k", "recall_ctx_chars", "recall_thin_len"}

DEFAULT_VEC = {k: v[0] for k, v in KNOBS.items()}

MIN_PROMOTION_DELTA = 0.02
MUTATION_RATE_DECAY = 0.85
MUTATION_RATE_GROW = 1.4
MUTATION_FLOOR = 0.10
MUTATION_CEIL = 2.0


def load_archive():
    if os.path.exists(ARCHIVE_PATH):
        try:
            return json.load(open(ARCHIVE_PATH))
        except Exception:
            pass
    return {"generation": 0, "best_fitness": 0.0, "mutation_rate": 1.0,
            "history": []}


def save_archive(a):
    os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
    with open(ARCHIVE_PATH, "w") as f:
        json.dump(a, f, indent=1)


def write_params(vec):
    """Merge recall knobs into the live bridge params file (atomic). The
    bridge reloads every request (mtime-gated); other params untouched."""
    if not os.path.exists(PARAMS_PATH):
        merged = {}
    else:
        merged = json.load(open(PARAMS_PATH))
    for k, v in vec.items():
        merged[k] = int(v) if k in _INT_KNOBS else round(float(v), 5)
    tmp = PARAMS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, PARAMS_PATH)
    print(f"  params -> {PARAMS_PATH}: top_k={merged['recall_top_k']} "
          f"ctx={merged['recall_ctx_chars']} gate={merged['recall_sim_gate']} "
          f"thin={merged['recall_thin_len']}", flush=True)


def mutate(parent, rate):
    child = dict(parent)
    for k in parent:
        if random.random() < rate:
            _def, lo, hi = KNOBS[k]
            if k in _INT_KNOBS:
                step = 1 if random.random() < 0.5 else -1
                child[k] = min(hi, max(lo, int(parent[k]) + step))
            else:
                span = hi - lo
                step = span * 0.05 * (1.0 + random.random() * 2.0)
                if random.random() < 0.5:
                    step = -step
                child[k] = min(hi, max(lo, parent[k] + step))
    return child


def evaluate(vec, model, port, label):
    write_params(vec)
    last_err = None
    for attempt in range(3):
        try:
            res = grade_arm(f"{label}_recall", "recall", port, model=model)
            return res["pass"], res
        except Exception as e:
            last_err = e
            print(f"  [eval retry {attempt + 1}/3] {type(e).__name__}: {e}",
                  flush=True)
            time.sleep(15 * (attempt + 1))
    raise last_err


def main():
    ap = argparse.ArgumentParser(description="darwin recall-evolve (LAYER 1)")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--port", type=int, default=BRIDGE_PORT)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    arch = load_archive()
    gen = arch.get("generation", 0)
    best_fit = arch.get("best_fitness", 0.0)
    mut_rate = arch.get("mutation_rate", 1.0)
    best_vec = arch.get("best_vec") or dict(DEFAULT_VEC)

    print(f"== darwin recall-evolve  gen={gen}  best_fit={best_fit:.3f}  "
          f"mut_rate={mut_rate:.2f}  model={args.model}  "
          f"ruler={len(TASKS)} tasks (recall arm)", flush=True)

    for g in range(gen + 1, gen + args.generations + 1):
        child = mutate(best_vec, mut_rate)
        label = f"darwin_recall_g{g}"
        print(f"\n--- generation {g}  parent_fit={best_fit:.3f} "
              f"mut_rate={mut_rate:.2f} ---", flush=True)
        t0 = time.time()
        fit, res = evaluate(child, args.model, args.port, label)
        elapsed = time.time() - t0
        print(f"  child fitness={fit:.3f}  pass={res['pass']:.3f} "
              f"solved={res['solved']}/{res['tasks']}  ({elapsed/60:.1f} min)",
              flush=True)

        delta = fit - best_fit
        arch["history"].append({
            "generation": g, "fitness": fit, "delta": delta,
            "solved": res["solved"], "tasks": res["tasks"],
            "vec": {k: round(v, 4) for k, v in child.items()},
        })
        if fit > best_fit + MIN_PROMOTION_DELTA:
            best_fit, best_vec = fit, child
            arch["best_fitness"] = fit
            arch["best_vec"] = best_vec
            arch["best_generation"] = g
            mut_rate = max(MUTATION_FLOOR, mut_rate * MUTATION_RATE_DECAY)
            print(f"  >> PROMOTED gen {g}: fitness {fit:.3f} "
                  f"(delta {delta:+.3f}), narrowing step", flush=True)
        else:
            mut_rate = min(MUTATION_CEIL, mut_rate * MUTATION_RATE_GROW)
            print(f"  >> gated: delta {delta:+.3f} < {MIN_PROMOTION_DELTA}, "
                  f"widening step", flush=True)
        arch["generation"] = g
        arch["mutation_rate"] = mut_rate
        save_archive(arch)

    print("\n== done ==", flush=True)
    print(f"best fitness {arch.get('best_fitness', 0.0):.3f} at gen "
          f"{arch.get('best_generation', '?')}", flush=True)
    print("best vec:", json.dumps({k: round(v, 4) for k, v in best_vec.items()}),
          flush=True)
    print("NOTE: last candidate remains in params; promote the winner by "
          "re-writing it if it differs.", flush=True)


if __name__ == "__main__":
    main()