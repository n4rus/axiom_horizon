#!/usr/bin/env python3
"""kai darwin bench-evolve — recursive self-improvement of the fused/recall
path, grounded in the VERIFIABLE code bench (not token-coherence proxies).

Loop (closes evolve -> params file -> live bridge -> bench delta):
  1. Mutate a physics-param vector (t_low/t_high escalation tiers, recall
     knobs, novelty_scale, tau bounds, top_p) around the current best.
  2. Write .kai_physics_params.json — the bridge live-reloads it per request,
     no restart needed (LAYER 2b escalation + LAYER 1 auto-recall read it).
  3. Fitness = physics-arm pass@1 + pass@K on a FIXED task subset, graded by
     hidden unit tests (objective, no judge, no rubric).
  4. Delta-gen gate: promote only strict improvements (parent->child fitness
     delta > min_delta); adapt mutation amplitude — narrow on improvement,
     widen on regression/plateau (mirrors darwin.rs::adapt_mutation_rate).
  5. Persist generation state in .axiom_state/darwin_bench_archive.json so a
     crash resumes instead of restarting from scratch.

The previous darwin archive (darwin_archive.json, gen 48) had 0/60 strict
promotions and empty patches because its fitness was token-level novelty /
coherence. This loop replaces that signal with task pass rates — the same
ruler that validated the escalation controller at 3b/7b/12b.
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

from kai_code_bench import grade_arm, TASKS, BRIDGE_PORT  # noqa: E402

PARAMS_PATH = os.path.join(ROOT, ".kai_physics_params.json")
ARCHIVE_PATH = os.path.join(ROOT, ".axiom_state", "darwin_bench_archive.json")

# Evolvable knobs: name -> (default, lo, hi). Bounds from empirical ranges:
# t_low/t_high from the escalation validation; recall knobs from the bridge
# defaults; tau from the VFE controller spec (section 6.2).
KNOBS = {
    "base_temperature": (0.33, 0.1, 1.2),
    "t_low": (0.15, 0.05, 0.60),
    "t_high": (1.60, 1.0, 2.4),
    "novelty_scale": (0.30, 0.0, 1.0),
    "top_p": (0.997, 0.85, 1.0),
    "tau_min": (0.855, 0.5, 1.2),
    "tau_max": (1.657, 1.2, 2.2),
    "vfe_tau_rate": (0.061, 0.01, 0.20),
    "recall_sim_gate": (0.45, 0.20, 0.80),
    "recall_thin_len": (40, 10, 120),
    "recall_top_k": (3, 1, 5),
}

DEFAULT_VEC = {k: v[0] for k, v in KNOBS.items()}

MIN_PROMOTION_DELTA = 0.02
MUTATION_RATE_DECAY = 0.85   # improvement -> narrow step
MUTATION_RATE_GROW = 1.4     # regression/plateau -> widen step
MUTATION_FLOOR = 0.10
MUTATION_CEIL = 2.0

# Fixed 12-task subset for fitness: mix of easy and hard, stable across
# generations so the delta is attributable to params, not task sampling.
# Chosen from the 29 to keep each candidate eval ~10-15 min at 3b.
SUBSET = [
    "fizzbuzz", "fib", "two_sum", "is_palindrome", "valid_parentheses",
    "first_uniq_char", "merge_intervals", "is_balanced", "matrix_transpose",
    "next_greater", "min_cost_stairs", "majority_element",
]


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
    """Write the param vector to the live bridge file. Bridge reloads on
    every request (mtime-gated), so no restart is needed."""
    merged = dict(DEFAULT_VEC)
    merged.update({k: round(float(v), 5) for k, v in vec.items()})
    tmp = PARAMS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, PARAMS_PATH)  # atomic; bridge never reads a torn file
    print(f"  params -> {PARAMS_PATH}: "
          f"t_low={merged['t_low']} t_high={merged['t_high']} "
          f"gate={merged['recall_sim_gate']} thin={merged['recall_thin_len']} "
          f"k={merged['recall_top_k']} ns={merged['novelty_scale']} "
          f"tau=[{merged['tau_min']},{merged['tau_max']}] rate={merged['vfe_tau_rate']}",
          flush=True)


def mutate(parent, rate):
    child = dict(parent)
    for k in parent:
        if random.random() < rate:
            _def, lo, hi = KNOBS[k]
            # Gaussian-ish jitter scaled to the knob's range.
            span = hi - lo
            step = span * 0.05 * (1.0 + random.random() * 2.0)
            if random.random() < 0.5:
                step = -step
            child[k] = min(hi, max(lo, parent[k] + step))
    return child


def evaluate(vec, model, port, label, k):
    """Physics-arm bench on the fixed subset. Returns (fitness, result).

    Resilience: the bridge can restart mid-eval (or ollama can drop a
    request). grade_arm's per-task ledger makes re-entry cheap, but a dead
    bridge raises URLError out of grade_arm — retry the whole eval a few
    times with backoff instead of crashing the generation.
    """
    write_params(vec)
    tasks = [t for t in TASKS if t["name"] in SUBSET]
    last_err = None
    for attempt in range(3):
        try:
            res = grade_arm(label, tasks, "physics", k, port, model=model)
            fit = res["pass1"] * 0.7 + res["pass_k"] * 0.3 - 0.01 * (res["mean_attempts"] or 0)
            return fit, res
        except Exception as e:
            last_err = e
            print(f"  [eval retry {attempt + 1}/3] {type(e).__name__}: {e}",
                  flush=True)
            time.sleep(15 * (attempt + 1))
    raise last_err


def main():
    ap = argparse.ArgumentParser(description="darwin bench-evolve")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
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

    print(f"== darwin bench-evolve  gen={gen}  best_fit={best_fit:.3f}  "
          f"mut_rate={mut_rate:.2f}  model={args.model}  subset={len(SUBSET)} "
          f"tasks  k={args.k}", flush=True)

    for g in range(gen + 1, gen + args.generations + 1):
        # 1 candidate per generation: parent = current best; single-offspring
        # selection keeps each eval cheap (~10-15 min) while the delta-gen
        # gate still enforces directed search. Population of 1 is enough to
        # test the loop; widen with --pop if desired later.
        child = mutate(best_vec, mut_rate)
        label = f"darwin_g{g}"
        print(f"\n--- generation {g}  parent_fit={best_fit:.3f} "
              f"mut_rate={mut_rate:.2f} ---", flush=True)
        t0 = time.time()
        fit, res = evaluate(child, args.model, args.port, label, args.k)
        elapsed = time.time() - t0
        print(f"  child fitness={fit:.3f}  pass@1={res['pass1']:.3f} "
              f"pass@K={res['pass_k']:.3f}  solved={res['solved']}/{res['tasks']} "
              f"mt={res['mean_temp']}  ({elapsed/60:.1f} min)", flush=True)

        delta = fit - best_fit
        arch["history"].append({
            "generation": g, "fitness": fit, "delta": delta,
            "pass1": res["pass1"], "pass_k": res["pass_k"],
            "solved": res["solved"], "mean_temp": res["mean_temp"],
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


if __name__ == "__main__":
    main()