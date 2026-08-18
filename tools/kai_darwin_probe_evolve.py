#!/usr/bin/env python3
"""kai darwin probe-evolve — evolve the LAYER 2c SELF-VERIFICATION probe so
the bridge closes the loop INTERNALLY (generate -> verify -> explore), with
NO external attempt field driving escalation.

Why this exists (Leverage 1, the highest-leverage cheap move):
  The validated escalation win (physics arm: 3b 0.93/1.00/29, 7b 0.966/29,
  12b 0.897/0.931/27) only exists because the BENCH sends `attempt=N` and
  re-asks after each failed grade. Production requests (opencode, ollama)
  never send that field -> the bridge serves everything at t_low and never
  escalates in real use. The loop is NOT actually closed in production.

  This script evolves the probe that lets the BRIDGE decide when to explore:
  after each internal generation it self-reviews its own code (majority vote
  over n_samples reviewer calls at probe_verify_temp) and escalates to
  t_high iff agreement < probe_agree_frac, bounded by auto_max_attempts.
  The knobs are darwin-evolvable; fitness is the honest closed-loop ruler:
  the bench's `auto` arm sends ONE request with NO attempt field and grades
  the bridge's self-decided final answer.

Fitness: auto-arm pass@1 (= pass@K, single external call) on the fixed
12-task subset, minus a small heat penalty proportional to the mean internal
auto_attempts (exploration is only worth its cost when it buys correctness).

The old probe_difficulty (one temp-0 review) was measured unreliable —
false-flagged easy fizzbuzz INCORRECT. This probe's majority vote + tunable
threshold is what darwin searches over here.
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
ARCHIVE_PATH = os.path.join(ROOT, ".axiom_state", "darwin_probe_archive.json")

# Evolvable knobs: name -> (default, lo, hi). Ints are marked via _INT_KNOBS.
# Probe semantics (LAYER 2c):
#   probe_verify_temp  — reviewer temperature (0 = strict/deterministic,
#                        0.4 = diverse reviews, noisier)
#   probe_n_verify     — number of independent reviewer calls (majority vote)
#   probe_agree_frac   — fraction of CORRECT verdicts required to trust
#   auto_max_attempts  — internal escalation depth (generate-verify rounds)
#   t_low/t_high       — escalation tiers (first shot vs explore)
KNOBS = {
    "t_low": (0.15, 0.05, 0.60),
    "t_high": (1.60, 1.0, 2.4),
    "probe_mode": (1, 0, 1),
    "probe_verify_temp": (0.0, 0.0, 0.80),
    "probe_n_verify": (1, 1, 3),
    "probe_agree_frac": (1.0, 0.40, 1.0),
    "probe_gen_temp": (0.4, 0.0, 1.0),
    "auto_max_attempts": (3, 1, 4),
}
_INT_KNOBS = {"probe_n_verify", "auto_max_attempts", "probe_mode"}

DEFAULT_VEC = {k: v[0] for k, v in KNOBS.items()}

MIN_PROMOTION_DELTA = 0.02
MUTATION_RATE_DECAY = 0.85   # improvement -> narrow step
MUTATION_RATE_GROW = 1.4     # regression/plateau -> widen step
MUTATION_FLOOR = 0.10
MUTATION_CEIL = 2.0

# Same fixed 12-task subset as the escalation darwin loop (stable ruler).
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
    every request (mtime-gated), so no restart is needed. auto_loop is FORCED
    on — this script only makes sense with the autonomous loop enabled."""
    merged = dict(DEFAULT_VEC)
    merged.update({k: round(float(v), 5) for k, v in vec.items()})
    merged["auto_loop"] = 1.0
    merged["probe_n_verify"] = int(merged["probe_n_verify"])
    merged["auto_max_attempts"] = int(merged["auto_max_attempts"])
    merged["probe_mode"] = int(merged["probe_mode"])
    tmp = PARAMS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, PARAMS_PATH)  # atomic; bridge never reads a torn file
    print(f"  params -> {PARAMS_PATH}: "
          f"t_low={merged['t_low']} t_high={merged['t_high']} "
          f"vtemp={merged['probe_verify_temp']} "
          f"n_verify={merged['probe_n_verify']} "
          f"agree={merged['probe_agree_frac']} "
          f"max_att={merged['auto_max_attempts']} auto_loop=1", flush=True)


def mutate(parent, rate):
    child = dict(parent)
    for k in parent:
        if random.random() < rate:
            _def, lo, hi = KNOBS[k]
            if k in _INT_KNOBS:
                # Discrete knobs: step by whole units.
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
    """Auto-arm bench on the fixed subset: ONE external call per task, no
    attempt field — the bridge's self-decided final answer is graded.
    Fitness rewards pass rate and penalizes internal sampling heat."""
    write_params(vec)
    tasks = [t for t in TASKS if t["name"] in SUBSET]
    last_err = None
    for attempt in range(3):
        try:
            res = grade_arm(label, tasks, "auto", 4, port, model=model)
            # mean internal auto_attempts (bridge self-verification rounds)
            auto_attempts = [t.get("auto_attempts", 1) for t in res["per_task"]]
            mean_auto = (sum(auto_attempts) / len(auto_attempts)
                         if auto_attempts else 1.0)
            # pass1 == pass_k for auto arm (single external call)
            fit = res["pass1"] - 0.01 * mean_auto
            res["mean_auto_attempts"] = round(mean_auto, 2)
            return fit, res
        except Exception as e:
            last_err = e
            print(f"  [eval retry {attempt + 1}/3] {type(e).__name__}: {e}",
                  flush=True)
            time.sleep(15 * (attempt + 1))
    raise last_err


def main():
    ap = argparse.ArgumentParser(description="darwin probe-evolve (LAYER 2c)")
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

    print(f"== darwin probe-evolve  gen={gen}  best_fit={best_fit:.3f}  "
          f"mut_rate={mut_rate:.2f}  model={args.model}  subset={len(SUBSET)} "
          f"tasks  (auto arm, closed loop)", flush=True)

    for g in range(gen + 1, gen + args.generations + 1):
        child = mutate(best_vec, mut_rate)
        label = f"darwin_probe_g{g}"
        print(f"\n--- generation {g}  parent_fit={best_fit:.3f} "
              f"mut_rate={mut_rate:.2f} ---", flush=True)
        t0 = time.time()
        fit, res = evaluate(child, args.model, args.port, label)
        elapsed = time.time() - t0
        print(f"  child fitness={fit:.3f}  pass@1={res['pass1']:.3f} "
              f"solved={res['solved']}/{res['tasks']} "
              f"mean_auto_attempts={res['mean_auto_attempts']} "
              f"mt={res['mean_temp']}  ({elapsed/60:.1f} min)", flush=True)

        delta = fit - best_fit
        arch["history"].append({
            "generation": g, "fitness": fit, "delta": delta,
            "pass1": res["pass1"], "solved": res["solved"],
            "mean_auto_attempts": res["mean_auto_attempts"],
            "mean_temp": res["mean_temp"],
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
    print("NOTE: last candidate remains in .kai_physics_params.json; promote "
          "the winner by re-writing it if it differs.", flush=True)


if __name__ == "__main__":
    main()
