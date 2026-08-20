#!/usr/bin/env python3
"""kai darwin unified-evolve — recursive self-improvement over the WHOLE knob
space (escalation + probe + recall), grounded in a COMBINED ruler.

Why unified (Part 2.D): the three darwin loops (bench-escalation, probe,
recall) each evolved disjoint knob sets against a single ruler. A knob vector
that wins its own ruler is never forced to generalize. This loop evolves ONE
vector across all three domains and scores it on BOTH rulers:

    fitness = w_code * code_pass1  +  w_mem * mem_pass

  - code arm:  AUTO (LAYER 2c closed loop) by default — ONE external call
               with no attempt index; the bridge self-verifies via the exec
               probe and escalates internally, and its FINAL self-decided
               answer is graded by hidden tests. This is the TRUE probe
               ruler: the physics arm never exercises the probe, so probe
               knobs evolved against it were unconstrained walks. The auto
               arm makes probe_n_verify / probe_agree_frac / probe_n_asserts
               / auto_max_attempts / t_low / t_mid / t_high all directly
               fitness-bearing (pass@1 is the bridge's own decision quality).
               Code subset = 12 hard tasks INCLUDING matrix_transpose.
  - mem arm:   recall (fused corpus memory) on the 21-task memory ruler,
               exact-match graded.

A promotion must therefore improve the closed loop AND not regress memory
(or vice versa) — generalization pressure across domains, which is the
closest thing to AGI pressure the stack can measure. Archive in
.axiom_state/darwin_unified_auto_archive.json.

Loop (closes evolve -> params file -> live bridge -> both rulers):
  1. Mutate the unified vector (mtime-gated reload, no bridge restart).
  2. Fitness = w_code*code_pass + w_mem*mem_pass.
  3. Delta-gen gate + mutation-rate adaptation (mirror darwin.rs).
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

from kai_recall_bench import grade_arm as mem_grade_arm, TASKS as MEM_TASKS  # noqa: E402
from kai_code_bench import (grade_arm as code_grade_arm, TASKS as CODE_TASKS,
                            CAPACITY_TASKS, BRIDGE_PORT)  # noqa: E402

PARAMS_PATH = os.path.join(ROOT, ".kai_physics_params.json")
ARCHIVE_PATH = os.path.join(ROOT, ".axiom_state", "darwin_unified_auto_archive.json")

# Evolvable knobs across all three domains: name -> (default, lo, hi).
# Ints via _INT_KNOBS. Domain tags for logging.
KNOBS = {
    # escalation ramp (LAYER 2b/2c)
    "t_low": (0.15, 0.05, 0.35),
    "t_mid": (0.60, 0.35, 0.95),
    "t_high": (1.60, 1.20, 2.00),
    # probe (LAYER 2c exec spec-verification)
    "probe_n_verify": (1, 1, 3),
    "probe_agree_frac": (1.0, 0.6, 1.0),
    "probe_gen_temp": (0.4, 0.0, 1.0),
    "probe_n_asserts": (5, 2, 8),
    "auto_max_attempts": (3, 2, 5),
    # recall / fuse (LAYER 1)
    "recall_top_k": (3, 1, 8),
    "recall_ctx_chars": (300, 50, 800),
    "recall_sim_gate": (0.45, 0.20, 0.90),
    "recall_thin_len": (40, 5, 150),
}
_INT_KNOBS = {"probe_n_verify", "probe_n_asserts", "auto_max_attempts",
              "recall_top_k", "recall_ctx_chars", "recall_thin_len"}

DEFAULT_VEC = {k: v[0] for k, v in KNOBS.items()}

MIN_PROMOTION_DELTA = 0.02
MUTATION_RATE_DECAY = 0.85
MUTATION_RATE_GROW = 1.4
MUTATION_FLOOR = 0.10
MUTATION_CEIL = 2.0

# Hard code subset: includes the persistent failures (matrix_transpose,
# rotate_list, merge_intervals, next_greater) so the ramp is really
# exercised. First 12 by curated order.
CODE_SUBSET = [
    t for t in CODE_TASKS
    if t["name"] in {
        "fizzbuzz", "is_palindrome", "two_sum", "flatten", "matrix_transpose",
        "rotate_list", "merge_intervals", "next_greater", "longest_common_prefix",
        "max_subarray", "majority_element", "min_cost_stairs",
    }
]
CODE_SUBSET.sort(key=lambda t: [x["name"] for x in CODE_TASKS].index(t["name"]))

# Capacity-tier subset (A4): the 15 harder stateful/adversarial tasks — the
# ruler that actually still has headroom at 7b (base tier saturates at ~1.0).
CAP_SUBSET = list(CAPACITY_TASKS)

W_CODE = 0.5
W_MEM = 0.5
CODE_K = 4


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
    """Merge the unified vector into the live bridge params file (atomic).
    The bridge reloads every request (mtime-gated); other params untouched."""
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
    print(f"  params -> {PARAMS_PATH}: t_low={merged['t_low']} "
          f"t_mid={merged['t_mid']} t_high={merged['t_high']} "
          f"n_verify={merged['probe_n_verify']} "
          f"n_asserts={merged['probe_n_asserts']} "
          f"top_k={merged['recall_top_k']} gate={merged['recall_sim_gate']}",
          flush=True)


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


def evaluate(vec, model, port, label, code_tasks=None):
    """Combined fitness: w_code * code_pass1 (AUTO closed-loop arm, hard
    subset) + w_mem * mem_pass (recall arm, full memory ruler). The auto
    arm is the probe ruler — the bridge self-verifies and its final answer
    is graded, so probe knobs are directly fitness-bearing."""
    if code_tasks is None:
        code_tasks = CODE_SUBSET
    write_params(vec)
    last_err = None
    for attempt in range(3):
        try:
            code_res = code_grade_arm(f"{label}_code", code_tasks, "auto",
                                      1, port, model=model)
            mem_res = mem_grade_arm(f"{label}_mem", "recall", port, model=model)
            fit = W_CODE * code_res["pass1"] + W_MEM * mem_res["pass"]
            return fit, code_res, mem_res
        except Exception as e:
            last_err = e
            print(f"  [eval retry {attempt + 1}/3] {type(e).__name__}: {e}",
                  flush=True)
            time.sleep(15 * (attempt + 1))
    raise last_err


def seed_from_live_params():
    """Seed the initial parent vector from the CURRENT live params file —
    today's validated state (t_low=0.2017, t_mid=0.61, probe relaxations,
    darwin-promoted recall knobs) instead of the raw defaults."""
    vec = dict(DEFAULT_VEC)
    if os.path.exists(PARAMS_PATH):
        try:
            live = json.load(open(PARAMS_PATH))
            for k in vec:
                if k in live and isinstance(live[k], (int, float)):
                    vec[k] = live[k]
        except Exception:
            pass
    return vec


def main():
    ap = argparse.ArgumentParser(description="darwin unified-evolve (all domains)")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--port", type=int, default=BRIDGE_PORT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tier", choices=["base", "capacity"], default="base",
                    help="code ruler tier: base=12 hard subset (saturating), "
                         "capacity=15 stateful/adversarial tasks (headroom)")
    args = ap.parse_args()
    random.seed(args.seed)
    code_tasks = CAP_SUBSET if args.tier == "capacity" else CODE_SUBSET

    arch = load_archive()
    gen = arch.get("generation", 0)
    best_fit = arch.get("best_fitness", 0.0)
    mut_rate = arch.get("mutation_rate", 1.0)
    if gen == 0:
        best_vec = arch.get("best_vec") or seed_from_live_params()
    else:
        best_vec = arch.get("best_vec") or dict(DEFAULT_VEC)

    print(f"== darwin unified-evolve  gen={gen}  best_fit={best_fit:.3f}  "
          f"mut_rate={mut_rate:.2f}  model={args.model}  tier={args.tier}",
          flush=True)
    print(f"   code subset: {len(code_tasks)} tasks (AUTO closed-loop arm, "
          f"w={W_CODE})  |  memory ruler: {len(MEM_TASKS)} tasks (recall arm, "
          f"w={W_MEM})", flush=True)

    for g in range(gen + 1, gen + args.generations + 1):
        child = mutate(best_vec, mut_rate)
        label = f"darwin_{args.tier}_g{g}"
        print(f"\n--- generation {g}  parent_fit={best_fit:.3f} "
              f"mut_rate={mut_rate:.2f} ---", flush=True)
        t0 = time.time()
        fit, code_res, mem_res = evaluate(child, args.model, args.port, label,
                                          code_tasks=code_tasks)
        elapsed = time.time() - t0
        print(f"  fitness={fit:.3f}  code_pass1={code_res['pass1']:.3f} "
              f"({code_res['solved']}/{len(CODE_SUBSET)})  "
              f"mem_pass={mem_res['pass']:.3f} "
              f"({mem_res['solved']}/{mem_res['tasks']})  ({elapsed/60:.1f} min)",
              flush=True)

        delta = fit - best_fit
        arch["history"].append({
            "generation": g, "fitness": fit, "delta": delta,
            "code_pass1": code_res["pass1"], "code_solved": code_res["solved"],
            "mem_pass": mem_res["pass"], "mem_solved": mem_res["solved"],
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
