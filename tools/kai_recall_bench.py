#!/usr/bin/env python3
"""kai recall bench — memory-grounded ruler for the recall/fuse path.

The code bench grades escalation (LAYER 2b/2c) but auto-recall (LAYER 1)
never fires there: code answers are never thin. This bench grades the
RECALL/FUSE path itself with tasks whose answers are IN the corpus but not
in the model's parametric memory:

  - Each task is a factual question whose gold answer is a distinctive
    token present in a wiki-corpus entry (verified retrievable, sim >= 0.69).
  - Greedy arm: raw ollama, NO corpus context -> measures parametric memory.
  - Recall arm: kai bridge (injects corpus context via recall_top_k /
    recall_ctx_chars knobs; thin-answer auto self-recall re-asks with the
    retrieved context) -> measures the fused memory path.
  - Grading: exact-match — the gold token must appear in the normalized
    answer (case/punct-insensitive). No judge, no rubric: the answer either
    contains the fact from memory or it does not.

This is the ruler for darwin-evolve of the recall knobs (recall_top_k,
recall_ctx_chars, recall_sim_gate, recall_thin_len).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from kai_code_bench import _ollama_chat, _bridge_chat, BRIDGE_PORT  # noqa: E402

RECORD = os.path.join(ROOT, ".kai_recall_bench.jsonl")

# (question, gold_token, source_article) — all gold tokens verified present
# in the corpus (see the inventory probe: sim 0.69-0.89, gold in top-3 hit).
TASKS = [
    ("What does the abbreviation TAI stand for?", ["International Atomic Time"], "international_atomic_time"),
    ("The Beaulieu Mine was a gold mining operation located near which city?", ["Yellowknife"], "beaulieu_mine"),
    ("Wrox Press, the computer book publisher, was originally based in which English city?", ["Birmingham"], "wrox_press"),
    ("The Economy Act of 1933 was officially titled the Act of which date?", ["March 20, 1933"], "economy_act_of_march_20_1933"),
    ("Aethalura is a genus of insects in which family?", ["Geometridae"], "aethalura_family"),
    ("Decebalus, also called Diurpaneus, was the last king of which ancient people?", ["Dacian", "Dacia"], "decebalus"),
    ("In Germanic mythology, Gram is the magical sword used by which hero?", ["Sigurd"], "gram_mythology"),
    ("Pāramitā is a Buddhist term often translated as what?", ["perfection"], "p_ramit"),
    ("Jerome A. Hammersmith was a political figure in which Canadian province?", ["Saskatchewan"], "jerome_hammersmith"),
    ("Greatorex was an electoral division in which Australian territory?", ["Northern Territory"], "electoral_division_of_greatorex"),
    ("What does CIM stand for in the manufacturing context?", ["Computer-integrated"], "computer_integrated_manufacturing"),
    ("Claudius Mamertinus was an official in which ancient empire?", ["Roman"], "claudius_mamertinus"),
    ("The Hardy-Littlewood Tauberian theorem is a theorem in which field?", ["analysis"], "hardy_littlewood_tauberian_theorem"),
    ("The four-barred grey moth belongs to which genus?", ["Aethalura"], "aethalura_genus"),
    # ── Harder tier: facts deeper in entries / at marginal sim, so the fuse
    #    knobs (top_k, ctx_chars, sim_gate) actually discriminate. All gold
    #    tokens verified present in the corpus at sim >= 0.68.
    ("What year was Wrox Press established?", ["1992"], "wrox_press_year"),
    ("In what year did the Beaulieu Mine enter production?", ["1947"], "beaulieu_mine_year"),
    ("The ingrailed clay moth belongs to which family?", ["Noctuidae"], "ingrailed_clay"),
    ("The viscous stress tensor is used to model what?", ["continuum", "stress"], "viscous_stress_tensor"),
    ("Which hero used the sword Gram in Germanic legend?", ["Sigurd"], "gram_mythology_hero"),
    ("Computer-integrated manufacturing is an approach using what to control the entire production process?", ["computers"], "computer_integrated_manufacturing_ctrl"),
    ("Plasma cosmology is a non-standard theory of what?", ["cosmology"], "non_standard_cosmology"),
]


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def grade(answer: str, gold) -> bool:
    """Exact-match: any gold token (normalized) must appear in the answer."""
    if isinstance(gold, str):
        gold = [gold]
    na = normalize(answer)
    return any(normalize(g) in na for g in gold)


def grade_arm(label: str, mode: str, port: int,
              model: str = "qwen2.5-coder:3b") -> dict:
    ledger_path = RECORD + f".{label}.ledger.jsonl"
    done = {}
    if os.path.exists(ledger_path):
        for line in open(ledger_path):
            try:
                e = json.loads(line)
                done[e["name"]] = e
            except Exception:
                pass
    passed = 0
    per_task = []
    t0 = time.time()
    ledger = open(ledger_path, "a")
    for ti, (q, gold, src) in enumerate(TASKS, 1):
        name = src
        if name in done:
            e = done[name]
            per_task.append(e)
            if e["pass"]:
                passed += 1
            print(f"  [{ti}/{len(TASKS)}] {name:42s} (resumed) pass={e['pass']}")
            continue
        if mode == "greedy":
            ans = _ollama_chat(model, q, 0.0, 120)
        else:  # recall — bridge fused memory path, NO external attempt
            ans, phys = _bridge_chat(q, 120, port, attempt=0, model=model)
        ok = grade(ans, gold)
        if ok:
            passed += 1
        entry = {"name": name, "pass": ok, "answer": ans[:200], "gold": gold,
                 "question": q, "mode": mode, "model": model}
        per_task.append(entry)
        ledger.write(json.dumps(entry) + "\n")
        ledger.flush()
        print(f"  [{ti}/{len(TASKS)}] {name:42s} pass={ok} | ans: {ans[:80]!r}")
    ledger.close()
    n = len(TASKS)
    res = {
        "t": time.time(), "label": label, "mode": mode, "worker": model,
        "tasks": n, "elapsed_s": round(time.time() - t0, 1),
        "pass": round(passed / n, 3), "solved": passed,
        "per_task": per_task,
    }
    return res


def main():
    ap = argparse.ArgumentParser(description="memory-grounded recall bench")
    ap.add_argument("--label", default="recall_bench")
    ap.add_argument("--arms", default="greedy,recall")
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--port", type=int, default=BRIDGE_PORT)
    args = ap.parse_args()

    print(f"==== kai recall bench  label='{args.label}'  model={args.model}  "
          f"arms={args.arms}")
    print(f"     {len(TASKS)} tasks, gold tokens from wiki corpus, "
          f"exact-match grading\n")
    for mode in [m.strip() for m in args.arms.split(",")]:
        print(f"--- {mode} ---")
        res = grade_arm(f"{args.label}_{mode}", mode, args.port, model=args.model)
        rec = json.dumps(res)
        with open(RECORD, "a") as f:
            f.write(rec + "\n")
        print(f"  {mode}: pass={res['pass']} solved={res['solved']}/{res['tasks']} "
              f"({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()