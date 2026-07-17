"""Continuous local AGI runner.

Runs the two-agent dialectic + Kai oracle in repeated epochs, accumulating
synthesis / emergence / phase-shift metrics to a JSONL log. This closes the
loop: the local AGI runs continuously and its progress toward an AGI-grade
synthesis becomes measurable over time.

Run:
    python3 agi_runner.py            # default: 4 epochs, Kai on
    python3 agi_runner.py --epochs 8 --no-kai
"""

from __future__ import annotations

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, "products/kai-insight/core")

from two_agent_dialogue import TwoAgentDialogue
from kai_local import AXIOM_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agi_runner")

LOG_PATH = AXIOM_STATE / "agi_runner.log"

TOPICS = [
    "How can free-energy minimization plus quantum superposition yield general intelligence?",
    "What internal metric best signals that an agent has truly understood a domain?",
    "How should a self-modifying mind bound the risk of its own edits?",
    "Can emergence of language between two agents be a sufficient signature of intelligence?",
    "How does curvature (Ricci) of the belief manifold relate to learning speed?",
    "What is the minimal loop (perceive-predict-act) that exhibits open-ended creativity?",
]


def run_epoch(dlg: TwoAgentDialogue, topic: str, turns: int) -> Dict[str, Any]:
    """Run a single dialogue epoch and return its summary."""
    summary = dlg.run(topic, turns=turns)
    kai_lines = [m for m in dlg.transcript if m["speaker"] == "KAI"]
    summary["topic"] = topic
    summary["kai_advices"] = len(kai_lines)
    summary["wall_clock"] = round(time.time(), 3)
    return summary


def main(epochs: int, turns: int, use_kai: bool, dilation: float):
    print(f"=== Continuous Local AGI Runner ===")
    print(f"epochs={epochs} turns={turns} use_kai={use_kai} dilation={dilation}\n")

    # One shared dialogue object so agents keep their belief graphs across epochs.
    dlg = TwoAgentDialogue(dilation=dilation, use_kai=use_kai)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        for i in range(epochs):
            topic = TOPICS[i % len(TOPICS)]
            print(f"--- Epoch {i+1}/{epochs}: {topic[:60]}... ---")
            summary = run_epoch(dlg, topic, turns)
            history.append(summary)

            line = json.dumps({
                "epoch": i + 1,
                "topic": topic,
                "synthesis_score": summary["synthesis_score"],
                "emergence_novelty": summary["emergence_novelty"],
                "phase_shifts": summary["phase_shifts"],
                "theorist_beliefs": summary["theorist_beliefs"],
                "engineer_beliefs": summary["engineer_beliefs"],
                "kai_advices": summary["kai_advices"],
                "wall_clock": summary["wall_clock"],
            })
            logf.write(line + "\n")

            print(f"  synthesis={summary['synthesis_score']} "
                  f"novelty={summary['emergence_novelty']} "
                  f"phase_shifts={summary['phase_shifts']} "
                  f"kai_advices={summary['kai_advices']}")

    # Trajectory of synthesis across epochs
    scores = [h["synthesis_score"] for h in history]
    trend = scores[-1] - scores[0] if len(scores) > 1 else 0.0
    print(f"\n=== Trajectory ===")
    print(f"synthesis: {scores}")
    print(f"delta over run: {round(trend, 4)} "
          f"({'rising' if trend > 0 else 'stable' if trend == 0 else 'falling'})")
    print(f"cumulative phase shifts: {sum(h['phase_shifts'] for h in history)}")
    print(f"log: {LOG_PATH}")

    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--dilation", type=float, default=0.5)
    ap.add_argument("--no-kai", action="store_true", help="disable the Kai oracle")
    args = ap.parse_args()

    main(args.epochs, args.turns, use_kai=not args.no_kai, dilation=args.dilation)
