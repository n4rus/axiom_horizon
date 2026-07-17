#!/usr/bin/env python3
"""Expand Kai's knowledge base breadth with authoritative live sources.

Targets the attractor's identified primary AGI bottleneck (knowledge breadth).
Dependency-free; uses KaiMind.kb_ingest_url. Safe to re-run (idempotent).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import kai_mind as km

# Curated, authoritative sources spanning AGI theory, ML, and physics.
SOURCES = [
    # AGI / cognitive architectures
    "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
    "https://en.wikipedia.org/wiki/Active_inference",
    "https://en.wikipedia.org/wiki/Predictive_coding",
    "https://en.wikipedia.org/wiki/Free_energy_principle",
    "https://en.wikipedia.org/wiki/Attractor_network",
    "https://en.wikipedia.org/wiki/Emergence",
    "https://en.wikipedia.org/wiki/Self-organization",
    "https://en.wikipedia.org/wiki/Integrated_information_theory",
    "https://en.wikipedia.org/wiki/Causal_inference",
    "https://en.wikipedia.org/wiki/World_model",
    # ML foundations
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
    "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Backpropagation",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Variational_Bayes",
    "https://en.wikipedia.org/wiki/Gradient_descent",
    # Information theory / complexity
    "https://en.wikipedia.org/wiki/Kolmogorov_complexity",
    "https://en.wikipedia.org/wiki/Algorithmic_information_theory",
    "https://en.wikipedia.org/wiki/Entropy_(information_theory)",
    # Physics grounding (complements the Halliday book)
    "https://en.wikipedia.org/wiki/Statistical_mechanics",
    "https://en.wikipedia.org/wiki/Thermodynamics",
    "https://en.wikipedia.org/wiki/General_relativity",
    "https://en.wikipedia.org/wiki/Quantum_field_theory",
    "https://en.wikipedia.org/wiki/Second_law_of_thermodynamics",
]

def main():
    mind = km.KaiMind()
    print(f"[feed] KB before: {mind.kb_summary()}")
    ok = 0
    for url in SOURCES:
        try:
            res = mind.kb_ingest_url(url, max_chunks=6)
            status = "OK" if res.startswith("[OK]") else "SKIP"
            if status == "OK":
                ok += 1
            print(f"  [{status}] {url.split('/wiki/')[-1]:<34} {res}")
        except Exception as e:
            print(f"  [ERR] {url}: {e}")
    print(f"[feed] ingested {ok}/{len(SOURCES)} sources")
    print(f"[feed] KB after:  {mind.kb_summary()}")

if __name__ == "__main__":
    main()
