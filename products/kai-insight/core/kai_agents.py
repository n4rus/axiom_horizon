"""Quantum agents integrated with live KaiMind inference.

Agents use real KaiMind select_policy for their responses instead of
simple string templates.
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

logger = logging.getLogger('kai_insight.agents')


class KaiMindAgent(Agent):
    """Agent that uses real KaiMind for inference."""

    def __init__(self, name: str, goal: str, role: AgentRole,
                 qm: QuantumSuperpositionManager, wfc: WavefunctionCollapse,
                 kai_instance=None):
        super().__init__(name, goal, role, qm, wfc)
        self._kai = kai_instance

    def set_kai(self, kai_instance):
        self._kai = kai_instance

    def process_message(self, message: str) -> str:
        if self._kai is None:
            return f"{self.name}({self.role.value}): {message}"

        try:
            policy, info = self._kai.select_policy([message, self.goal])
            efe = info.get('G', 0.0)
            return f"{self.name}({self.role.value}): [EFE={efe:.3f}] {policy}"
        except Exception as e:
            logger.warning(f"KaiMind inference failed for {self.name}: {e}")
            return f"{self.name}({self.role.value}): {message}"


def create_kai_agents(kai_instance=None) -> tuple:
    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)

    explorer = KaiMindAgent(
        "Explorer", "discover new patterns in data",
        AgentRole.DISCOVERY, qm, wfc, kai_instance
    )
    builder = KaiMindAgent(
        "Builder", "turn patterns into working solutions",
        AgentRole.APPLICATION, qm, wfc, kai_instance
    )
    meta = KaiMindAgent(
        "Meta", "optimize the discovery and application process",
        AgentRole.META, qm, wfc, kai_instance
    )

    explorer.connect_to(builder)
    explorer.connect_to(meta)
    builder.connect_to(meta)

    return explorer, builder, meta, qm, wfc


def run_kai_dialogue(kai_instance, topic: str = "knowledge exploration", rounds: int = 3):
    explorer, builder, meta, qm, wfc = create_kai_agents(kai_instance)

    results = []
    for i in range(rounds):
        round_result = {'round': i + 1, 'interactions': []}

        for sender, receiver in [(explorer, builder), (builder, meta), (meta, explorer)]:
            result = sender.interact_with(receiver, topic)
            if result:
                round_result['interactions'].append({
                    'sender': result['sender'],
                    'receiver': result['receiver'],
                    'type': result['type'],
                    'response': result['response'][:80],
                })

        results.append(round_result)

    summary = {
        'rounds': rounds,
        'total_interactions': sum(len(r['interactions']) for r in results),
        'explorer_memory': len(explorer.memory),
        'builder_memory': len(builder.memory),
        'meta_memory': len(meta.memory),
        'results': results,
    }
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
    from adapter import get_kai_instance

    print("=== Kai-Integrated Quantum Agents ===\n")

    kai = get_kai_instance()
    print(f"KaiMind loaded: mode={kai._mode}, vfe={kai.vfe:.4f}\n")

    summary = run_kai_dialogue(kai, "How can we achieve AGI locally?", rounds=2)

    for r in summary['results']:
        print(f"--- Round {r['round']} ---")
        for interaction in r['interactions']:
            print(f"  [{interaction['type']:12s}] {interaction['sender']:10s} -> {interaction['receiver']:10s}: {interaction['response']}")

    print(f"\n--- Summary ---")
    print(f"  Total interactions: {summary['total_interactions']}")
    print(f"  Explorer memory: {summary['explorer_memory']}")
    print(f"  Builder memory: {summary['builder_memory']}")
    print(f"  Meta memory: {summary['meta_memory']}")
