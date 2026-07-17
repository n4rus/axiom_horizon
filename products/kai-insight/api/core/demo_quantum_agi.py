#!/usr/bin/env python3
"""
Quantum-Inspired AGI System - Comprehensive Demo
Tests all core features: superposition, entanglement, wavefunction collapse, multi-agent coordination
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from agent_framework import (
    Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse
)


def demo_superposition():
    print("=" * 60)
    print("DEMO 1: Quantum Superposition & Collapse")
    print("=" * 60)

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)
    a1 = Agent("Researcher", "discover new insights", AgentRole.DISCOVERY, qm, wfc)
    a2 = Agent("Engineer", "build solutions from insights", AgentRole.APPLICATION, qm, wfc)

    sup = qm.create_superposition("How do we solve X?", [a1, a2])
    print(f"  Created superposition with {len(sup['paths'])} possible paths")
    for p in sup['paths'][:4]:
        print(f"    [{p['response_type']}] {p['sender']}: {p['response_content'][:50]}...")

    collapse = wfc.collapse_superposition(sup['superposition_id'])
    print(f"\n  Collapsed to: [{collapse['selected_path']['response_type']}]")
    print(f"  Response: {collapse['selected_path']['response_content']}")
    print(f"  Entropy: {collapse['collapse_entropy']:.3f}")
    print()


def demo_entanglement():
    print("=" * 60)
    print("DEMO 2: Agent Entanglement")
    print("=" * 60)

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)
    agents = [
        Agent("Alpha", "explore the universe", AgentRole.DISCOVERY, qm, wfc),
        Agent("Beta", "explore data patterns", AgentRole.DISCOVERY, qm, wfc),
        Agent("Gamma", "build universe models", AgentRole.APPLICATION, qm, wfc),
        Agent("Delta", "optimize exploration", AgentRole.META, qm, wfc),
    ]
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            a.connect_to(b)

    print("\n  Entanglement Matrix:")
    for name, partners in qm.entanglement_matrix.items():
        scores = [f"{k}={v:.2f}" for k, v in partners.items()]
        print(f"    {name}: {', '.join(scores)}")
    print()


def demo_multi_agent_coordination():
    print("=" * 60)
    print("DEMO 3: Multi-Agent Coordination (3 agents, 5 rounds)")
    print("=" * 60)

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)
    agents = [
        Agent("Discoverer", "find patterns in complex systems", AgentRole.DISCOVERY, qm, wfc),
        Agent("Builder", "turn patterns into working code", AgentRole.APPLICATION, qm, wfc),
        Agent("Coordinator", "optimize team performance", AgentRole.META, qm, wfc),
    ]
    agents[0].connect_to(agents[1])
    agents[0].connect_to(agents[2])
    agents[1].connect_to(agents[2])

    interactions = [
        ("Discoverer", "Builder", "What patterns did you find in the dataset?"),
        ("Builder", "Coordinator", "How should we prioritize the implementation?"),
        ("Coordinator", "Discoverer", "What areas need more exploration?"),
        ("Discoverer", "Coordinator", "The system shows high variance in sector 7"),
        ("Builder", "Discoverer", "Can you validate this new hypothesis?"),
    ]

    results = []
    for sender_name, receiver_name, query in interactions:
        sender = next(a for a in agents if a.name == sender_name)
        receiver = next(a for a in agents if a.name == receiver_name)
        result = sender.interact_with(receiver, query)
        if result:
            results.append(result)
            print(f"  [{result['type']:12s}] {result['sender']:12s} -> {result['receiver']:12s}: {result['response'][:50]}...")

    print(f"\n  Total interactions: {len(results)}")
    print(f"  Average entropy: {np.mean([r['entropy'] for r in results]):.3f}")

    print("\n  Agent Memory Summary:")
    for a in agents:
        print(f"    {a.name}: {len(a.memory)} interactions, coherence={a.quantum_state['coherence']:.2f}")
    print()


def demo_quantum_tunneling():
    print("=" * 60)
    print("DEMO 4: Quantum Tunneling (rare events)")
    print("=" * 60)

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)
    a1 = Agent("Agent1", "standard goal", AgentRole.APPLICATION, qm, wfc)
    a2 = Agent("Agent2", "related goal", AgentRole.APPLICATION, qm, wfc)
    a1.connect_to(a2)

    tunnel_count = 0
    total = 50
    for i in range(total):
        result = a1.interact_with(a2, f"Test message {i}")
        if result and result['type'] == 'tunnel':
            tunnel_count += 1
            print(f"  [TUNNEL] Round {i}: {result['response'][:60]}...")

    print(f"\n  Tunnel events: {tunnel_count}/{total} ({100 * tunnel_count / total:.1f}%)")
    print()


def demo_adaptive_calibration():
    print("=" * 60)
    print("DEMO 5: Adaptive Quantum Environment Calibration")
    print("=" * 60)

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)
    agents = [
        Agent("A1", "goal A", AgentRole.DISCOVERY, qm, wfc),
        Agent("A2", "goal B", AgentRole.APPLICATION, qm, wfc),
        Agent("A3", "goal C", AgentRole.META, qm, wfc),
    ]

    calibrations = [0.1, 0.5, 1.0, 1.5, 2.0]
    for cal in calibrations:
        wfc.calibration = cal
        entropies = []
        for _ in range(10):
            sup = qm.create_superposition("calibration test", agents)
            collapse = wfc.collapse_superposition(sup['superposition_id'])
            entropies.append(collapse['collapse_entropy'])
        print(f"  Calibration={cal:.1f}: avg entropy={np.mean(entropies):.3f} (std={np.std(entropies):.3f})")
    print()


def main():
    print("\n" + "=" * 60)
    print("  QUANTUM-INSPIRED AGI SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 60 + "\n")

    demo_superposition()
    demo_entanglement()
    demo_multi_agent_coordination()
    demo_quantum_tunneling()
    demo_adaptive_calibration()

    print("=" * 60)
    print("  ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
