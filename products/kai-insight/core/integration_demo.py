"""Full System Integration Demo for Kai AGI.

Demonstrates all components working together:
- Quantum reasoning → Evolution → Knowledge transfer → Meta-cognition → Emergence tracking
"""

import sys
import os
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))


def run_integration_demo():
    print("=== Kai AGI Full System Integration ===\n")

    from quantum_reasoning import QuantumReasoner
    from evolution_engine import EvolutionEngine
    from knowledge_transfer import KnowledgeTransfer
    from meta_cognition import MetaCognitiveAgent
    from emergence_tracker import EmergenceTracker
    from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)

    agents = []
    for name, role in [("Alpha", AgentRole.DISCOVERY), ("Beta", AgentRole.APPLICATION), ("Gamma", AgentRole.META)]:
        agent = Agent(name, f"{name.lower()}_goal", role, qm, wfc)
        meta = MetaCognitiveAgent(f"Meta{name}", agent)
        agents.append((agent, meta))

    for i, (a1, _) in enumerate(agents):
        for j, (a2, _) in enumerate(agents):
            if i != j:
                a1.connect_to(a2)

    qr = QuantumReasoner(n_superpositions=3)
    engine = EvolutionEngine(population_size=3)
    kt = KnowledgeTransfer()
    tracker = EmergenceTracker(window_size=50)

    domains = ['physics', 'biology', 'computer_science', 'mathematics', 'philosophy']
    for d in domains:
        kt.add_knowledge(f"concept_{d}", d)

    print("1. Quantum Reasoning")
    context = {'efe': -0.5, 'ricci': 4.0, 'vfe': -0.1, 'previous_goals': []}
    for goal in ['explore_universe', 'build_system', 'understand_consciousness']:
        result = qr.reason(goal, context['efe'], context)
        context['previous_goals'].append(goal)
        hyp = result.get('hypothesis', str(result))[:30] if isinstance(result, dict) else str(result)[:30]
        score = result.get('score', 0) if isinstance(result, dict) else 0
        print(f"   {goal}: hypothesis={hyp}... score={score:.3f}")

    print("\n2. Multi-Agent Interaction with Meta-Cognition")
    for (agent, meta) in agents:
        result = meta.reason("What should we focus on?", context)
        print(f"   {meta.name}: confidence={result.get('confidence', 0):.3f}")

    print("\n3. Emergent Behavior Detection")
    for _ in range(30):
        (a1, _), (a2, _) = random.sample(agents, 2)
        action = random.choice(['explore', 'build', 'optimize', 'question'])
        result = {'success': random.random() > 0.2}
        tracker.record_interaction(a1.name, a2.name, action, result)

    stats = tracker.get_stats()
    print(f"   Behaviors tracked: {stats['total_behaviors']}")
    print(f"   Emergence events: {stats['emergence_events']}")
    print(f"   Novel behaviors: {stats['novel_behaviors']}")

    print("\n4. Knowledge Transfer")
    for d1 in random.sample(domains, 2):
        for d2 in random.sample(domains, 2):
            if d1 != d2:
                transfers = kt.transfer(d1, d2)
    kt_stats = kt.get_stats()
    print(f"   Knowledge nodes: {kt_stats['total_nodes']}")
    print(f"   Cross-domain links: {kt_stats['cross_domain_links']}")

    print("\n5. Evolution")
    tasks = [{'goal': f'task_{i}', 'difficulty': random.uniform(0.3, 0.8)} for i in range(3)]
    engine.run_competition(tasks)
    engine.evolve()
    evo_stats = engine.get_stats()
    print(f"   Generation: {evo_stats['generation']}")
    print(f"   Avg fitness: {evo_stats['avg_fitness']:.3f}")

    print("\n6. Quantum State Entanglement")
    if hasattr(qm, 'entangled_pairs'):
        print(f"   Entangled pairs: {len(qm.entangled_pairs)}")
        for pair, score in list(qm.entangled_pairs.items())[:3]:
            print(f"     {pair[0]} <-> {pair[1]}: {score:.3f}")
    else:
        print(f"   Entangled pairs: N/A (no entangled_pairs attribute)")

    print("\n7. Meta-Cognitive Summary")
    for _, meta in agents:
        ms = meta.get_meta_summary()
        print(f"   {ms['name']}: adjustments={ms['adjustments']}, quality_trend={ms['reasoning_trace']['quality_trend']:.3f}")

    print("\n=== Integration Complete ===")
    return True


if __name__ == "__main__":
    success = run_integration_demo()
    sys.exit(0 if success else 1)
