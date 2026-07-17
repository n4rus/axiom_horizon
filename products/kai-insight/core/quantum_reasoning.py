"""Quantum Reasoning Integration for Kai AGI.

Advanced quantum-inspired reasoning with:
- Hypothesis superposition and collapse
- Entanglement learning from successful patterns
- Quantum interference for hypothesis evolution
- Decoherence tracking and management
- Cross-domain quantum transfer
"""

from __future__ import annotations

import math
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Represents a quantum state of a hypothesis."""
    id: str
    goal: str
    phase: float
    coherence: float
    energy: float
    probability: float
    depth: int
    exploration_bias: float
    entanglements: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    collapse_count: int = 0
    success_rate: float = 0.0


@dataclass
class Entanglement:
    """Represents quantum entanglement between two states."""
    state1_id: str
    state2_id: str
    strength: float
    created_at: float = field(default_factory=time.time)
    interactions: int = 0
    success_correlation: float = 0.0


class QuantumMemory:
    """Quantum memory with superposition retention."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.states: Dict[str, QuantumState] = {}
        self.entanglements: List[Entanglement] = []
        self._success_history: List[Tuple[str, bool]] = []

    def add_state(self, state: QuantumState) -> bool:
        """Add a quantum state to memory."""
        if len(self.states) >= self.max_size:
            self._decay_oldest()

        self.states[state.id] = state
        return True

    def get_state(self, state_id: str) -> Optional[QuantumState]:
        """Get a quantum state from memory."""
        return self.states.get(state_id)

    def record_success(self, state_id: str, success: bool) -> None:
        """Record success/failure for a state."""
        if state_id in self.states:
            state = self.states[state_id]
            # Update success rate with exponential moving average
            state.success_rate = 0.9 * state.success_rate + 0.1 * (1.0 if success else 0.0)
            state.collapse_count += 1
            self._success_history.append((state_id, success))

            # Update entanglement correlations
            for ent in self.entanglements:
                if ent.state1_id == state_id or ent.state2_id == state_id:
                    ent.interactions += 1
                    ent.success_correlation = 0.9 * ent.success_correlation + 0.1 * (1.0 if success else 0.0)

    def _decay_oldest(self) -> None:
        """Decay oldest states to make room for new ones."""
        if not self.states:
            return

        # Find states with lowest success rate and oldest creation time
        sorted_states = sorted(
            self.states.values(),
            key=lambda s: (s.success_rate, s.created_at)
        )

        # Remove bottom 10%
        n_remove = max(1, len(sorted_states) // 10)
        for state in sorted_states[:n_remove]:
            del self.states[state.id]

    def get_successful_patterns(self, min_success_rate: float = 0.7) -> List[QuantumState]:
        """Get states with high success rates."""
        return [
            state for state in self.states.values()
            if state.success_rate >= min_success_rate and state.collapse_count >= 3
        ]


class QuantumReasoner:
    """Advanced quantum-inspired reasoning layer for KaiMind."""

    def __init__(self, n_superpositions: int = 4, memory_size: int = 1000):
        self.n_superpositions = n_superpositions
        self.memory = QuantumMemory(max_size=memory_size)
        self.hypothesis_memory: List[Dict] = []
        self._interference_pattern: Dict[str, float] = defaultdict(float)
        self._decoherence_rate: float = 0.01

    def create_hypothesis_superposition(self, goal: str, context: Dict) -> List[QuantumState]:
        """Create multiple hypothesis states for a goal with quantum enhancement."""
        hypotheses = []
        base_energy = context.get('efe', 0.5)
        ricci = context.get('ricci', 4.0)
        vfe = context.get('vfe', 0.0)
        previous_goals = context.get('previous_goals', [])

        # Learn from successful past patterns
        successful_patterns = self.memory.get_successful_patterns()

        for i in range(self.n_superpositions):
            phase = random.random() * 2 * math.pi
            coherence = 0.5 + 0.5 * math.cos(phase)

            # Quantum interference from past successes
            interference = self._compute_interference(goal, previous_goals)
            energy = base_energy * (1.0 + 0.2 * math.sin(phase)) - interference

            # Learn from successful patterns
            pattern_bonus = self._compute_pattern_bonus(goal, successful_patterns)

            hypothesis = QuantumState(
                id=f"h_{len(self.hypothesis_memory)}_{i}",
                goal=goal,
                phase=phase,
                coherence=coherence,
                energy=energy,
                probability=self._boltzmann_probability(energy, ricci),
                depth=i + 1,
                exploration_bias=0.3 + 0.7 * random.random(),
            )

            # Boost probability based on pattern similarity
            hypothesis.probability *= (1.0 + pattern_bonus)
            hypotheses.append(hypothesis)

        # Normalize probabilities
        total_prob = sum(h.probability for h in hypotheses)
        for h in hypotheses:
            h.probability /= total_prob if total_prob > 0 else 1.0

        # Add to memory
        for h in hypotheses:
            self.memory.add_state(h)

        return hypotheses

    def collapse_hypotheses(self, hypotheses: List[QuantumState]) -> QuantumState:
        """Collapse hypothesis superposition to select winning hypothesis."""
        rand = random.random()
        cumulative = 0.0

        for h in hypotheses:
            cumulative += h.probability
            if rand <= cumulative:
                # Create entanglement with recently collapsed states
                self._create_entanglement(h)
                return h

        # Fallback to last hypothesis
        selected = hypotheses[-1]
        self._create_entanglement(selected)
        return selected

    def _compute_interference(self, goal: str, previous_goals: List[str]) -> float:
        """Compute quantum interference from previous goals."""
        if not previous_goals:
            return 0.0

        interference = 0.0
        for prev_goal in previous_goals[-5:]:  # Last 5 goals
            similarity = self._goal_similarity(goal, prev_goal)
            phase_diff = self._interference_pattern[prev_goal]
            interference += similarity * math.cos(phase_diff) * 0.1

        return interference

    def _compute_pattern_bonus(self, goal: str, patterns: List[QuantumState]) -> float:
        """Compute bonus from successful patterns."""
        if not patterns:
            return 0.0

        bonus = 0.0
        for pattern in patterns:
            similarity = self._goal_similarity(goal, pattern.goal)
            bonus += similarity * pattern.success_rate * 0.2

        return min(0.5, bonus)  # Cap the bonus

    def _goal_similarity(self, goal1: str, goal2: str) -> float:
        """Compute similarity between two goals."""
        if goal1 == goal2:
            return 1.0

        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _create_entanglement(self, state: QuantumState) -> None:
        """Create entanglement between newly collapsed state and recent states."""
        recent_states = [
            s for s in self.memory.states.values()
            if s.id != state.id and time.time() - s.created_at < 300  # Last 5 minutes
        ]

        for recent in recent_states[:3]:  # Max 3 entanglements
            similarity = self._goal_similarity(state.goal, recent.goal)
            if similarity > 0.3:  # Only entangle if some similarity
                entanglement = Entanglement(
                    state1_id=state.id,
                    state2_id=recent.id,
                    strength=similarity,
                )
                self.memory.entanglements.append(entanglement)

                # Update state entanglements
                state.entanglements[recent.id] = similarity
                recent.entanglements[state.id] = similarity

    def _boltzmann_probability(self, energy: float, temperature: float) -> float:
        """Boltzmann probability distribution."""
        temp = max(0.1, temperature)
        return math.exp(-abs(energy) / temp)

    def compute_entanglement(self, goal1: str, goal2: str) -> float:
        """Compute entanglement strength between two goals."""
        if goal1 == goal2:
            return 1.0

        # Check if we have recorded entanglement
        for ent in self.memory.entanglements:
            s1 = self.memory.get_state(ent.state1_id)
            s2 = self.memory.get_state(ent.state2_id)
            if s1 and s2:
                if (self._goal_similarity(goal1, s1.goal) > 0.5 and
                    self._goal_similarity(goal2, s2.goal) > 0.5):
                    return ent.strength * ent.success_correlation
                elif (self._goal_similarity(goal1, s2.goal) > 0.5 and
                      self._goal_similarity(goal2, s1.goal) > 0.5):
                    return ent.strength * ent.success_correlation

        # Compute from scratch
        similarity = self._goal_similarity(goal1, goal2)
        return similarity * 0.8 + random.random() * 0.2

    def quantum_enhanced_score(self, base_efe: float, hypothesis: QuantumState,
                                context: Dict) -> float:
        """Enhance EFE score with quantum reasoning factors."""
        coherence_bonus = hypothesis.coherence * 0.1
        exploration_bonus = hypothesis.exploration_bias * 0.05
        depth_factor = 1.0 + hypothesis.depth * 0.02

        # Entanglement bonus from memory
        entanglement_bonus = 0.0
        prev_goals = context.get('previous_goals', [])
        if prev_goals:
            entanglement_bonus = sum(
                self.compute_entanglement(hypothesis.goal, g)
                for g in prev_goals
            ) / len(prev_goals) * 0.1

        # Success rate bonus
        success_bonus = hypothesis.success_rate * 0.15

        enhanced = (base_efe * depth_factor - coherence_bonus -
                   exploration_bonus - entanglement_bonus - success_bonus)
        return enhanced

    def reason(self, goal: str, base_efe: float, context: Dict) -> Tuple[Dict, float]:
        """Full quantum reasoning cycle."""
        hypotheses = self.create_hypothesis_superposition(goal, context)
        winning = self.collapse_hypotheses(hypotheses)
        enhanced_score = self.quantum_enhanced_score(base_efe, winning, context)

        # Update interference pattern
        self._interference_pattern[goal] = winning.phase

        # Apply decoherence
        self._apply_decoherence()

        return winning, enhanced_score

    def _apply_decoherence(self) -> None:
        """Apply decoherence to all states."""
        for state in list(self.memory.states.values()):
            state.coherence *= (1.0 - self._decoherence_rate)
            if state.coherence < 0.1:
                # State has decohered, remove from active memory
                if state.id in self.memory.states:
                    del self.memory.states[state.id]

    def get_memory_stats(self) -> Dict:
        """Get quantum memory statistics."""
        states = list(self.memory.states.values())
        return {
            'total_states': len(states),
            'entanglements': len(self.memory.entanglements),
            'avg_coherence': sum(s.coherence for s in states) / max(1, len(states)),
            'avg_success_rate': sum(s.success_rate for s in states) / max(1, len(states)),
            'successful_patterns': len(self.memory.get_successful_patterns()),
        }


class QuantumInspiredAgent:
    """Agent using quantum reasoning for decision-making."""

    def __init__(self, name: str, role: str, reasoner: Optional[QuantumReasoner] = None):
        self.name = name
        self.role = role
        self.reasoner = reasoner or QuantumReasoner(n_superpositions=4)
        self.memory: List[Dict] = []
        self.performance_score: float = 0.5
        self._decision_history: List[Dict] = []

    def decide(self, goal: str, context: Dict) -> Dict:
        """Make a quantum-enhanced decision."""
        base_efe = context.get('efe', 0.5)
        hypothesis, score = self.reasoner.reason(goal, base_efe, context)

        decision = {
            'agent': self.name,
            'role': self.role,
            'goal': goal,
            'hypothesis': hypothesis.__dict__ if hasattr(hypothesis, '__dict__') else hypothesis,
            'score': score,
            'timestamp': context.get('timestamp', time.time()),
        }

        self.memory.append(decision)
        self._decision_history.append({
            'goal': goal,
            'score': score,
            'success': None,  # To be updated later
        })

        return decision

    def update_performance(self, reward: float, decision_index: Optional[int] = None):
        """Update performance based on outcome."""
        self.performance_score = 0.9 * self.performance_score + 0.1 * reward

        # Update decision history
        if decision_index is not None and 0 <= decision_index < len(self._decision_history):
            self._decision_history[decision_index]['success'] = reward > 0.5

            # Update quantum memory with success/failure
            if 'hypothesis' in self.memory[decision_index]:
                h_id = self.memory[decision_index]['hypothesis'].get('id')
                if h_id:
                    self.reasoner.memory.record_success(h_id, reward > 0.5)

    def get_insights(self) -> Dict:
        """Get insights from decision history."""
        if not self._decision_history:
            return {'insights': []}

        successful = [d for d in self._decision_history if d['success'] is True]
        failed = [d for d in self._decision_history if d['success'] is False]

        return {
            'total_decisions': len(self._decision_history),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / max(1, len(self._decision_history)),
            'avg_score_successful': sum(d['score'] for d in successful) / max(1, len(successful)),
            'avg_score_failed': sum(d['score'] for d in failed) / max(1, len(failed)),
            'quantum_memory': self.reasoner.get_memory_stats(),
        }


if __name__ == "__main__":
    print("=== Quantum Reasoning Integration ===\n")

    reasoner = QuantumReasoner(n_superpositions=4)
    agent = QuantumInspiredAgent("QuantumAgent", "discovery", reasoner)

    goals = [
        "discover patterns in complex systems",
        "build working solutions from insights",
        "optimize system performance",
        "analyze emergent behaviors",
        "synthesize knowledge across domains",
    ]

    context = {'efe': -0.4, 'ricci': 4.0, 'vfe': -0.1, 'previous_goals': []}

    for i, goal in enumerate(goals):
        decision = agent.decide(goal, context)
        h = decision['hypothesis']
        print(f"Goal: {goal}")
        print(f"  Hypothesis: depth={h['depth']}, coherence={h['coherence']:.3f}, "
              f"energy={h['energy']:.3f}, prob={h['probability']:.3f}")
        print(f"  Enhanced score: {decision['score']:.4f}")

        # Simulate success/failure
        success = random.random() > 0.3
        agent.update_performance(1.0 if success else 0.0, i)
        print(f"  Outcome: {'SUCCESS' if success else 'FAILED'}")

        context['previous_goals'].append(goal)
        print()

    # Get insights
    insights = agent.get_insights()
    print("Agent Insights:")
    print(f"  Success rate: {insights['success_rate']:.3f}")
    print(f"  Quantum memory: {insights['quantum_memory']}")
