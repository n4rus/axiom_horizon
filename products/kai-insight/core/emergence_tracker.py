"""Emergent Behavior Tracking for Kai AGI.

Advanced emergent behavior detection with:
- Novelty scoring and tracking
- Pattern complexity analysis
- Emergence event detection
- Anomaly detection with statistical significance
- Causal relationship inference
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)


class EmergenceType(Enum):
    """Types of emergent behaviors."""
    NOVEL_PATTERN = "novel_pattern"
    PATTERN_CONCENTRATION = "pattern_concentration"
    UNEXPECTED_OUTCOME = "unexpected_outcome"
    FEEDBACK_LOOP = "feedback_loop"
    PHASE_SHIFT = "phase_shift"


@dataclass
class BehaviorSignature:
    """Signature of an emergent behavior."""
    pattern: str
    frequency: int = 1
    novelty: float = 1.0
    complexity: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    contexts: List[Dict] = field(default_factory=list)
    success_rate: float = 0.5
    causal_links: List[str] = field(default_factory=list)

    def update(self, context: Dict, success: bool = True):
        """Update behavior signature with new observation."""
        self.frequency += 1
        self.last_seen = time.time()
        self.novelty *= 0.95  # Decay novelty over time
        self.success_rate = 0.9 * self.success_rate + 0.1 * (1.0 if success else 0.0)

        self.contexts.append(context)
        if len(self.contexts) > 10:
            self.contexts.pop(0)

    def get_age(self) -> float:
        """Get age of behavior in seconds."""
        return time.time() - self.first_seen

    def get_recency(self) -> float:
        """Get recency score (1.0 = very recent, 0.0 = very old)."""
        age = time.time() - self.last_seen
        return math.exp(-age / 3600)  # Exponential decay over 1 hour


@dataclass
class EmergenceEvent:
    """An emergent behavior event."""
    event_type: EmergenceType
    pattern: str
    description: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergenceTracker:
    """Advanced emergent behavior tracking system."""

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.behaviors: Dict[str, BehaviorSignature] = {}
        self.interaction_log: deque = deque(maxlen=window_size)
        self.emergence_events: List[EmergenceEvent] = []
        self.anomaly_threshold: float = 2.0
        self._causal_graph: Dict[str, List[str]] = defaultdict(list)
        self._phase_history: List[Tuple[float, str]] = []

    def record_interaction(
        self,
        agent1: str,
        agent2: str,
        action: str,
        result: Dict,
        context: Optional[Dict] = None,
    ):
        """Record an interaction between agents."""
        interaction = {
            'agent1': agent1,
            'agent2': agent2,
            'action': action,
            'result': result,
            'context': context or {},
            'timestamp': time.time(),
        }
        self.interaction_log.append(interaction)

        # Create pattern signature
        pattern = f"{agent1}:{action}:{agent2}"
        success = result.get('success', True)

        if pattern in self.behaviors:
            self.behaviors[pattern].update(interaction, success)
        else:
            self.behaviors[pattern] = BehaviorSignature(
                pattern=pattern,
                frequency=1,
                novelty=1.0,
                success_rate=0.5 if success else 0.0,
            )
            self._detect_emergence(pattern, interaction)

        # Update causal graph
        self._update_causal_graph(pattern, interaction)

        # Detect phase shifts
        self._detect_phase_shifts()

    def _detect_emergence(self, pattern: str, interaction: Dict):
        """Detect emergent behaviors."""
        if len(self.interaction_log) < 10:
            return

        recent_interactions = list(self.interaction_log)[-20:]
        recent_actions = [i['action'] for i in recent_interactions]

        # Check for pattern concentration
        action_freq = recent_actions.count(interaction['action'])
        if action_freq > len(recent_actions) * 0.3:
            event = EmergenceEvent(
                event_type=EmergenceType.PATTERN_CONCENTRATION,
                pattern=pattern,
                description=f"Action '{interaction['action']}' appeared {action_freq}/{len(recent_actions)} times",
                confidence=min(1.0, action_freq / len(recent_actions)),
                metadata={'frequency': action_freq, 'total': len(recent_actions)},
            )
            self.emergence_events.append(event)

        # Check for novel patterns (high complexity, low frequency)
        if self.behaviors[pattern].frequency == 1:
            complexity = self._calculate_pattern_complexity(pattern)
            if complexity > 0.7:
                event = EmergenceEvent(
                    event_type=EmergenceType.NOVEL_PATTERN,
                    pattern=pattern,
                    description=f"Novel complex pattern detected: {pattern}",
                    confidence=complexity,
                    metadata={'complexity': complexity},
                )
                self.emergence_events.append(event)

    def _calculate_pattern_complexity(self, pattern: str) -> float:
        """Calculate complexity of a pattern."""
        parts = pattern.split(':')
        if len(parts) < 2:
            return 0.0

        # Complexity based on:
        # - Number of components
        # - Uniqueness of components
        # - Length of pattern
        component_count = len(parts)
        unique_components = len(set(parts))
        pattern_length = len(pattern)

        complexity = (
            min(1.0, component_count / 3) * 0.4 +
            min(1.0, unique_components / component_count) * 0.3 +
            min(1.0, pattern_length / 30) * 0.3
        )

        return complexity

    def _update_causal_graph(self, pattern: str, interaction: Dict):
        """Update causal relationships between patterns."""
        if len(self.interaction_log) < 2:
            return

        prev_interaction = self.interaction_log[-2]
        prev_pattern = f"{prev_interaction['agent1']}:{prev_interaction['action']}:{prev_interaction['agent2']}"

        if prev_pattern != pattern:
            self._causal_graph[prev_pattern].append(pattern)
            # Keep only recent causal links
            if len(self._causal_graph[prev_pattern]) > 10:
                self._causal_graph[prev_pattern] = self._causal_graph[prev_pattern][-10:]

    def _detect_phase_shifts(self):
        """Detect phase shifts in system behavior."""
        if len(self.interaction_log) < 50:
            return

        # Calculate recent action distribution
        recent_actions = [i['action'] for i in list(self.interaction_log)[-25:]]
        older_actions = [i['action'] for i in list(self.interaction_log)[-50:-25]]

        recent_dist = defaultdict(int)
        older_dist = defaultdict(int)

        for a in recent_actions:
            recent_dist[a] += 1
        for a in older_actions:
            older_dist[a] += 1

        # Calculate KL divergence
        kl_div = 0.0
        all_actions = set(recent_dist.keys()) | set(older_dist.keys())

        for action in all_actions:
            p = recent_dist[action] / max(1, len(recent_actions))
            q = older_dist[action] / max(1, len(older_actions))

            if p > 0 and q > 0:
                kl_div += p * math.log(p / q)

        # If KL divergence is high, we have a phase shift
        if kl_div > 0.5:
            event = EmergenceEvent(
                event_type=EmergenceType.PHASE_SHIFT,
                pattern="system_behavior",
                description=f"Phase shift detected: KL divergence = {kl_div:.3f}",
                confidence=min(1.0, kl_div),
                metadata={'kl_divergence': kl_div},
            )
            self.emergence_events.append(event)

    def detect_anomalies(self) -> List[Dict]:
        """Detect statistical anomalies in behavior."""
        anomalies = []
        if len(self.interaction_log) < 20:
            return anomalies

        actions = [i['action'] for i in self.interaction_log]
        action_counts = defaultdict(int)
        for a in actions:
            action_counts[a] += 1

        mean_freq = sum(action_counts.values()) / max(1, len(action_counts))
        variance = sum((f - mean_freq)**2 for f in action_counts.values()) / max(1, len(action_counts))
        std = math.sqrt(variance) if variance > 0 else 1.0

        for action, count in action_counts.items():
            z_score = (count - mean_freq) / std if std > 0 else 0
            if abs(z_score) > self.anomaly_threshold:
                anomalies.append({
                    'action': action,
                    'count': count,
                    'z_score': z_score,
                    'type': 'overrepresented' if z_score > 0 else 'underrepresented',
                    'significance': min(1.0, abs(z_score) / 3),
                })

        return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)

    def get_novel_behaviors(self, min_novelty: float = 0.5, limit: int = 10) -> List[Dict]:
        """Get novel behaviors sorted by novelty score."""
        novel = []
        for pattern, sig in self.behaviors.items():
            if sig.novelty >= min_novelty:
                novelty_score = (
                    sig.novelty * 0.4 +
                    sig.get_recency() * 0.3 +
                    (1.0 - sig.success_rate) * 0.3  # Novel things might fail more
                )
                novel.append({
                    'pattern': pattern,
                    'novelty': sig.novelty,
                    'complexity': sig.complexity,
                    'frequency': sig.frequency,
                    'age': sig.get_age(),
                    'recency': sig.get_recency(),
                    'success_rate': sig.success_rate,
                    'novelty_score': novelty_score,
                })

        return sorted(novel, key=lambda x: x['novelty_score'], reverse=True)[:limit]

    def get_causal_relationships(self, pattern: str, depth: int = 2) -> Dict[str, Any]:
        """Get causal relationships for a pattern."""
        if pattern not in self._causal_graph:
            return {'pattern': pattern, 'causes': [], 'effects': []}

        effects = self._causal_graph[pattern]

        # Get deeper effects
        all_effects = list(effects)
        if depth > 1:
            for effect in effects:
                deeper_effects = self._causal_graph.get(effect, [])
                all_effects.extend(deeper_effects)

        return {
            'pattern': pattern,
            'causes': list(set(effects)),
            'effects': list(set(all_effects)),
            'causal_strength': len(effects) / max(1, len(self._causal_graph)),
        }

    def get_emergence_events(self, event_type: Optional[EmergenceType] = None, limit: int = 10) -> List[EmergenceEvent]:
        """Get emergence events, optionally filtered by type."""
        events = self.emergence_events
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return sorted(events, key=lambda e: e.confidence, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive emergence tracking statistics."""
        if not self.behaviors:
            return {'total_behaviors': 0, 'total_interactions': 0}

        # Calculate metrics
        total_novelty = sum(b.novelty for b in self.behaviors.values())
        avg_novelty = total_novelty / len(self.behaviors)

        total_complexity = sum(
            self._calculate_pattern_complexity(b.pattern)
            for b in self.behaviors.values()
        )
        avg_complexity = total_complexity / len(self.behaviors)

        return {
            'total_behaviors': len(self.behaviors),
            'total_interactions': len(self.interaction_log),
            'emergence_events': len(self.emergence_events),
            'novel_behaviors': len(self.get_novel_behaviors()),
            'anomalies': len(self.detect_anomalies()),
            'avg_novelty': avg_novelty,
            'avg_complexity': avg_complexity,
            'causal_relationships': len(self._causal_graph),
            'event_types': {
                et.value: len([e for e in self.emergence_events if e.event_type == et])
                for et in EmergenceType
            },
        }


if __name__ == "__main__":
    print("=== Advanced Emergent Behavior Tracking ===\n")

    tracker = EmergenceTracker(window_size=100)

    agents = ['Explorer', 'Builder', 'Critic', 'Synthesizer']
    actions = ['explore', 'build', 'analyze', 'optimize', 'synthesize', 'question']

    import random

    # Simulate interactions with some patterns
    for i in range(150):
        a1, a2 = random.sample(agents, 2)

        # Create some concentrated patterns
        if i < 50:
            action = 'explore'  # High frequency early
        elif i < 100:
            action = random.choice(['build', 'analyze'])
        else:
            action = random.choice(actions)

        result = {'success': random.random() > 0.2}
        tracker.record_interaction(a1, a2, action, result)

    print("Statistics:")
    stats = tracker.get_stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")

    print("\nNovel Behaviors:")
    for nb in tracker.get_novel_behaviors(min_novelty=0.3, limit=5):
        print(f"  {nb['pattern']}:")
        print(f"    Novelty: {nb['novelty']:.3f}, Complexity: {nb['complexity']:.3f}")
        print(f"    Score: {nb['novelty_score']:.3f}, Success Rate: {nb['success_rate']:.3f}")

    print("\nAnomalies:")
    for a in tracker.detect_anomalies()[:3]:
        print(f"  {a['action']}: {a['type']} (z={a['z_score']:.2f}, sig={a['significance']:.3f})")

    print("\nEmergence Events:")
    for e in tracker.get_emergence_events(limit=5):
        print(f"  {e.event_type.value}: {e.description}")
        print(f"    Confidence: {e.confidence:.3f}")
