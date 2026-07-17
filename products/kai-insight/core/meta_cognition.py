"""Meta-Cognitive Agent Layer for Kai AGI.

Advanced meta-cognition with:
- Self-reflection and introspection
- Reasoning quality tracking
- Bias detection and correction
- Strategy adaptation based on meta-observations
- Assumption questioning
- Learning from mistakes
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of self-reflection."""
    ASSUMPTION_CHECK = "assumption_check"
    STRATEGY_EVALUATION = "strategy_evaluation"
    BIAS_CORRECTION = "bias_correction"
    LEARNING_EXTRACTION = "learning_extraction"
    GOAL_ALIGNMENT = "goal_alignment"


@dataclass
class Reflection:
    """A self-reflection event."""
    reflection_type: ReflectionType
    content: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[float] = None


@dataclass
class Learning:
    """A learning from experience."""
    insight: str
    source: str
    confidence: float
    applicable_contexts: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0


class ReasoningTrace:
    """Tracks reasoning quality and patterns over time."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.steps: List[Dict] = []
        self.quality_scores: deque = deque(maxlen=max_size)
        self.bias_detection: Dict[str, int] = {
            'confirmation': 0,
            'anchoring': 0,
            'availability': 0,
            'overconfidence': 0,
            'hindsight': 0,
        }
        self._reasoning_patterns: Dict[str, int] = {}

    def add_step(self, reasoning: str, confidence: float, outcome: Optional[float] = None):
        """Add a reasoning step."""
        step = {
            'reasoning': reasoning,
            'confidence': confidence,
            'outcome': outcome,
            'step_num': len(self.steps),
            'timestamp': time.time(),
        }
        self.steps.append(step)

        # Keep only recent steps
        if len(self.steps) > self.max_size:
            self.steps = self.steps[-self.max_size:]

        if outcome is not None:
            quality = 1.0 - abs(confidence - outcome)
            self.quality_scores.append(quality)
            self._detect_biases(confidence, outcome)
            self._extract_pattern(reasoning)

    def _detect_biases(self, confidence: float, outcome: float):
        """Detect cognitive biases."""
        if confidence > 0.8 and outcome < 0.5:
            self.bias_detection['overconfidence'] += 1
        if outcome > 0.7 and confidence < 0.3:
            self.bias_detection['anchoring'] += 1
        if confidence > 0.7 and outcome > 0.7:
            # Could be confirmation bias if always confident and right
            self.bias_detection['confirmation'] += 1
        if outcome > 0.8 and confidence < 0.5:
            self.bias_detection['hindsight'] += 1

    def _extract_pattern(self, reasoning: str):
        """Extract reasoning patterns."""
        # Simple pattern extraction based on key words
        words = reasoning.lower().split()
        for word in ['because', 'therefore', 'however', 'if', 'then']:
            if word in words:
                self._reasoning_patterns[word] = self._reasoning_patterns.get(word, 0) + 1

    def get_quality_trend(self, window: int = 10) -> float:
        """Get quality trend over recent window."""
        if len(self.quality_scores) < window:
            return 0.0

        scores = list(self.quality_scores)
        recent = scores[-window:]
        older = scores[-window*2:-window] if len(scores) >= window*2 else scores[:window]

        if not older:
            return 0.0

        return sum(recent)/len(recent) - sum(older)/len(older)

    def get_confidence_calibration(self) -> float:
        """Measure how well confidence predicts outcomes."""
        calibrated_steps = [
            s for s in self.steps
            if s['outcome'] is not None
        ]

        if len(calibrated_steps) < 5:
            return 0.5  # Neutral

        # Calculate correlation between confidence and outcome
        confidences = [s['confidence'] for s in calibrated_steps]
        outcomes = [s['outcome'] for s in calibrated_steps]

        mean_conf = sum(confidences) / len(confidences)
        mean_out = sum(outcomes) / len(outcomes)

        numerator = sum((c - mean_conf) * (o - mean_out) for c, o in zip(confidences, outcomes))
        denom_conf = sum((c - mean_conf) ** 2 for c in confidences)
        denom_out = sum((o - mean_out) ** 2 for o in outcomes)

        if denom_conf == 0 or denom_out == 0:
            return 0.5

        correlation = numerator / math.sqrt(denom_conf * denom_out)
        return (correlation + 1) / 2  # Normalize to [0, 1]

    def get_summary(self) -> Dict:
        """Get summary of reasoning trace."""
        return {
            'total_steps': len(self.steps),
            'avg_quality': sum(self.quality_scores)/len(self.quality_scores) if self.quality_scores else 0,
            'quality_trend': self.get_quality_trend(),
            'confidence_calibration': self.get_confidence_calibration(),
            'biases': dict(self.bias_detection),
            'dominant_bias': max(self.bias_detection, key=self.bias_detection.get) if any(self.bias_detection.values()) else None,
            'reasoning_patterns': dict(self._reasoning_patterns),
        }


class MetaCognitiveAgent:
    """Agent with advanced self-reflection and introspection."""

    def __init__(self, name: str, base_agent: Any):
        self.name = name
        self.base_agent = base_agent
        self.trace = ReasoningTrace()
        self.meta_strategies: List[str] = []
        self.self_queries: List[str] = []
        self.adjustments: List[Dict] = []
        self.reflections: List[Reflection] = []
        self.learnings: List[Learning] = []
        self._assumptions: Dict[str, float] = {}  # assumption -> confidence
        self._goals: List[str] = []

    def reason(self, query: str, context: Dict) -> Dict:
        """Reason with self-reflection."""
        # Pre-reasoning reflection
        self._reflect(ReflectionType.ASSUMPTION_CHECK, f"Checking assumptions for: {query}")

        # Base reasoning
        base_result = self._base_reason(query, context)

        # Track reasoning
        confidence = base_result.get('confidence', 0.5)
        self.trace.add_step(query, confidence)

        # Post-reasoning reflection
        meta_observation = self._meta_observe(base_result, context)

        if meta_observation['suggests_adjustment']:
            adjusted = self._adjust_reasoning(base_result, meta_observation)
            self.adjustments.append(adjusted)

            # Reflect on the adjustment
            self._reflect(
                ReflectionType.BIAS_CORRECTION,
                f"Adjusted reasoning due to: {meta_observation['reason']}"
            )

            return adjusted

        return base_result

    def _base_reason(self, query: str, context: Dict) -> Dict:
        """Delegate to base agent for reasoning."""
        if hasattr(self.base_agent, 'decide'):
            return self.base_agent.decide(query, context)
        return {
            'agent': self.name,
            'query': query,
            'response': f"Reasoning about: {query}",
            'confidence': 0.5,
        }

    def _reflect(self, reflection_type: ReflectionType, content: str, context: Optional[Dict] = None):
        """Perform self-reflection."""
        reflection = Reflection(
            reflection_type=reflection_type,
            content=content,
            confidence=0.8,
            context=context or {},
        )
        self.reflections.append(reflection)

        # Keep only recent reflections
        if len(self.reflections) > 100:
            self.reflections = self.reflections[-100:]

    def _meta_observe(self, result: Dict, context: Dict) -> Dict:
        """Observe and analyze own reasoning."""
        quality_trend = self.trace.get_quality_trend()
        biases = self.trace.bias_detection
        calibration = self.trace.get_confidence_calibration()

        suggests_adjustment = False
        reason = ""

        # Check for quality decline
        if quality_trend < -0.1:
            suggests_adjustment = True
            reason = "quality_declining"

        # Check for significant biases
        elif biases.get('overconfidence', 0) > 3:
            suggests_adjustment = True
            reason = "overconfidence_detected"
        elif biases.get('anchoring', 0) > 3:
            suggests_adjustment = True
            reason = "anchoring_bias"

        # Check for poor calibration
        elif calibration < 0.3:
            suggests_adjustment = True
            reason = "poor_calibration"

        return {
            'quality_trend': quality_trend,
            'biases': biases,
            'calibration': calibration,
            'suggests_adjustment': suggests_adjustment,
            'reason': reason,
        }

    def _adjust_reasoning(self, original: Dict, meta: Dict) -> Dict:
        """Adjust reasoning based on meta-observations."""
        adjusted = dict(original)
        adjusted['meta_adjusted'] = True
        adjusted['adjustment_reason'] = meta['reason']

        if meta['reason'] == 'overconfidence_detected':
            adjusted['confidence'] = original.get('confidence', 0.5) * 0.7
            self.meta_strategies.append('reduce_confidence')
            self._learn("Reduce confidence when overconfident", "bias_correction")

        elif meta['reason'] == 'quality_declining':
            adjusted['confidence'] = original.get('confidence', 0.5) * 0.8
            self.meta_strategies.append('be_more_cautious')
            self._learn("Be more cautious when quality is declining", "strategy_adjustment")

        elif meta['reason'] == 'poor_calibration':
            # Adjust confidence based on calibration
            calibration = meta.get('calibration', 0.5)
            adjusted['confidence'] = original.get('confidence', 0.5) * (0.5 + calibration)
            self.meta_strategies.append('recalibrate')
            self._learn("Recalibrate confidence based on historical accuracy", "calibration")

        return adjusted

    def _learn(self, insight: str, source: str, confidence: float = 0.7):
        """Record a learning."""
        learning = Learning(
            insight=insight,
            source=source,
            confidence=confidence,
        )
        self.learnings.append(learning)

        # Keep only recent learnings
        if len(self.learnings) > 50:
            self.learnings = self.learnings[-50:]

    def set_assumption(self, assumption: str, confidence: float = 0.5):
        """Set an assumption with confidence."""
        self._assumptions[assumption] = confidence

    def question_assumption(self, assumption: str) -> Dict:
        """Question an assumption and reflect on it."""
        confidence = self._assumptions.get(assumption, 0.5)

        self._reflect(
            ReflectionType.ASSUMPTION_CHECK,
            f"Questioning assumption: {assumption} (confidence: {confidence:.2f})",
            {'assumption': assumption, 'confidence': confidence}
        )

        # Reduce confidence in assumption
        self._assumptions[assumption] = confidence * 0.8

        return {
            'assumption': assumption,
            'previous_confidence': confidence,
            'new_confidence': self._assumptions[assumption],
            'reflection': f"Questioned assumption: {assumption}",
        }

    def set_goal(self, goal: str):
        """Set a goal for alignment checking."""
        if goal not in self._goals:
            self._goals.append(goal)

    def check_goal_alignment(self, action: str) -> Dict:
        """Check if an action aligns with goals."""
        if not self._goals:
            return {'aligned': True, 'reason': 'no_goals_set'}

        # Simple alignment check based on keyword matching
        action_words = set(action.lower().split())
        aligned_goals = []

        for goal in self._goals:
            goal_words = set(goal.lower().split())
            overlap = len(action_words & goal_words)
            if overlap > 0:
                aligned_goals.append(goal)

        aligned = len(aligned_goals) > 0

        self._reflect(
            ReflectionType.GOAL_ALIGNMENT,
            f"Action '{action[:50]}...' {'aligns' if aligned else 'does not align'} with goals",
            {'action': action, 'aligned': aligned, 'goals': aligned_goals}
        )

        return {
            'aligned': aligned,
            'matching_goals': aligned_goals,
            'total_goals': len(self._goals),
        }

    def get_meta_summary(self) -> Dict:
        """Get comprehensive meta-cognitive summary."""
        return {
            'name': self.name,
            'reasoning_trace': self.trace.get_summary(),
            'self_queries': len(self.self_queries),
            'adjustments': len(self.adjustments),
            'reflections': len(self.reflections),
            'learnings': len(self.learnings),
            'assumptions': len(self._assumptions),
            'goals': len(self._goals),
            'meta_strategies': self.meta_strategies[-5:],
            'recent_reflections': [
                {'type': r.reflection_type.value, 'content': r.content[:50]}
                for r in self.reflections[-3:]
            ],
            'recent_learnings': [
                {'insight': l.insight[:50], 'source': l.source}
                for l in self.learnings[-3:]
            ],
        }

    def introspect(self) -> Dict:
        """Perform deep introspection."""
        self._reflect(
            ReflectionType.STRATEGY_EVALUATION,
            "Performing deep introspection on reasoning patterns"
        )

        return {
            'reasoning_quality': self.trace.get_summary(),
            'reflection_count': len(self.reflections),
            'learning_count': len(self.learnings),
            'strategy_effectiveness': self._evaluate_strategies(),
            'improvement_areas': self._identify_improvement_areas(),
        }

    def _evaluate_strategies(self) -> Dict:
        """Evaluate effectiveness of meta-strategies."""
        strategy_counts = {}
        for strategy in self.meta_strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            'total_strategies': len(self.meta_strategies),
            'unique_strategies': len(strategy_counts),
            'most_used': max(strategy_counts, key=strategy_counts.get) if strategy_counts else None,
            'distribution': strategy_counts,
        }

    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas for improvement."""
        areas = []

        biases = self.trace.bias_detection
        if biases.get('overconfidence', 0) > 5:
            areas.append("Reduce overconfidence in predictions")
        if biases.get('anchoring', 0) > 5:
            areas.append("Avoid anchoring bias")
        if self.trace.get_quality_trend() < -0.2:
            areas.append("Improve reasoning quality trend")
        if self.trace.get_confidence_calibration() < 0.4:
            areas.append("Improve confidence calibration")

        return areas


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
    from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

    print("=== Advanced Meta-Cognitive Agent Test ===\n")

    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)
    base = Agent("Explorer", "discover patterns", AgentRole.DISCOVERY, qm, wfc)

    meta_agent = MetaCognitiveAgent("MetaExplorer", base)

    # Set some assumptions
    meta_agent.set_assumption("Data is representative", 0.8)
    meta_agent.set_assumption("Model will generalize", 0.6)

    # Set goals
    meta_agent.set_goal("discover novel patterns")
    meta_agent.set_goal("improve system performance")

    queries = [
        "What patterns exist in this data?",
        "How can we optimize the system?",
        "What is the best approach to AGI?",
        "Should we explore or converge?",
        "What are the risks of this strategy?",
    ]

    for q in queries:
        result = meta_agent.reason(q, {'efe': -0.4, 'ricci': 4.0})
        print(f"Query: {q}")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Meta-adjusted: {result.get('meta_adjusted', False)}")
        print()

    # Question assumptions
    print("Questioning assumptions:")
    for assumption in list(meta_agent._assumptions.keys())[:2]:
        result = meta_agent.question_assumption(assumption)
        print(f"  {result['assumption']}: {result['previous_confidence']:.2f} -> {result['new_confidence']:.2f}")

    # Check goal alignment
    print("\nGoal alignment check:")
    alignment = meta_agent.check_goal_alignment("discover patterns in quantum systems")
    print(f"  Aligned: {alignment['aligned']}")
    print(f"  Matching goals: {alignment['matching_goals']}")

    # Get comprehensive summary
    summary = meta_agent.get_meta_summary()
    print(f"\nMeta Summary:")
    print(f"  Reasoning steps: {summary['reasoning_trace']['total_steps']}")
    print(f"  Reflections: {summary['reflections']}")
    print(f"  Learnings: {summary['learnings']}")
    print(f"  Assumptions: {summary['assumptions']}")

    # Introspect
    introspection = meta_agent.introspect()
    print(f"\nIntrospection:")
    print(f"  Improvement areas: {introspection['improvement_areas']}")
