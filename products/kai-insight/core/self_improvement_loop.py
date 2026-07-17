"""Continuous Self-Improvement Feedback Loop for Kai AGI.

Advanced self-improvement system with:
- Performance monitoring and analysis
- Strategy adaptation based on outcomes
- Learning rate optimization
- Goal progression tracking
- Automated parameter tuning
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ImprovementType(Enum):
    """Types of improvements."""
    PARAMETER_TUNING = "parameter_tuning"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    GOAL_ADJUSTMENT = "goal_adjustment"
    LEARNING_OPTIMIZATION = "learning_optimization"
    ARCHITECTURE_CHANGE = "architecture_change"


@dataclass
class PerformanceMetric:
    """A performance metric measurement."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementAction:
    """An improvement action taken."""
    improvement_type: ImprovementType
    description: str
    parameters_changed: Dict[str, Any]
    expected_benefit: float
    timestamp: float = field(default_factory=time.time)
    success: Optional[bool] = None


@dataclass
class GoalProgress:
    """Progress toward a goal."""
    goal: str
    current_value: float
    target_value: float
    progress: float  # 0.0 to 1.0
    history: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SelfImprovementLoop:
    """Continuous self-improvement feedback loop."""

    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self.performance_history: deque = deque(maxlen=history_size)
        self.improvement_actions: List[ImprovementAction] = []
        self.goal_progress: Dict[str, GoalProgress] = {}
        self._parameter_history: Dict[str, deque] = {}
        self._learning_rate: float = 0.01
        self._momentum: float = 0.9
        self._velocity: Dict[str, float] = {}

    def record_metric(self, name: str, value: float, context: Optional[Dict] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            context=context or {},
        )
        self.performance_history.append(metric)

        # Update parameter history
        if name not in self._parameter_history:
            self._parameter_history[name] = deque(maxlen=100)
        self._parameter_history[name].append(value)

    def set_goal(self, goal: str, target_value: float, current_value: float = 0.0):
        """Set a goal to track."""
        progress = min(1.0, current_value / target_value) if target_value > 0 else 0.0
        self.goal_progress[goal] = GoalProgress(
            goal=goal,
            current_value=current_value,
            target_value=target_value,
            progress=progress,
            history=[current_value],
        )

    def update_goal(self, goal: str, current_value: float):
        """Update goal progress."""
        if goal not in self.goal_progress:
            self.set_goal(goal, current_value * 2)  # Default target

        gp = self.goal_progress[goal]
        gp.current_value = current_value
        gp.progress = min(1.0, current_value / gp.target_value) if gp.target_value > 0 else 0.0
        gp.history.append(current_value)
        gp.timestamp = time.time()

    def analyze_performance(self, window: int = 50) -> Dict[str, Any]:
        """Analyze recent performance."""
        if len(self.performance_history) < window:
            return {'status': 'insufficient_data'}

        recent = list(self.performance_history)[-window:]
        metrics = {}

        # Group by metric name
        for metric in recent:
            if metric.name not in metrics:
                metrics[metric.name] = []
            metrics[metric.name].append(metric.value)

        analysis = {}
        for name, values in metrics.items():
            if len(values) < 5:
                continue

            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.001

            # Calculate trend
            x_vals = list(range(len(values)))
            mean_x = sum(x_vals) / len(x_vals)
            numerator = sum((x - mean_x) * (y - mean) for x, y in zip(x_vals, values))
            denominator = sum((x - mean_x) ** 2 for x in x_vals)
            trend = numerator / denominator if denominator > 0 else 0.0

            analysis[name] = {
                'mean': mean,
                'std': std,
                'trend': trend,
                'min': min(values),
                'max': max(values),
                'recent': values[-5:],
                'improving': trend > 0 if 'error' in name.lower() or 'loss' in name.lower() else trend < 0,
            }

        return analysis

    def suggest_improvements(self) -> List[ImprovementAction]:
        """Suggest improvements based on performance analysis."""
        analysis = self.analyze_performance()
        suggestions = []

        for metric_name, stats in analysis.items():
            if isinstance(stats, dict) and 'trend' in stats:
                # Check if metric is declining
                if not stats.get('improving', True):
                    if 'error' in metric_name.lower() or 'loss' in metric_name.lower():
                        # Increase learning rate for error metrics
                        suggestions.append(ImprovementAction(
                            improvement_type=ImprovementType.LEARNING_OPTIMIZATION,
                            description=f"Increase learning rate for {metric_name}",
                            parameters_changed={'learning_rate': self._learning_rate * 1.5},
                            expected_benefit=0.1,
                        ))
                    elif 'accuracy' in metric_name.lower() or 'score' in metric_name.lower():
                        # Decrease learning rate for accuracy metrics
                        suggestions.append(ImprovementAction(
                            improvement_type=ImprovementType.PARAMETER_TUNING,
                            description=f"Fine-tune parameters for {metric_name}",
                            parameters_changed={'momentum': min(0.99, self._momentum + 0.05)},
                            expected_benefit=0.05,
                        ))

        # Check goal progress
        for goal, progress in self.goal_progress.items():
            if progress.progress < 0.3 and len(progress.history) > 10:
                suggestions.append(ImprovementAction(
                    improvement_type=ImprovementType.GOAL_ADJUSTMENT,
                    description=f"Adjust goal '{goal}' - progress too slow",
                    parameters_changed={'target_adjustment': 0.8},
                    expected_benefit=0.15,
                ))

        return suggestions

    def apply_improvement(self, action: ImprovementAction) -> bool:
        """Apply an improvement action."""
        try:
            # Apply parameter changes
            for param, value in action.parameters_changed.items():
                if param == 'learning_rate':
                    self._learning_rate = max(0.001, min(0.1, value))
                elif param == 'momentum':
                    self._momentum = max(0.5, min(0.99, value))
                elif param == 'target_adjustment':
                    for gp in self.goal_progress.values():
                        gp.target_value *= value

            action.success = True
            self.improvement_actions.append(action)

            # Keep only recent actions
            if len(self.improvement_actions) > 100:
                self.improvement_actions = self.improvement_actions[-100:]

            logger.info(f"Applied improvement: {action.description}")
            return True

        except Exception as e:
            action.success = False
            self.improvement_actions.append(action)
            logger.error(f"Failed to apply improvement: {e}")
            return False

    def get_optimal_parameters(self) -> Dict[str, float]:
        """Get optimal parameters based on history."""
        if not self._parameter_history:
            return {'learning_rate': self._learning_rate, 'momentum': self._momentum}

        # Find parameter values that correlate with best performance
        best_performance = -float('inf')
        best_params = {}

        for metric_name, values in self._parameter_history.items():
            if len(values) < 10:
                continue

            # Find peak performance
            peak_idx = values.index(max(values))
            if peak_idx > 0:
                # Use parameters from before peak
                best_params[metric_name] = values[peak_idx - 1]

        return {
            'learning_rate': self._learning_rate,
            'momentum': self._momentum,
            'parameter_peaks': best_params,
        }

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvement activities."""
        if not self.improvement_actions:
            return {'total_actions': 0}

        successful = [a for a in self.improvement_actions if a.success is True]
        failed = [a for a in self.improvement_actions if a.success is False]

        return {
            'total_actions': len(self.improvement_actions),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / max(1, len(self.improvement_actions)),
            'improvement_types': {
                it.value: len([a for a in self.improvement_actions if a.improvement_type == it])
                for it in ImprovementType
            },
            'current_parameters': {
                'learning_rate': self._learning_rate,
                'momentum': self._momentum,
            },
            'goal_progress': {
                goal: {
                    'progress': gp.progress,
                    'current': gp.current_value,
                    'target': gp.target_value,
                }
                for goal, gp in self.goal_progress.items()
            },
        }

    def run_improvement_cycle(self) -> Dict[str, Any]:
        """Run a complete improvement cycle."""
        # Analyze performance
        analysis = self.analyze_performance()

        # Suggest improvements
        suggestions = self.suggest_improvements()

        # Apply top suggestions
        applied = 0
        for suggestion in suggestions[:3]:  # Apply top 3
            if self.apply_improvement(suggestion):
                applied += 1

        return {
            'analysis': analysis,
            'suggestions': len(suggestions),
            'applied': applied,
            'summary': self.get_improvement_summary(),
        }


if __name__ == "__main__":
    print("=== Continuous Self-Improvement Test ===\n")

    loop = SelfImprovementLoop()

    # Set goals
    loop.set_goal("accuracy", 0.95, 0.7)
    loop.set_goal("latency", 50.0, 100.0)

    # Simulate performance metrics
    import random
    for i in range(100):
        # Record metrics with some improvement trend
        accuracy = 0.7 + 0.002 * i + random.gauss(0, 0.02)
        latency = 100 - 0.5 * i + random.gauss(0, 5)

        loop.record_metric("accuracy", accuracy)
        loop.record_metric("latency", latency)

        # Update goals
        loop.update_goal("accuracy", accuracy)
        loop.update_goal("latency", latency)

    # Analyze and improve
    print("Performance Analysis:")
    analysis = loop.analyze_performance()
    for metric, stats in analysis.items():
        if isinstance(stats, dict):
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Trend: {stats['trend']:.6f}")
            print(f"    Improving: {stats['improving']}")

    # Run improvement cycle
    print("\nRunning Improvement Cycle:")
    result = loop.run_improvement_cycle()
    print(f"  Suggestions: {result['suggestions']}")
    print(f"  Applied: {result['applied']}")

    # Get summary
    summary = loop.get_improvement_summary()
    print(f"\nImprovement Summary:")
    print(f"  Total actions: {summary['total_actions']}")
    print(f"  Success rate: {summary['success_rate']:.3f}")
    print(f"  Current parameters: {summary['current_parameters']}")

    print("\nGoal Progress:")
    for goal, progress in summary['goal_progress'].items():
        print(f"  {goal}: {progress['progress']:.3f} ({progress['current']:.3f}/{progress['target']:.3f})")
