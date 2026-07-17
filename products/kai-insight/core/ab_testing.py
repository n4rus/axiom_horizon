"""A/B Testing Framework for Kai AGI System.

Production-quality framework for comparing inference approaches,
feature flags, and performance analysis.
"""

from __future__ import annotations

import time
import random
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Variant(Enum):
    """A/B test variant."""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str = ""
    variants: List[Variant] = field(default_factory=lambda: [Variant.CONTROL, Variant.TREATMENT])
    traffic_split: float = 0.5  # Percentage of traffic for treatment
    enabled: bool = True
    start_time: float = 0.0
    end_time: float = 0.0  # 0 = no end
    min_samples: int = 100


@dataclass
class ExperimentResult:
    """Result of an A/B test experiment."""
    experiment: str
    variant: Variant
    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentStats:
    """Statistics for an A/B test experiment."""
    experiment: str
    variant: Variant
    metric_name: str
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    p_value: float = 1.0
    significant: bool = False


class ABTestFramework:
    """A/B testing framework for comparing inference approaches."""

    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: List[ExperimentResult] = []
        self._user_assignments: Dict[str, Dict[str, Variant]] = {}

    def create_experiment(
        self,
        name: str,
        description: str = "",
        traffic_split: float = 0.5,
        min_samples: int = 100,
    ) -> ExperimentConfig:
        """Create a new A/B test experiment."""
        config = ExperimentConfig(
            name=name,
            description=description,
            traffic_split=max(0.0, min(1.0, traffic_split)),
            min_samples=max(1, min_samples),
            start_time=time.time(),
        )
        self.experiments[name] = config
        logger.info(f"Created experiment: {name}")
        return config

    def assign_variant(self, user_id: str, experiment_name: str) -> Variant:
        """Assign a variant to a user for an experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_name}")

        config = self.experiments[experiment_name]

        # Check if already assigned
        if user_id in self._user_assignments:
            if experiment_name in self._user_assignments[user_id]:
                return self._user_assignments[user_id][experiment_name]

        # Deterministic assignment based on user_id hash
        hash_value = int(hashlib.md5(f"{user_id}:{experiment_name}".encode()).hexdigest(), 16)
        assignment_value = (hash_value % 100) / 100.0

        variant = Variant.TREATMENT if assignment_value < config.traffic_split else Variant.CONTROL

        # Store assignment
        if user_id not in self._user_assignments:
            self._user_assignments[user_id] = {}
        self._user_assignments[user_id][experiment_name] = variant

        return variant

    def record_result(
        self,
        experiment_name: str,
        variant: Variant,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a result for an experiment."""
        result = ExperimentResult(
            experiment=experiment_name,
            variant=variant,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {},
        )
        self.results.append(result)

    def get_stats(self, experiment_name: str, metric_name: str) -> Dict[Variant, ExperimentStats]:
        """Get statistics for an experiment."""
        experiment_results = [
            r for r in self.results
            if r.experiment == experiment_name and r.metric_name == metric_name
        ]

        stats = {}
        for variant in Variant:
            variant_results = [r.value for r in experiment_results if r.variant == variant]
            if variant_results:
                n = len(variant_results)
                mean = sum(variant_results) / n
                variance = sum((x - mean) ** 2 for x in variant_results) / max(1, n - 1)

                stats[variant] = ExperimentStats(
                    experiment=experiment_name,
                    variant=variant,
                    metric_name=metric_name,
                    count=n,
                    mean=mean,
                    variance=variance,
                    min_value=min(variant_results),
                    max_value=max(variant_results),
                )

        # Calculate significance (simplified t-test)
        if Variant.CONTROL in stats and Variant.TREATMENT in stats:
            control = stats[Variant.CONTROL]
            treatment = stats[Variant.TREATMENT]

            if control.count > 1 and treatment.count > 1:
                # Simple t-test approximation
                pooled_se = (
                    (control.variance / control.count) +
                    (treatment.variance / treatment.count)
                ) ** 0.5

                if pooled_se > 0:
                    t_stat = abs(treatment.mean - control.mean) / pooled_se
                    # Rough p-value approximation
                    p_value = max(0.001, 1.0 - min(1.0, t_stat / 3.0))

                    stats[Variant.CONTROL].p_value = p_value
                    stats[Variant.TREATMENT].p_value = p_value

                    significant = p_value < 0.05 and min(control.count, treatment.count) >= 10
                    stats[Variant.CONTROL].significant = significant
                    stats[Variant.TREATMENT].significant = significant

        return stats

    def is_experiment_active(self, experiment_name: str) -> bool:
        """Check if an experiment is currently active."""
        if experiment_name not in self.experiments:
            return False

        config = self.experiments[experiment_name]
        if not config.enabled:
            return False

        now = time.time()
        if config.start_time > 0 and now < config.start_time:
            return False
        if config.end_time > 0 and now > config.end_time:
            return False

        return True

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            name: {
                'enabled': config.enabled,
                'traffic_split': config.traffic_split,
                'total_results': sum(
                    1 for r in self.results if r.experiment == name
                ),
            }
            for name, config in self.experiments.items()
        }


class FeatureFlag:
    """Feature flag for gradual rollouts."""

    def __init__(self, name: str, default_enabled: bool = False):
        self.name = name
        self.enabled = default_enabled
        self.rollout_percentage: float = 100.0 if default_enabled else 0.0
        self._user_rollout: Dict[str, bool] = {}

    def is_enabled(self, user_id: str = "") -> bool:
        """Check if feature is enabled for a user."""
        if not self.enabled:
            return False

        if not user_id:
            return self.rollout_percentage >= 100.0

        # Deterministic rollout based on user_id
        hash_value = int(hashlib.md5(f"{user_id}:{self.name}".encode()).hexdigest(), 16)
        user_percentage = (hash_value % 100) / 100.0

        return user_percentage < (self.rollout_percentage / 100.0)

    def set_rollout(self, percentage: float) -> None:
        """Set rollout percentage."""
        self.rollout_percentage = max(0.0, min(100.0, percentage))
        self.enabled = self.rollout_percentage > 0

    def enable(self) -> None:
        """Enable the feature flag."""
        self.enabled = True
        self.rollout_percentage = 100.0

    def disable(self) -> None:
        """Disable the feature flag."""
        self.enabled = False
        self.rollout_percentage = 0.0


if __name__ == "__main__":
    print("=== A/B Testing Framework Test ===\n")

    framework = ABTestFramework()

    # Create experiment
    exp = framework.create_experiment(
        name="inference_comparison",
        description="Compare two inference approaches",
        traffic_split=0.5,
    )

    # Simulate results
    for i in range(200):
        user_id = f"user_{i}"
        variant = framework.assign_variant(user_id, "inference_comparison")

        # Simulate metric
        if variant == Variant.CONTROL:
            value = random.gauss(0.5, 0.1)
        else:
            value = random.gauss(0.55, 0.1)

        framework.record_result(
            "inference_comparison",
            variant,
            "response_time",
            value,
        )

    # Get stats
    stats = framework.get_stats("inference_comparison", "response_time")
    print("Experiment Statistics:")
    for variant, stat in stats.items():
        print(f"  {variant.value}: count={stat.count}, mean={stat.mean:.4f}, "
              f"variance={stat.variance:.4f}, significant={stat.significant}")

    # Test feature flags
    print("\nFeature Flags:")
    flag = FeatureFlag("new_algorithm", default_enabled=True)
    print(f"  new_algorithm enabled: {flag.enabled}")
    print(f"  User 'user_1' sees feature: {flag.is_enabled('user_1')}")
    flag.set_rollout(50.0)
    print(f"  After 50% rollout: {flag.is_enabled('user_2')}")
