"""Runtime Constant Grounding for Kai AGI System.

Dynamically tunes remaining heuristic constants based on
actual runtime performance data, replacing guessed values
with measured outcomes.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ConstantConfig:
    """Configuration for a tunable constant."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    description: str = ""
    history: List[float] = field(default_factory=list)


class RuntimeConstantTuner:
    """Tunes constants based on runtime performance."""

    def __init__(self):
        self.constants: Dict[str, ConstantConfig] = {}
        self._performance_history: List[Dict[str, float]] = []
        self._update_interval: float = 60.0
        self._last_update: float = 0.0

    def register_constant(
        self,
        name: str,
        default_value: float,
        min_value: float,
        max_value: float,
        description: str = "",
    ) -> None:
        """Register a tunable constant."""
        self.constants[name] = ConstantConfig(
            name=name,
            current_value=default_value,
            min_value=min_value,
            max_value=max_value,
            default_value=default_value,
            description=description,
        )

    def record_performance(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics for tuning."""
        self._performance_history.append({
            **metrics,
            'timestamp': time.time(),
        })
        # Keep last 1000 records
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

    def tune_constants(self) -> Dict[str, float]:
        """Tune constants based on performance history."""
        now = time.time()
        if now - self._last_update < self._update_interval:
            return {name: c.current_value for name, c in self.constants.items()}

        self._last_update = now

        if len(self._performance_history) < 10:
            return {name: c.current_value for name, c in self.constants.items()}

        # Analyze recent performance
        recent = self._performance_history[-50:]
        avg_vfe = sum(r.get('vfe', 0) for r in recent) / len(recent)
        avg_latency = sum(r.get('latency', 0) for r in recent) / len(recent)
        avg_ricci = sum(r.get('ricci', 0) for r in recent) / len(recent)
        vfe_trend = self._calculate_trend([r.get('vfe', 0) for r in recent])

        # Tune each constant
        tuned_values = {}

        # Mode bias easing (0.34 default)
        if 'mode_eps' in self.constants:
            c = self.constants['mode_eps']
            # Faster easing when VFE is decreasing well
            if vfe_trend < -0.001:
                new_value = c.current_value * 1.1
            elif vfe_trend > 0.001:
                new_value = c.current_value * 0.9
            else:
                new_value = c.current_value
            c.current_value = max(c.min_value, min(c.max_value, new_value))
            tuned_values['mode_eps'] = c.current_value

        # Ricci EMA smoothing (0.05 default)
        if 'ricci_ema_lambda' in self.constants:
            c = self.constants['ricci_ema_lambda']
            # Smoother when ricci is volatile
            ricci_variance = self._calculate_variance([r.get('ricci', 0) for r in recent])
            if ricci_variance > 1.0:
                new_value = c.current_value * 0.9
            else:
                new_value = c.current_value * 1.05
            c.current_value = max(c.min_value, min(c.max_value, new_value))
            tuned_values['ricci_ema_lambda'] = c.current_value

        # Velocity threshold (0.005 default)
        if 'velocity_threshold' in self.constants:
            c = self.constants['velocity_threshold']
            # Lower threshold when learning is slow
            if abs(vfe_trend) < 0.0001:
                new_value = c.current_value * 0.9
            else:
                new_value = c.current_value * 1.1
            c.current_value = max(c.min_value, min(c.max_value, new_value))
            tuned_values['velocity_threshold'] = c.current_value

        # WM learning rate
        if 'wm_alpha' in self.constants:
            c = self.constants['wm_alpha']
            # Adjust based on world model error
            wm_error = sum(r.get('wm_error', 0.01) for r in recent) / len(recent)
            if wm_error > 0.1:
                new_value = c.current_value * 1.2
            else:
                new_value = c.current_value * 0.95
            c.current_value = max(c.min_value, min(c.max_value, new_value))
            tuned_values['wm_alpha'] = c.current_value

        # Anneal temperature bounds
        if 't_min' in self.constants:
            c = self.constants['t_min']
            # Adjust based on exploration success
            exploration_success = sum(1 for r in recent if r.get('vfe', 0) < 0) / len(recent)
            if exploration_success > 0.7:
                new_value = c.current_value * 0.95
            else:
                new_value = c.current_value * 1.05
            c.current_value = max(c.min_value, min(c.max_value, new_value))
            tuned_values['t_min'] = c.current_value

        if 't_max' in self.constants:
            c = self.constants['t_max']
            # Opposite of t_min
            if 't_min' in self.constants:
                t_min_val = self.constants['t_min'].current_value
                new_value = max(t_min_val + 0.5, c.current_value)
            else:
                new_value = c.current_value
            c.current_value = max(c.min_value, min(c.max_value, new_value))
            tuned_values['t_max'] = c.current_value

        logger.debug(f"Tuned constants: {tuned_values}")
        return tuned_values

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values."""
        if len(values) < 2:
            return 0.0
        x_vals = list(range(len(values)))
        mean_x = sum(x_vals) / len(x_vals)
        mean_y = sum(values) / len(values)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, values))
        denominator = sum((x - mean_x) ** 2 for x in x_vals)
        return numerator / denominator if denominator > 0 else 0.0

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def get_constants(self) -> Dict[str, float]:
        """Get current constant values."""
        return {name: c.current_value for name, c in self.constants.items()}

    def get_status(self) -> Dict[str, Any]:
        """Get tuner status."""
        return {
            'constants': {
                name: {
                    'current': c.current_value,
                    'default': c.default_value,
                    'min': c.min_value,
                    'max': c.max_value,
                }
                for name, c in self.constants.items()
            },
            'performance_records': len(self._performance_history),
            'last_update': self._last_update,
        }


def create_default_tuner() -> RuntimeConstantTuner:
    """Create a tuner with default constants."""
    tuner = RuntimeConstantTuner()

    # Register all tunable constants
    tuner.register_constant('mode_eps', 0.34, 0.05, 1.0, "Mode bias easing rate")
    tuner.register_constant('ricci_ema_lambda', 0.05, 0.01, 0.2, "Ricci EMA smoothing")
    tuner.register_constant('velocity_threshold', 0.005, 0.0001, 0.1, "VFE velocity threshold")
    tuner.register_constant('wm_alpha', 0.01, 0.001, 0.05, "World model learning rate")
    tuner.register_constant('t_min', 0.15, 0.01, 1.0, "Minimum anneal temperature")
    tuner.register_constant('t_max', 1.5, 0.5, 5.0, "Maximum anneal temperature")

    return tuner


if __name__ == "__main__":
    print("=== Runtime Constant Grounding Test ===\n")

    tuner = create_default_tuner()

    # Simulate performance data
    import random
    for i in range(100):
        tuner.record_performance({
            'vfe': -0.1 + random.gauss(0, 0.02),
            'latency': 50 + random.gauss(0, 10),
            'ricci': 3.5 + random.gauss(0, 0.5),
            'wm_error': 0.05 + random.gauss(0, 0.01),
        })

    # Tune constants
    tuned = tuner.tune_constants()
    print("Tuned constants:")
    for name, value in tuned.items():
        default = tuner.constants[name].default_value
        change = ((value - default) / default) * 100
        print(f"  {name}: {value:.4f} (was {default:.4f}, {change:+.1f}%)")

    print(f"\nStatus: {tuner.get_status()}")
