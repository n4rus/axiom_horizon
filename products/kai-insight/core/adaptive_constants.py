"""Adaptive constants for Kai AGI system.

Replaces hardcoded heuristic values with self-tuning parameters
driven by runtime VFE, ricci, trend, and learning signals.

Production-quality module with full error handling, validation,
and comprehensive type hints.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveConstants:
    """Immutable container for adaptive constant values."""
    explore_span: int = 8
    converge_span: int = 5
    mode_eps: float = 0.34
    ricci_ema_lambda: float = 0.05
    wm_alpha: float = 0.01
    t_min: float = 0.15
    t_max: float = 1.5
    alpha_weight: float = 0.05
    velocity_threshold: float = 0.005

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'explore_span': self.explore_span,
            'converge_span': self.converge_span,
            'mode_eps': self.mode_eps,
            'ricci_ema_lambda': self.ricci_ema_lambda,
            'wm_alpha': self.wm_alpha,
            't_min': self.t_min,
            't_max': self.t_max,
            'alpha_weight': self.alpha_weight,
            'velocity_threshold': self.velocity_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AdaptiveConstants':
        """Create from dictionary with validation."""
        return cls(
            explore_span=max(2, min(20, int(d.get('explore_span', 8)))),
            converge_span=max(1, min(15, int(d.get('converge_span', 5)))),
            mode_eps=max(0.05, min(1.0, float(d.get('mode_eps', 0.34)))),
            ricci_ema_lambda=max(0.01, min(0.2, float(d.get('ricci_ema_lambda', 0.05)))),
            wm_alpha=max(0.001, min(0.05, float(d.get('wm_alpha', 0.01)))),
            t_min=max(0.01, min(1.0, float(d.get('t_min', 0.15)))),
            t_max=max(0.5, min(5.0, float(d.get('t_max', 1.5)))),
            alpha_weight=max(0.01, min(0.3, float(d.get('alpha_weight', 0.05)))),
            velocity_threshold=max(0.0001, min(0.1, float(d.get('velocity_threshold', 0.005)))),
        )


def compute_adaptive_constants(
    time_secs: float,
    vfe_history: List[float],
    ricci: float,
    vfe_trend: float,
    wm_errors: Optional[List[float]] = None,
    weight_deltas: Optional[List[float]] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> AdaptiveConstants:
    """Compute adaptive constants from runtime observations.

    Args:
        time_secs: System uptime in seconds (must be >= 0)
        vfe_history: Recent VFE values (last 50-100 points)
        ricci: Current Ricci curvature (must be finite)
        vfe_trend: Current VFE trend slope
        wm_errors: Recent world model errors (optional)
        weight_deltas: Recent weight changes (optional)
        overrides: Measured optimal values from ``constant_tuner`` (optional).
            Any name present here pins that constant to its measured value
            instead of the heuristic formula — this is how real measurement
            replaces guesswork.

    Returns:
        AdaptiveConstants with computed values

    Raises:
        ValueError: If inputs are invalid (nan/inf or negative time)
    """
    if wm_errors is None:
        wm_errors = []
    if weight_deltas is None:
        weight_deltas = []

    # Input validation
    if not math.isfinite(time_secs) or time_secs < 0:
        raise ValueError(f"time_secs must be finite and >= 0, got {time_secs}")
    if not math.isfinite(ricci):
        raise ValueError(f"ricci must be finite, got {ricci}")
    if not math.isfinite(vfe_trend):
        raise ValueError(f"vfe_trend must be finite, got {vfe_trend}")
    for i, v in enumerate(vfe_history):
        if not math.isfinite(v):
            raise ValueError(f"vfe_history[{i}] must be finite, got {v}")
    for i, e in enumerate(wm_errors):
        if not math.isfinite(e):
            raise ValueError(f"wm_errors[{i}] must be finite, got {e}")
    for i, d in enumerate(weight_deltas):
        if not math.isfinite(d):
            raise ValueError(f"weight_deltas[{i}] must be finite, got {d}")

    try:
        n_vfe = len(vfe_history)

        # --- Explore/Converge spans: scale by chaos (ricci) ---
        chaos_factor = 1.0 + max(0, ricci) / 10.0
        explore_span = max(3, int(8 * chaos_factor))
        converge_span = max(2, int(5 * chaos_factor))

        # --- Mode bias ease (0.34 base): faster when trend is strong ---
        trend_strength = min(1.0, abs(vfe_trend) / 0.01) if abs(vfe_trend) > 1e-6 else 0.0
        mode_eps = 0.15 + (0.6 - 0.15) * trend_strength

        # --- Ricci EMA smoothing: slow when ricci volatile ---
        volatility = 0.5
        if n_vfe >= 5:
            recent = vfe_history[-10:] if n_vfe >= 10 else vfe_history
            mean_vfe = sum(recent) / len(recent)
            variance = sum((v - mean_vfe) ** 2 for v in recent) / len(recent)
            volatility = min(1.0, math.sqrt(max(0, variance)) / 0.1)

        ricci_ema_lambda = 0.02 + (0.1 - 0.02) * (1.0 - volatility)

        # --- WM learning rate schedule: error-driven ---
        wm_alpha = 0.01
        if wm_errors:
            avg_abs_error = sum(abs(e) for e in wm_errors[-20:]) / min(20, len(wm_errors))
            wm_alpha = min(0.02, max(0.001, 0.01 * (avg_abs_error / 0.1)))

        # --- Anneal temperature bounds: driven by curiosity signal ---
        curiosity = ricci * 4.0 * (1.0 + abs(vfe_trend) / 0.01) if abs(vfe_trend) > 1e-6 else ricci * 4.0
        t_min = 0.15 + (curiosity - 4.0) * 0.02
        t_max = 1.5 + (curiosity - 4.0) * 0.03
        t_min = max(0.05, min(0.5, t_min))
        t_max = max(1.0, min(3.0, t_max))

        # --- VFE weight alpha: faster when weights changing ---
        alpha_weight = 0.05
        if weight_deltas:
            recent_delta = sum(abs(d) for d in weight_deltas[-5:]) / min(5, len(weight_deltas))
            alpha_weight = min(0.15, max(0.02, 0.05 * (1.0 + recent_delta / 0.01)))

        # --- Velocity threshold: scale with phase ---
        phase_factor = min(5.0, 0.1 + time_secs / 120.0)
        velocity_threshold = 0.005 * phase_factor * (1.0 + max(0, ricci) / 10.0)

        # Pin any measured overrides so research-grounded values replace the
        # heuristic formulas above. Clamp to the same safe bounds as from_dict.
        if overrides:
            _ov = {k: float(v) for k, v in overrides.items()}
            explore_span = max(2, min(20, int(_ov.get("explore_span", explore_span))))
            converge_span = max(1, min(15, int(_ov.get("converge_span", converge_span))))
            mode_eps = max(0.05, min(1.0, _ov.get("mode_eps", mode_eps)))
            ricci_ema_lambda = max(0.01, min(0.2, _ov.get("ricci_ema_lambda", ricci_ema_lambda)))
            wm_alpha = max(0.001, min(0.05, _ov.get("wm_alpha", wm_alpha)))
            t_min = max(0.01, min(1.0, _ov.get("t_min", t_min)))
            t_max = max(0.5, min(5.0, _ov.get("t_max", t_max)))
            alpha_weight = max(0.01, min(0.3, _ov.get("alpha_weight", alpha_weight)))
            velocity_threshold = max(0.0001, min(0.1, _ov.get("velocity_threshold", velocity_threshold)))

        return AdaptiveConstants(
            explore_span=explore_span,
            converge_span=converge_span,
            mode_eps=round(mode_eps, 4),
            ricci_ema_lambda=round(ricci_ema_lambda, 4),
            wm_alpha=round(wm_alpha, 5),
            t_min=round(t_min, 4),
            t_max=round(t_max, 4),
            alpha_weight=round(alpha_weight, 4),
            velocity_threshold=round(velocity_threshold, 5),
        )

    except Exception as e:
        logger.error(f"Failed to compute adaptive constants: {e}")
        return AdaptiveConstants()


def get_default_constants() -> AdaptiveConstants:
    """Return the original hardcoded defaults for comparison."""
    return AdaptiveConstants()


if __name__ == "__main__":
    import random

    print("=== Adaptive Constants Demo ===\n")

    defaults = get_default_constants()
    print("Default (hardcoded) values:")
    for k, v in defaults.to_dict().items():
        print(f"  {k}: {v}")

    print("\n--- Simulating runtime adaptation ---\n")

    vfe_hist = [0.5 - 0.01 * i + random.gauss(0, 0.02) for i in range(50)]
    ricci_val = 3.5
    trend_val = -0.008
    wm_errs = [0.1 * math.exp(-0.05 * i) + random.gauss(0, 0.01) for i in range(20)]
    w_deltas = [0.005 * math.exp(-0.1 * i) for i in range(10)]

    adapted = compute_adaptive_constants(
        time_secs=300.0,
        vfe_history=vfe_hist,
        ricci=ricci_val,
        vfe_trend=trend_val,
        wm_errors=wm_errs,
        weight_deltas=w_deltas,
    )

    print("Adaptive (runtime) values:")
    defaults_dict = defaults.to_dict()
    adapted_dict = adapted.to_dict()
    for k, v in adapted_dict.items():
        d = defaults_dict[k]
        direction = "UP" if v > d else "DOWN" if v < d else "SAME"
        print(f"  {k}: {v}  (was {d}, {direction})")
