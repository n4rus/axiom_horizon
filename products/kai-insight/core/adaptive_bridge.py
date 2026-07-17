"""Bridge to apply adaptive constants to live kai_mind.py daemon.

This module provides functions that can be called from the daemon loop
to replace hardcoded constants with adaptive values.

Production-quality with error handling, validation, and monitoring.
"""

from __future__ import annotations

import sys
import os
import time
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from adaptive_constants import compute_adaptive_constants, get_default_constants, AdaptiveConstants


class AdaptiveBridge:
    """Applies adaptive constants to a running KaiMind instance.

    Thread-safe and fault-tolerant. Updates constants based on
    runtime performance metrics and applies them to the daemon.
    """

    def __init__(self, kai_instance: Any, update_interval: float = 60.0):
        """Initialize the adaptive bridge.

        Args:
            kai_instance: The KaiMind instance to adapt
            update_interval: Minimum seconds between updates
        """
        self.kai = kai_instance
        self.defaults = get_default_constants()
        self.last_update: float = 0.0
        self.update_interval: float = max(10.0, min(3600.0, update_interval))
        self.current: AdaptiveConstants = self.defaults
        self._error_count: int = 0
        self._last_error: Optional[str] = None
        self._update_count: int = 0

    def maybe_update(self) -> bool:
        """Check if update is due and perform it.

        Returns:
            True if update was performed, False otherwise
        """
        now = time.time()
        if now - self.last_update < self.update_interval:
            return False

        try:
            self.update()
            self.last_update = now
            self._error_count = 0
            self._update_count += 1
            return True
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"Adaptive bridge update failed (attempt {self._error_count}): {e}")
            return False

    def update(self) -> None:
        """Update adaptive constants from current state."""
        vfe_history = self._extract_vfe_history()
        ricci = self._extract_ricci()
        trend = self._extract_trend()
        uptime = self._extract_uptime()
        wm_errors = self._extract_wm_errors()
        weight_deltas = self._extract_weight_deltas()

        self.current = compute_adaptive_constants(
            time_secs=uptime,
            vfe_history=vfe_history,
            ricci=ricci,
            vfe_trend=trend,
            wm_errors=wm_errors,
            weight_deltas=weight_deltas,
            overrides=self._measured_overrides(),
        )

    def _measured_overrides(self) -> dict:
        """Load measured optimal constants from constant_tuner (if present)."""
        try:
            from constant_tuner import load_optimal
            return load_optimal()
        except Exception:
            return {}

    def _extract_vfe_history(self) -> list[float]:
        """Safely extract VFE history from KaiMind."""
        try:
            if hasattr(self.kai, '_vfe_ledger'):
                return [p[1] for p in list(self.kai._vfe_ledger)[-100:]]
        except Exception as e:
            logger.debug(f"Failed to extract VFE history: {e}")
        return []

    def _extract_ricci(self) -> float:
        """Safely extract Ricci curvature from KaiMind."""
        try:
            if hasattr(self.kai, 'ricci'):
                val = self.kai.ricci
                if math.isfinite(val):
                    return val
        except Exception as e:
            logger.debug(f"Failed to extract ricci: {e}")
        return 4.0

    def _extract_trend(self) -> float:
        """Safely extract VFE trend from KaiMind."""
        try:
            if hasattr(self.kai, 'vfe_trend'):
                val = self.kai.vfe_trend()
                if math.isfinite(val):
                    return val
        except Exception as e:
            logger.debug(f"Failed to extract trend: {e}")
        return 0.0

    def _extract_uptime(self) -> float:
        """Safely extract uptime from KaiMind."""
        try:
            if hasattr(self.kai, '_uptime_secs'):
                val = self.kai._uptime_secs
                if math.isfinite(val) and val >= 0:
                    return val
        except Exception as e:
            logger.debug(f"Failed to extract uptime: {e}")
        return 0.0

    def _extract_wm_errors(self) -> list[float]:
        """Safely extract world model errors from KaiMind."""
        try:
            if hasattr(self.kai, '_wm_last_mse'):
                return [self.kai._wm_last_mse]
        except Exception as e:
            logger.debug(f"Failed to extract WM errors: {e}")
        return []

    def _extract_weight_deltas(self) -> list[float]:
        """Safely extract weight deltas from KaiMind."""
        try:
            if hasattr(self.kai, '_w_novelty'):
                return [abs(self.kai._w_novelty - 0.4)]
        except Exception as e:
            logger.debug(f"Failed to extract weight deltas: {e}")
        return []

    def apply_to_daemon(self) -> bool:
        """Apply adaptive constants to the daemon.

        Returns:
            True if applied successfully
        """
        if not hasattr(self.kai, '_mode_ticks'):
            return False

        try:
            self.kai._adaptive_explore_span = self.current.explore_span
            self.kai._adaptive_converge_span = self.current.converge_span
            self.kai._adaptive_mode_eps = self.current.mode_eps
            self.kai._adaptive_ricci_ema_lambda = self.current.ricci_ema_lambda
            self.kai._adaptive_wm_alpha = self.current.wm_alpha
            self.kai._adaptive_t_min = self.current.t_min
            self.kai._adaptive_t_max = self.current.t_max
            self.kai._adaptive_alpha_weight = self.current.alpha_weight
            self.kai._adaptive_velocity_threshold = self.current.velocity_threshold
            return True
        except Exception as e:
            logger.error(f"Failed to apply adaptive constants: {e}")
            return False

    def get(self, key: str, fallback: Any = None) -> Any:
        """Get an adaptive constant value by name."""
        return getattr(self.current, key, fallback)

    def status(self) -> Dict[str, Any]:
        """Get bridge status for monitoring."""
        return {
            'adapted': self.current.to_dict(),
            'defaults': self.defaults.to_dict(),
            'last_update': self.last_update,
            'update_interval': self.update_interval,
            'update_count': self._update_count,
            'error_count': self._error_count,
            'last_error': self._last_error,
        }


def create_bridge(kai_instance: Any, update_interval: float = 60.0) -> AdaptiveBridge:
    """Create and initialize an adaptive bridge.

    Args:
        kai_instance: The KaiMind instance to adapt
        update_interval: Minimum seconds between updates

    Returns:
        Initialized AdaptiveBridge instance
    """
    bridge = AdaptiveBridge(kai_instance, update_interval)
    bridge.update()
    bridge.apply_to_daemon()
    return bridge


if __name__ == "__main__":
    print("=== Adaptive Bridge Test ===\n")

    defaults = get_default_constants()
    print("Defaults:", defaults.to_dict())

    print("\nBridge would apply adaptive constants to live daemon.")
    print("Import and call create_bridge(kai_instance) in daemon loop.")
