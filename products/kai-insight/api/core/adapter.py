"""Inference adapter for Kai AGI system.

Provides clean interface to Kai's inference capabilities with fallback support.
"""

import sys
import os
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_kai_instance = None
_adapter_lock = threading.Lock()


class SimpleKaiMind:
    """Fallback Kai implementation for testing."""

    def __init__(self):
        self._mode = 'explore'
        self._mode_bias = 0.5
        self._prime_goal = 'Market_data_gen'
        self.vfe = 0.1
        self.ricci = 4.0
        self._last_efe = None
        self._vfe_ledger = []
        self._w_novelty = 0.4
        self._w_curvature = 0.3
        self._w_pnl = 0.2
        self._w_test = 0.1

    def vfe_trend(self, window: int = 500) -> float:
        pts = list(self._vfe_ledger)[-window:]
        n = len(pts)
        if n < 3:
            return 0.0
        xs = list(range(n))
        ys = [p[1] for p in pts]
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        if vx <= 1e-12:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / vx

    def select_policy(self, candidates: List[str]):
        if not candidates:
            return 'Market_data_gen', {'G': 0.5, 'risk': 0.3, 'epistemic': 0.4}
        return candidates[0], {
            'G': 0.5, 'risk': 0.3, 'epistemic': 0.4,
            'n': len(candidates)
        }


def _import_kai_mind():
    try:
        kai_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
        if kai_path not in sys.path:
            sys.path.insert(0, os.path.abspath(kai_path))
        from kai_mind import KaiMind
        return KaiMind
    except ImportError:
        return None


def get_kai_instance():
    global _kai_instance
    with _adapter_lock:
        if _kai_instance is None:
            KaiMind = _import_kai_mind()
            if KaiMind:
                try:
                    _kai_instance = KaiMind()
                    logger.info("Initialized real KaiMind")
                except Exception as e:
                    logger.warning(f"KaiMind init failed: {e}, using fallback")
                    _kai_instance = SimpleKaiMind()
            else:
                logger.info("Using SimpleKaiMind fallback")
                _kai_instance = SimpleKaiMind()
    return _kai_instance


def infer(state: List[float] = None, goal: str = None, depth: int = 3) -> Dict[str, Any]:
    if state is None:
        state = [0.0] * 8
    if goal is None:
        goal = 'Market_data_gen'

    kai = get_kai_instance()
    try:
        policy, info = kai.select_policy([goal])
        return {
            'policy': policy,
            'efe': info.get('G', 0.5),
            'risk': info.get('risk', 0.3),
            'epistemic': info.get('epistemic', 0.4),
            'mode': getattr(kai, '_mode', 'explore'),
            'vfe': getattr(kai, 'vfe', 0.1),
            'ricci': getattr(kai, 'ricci', 4.0),
            'trend': kai.vfe_trend() if hasattr(kai, 'vfe_trend') else 0.0,
            'info': info,
            '_fallback_used': isinstance(kai, SimpleKaiMind),
            'timestamp': datetime.now().isoformat(),
            'success': True,
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {
            'policy': goal, 'efe': 0.5, 'risk': 0.3, 'epistemic': 0.4,
            'mode': 'explore', 'vfe': 0.1, 'ricci': 4.0, 'trend': -0.01,
            '_fallback_used': True, 'error': str(e), 'success': False,
            'timestamp': datetime.now().isoformat(),
        }


def health_check() -> Dict[str, Any]:
    try:
        kai = get_kai_instance()
        return {
            'status': 'healthy',
            'kai_type': type(kai).__name__,
            'vfe': getattr(kai, 'vfe', 0.0),
            'trend': kai.vfe_trend() if hasattr(kai, 'vfe_trend') else 0.0,
            'timestamp': datetime.now().isoformat(),
            'inference_available': True,
            'fallback_mode': isinstance(kai, SimpleKaiMind),
        }
    except Exception as e:
        return {
            'status': 'unhealthy', 'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }


__all__ = ['infer', 'get_kai_instance', 'health_check', 'SimpleKaiMind']
