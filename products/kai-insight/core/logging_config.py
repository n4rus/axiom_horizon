"""Structured logging for Kai Insight production system."""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class KaiJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'kai_data'):
            log_data['kai'] = record.kai_data
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(log_dir: str = None, level: str = 'INFO') -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.axiom_state', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger('kai_insight')
    root_logger.setLevel(getattr(logging, level.upper()))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(KaiJSONFormatter())
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, 'kai_insight.log'))
    file_handler.setFormatter(KaiJSONFormatter())
    root_logger.addHandler(file_handler)

    return root_logger


def log_inference(logger: logging.Logger, goal: str, result: Dict[str, Any], duration_ms: float):
    extra = {'kai_data': {
        'event': 'inference',
        'goal': goal,
        'policy': result.get('policy'),
        'efe': result.get('efe'),
        'success': result.get('success'),
        'fallback': result.get('_fallback_used'),
        'duration_ms': round(duration_ms, 2),
    }}
    logger.info(f"inference goal={goal} efe={result.get('efe')} dur={duration_ms:.1f}ms", extra=extra)


def log_health(logger: logging.Logger, health: Dict[str, Any]):
    extra = {'kai_data': {
        'event': 'health_check',
        'status': health.get('status'),
        'kai_type': health.get('kai_type'),
        'vfe': health.get('vfe'),
    }}
    logger.info(f"health status={health.get('status')} type={health.get('kai_type')}", extra=extra)


def log_mode_transition(logger: logging.Logger, old_mode: str, new_mode: str, vfe: float, trend: float):
    extra = {'kai_data': {
        'event': 'mode_transition',
        'from': old_mode,
        'to': new_mode,
        'vfe': vfe,
        'trend': trend,
    }}
    logger.info(f"mode_transition {old_mode}->{new_mode} vfe={vfe:.4f} trend={trend:.6f}", extra=extra)


if __name__ == "__main__":
    logger = setup_logging(level='DEBUG')
    logger.info("Kai Insight logging initialized")
    log_inference(logger, "test_goal", {"policy": "test", "efe": -0.5, "success": True}, 42.5)
    log_health(logger, {"status": "healthy", "kai_type": "KaiMind", "vfe": -0.1})
    log_mode_transition(logger, "explore", "converge", -0.08, -0.005)
    print("Logging test complete - check .axiom_state/logs/kai_insight.log")
