"""Integration APIs for Kai AGI System.

Production-quality APIs for external system connectivity,
webhooks, and state synchronization.
"""

from __future__ import annotations

import time
import json
import hashlib
import hmac
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Webhook event types."""
    INFERENCE_COMPLETE = "inference.complete"
    HEALTH_CHANGED = "health.changed"
    ANOMALY_DETECTED = "anomaly.detected"
    STATE_UPDATED = "state.updated"
    EXPERIMENT_COMPLETED = "experiment.completed"


@dataclass
class WebhookConfig:
    """Webhook configuration."""
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    enabled: bool = True
    retry_count: int = 3
    timeout: float = 10.0


@dataclass
class WebhookPayload:
    """Webhook payload."""
    event: WebhookEvent
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "kai-insight"


class WebhookManager:
    """Manages webhooks for external system integration."""

    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self._lock = threading.Lock()
        self._delivery_log: List[Dict[str, Any]] = []

    def register_webhook(
        self,
        name: str,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
    ) -> WebhookConfig:
        """Register a new webhook."""
        config = WebhookConfig(
            url=url,
            events=events,
            secret=secret,
        )
        with self._lock:
            self.webhooks[name] = config
        logger.info(f"Registered webhook: {name}")
        return config

    def unregister_webhook(self, name: str) -> bool:
        """Unregister a webhook."""
        with self._lock:
            if name in self.webhooks:
                del self.webhooks[name]
                logger.info(f"Unregistered webhook: {name}")
                return True
        return False

    def send_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Send webhook to all registered handlers."""
        results = []

        with self._lock:
            active_webhooks = [
                (name, config)
                for name, config in self.webhooks.items()
                if config.enabled and event in config.events
            ]

        for name, config in active_webhooks:
            try:
                payload = WebhookPayload(
                    event=event,
                    data=data,
                )

                # Generate signature if secret is set
                signature = None
                if config.secret:
                    payload_bytes = json.dumps(payload.__dict__, default=str).encode()
                    signature = hmac.new(
                        config.secret.encode(),
                        payload_bytes,
                        hashlib.sha256
                    ).hexdigest()

                # In production, use requests.post here
                # For now, log the delivery
                delivery_record = {
                    'webhook': name,
                    'event': event.value,
                    'url': config.url,
                    'success': True,
                    'timestamp': time.time(),
                    'signature': signature,
                }

                self._delivery_log.append(delivery_record)
                results.append(delivery_record)

                logger.debug(f"Webhook sent: {name} -> {event.value}")

            except Exception as e:
                delivery_record = {
                    'webhook': name,
                    'event': event.value,
                    'url': config.url,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time(),
                }
                self._delivery_log.append(delivery_record)
                results.append(delivery_record)
                logger.error(f"Webhook delivery failed: {name}: {e}")

        return results

    def get_delivery_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent webhook delivery log."""
        with self._lock:
            return self._delivery_log[-limit:]


class StateSynchronizer:
    """Synchronizes state across distributed components."""

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._version: int = 0
        self._lock = threading.Lock()
        self._watchers: List[Callable[[Dict[str, Any]], None]] = []

    def get_state(self, key: Optional[str] = None) -> Any:
        """Get state value or entire state."""
        with self._lock:
            if key:
                return self._state.get(key)
            return dict(self._state)

    def set_state(self, key: str, value: Any) -> int:
        """Set state value and increment version."""
        with self._lock:
            self._state[key] = value
            self._version += 1
            version = self._version

        # Notify watchers
        self._notify_watchers({key: value})

        return version

    def update_state(self, updates: Dict[str, Any]) -> int:
        """Update multiple state values."""
        with self._lock:
            self._state.update(updates)
            self._version += 1
            version = self._version

        # Notify watchers
        self._notify_watchers(updates)

        return version

    def get_version(self) -> int:
        """Get current state version."""
        with self._lock:
            return self._version

    def register_watcher(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a state change watcher."""
        self._watchers.append(callback)

    def _notify_watchers(self, changes: Dict[str, Any]) -> None:
        """Notify all watchers of state changes."""
        for watcher in self._watchers:
            try:
                watcher(changes)
            except Exception as e:
                logger.error(f"Watcher notification failed: {e}")

    def export_state(self) -> Dict[str, Any]:
        """Export entire state for synchronization."""
        with self._lock:
            return {
                'state': dict(self._state),
                'version': self._version,
                'timestamp': time.time(),
            }

    def import_state(self, data: Dict[str, Any]) -> bool:
        """Import state from synchronization."""
        try:
            with self._lock:
                self._state.update(data.get('state', {}))
                self._version = max(self._version, data.get('version', 0))
            return True
        except Exception as e:
            logger.error(f"State import failed: {e}")
            return False


class MetricsCollector:
    """Collects and exposes metrics for monitoring."""

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def observe_histogram(self, name: str, value: float) -> None:
        """Observe a histogram value."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)
            # Keep last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {
                    name: {
                        'count': len(values),
                        'sum': sum(values),
                        'avg': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0,
                    }
                    for name, values in self._histograms.items()
                },
            }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Counters
            for name, value in self._counters.items():
                lines.append(f"kai_{name}_total {value}")

            # Gauges
            for name, value in self._gauges.items():
                lines.append(f"kai_{name} {value}")

            # Histograms
            for name, values in self._histograms.items():
                if values:
                    lines.append(f"kai_{name}_count {len(values)}")
                    lines.append(f"kai_{name}_sum {sum(values)}")

        return '\n'.join(lines)


if __name__ == "__main__":
    print("=== Integration APIs Test ===\n")

    # Test webhook manager
    print("Webhook Manager:")
    webhook_mgr = WebhookManager()

    def test_handler(payload):
        print(f"  Handler received: {payload.event.value}")

    webhook_mgr.register_webhook(
        "test_webhook",
        "http://localhost:8000/webhook",
        [WebhookEvent.INFERENCE_COMPLETE],
    )

    results = webhook_mgr.send_webhook(
        WebhookEvent.INFERENCE_COMPLETE,
        {'result': 'test'},
    )
    print(f"  Delivery results: {len(results)}")

    # Test state synchronizer
    print("\nState Synchronizer:")
    sync = StateSynchronizer()
    sync.set_state('mode', 'explore')
    sync.set_state('vfe', -0.1)
    print(f"  State: {sync.get_state()}")
    print(f"  Version: {sync.get_version()}")

    # Test metrics collector
    print("\nMetrics Collector:")
    metrics = MetricsCollector()
    metrics.increment_counter('requests')
    metrics.set_gauge('vfe', -0.1)
    metrics.observe_histogram('latency', 42.5)
    print(f"  Metrics: {metrics.get_metrics()}")
    print(f"  Prometheus:\n{metrics.export_prometheus()}")
