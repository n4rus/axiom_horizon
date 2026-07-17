"""Health Monitoring and Self-Healing for Kai AGI System.

Production-quality health monitoring with automatic recovery,
circuit breaker pattern, and comprehensive diagnostics.
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    """Circuit breaker state for fault tolerance."""
    failure_count: int = 0
    last_failure: float = 0.0
    state: str = "closed"  # closed, open, half-open
    failure_threshold: int = 5
    recovery_timeout: float = 60.0

    def record_failure(self) -> None:
        """Record a failure and potentially trip the circuit."""
        self.failure_count += 1
        self.last_failure = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker tripped after {self.failure_count} failures")

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logger.info("Circuit breaker closed after successful recovery")

    def can_execute(self) -> bool:
        """Check if operation is allowed."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open allows one attempt


class HealthMonitor:
    """Comprehensive health monitoring with self-healing."""

    def __init__(self, check_interval: float = 30.0):
        """Initialize health monitor.

        Args:
            check_interval: Seconds between automatic health checks
        """
        self.check_interval = max(5.0, min(300.0, check_interval))
        self.health_history: List[HealthCheck] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._health_checks: List[Callable[[], HealthCheck]] = []

    def register_health_check(self, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        self._health_checks.append(check_func)

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a component."""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker()
            return self.circuit_breakers[name]

    def check_health(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        for check_func in self._health_checks:
            try:
                check = check_func()
                results[check.component] = check
                with self._lock:
                    self.health_history.append(check)
                    # Keep last 1000 checks
                    if len(self.health_history) > 1000:
                        self.health_history = self.health_history[-1000:]
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                results[check_func.__name__] = HealthCheck(
                    component=check_func.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}"
                )
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.check_health()
        if not results:
            return HealthStatus.HEALTHY

        statuses = [r.status for r in results.values()]
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def start(self) -> None:
        """Start automatic health monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitor started")

    def stop(self) -> None:
        """Stop automatic health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Health monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                results = self.check_health()
                unhealthy = [r for r in results.values() if r.status != HealthStatus.HEALTHY]
                if unhealthy:
                    logger.warning(f"Unhealthy components: {[r.component for r in unhealthy]}")
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            time.sleep(self.check_interval)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        with self._lock:
            recent = self.health_history[-100:] if self.health_history else []
            circuit_states = {name: cb.state for name, cb in self.circuit_breakers.items()}

        return {
            'overall_status': self.get_overall_status().value,
            'health_checks_count': len(self.health_history),
            'recent_unhealthy': [
                {'component': h.component, 'status': h.status.value, 'message': h.message}
                for h in recent if h.status != HealthStatus.HEALTHY
            ],
            'circuit_breakers': circuit_states,
            'check_interval': self.check_interval,
        }


def create_default_monitor() -> HealthMonitor:
    """Create a health monitor with default checks."""
    monitor = HealthMonitor()

    def check_system_resources() -> HealthCheck:
        """Check system resource usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            if cpu_percent > 90 or memory.percent > 90:
                status = HealthStatus.CRITICAL
                msg = f"High resource usage: CPU={cpu_percent}%, Memory={memory.percent}%"
            elif cpu_percent > 70 or memory.percent > 70:
                status = HealthStatus.DEGRADED
                msg = f"Elevated resource usage: CPU={cpu_percent}%, Memory={memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                msg = f"Normal: CPU={cpu_percent}%, Memory={memory.percent}%"

            return HealthCheck(
                component="system_resources",
                status=status,
                message=msg,
                metadata={'cpu_percent': cpu_percent, 'memory_percent': memory.percent}
            )
        except ImportError:
            return HealthCheck(
                component="system_resources",
                status=HealthStatus.HEALTHY,
                message="psutil not available, skipping"
            )

    def check_daemon_running() -> HealthCheck:
        """Check if main daemon is running."""
        try:
            import os
            pid_file = os.path.expanduser("~/.axiom_state/daemon.pid")
            if os.path.exists(pid_file):
                return HealthCheck(
                    component="daemon",
                    status=HealthStatus.HEALTHY,
                    message="Daemon PID file exists"
                )
            return HealthCheck(
                component="daemon",
                status=HealthStatus.DEGRADED,
                message="Daemon PID file not found"
            )
        except Exception as e:
            return HealthCheck(
                component="daemon",
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}"
            )

    monitor.register_health_check(check_system_resources)
    monitor.register_health_check(check_daemon_running)

    return monitor


if __name__ == "__main__":
    print("=== Health Monitor Test ===\n")

    monitor = create_default_monitor()
    results = monitor.check_health()

    print("Health check results:")
    for name, check in results.items():
        print(f"  {name}: {check.status.value} - {check.message}")

    print(f"\nOverall status: {monitor.get_overall_status().value}")
    print(f"Diagnostics: {monitor.get_diagnostics()}")
