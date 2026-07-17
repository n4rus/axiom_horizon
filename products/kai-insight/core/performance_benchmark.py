"""Performance Benchmarking System for Kai AGI.

Tracks system performance over time and provides insights:
- Benchmark execution times
- Track learning progress
- Monitor resource usage
- Generate performance reports
"""

from __future__ import annotations

import time
import psutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    name: str
    duration: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    timestamp: float = field(default_factory=time.time)


class PerformanceBenchmark:
    """Performance benchmarking system."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.benchmark_results: deque = deque(maxlen=history_size)
        self.performance_snapshots: deque = deque(maxlen=history_size)
        self._active_benchmarks: Dict[str, float] = {}

    def start_benchmark(self, name: str):
        """Start a benchmark."""
        self._active_benchmarks[name] = time.perf_counter()

    def end_benchmark(self, name: str, success: bool = True, metadata: Optional[Dict] = None) -> BenchmarkResult:
        """End a benchmark and record result."""
        if name not in self._active_benchmarks:
            raise ValueError(f"Benchmark '{name}' not started")

        start_time = self._active_benchmarks.pop(name)
        duration = time.perf_counter() - start_time

        result = BenchmarkResult(
            name=name,
            duration=duration,
            success=success,
            metadata=metadata or {},
        )

        self.benchmark_results.append(result)
        return result

    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            snapshot = PerformanceSnapshot(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
            )

            self.performance_snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            logger.error(f"Failed to take snapshot: {e}")
            return PerformanceSnapshot(cpu_percent=0, memory_percent=0, memory_used_mb=0)

    def get_benchmark_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get benchmark statistics."""
        results = list(self.benchmark_results)
        if name:
            results = [r for r in results if r.name == name]

        if not results:
            return {'count': 0}

        durations = [r.duration for r in results]
        successes = [r for r in results if r.success]

        return {
            'count': len(results),
            'success_count': len(successes),
            'success_rate': len(successes) / len(results),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations),
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_snapshots:
            return {'count': 0}

        snapshots = list(self.performance_snapshots)
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]

        return {
            'count': len(snapshots),
            'avg_cpu': sum(cpu_values) / len(cpu_values),
            'max_cpu': max(cpu_values),
            'avg_memory': sum(memory_values) / len(memory_values),
            'max_memory': max(memory_values),
            'current_cpu': cpu_values[-1] if cpu_values else 0,
            'current_memory': memory_values[-1] if memory_values else 0,
        }

    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'benchmarks': self.get_benchmark_stats(),
            'performance': self.get_performance_stats(),
            'total_benchmarks': len(self.benchmark_results),
            'total_snapshots': len(self.performance_snapshots),
        }


if __name__ == "__main__":
    print("=== Performance Benchmark Test ===\n")

    benchmark = PerformanceBenchmark()

    # Run some benchmarks
    for i in range(5):
        benchmark.start_benchmark(f"test_benchmark_{i}")
        time.sleep(0.01)  # Simulate work
        benchmark.end_benchmark(f"test_benchmark_{i}", success=i % 2 == 0)

    # Take performance snapshots
    for _ in range(3):
        benchmark.take_snapshot()
        time.sleep(0.1)

    # Get report
    report = benchmark.get_report()
    print("Performance Report:")
    print(f"  Benchmarks: {report['benchmarks']}")
    print(f"  Performance: {report['performance']}")
