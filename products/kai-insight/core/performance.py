"""Performance Optimization and Memory Management for Kai AGI System.

Production-quality performance utilities with caching, memory management,
and profiling capabilities.
"""

from __future__ import annotations

import time
import gc
import logging
import functools
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_call_time: float = 0.0

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / max(1, self.call_count)

    def record(self, duration: float) -> None:
        """Record a call duration."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_call_time = duration

    def reset(self) -> None:
        """Reset metrics."""
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.last_call_time = 0.0


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        """Initialize LRU cache.

        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max(1, min(100000, max_size))
        self.ttl_seconds = max(0.0, min(3600.0, ttl_seconds))
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / max(1, self._hits + self._misses),
            }


def timed(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    metrics = PerformanceMetrics()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            metrics.record(duration)

    wrapper.metrics = metrics
    return wrapper


def cached(max_size: int = 128, ttl_seconds: float = 300.0) -> Callable:
    """Decorator for caching function results."""
    cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_parts = [func.__name__] + [str(a) for a in args]
            key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
            cache_key = ":".join(key_parts)

            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        return wrapper

    return decorator


def memoize(func: Callable) -> Callable:
    """Simple memoization decorator (no TTL, infinite size)."""
    cache: Dict[str, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper


class MemoryManager:
    """Memory management utilities."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                'rss_mb': mem_info.rss / 1024 / 1024,
                'vms_mb': mem_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
            }
        except ImportError:
            return {'error': 'psutil not available'}

    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return stats."""
        collected = gc.collect()
        return {
            'objects_collected': collected,
            'gc_counts': gc.get_count(),
            'gc_thresholds': gc.get_threshold(),
        }

    @staticmethod
    def get_object_sizes(obj: Any, max_depth: int = 3) -> Dict[str, int]:
        """Get approximate size of objects in memory."""
        import sys

        def _size(obj, seen):
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(_size(k, seen) + _size(v, seen) for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum(_size(i, seen) for i in obj)
            elif hasattr(obj, '__dict__'):
                size += _size(obj.__dict__, seen)
            return size

        if max_depth <= 0:
            return {'object': sys.getsizeof(obj)}

        return {'object': _size(obj, set())}


class Profiler:
    """Simple profiler for identifying bottlenecks."""

    def __init__(self):
        self.profiles: Dict[str, PerformanceMetrics] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start profiling a section."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop profiling and record duration."""
        if name not in self._start_times:
            return 0.0

        duration = time.perf_counter() - self._start_times.pop(name)

        if name not in self.profiles:
            self.profiles[name] = PerformanceMetrics()
        self.profiles[name].record(duration)

        return duration

    def get_report(self) -> Dict[str, Dict[str, float]]:
        """Get profiling report."""
        return {
            name: {
                'calls': metrics.call_count,
                'total_ms': metrics.total_time * 1000,
                'avg_ms': metrics.avg_time * 1000,
                'min_ms': metrics.min_time * 1000,
                'max_ms': metrics.max_time * 1000,
            }
            for name, metrics in self.profiles.items()
        }

    def reset(self) -> None:
        """Reset all profiles."""
        self.profiles.clear()
        self._start_times.clear()


if __name__ == "__main__":
    print("=== Performance Module Test ===\n")

    # Test LRU cache
    print("LRU Cache:")
    cache = LRUCache(max_size=3, ttl_seconds=10.0)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    print(f"  Get key1: {cache.get('key1')}")
    print(f"  Stats: {cache.stats()}")

    # Test timed decorator
    print("\nTimed decorator:")

    @timed
    def slow_function():
        time.sleep(0.01)
        return "done"

    result = slow_function()
    print(f"  Result: {result}")
    print(f"  Metrics: calls={slow_function.metrics.call_count}, avg={slow_function.metrics.avg_time*1000:.2f}ms")

    # Test memory manager
    print("\nMemory usage:")
    mem = MemoryManager.get_memory_usage()
    for k, v in mem.items():
        print(f"  {k}: {v}")

    # Test profiler
    print("\nProfiler:")
    profiler = Profiler()
    profiler.start("test")
    time.sleep(0.01)
    profiler.stop("test")
    print(f"  Report: {profiler.get_report()}")
