//! Phase 5.2: Distributed Computation — parallel evaluation harness.
//!
//! From AGI_PLAN.md §5.2:
//! "Immediate goal: Use CPU cores for parallel candidate testing.
//! GPU is the bottleneck for LLM inference only."
//!
//! This module provides a thread-pool based parallel evaluation
//! harness that wraps rayon for CPU-bound tasks like Darwin
//! candidate fitness evaluation.

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics from a parallel evaluation run.
#[derive(Debug, Clone, Default)]
pub struct EvalStats {
    /// Total tasks submitted
    pub total: usize,
    /// Tasks completed successfully
    pub completed: usize,
    /// Tasks that failed
    pub failed: usize,
    /// Total wall time in milliseconds
    pub wall_ms: u64,
    /// Total CPU time in milliseconds (sum of all workers)
    pub cpu_ms: u64,
    /// Worker count used
    pub workers: usize,
}

impl EvalStats {
    pub fn throughput(&self) -> f32 {
        if self.wall_ms == 0 { return 0.0; }
        self.completed as f32 / (self.wall_ms as f32 / 1000.0)
    }

    pub fn efficiency(&self) -> f32 {
        if self.cpu_ms == 0 || self.workers == 0 { return 0.0; }
        self.cpu_ms as f32 / (self.wall_ms as f32 * self.workers as f32)
    }

    pub fn summary(&self) -> String {
        format!(
            "tasks={}/{} ({} failed) | {:.1}ms wall | {:.1} tasks/s | workers={} eff={:.0}%",
            self.completed, self.total, self.failed,
            self.wall_ms, self.throughput(),
            self.workers, self.efficiency() * 100.0,
        )
    }
}

/// Parallel evaluator — runs tasks across multiple CPU cores using rayon.
pub struct ParallelEval {
    workers: usize,
}

impl ParallelEval {
    pub fn new() -> Self {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        ParallelEval { workers }
    }

    pub fn with_workers(workers: usize) -> Self {
        ParallelEval { workers }
    }

    /// Run tasks in parallel and collect results.
    /// Each task is a closure that returns an Option<T>.
    pub fn eval<F, T>(&self, tasks: Vec<F>) -> EvalStats
    where
        F: Fn() -> Option<T> + Send + Sync,
        T: Send,
    {
        let total = tasks.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));

        let start = std::time::Instant::now();

        // Use rayon for parallel iteration
        let tasks = Arc::new(tasks);
        let chunk_size = (total / self.workers.max(1)).max(1);

        use rayon::prelude::*;
        let _: Vec<_> = tasks.par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.par_iter().filter_map(|task| {
                    let result = task();
                    if result.is_some() {
                        completed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                    result
                }).collect::<Vec<_>>()
            })
            .collect();

        let wall_ms = start.elapsed().as_millis() as u64;
        let completed_count = completed.load(Ordering::Relaxed);
        let failed_count = failed.load(Ordering::Relaxed);

        EvalStats {
            total,
            completed: completed_count,
            failed: failed_count,
            wall_ms,
            cpu_ms: wall_ms * self.workers as u64,
            workers: self.workers,
        }
    }

    /// Run tasks and collect all results (including failures as None).
    pub fn eval_collect<F, T>(&self, tasks: Vec<F>) -> (Vec<Option<T>>, EvalStats)
    where
        F: Fn() -> Option<T> + Send + Sync,
        T: Send,
    {
        let total = tasks.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));

        let start = std::time::Instant::now();
        let tasks = Arc::new(tasks);
        let chunk_size = (total / self.workers.max(1)).max(1);

        use rayon::prelude::*;
        let results: Vec<Option<T>> = tasks.par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.par_iter().map(|task| {
                    let result = task();
                    if result.is_some() {
                        completed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                    result
                }).collect::<Vec<_>>()
            })
            .collect();

        let wall_ms = start.elapsed().as_millis() as u64;

        let stats = EvalStats {
            total,
            completed: completed.load(Ordering::Relaxed),
            failed: failed.load(Ordering::Relaxed),
            wall_ms,
            cpu_ms: wall_ms * self.workers as u64,
            workers: self.workers,
        };

        (results, stats)
    }

    pub fn workers(&self) -> usize {
        self.workers
    }
}

impl Default for ParallelEval {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequential evaluator (single-threaded) for comparison.
pub struct SequentialEval;

impl SequentialEval {
    pub fn eval<F, T>(tasks: Vec<F>) -> EvalStats
    where
        F: Fn() -> Option<T>,
    {
        let total = tasks.len();
        let start = std::time::Instant::now();
        let mut completed = 0usize;
        let mut failed = 0usize;

        for task in &tasks {
            if task().is_some() {
                completed += 1;
            } else {
                failed += 1;
            }
        }

        let wall_ms = start.elapsed().as_millis() as u64;

        EvalStats {
            total,
            completed,
            failed,
            wall_ms,
            cpu_ms: wall_ms,
            workers: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_eval_basic() {
        let eval = ParallelEval::with_workers(2);
        let tasks: Vec<_> = (0..100)
            .map(|i| move || Some(i * 2))
            .collect();
        let stats = eval.eval(tasks);
        assert_eq!(stats.completed, 100);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.workers, 2);
    }

    #[test]
    fn test_parallel_eval_with_failures() {
        let eval = ParallelEval::with_workers(2);
        let tasks: Vec<_> = (0..10)
            .map(|i| move || {
                if i % 3 == 0 { None } else { Some(i) }
            })
            .collect();
        let stats = eval.eval(tasks);
        assert!(stats.failed > 0);
        assert!(stats.completed > 0);
        assert_eq!(stats.completed + stats.failed, 10);
    }

    #[test]
    fn test_eval_collect() {
        let eval = ParallelEval::with_workers(2);
        let tasks: Vec<_> = (0..5)
            .map(|i| move || Some(i.to_string()))
            .collect();
        let (results, stats) = eval.eval_collect(tasks);
        assert_eq!(results.len(), 5);
        assert_eq!(stats.completed, 5);
        for r in &results {
            assert!(r.is_some());
        }
    }

    #[test]
    fn test_sequential_eval() {
        let tasks: Vec<_> = (0..50)
            .map(|i| move || Some(i))
            .collect();
        let stats = SequentialEval::eval(tasks);
        assert_eq!(stats.completed, 50);
        assert_eq!(stats.workers, 1);
    }

    #[test]
    fn test_eval_stats_summary() {
        let stats = EvalStats {
            total: 10,
            completed: 8,
            failed: 2,
            wall_ms: 100,
            cpu_ms: 400,
            workers: 4,
        };
        let s = stats.summary();
        assert!(s.contains("tasks="));
        assert!(s.contains("workers=4"));
    }

    #[test]
    fn test_throughput() {
        let stats = EvalStats {
            total: 100, completed: 100, failed: 0,
            wall_ms: 1000, cpu_ms: 4000, workers: 4,
        };
        assert!((stats.throughput() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_auto_detect_workers() {
        let eval = ParallelEval::new();
        assert!(eval.workers() >= 1);
    }
}
