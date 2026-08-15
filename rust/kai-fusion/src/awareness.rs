//! Phase 4.1: Self-Awareness Monitor — meta-cognitive layer that monitors
//! the agent itself. Tracks performance history, detects plateaus,
//! recommends strategy switches, and detects regressions.
//!
//! Mirrors the Python spec from AGI_PLAN.md §4.1.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single observation of the agent's performance at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Monotonic timestamp (seconds since epoch)
    pub timestamp: u64,
    /// Average VFE over the last N turns (lower = better)
    pub avg_vfe: f32,
    /// Fitness score from Darwin evolution (higher = better)
    pub fitness: f32,
    /// Tokens per second (throughput)
    pub tokens_per_sec: f32,
    /// Novelty score (higher = more novel)
    pub novelty: f32,
    /// Current generation counter
    pub generation: usize,
    /// Current tau (subjective time dilation)
    pub tau: f32,
}

/// Detected plateau characteristics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PlateauInfo {
    /// How many consecutive snapshots show no significant improvement
    pub duration: usize,
    /// The metric that has plateaued (e.g., "fitness", "vfe")
    pub metric: String,
    /// The improvement rate over the plateau period (should be ~0)
    pub improvement_rate: f32,
}

/// Strategy recommendations when a plateau is detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Strategy {
    /// Increase mutation rate to explore more
    IncreaseExploration,
    /// Decrease mutation rate to consolidate
    DecreaseExploration,
    /// Switch to a different mutation operator
    SwitchMutationOperator,
    /// Increase temperature for more diverse sampling
    IncreaseTemperature,
    /// Decrease temperature for more focused sampling
    DecreaseTemperature,
    /// Ingest new knowledge to expand the knowledge base
    IngestKnowledge,
    /// Run a longer evaluation with a different benchmark prompt
    ChangeBenchmark,
    /// No action needed
    Continue,
}

/// Self-Awareness Monitor — the meta-cognitive layer.
pub struct SelfAwarenessMonitor {
    /// History of performance observations (max 1000)
    history: VecDeque<PerformanceSnapshot>,
    /// Maximum history size
    max_history: usize,
    /// Plateau detection window (number of snapshots)
    window_size: usize,
    /// Improvement threshold: improvement below this is considered a plateau
    plateau_threshold: f32,
    /// Number of plateaued metrics before triggering a strategy switch
    plateau_trigger: usize,
}

impl SelfAwarenessMonitor {
    pub fn new() -> Self {
        SelfAwarenessMonitor {
            history: VecDeque::with_capacity(1000),
            max_history: 1000,
            window_size: 10,
            plateau_threshold: 0.01,
            plateau_trigger: 2,
        }
    }

    /// Create with custom parameters.
    #[allow(dead_code)]
    pub fn with_params(window_size: usize, plateau_threshold: f32) -> Self {
        SelfAwarenessMonitor {
            history: VecDeque::with_capacity(1000),
            max_history: 1000,
            window_size,
            plateau_threshold,
            plateau_trigger: 2,
        }
    }

    /// Record a new performance observation.
    pub fn record(&mut self, snapshot: PerformanceSnapshot) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(snapshot);
    }

    /// Get the full performance history.
    pub fn history(&self) -> &VecDeque<PerformanceSnapshot> {
        &self.history
    }

    /// Main assessment: analyze current state and recommend an action.
    /// This is the core of §4.1's `assess()` method.
    pub fn assess(&self) -> Assessment {
        let mut plateaued = Vec::new();

        // Check fitness plateau
        if let Some(info) = self.detect_plateau(|s| s.fitness) {
            plateaued.push(info);
        }
        // Check VFE plateau (should be decreasing; if flat, learning stalled)
        if let Some(info) = self.detect_plateau(|s| -s.avg_vfe) {
            plateaued.push(info);
        }
        // Check novelty plateau
        if let Some(info) = self.detect_plateau(|s| s.novelty) {
            plateaued.push(info);
        }

        // Check for regression
        let regression = self.detect_regression();

        // Decide action
        if regression {
            Assessment {
                action: Action::Rollback,
                reason: "Performance regression detected — rolling back last change".to_string(),
                plateaus: plateaued,
            }
        } else if plateaued.len() >= self.plateau_trigger {
            let strategy = self.recommend_strategy(&plateaued);
            Assessment {
                action: Action::SwitchStrategy(strategy),
                reason: format!("{} metrics plateaued for {}+ snapshots",
                    plateaued.len(), self.window_size),
                plateaus: plateaued,
            }
        } else {
            Assessment {
                action: Action::Continue,
                reason: "Performance is improving or stable".to_string(),
                plateaus: plateaued,
            }
        }
    }

    /// Detect if a specific metric has plateaued over the window.
    fn detect_plateau<F>(&self, metric_fn: F) -> Option<PlateauInfo>
    where
        F: Fn(&PerformanceSnapshot) -> f32,
    {
        if self.history.len() < self.window_size {
            return None;
        }

        let window: Vec<f32> = self.history
            .iter()
            .rev()
            .take(self.window_size)
            .map(&metric_fn)
            .collect();

        if window.len() < 2 {
            return None;
        }

        // Compute linear regression slope
        let n = window.len() as f32;
        let sum_x: f32 = (0..window.len()).map(|i| i as f32).sum();
        let sum_y: f32 = window.iter().sum();
        let sum_xy: f32 = window.iter().enumerate().map(|(i, &v)| i as f32 * v).sum();
        let sum_x2: f32 = (0..window.len()).map(|i| (i as f32).powi(2)).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return None;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let improvement_rate = slope.abs();

        if improvement_rate < self.plateau_threshold {
            Some(PlateauInfo {
                duration: self.window_size,
                metric: "unknown".to_string(),
                improvement_rate,
            })
        } else {
            None
        }
    }

    /// Detect if performance has regressed compared to the historical best.
    fn detect_regression(&self) -> bool {
        if self.history.len() < self.window_size + 5 {
            return false;
        }

        // Compare recent average fitness to the best average in the full history
        let recent: Vec<f32> = self.history.iter().rev().take(self.window_size).map(|s| s.fitness).collect();
        let recent_avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;

        // Find the best average over any window_size-length sliding window
        let all: Vec<f32> = self.history.iter().map(|s| s.fitness).collect();
        let mut best_avg = f32::MIN;
        for w in all.windows(self.window_size) {
            let avg = w.iter().sum::<f32>() / self.window_size as f32;
            if avg > best_avg { best_avg = avg; }
        }

        // Regression if recent is significantly worse than the historical best
        best_avg > 0.0 && recent_avg < best_avg * 0.9
    }

    /// Recommend a strategy based on which metrics have plateaued.
    fn recommend_strategy(&self, _plateaus: &[PlateauInfo]) -> Strategy {
        // Simple heuristic: alternate between exploration and consolidation
        let recent_novelty = self.history.back().map(|s| s.novelty).unwrap_or(0.5);

        if recent_novelty < 0.3 {
            // Low novelty → need more exploration
            Strategy::IncreaseExploration
        } else if recent_novelty > 0.7 {
            // High novelty but plateau → consolidate
            Strategy::DecreaseExploration
        } else {
            // Medium novelty → try changing the benchmark or mutation operator
            Strategy::SwitchMutationOperator
        }
    }

    /// Get a summary of recent performance.
    pub fn summary(&self) -> PerformanceSummary {
        if self.history.is_empty() {
            return PerformanceSummary::default();
        }

        let n = self.history.len();
        let window = 10.min(n);
        let recent: Vec<&PerformanceSnapshot> = self.history.iter().rev().take(window).collect();

        // For trend: compare average of oldest window to average of newest window
        let half = window / 2;
        let first_avg: f32 = self.history.iter().take(half.max(1)).map(|s| s.fitness).sum::<f32>() / (half.max(1) as f32);
        let last_avg: f32 = self.history.iter().rev().take(half.max(1)).map(|s| s.fitness).sum::<f32>() / (half.max(1) as f32);

        let avg_fitness: f32 = recent.iter().map(|s| s.fitness).sum::<f32>() / recent.len() as f32;
        let avg_vfe: f32 = recent.iter().map(|s| s.avg_vfe).sum::<f32>() / recent.len() as f32;
        let avg_novelty: f32 = recent.iter().map(|s| s.novelty).sum::<f32>() / recent.len() as f32;

        PerformanceSummary {
            total_snapshots: n,
            avg_fitness,
            avg_vfe,
            avg_novelty,
            fitness_trend: last_avg - first_avg,
            latest_generation: self.history.back().map(|s| s.generation).unwrap_or(0),
            latest_tau: self.history.back().map(|s| s.tau).unwrap_or(1.0),
        }
    }
}

impl Default for SelfAwarenessMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// The action recommended by the monitor.
#[derive(Debug)]
pub enum Action {
    /// Continue with current strategy
    Continue,
    /// Roll back the last change (regression detected)
    Rollback,
    /// Switch to a new strategy
    SwitchStrategy(Strategy),
}

/// Full assessment result including action, reason, and plateaus.
#[derive(Debug)]
pub struct Assessment {
    pub action: Action,
    pub reason: String,
    pub plateaus: Vec<PlateauInfo>,
}

/// A performance summary for display.
#[derive(Debug, Default)]
pub struct PerformanceSummary {
    pub total_snapshots: usize,
    pub avg_fitness: f32,
    pub avg_vfe: f32,
    pub avg_novelty: f32,
    pub fitness_trend: f32,
    pub latest_generation: usize,
    pub latest_tau: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_records_history() {
        let mut monitor = SelfAwarenessMonitor::new();
        for i in 0..5usize {
            monitor.record(PerformanceSnapshot {
                timestamp: i as u64,
                avg_vfe: 0.5,
                fitness: 0.5 + i as f32 * 0.1,
                tokens_per_sec: 10.0,
                novelty: 0.5,
                generation: i,
                tau: 1.0,
            });
        }
        assert_eq!(monitor.history().len(), 5);
    }

    #[test]
    fn test_plateau_detection() {
        let mut monitor = SelfAwarenessMonitor::with_params(5, 0.001);
        // 20 identical fitness values → plateau
        for i in 0..20usize {
            monitor.record(PerformanceSnapshot {
                timestamp: i as u64,
                avg_vfe: 0.5,
                fitness: 0.5,
                tokens_per_sec: 10.0,
                novelty: 0.5,
                generation: i,
                tau: 1.0,
            });
        }
        let assessment = monitor.assess();
        match assessment.action {
            Action::Continue => {} // OK — not enough plateaus yet
            Action::SwitchStrategy(_) => {} // Also OK — plateaus detected
            Action::Rollback => panic!("should not rollback on plateau"),
        }
    }

    #[test]
    fn test_regression_detection() {
        let mut monitor = SelfAwarenessMonitor::with_params(5, 0.01);
        // First 10 at high fitness
        for i in 0..10usize {
            monitor.record(PerformanceSnapshot {
                timestamp: i as u64,
                avg_vfe: 0.3,
                fitness: 0.9,
                tokens_per_sec: 10.0,
                novelty: 0.5,
                generation: i,
                tau: 1.0,
            });
        }
        // Next 10 at low fitness (regression)
        for i in 10..20usize {
            monitor.record(PerformanceSnapshot {
                timestamp: i as u64,
                avg_vfe: 0.3,
                fitness: 0.3,
                tokens_per_sec: 10.0,
                novelty: 0.5,
                generation: i,
                tau: 1.0,
            });
        }
        let assessment = monitor.assess();
        assert!(matches!(assessment.action, Action::Rollback),
            "should detect regression, got: {:?}", assessment.action);
    }

    #[test]
    fn test_summary() {
        let mut monitor = SelfAwarenessMonitor::new();
        for i in 0..20usize {
            monitor.record(PerformanceSnapshot {
                timestamp: i as u64,
                avg_vfe: 0.5 - i as f32 * 0.01,
                fitness: 0.3 + i as f32 * 0.03,
                tokens_per_sec: 10.0,
                novelty: 0.5,
                generation: i,
                tau: 1.0,
            });
        }
        let summary = monitor.summary();
        assert_eq!(summary.total_snapshots, 20);
        assert!(summary.fitness_trend > 0.0, "fitness should be trending up, got {}", summary.fitness_trend);
    }

    #[test]
    fn test_history_overflow() {
        let mut monitor = SelfAwarenessMonitor { max_history: 5, ..SelfAwarenessMonitor::new() };
        for i in 0..10usize {
            monitor.record(PerformanceSnapshot {
                timestamp: i as u64,
                avg_vfe: 0.5,
                fitness: i as f32,
                tokens_per_sec: 10.0,
                novelty: 0.5,
                generation: i,
                tau: 1.0,
            });
        }
        assert_eq!(monitor.history().len(), 5);
        // Should keep the most recent 5
        assert_eq!(monitor.history().front().unwrap().timestamp, 5);
    }
}
