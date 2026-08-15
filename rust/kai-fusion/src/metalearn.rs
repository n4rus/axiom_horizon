//! Phase 5.1: Meta-Learning — the self-improvement engine improves itself.
//!
//! From AGI_PLAN.md §5.1:
//! "The self-improvement engine itself must improve.
//! Feedback loop:
//!   1. Current mutation rate = X
//!   2. Generate N alternative mutation strategies
//!   3. Test each for M cycles
//!   4. Promote best strategy
//!   5. Repeat — the improvement loop is itself evolving
//!
//! Risk: The improvement loop could diverge. Safety protocols must
//! monitor entropy and roll back unstable strategies."

use serde::{Deserialize, Serialize};

/// A meta-learning strategy — parameters that control the improvement loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaStrategy {
    /// Strategy name/identifier
    pub name: String,
    /// Mutation rate (probability of mutating a gene)
    pub mutation_rate: f32,
    /// Selection pressure (higher = more aggressive selection)
    pub selection_pressure: f32,
    /// Population size for parallel evaluation
    pub population_size: usize,
    /// Number of generations per meta-cycle
    pub generations: usize,
    /// Crossover probability
    pub crossover_rate: f32,
    /// Elitism fraction (top N% always survive)
    pub elitism: f32,
    /// Fitness score (accumulated from cycles)
    pub fitness: f32,
    /// Number of cycles tested
    pub cycles_tested: u32,
    /// Whether this strategy is currently active
    pub active: bool,
}

impl MetaStrategy {
    pub fn default_strategy() -> Self {
        MetaStrategy {
            name: "default".to_string(),
            mutation_rate: 0.1,
            selection_pressure: 0.5,
            population_size: 8,
            generations: 10,
            crossover_rate: 0.7,
            elitism: 0.1,
            fitness: 0.0,
            cycles_tested: 0,
            active: true,
        }
    }

    /// Create a random mutant of this strategy.
    pub fn mutate(&self) -> Self {
        let mut mutant = self.clone();
        mutant.name = format!("{}_mut", self.name);
        mutant.active = false;
        mutant.fitness = 0.0;
        mutant.cycles_tested = 0;

        // Mutate one parameter randomly
        let rng_val = pseudo_random();
        if rng_val < 0.25 {
            mutant.mutation_rate = (self.mutation_rate + (pseudo_random() - 0.5) * 0.1).clamp(0.01, 0.5);
        } else if rng_val < 0.5 {
            mutant.selection_pressure = (self.selection_pressure + (pseudo_random() - 0.5) * 0.2).clamp(0.1, 1.0);
        } else if rng_val < 0.75 {
            mutant.population_size = ((self.population_size as f32 + (pseudo_random() - 0.5) * 4.0) as usize).max(2).min(32);
        } else {
            mutant.crossover_rate = (self.crossover_rate + (pseudo_random() - 0.5) * 0.2).clamp(0.1, 1.0);
        }

        mutant
    }

    /// Create offspring from two parent strategies.
    pub fn crossover(a: &MetaStrategy, b: &MetaStrategy) -> Self {
        let mut child = a.clone();
        child.name = format!("{}_x_{}", a.name, b.name);
        child.active = false;
        child.fitness = 0.0;
        child.cycles_tested = 0;

        // Uniform crossover
        if pseudo_random() < 0.5 { child.mutation_rate = b.mutation_rate; }
        if pseudo_random() < 0.5 { child.selection_pressure = b.selection_pressure; }
        if pseudo_random() < 0.5 { child.population_size = b.population_size; }
        if pseudo_random() < 0.5 { child.crossover_rate = b.crossover_rate; }
        if pseudo_random() < 0.5 { child.elitism = b.elitism; }

        child
    }

    /// Record a cycle result.
    pub fn record_cycle(&mut self, cycle_fitness: f32) {
        self.cycles_tested += 1;
        // Exponential moving average of fitness
        let alpha = 0.2;
        self.fitness = self.fitness * (1.0 - alpha) + cycle_fitness * alpha;
    }
}

/// Meta-Learner — manages a population of improvement strategies.
pub struct MetaLearner {
    strategies: Vec<MetaStrategy>,
    current_idx: usize,
    generation: u32,
    max_strategies: usize,
}

impl MetaLearner {
    pub fn new() -> Self {
        MetaLearner {
            strategies: vec![MetaStrategy::default_strategy()],
            current_idx: 0,
            generation: 0,
            max_strategies: 100,
        }
    }

    /// Get the currently active strategy.
    pub fn active_strategy(&self) -> &MetaStrategy {
        &self.strategies[self.current_idx]
    }

    /// Get all strategies.
    pub fn strategies(&self) -> &[MetaStrategy] {
        &self.strategies
    }

    /// Record a cycle result for the active strategy.
    pub fn record_cycle(&mut self, fitness: f32) {
        self.strategies[self.current_idx].record_cycle(fitness);
    }

    /// Evolve: generate mutants and crossovers, keep the best.
    pub fn evolve(&mut self) {
        self.generation += 1;

        // Generate mutants from top strategies
        let mut new_strategies: Vec<MetaStrategy> = Vec::new();
        let sorted: Vec<_> = {
            let mut s: Vec<_> = self.strategies.iter().enumerate().collect();
            s.sort_by(|a, b| b.1.fitness.partial_cmp(&a.1.fitness).unwrap());
            s
        };

        // Mutate top 3
        for (idx, _) in sorted.iter().take(3) {
            let mutant = self.strategies[*idx].mutate();
            new_strategies.push(mutant);
        }

        // Crossover top 2 pairs
        if sorted.len() >= 2 {
            let child = MetaStrategy::crossover(
                &self.strategies[sorted[0].0],
                &self.strategies[sorted[1].0],
            );
            new_strategies.push(child);
        }

        self.strategies.extend(new_strategies);

        // Prune: keep only the best max_strategies
        if self.strategies.len() > self.max_strategies {
            self.strategies.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            self.strategies.truncate(self.max_strategies);
        }

        // Activate the best
        if let Some(best_idx) = self.strategies.iter()
            .enumerate()
            .max_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap())
            .map(|(i, _)| i)
        {
            for s in &mut self.strategies { s.active = false; }
            self.strategies[best_idx].active = true;
            self.current_idx = best_idx;
        }
    }

    /// Check for divergence: if fitness variance is too high.
    pub fn divergence_detected(&self) -> bool {
        if self.strategies.len() < 3 { return false; }
        let mean: f32 = self.strategies.iter().map(|s| s.fitness).sum::<f32>()
            / self.strategies.len() as f32;
        let variance: f32 = self.strategies.iter()
            .map(|s| (s.fitness - mean).powi(2))
            .sum::<f32>() / self.strategies.len() as f32;
        variance > 0.05 // Threshold for divergence
    }

    /// Rollback: reset to default strategy.
    pub fn rollback(&mut self) {
        self.strategies.clear();
        self.strategies.push(MetaStrategy::default_strategy());
        self.current_idx = 0;
        self.generation = 0;
    }

    /// Current generation.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let active = self.active_strategy();
        format!(
            "gen={} strategies={} active='{}' fitness={:.3} div={}",
            self.generation,
            self.strategies.len(),
            active.name,
            active.fitness,
            if self.divergence_detected() { "YES" } else { "no" },
        )
    }
}

impl Default for MetaLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Phase 5.1 Meta-Evolution Driver — wraps DarwinArchive in a meta-loop
// ---------------------------------------------------------------------------

/// Configuration for meta-evolution cycles.
#[derive(Debug, Clone)]
pub struct MetaEvolveConfig {
    /// Number of meta-generations
    pub meta_generations: usize,
    /// Darwin cycles per strategy evaluation
    pub cycles_per_strategy: usize,
    /// Maximum strategies to keep
    #[allow(dead_code)]
    pub max_strategies: usize,
    /// Whether to reset archive between strategy tests
    #[allow(dead_code)]
    pub reset_archive: bool,
}

impl Default for MetaEvolveConfig {
    fn default() -> Self {
        Self {
            meta_generations: 10,
            cycles_per_strategy: 5,
            max_strategies: 20,
            reset_archive: true,
        }
    }
}

/// Run a full meta-evolution loop.
///
/// For each meta-generation:
/// 1. Get the active MetaStrategy from the meta-learner
/// 2. Run `cycles_per_strategy` Darwin cycles using that strategy
/// 3. Measure average fitness improvement
/// 4. Record the result in the meta-learner
/// 5. Evolve the meta-learner (generate new strategies, promote best)
/// 6. Detect divergence and rollback if needed
///
/// `run_darwin_cycle` is a closure that:
/// - Takes a `&MetaStrategy` to configure the evolution
/// - Runs one Darwin generation
/// - Returns the best fitness after that generation
pub fn run_meta_evolution<F>(
    meta_learner: &mut MetaLearner,
    config: &MetaEvolveConfig,
    mut run_darwin_cycle: F,
) -> Result<MetaStrategy, String>
where
    F: FnMut(&MetaStrategy) -> f32,
{
    let mut best_overall = MetaStrategy::default_strategy();
    let mut best_fitness = 0.0f32;

    for meta_gen in 0..config.meta_generations {
        println!("\n=== Meta-Generation {}/{} ===", meta_gen + 1, config.meta_generations);
        println!("  Current active: {}", meta_learner.summary());

        // Check for divergence before running
        if meta_learner.divergence_detected() {
            println!("  ⚠ Divergence detected! Rolling back to default strategy.");
            meta_learner.rollback();
            println!("  => Rolled back: {}", meta_learner.summary());
        }

        // Run cycles_per_strategy Darwin cycles with the active strategy
        let active = meta_learner.active_strategy().clone();
        let mut cycle_fitnesses = Vec::with_capacity(config.cycles_per_strategy);
        for cycle in 0..config.cycles_per_strategy {
            let fit = run_darwin_cycle(&active);
            cycle_fitnesses.push(fit);
            print!("  Cycle {}/{}: fitness={:.4}", cycle + 1, config.cycles_per_strategy, fit);
            if cycle < config.cycles_per_strategy - 1 {
                println!();
            }
        }
        let avg_cycle_fitness = if cycle_fitnesses.is_empty() {
            0.0
        } else {
            cycle_fitnesses.iter().sum::<f32>() / cycle_fitnesses.len() as f32
        };

        // Track best overall
        if avg_cycle_fitness > best_fitness {
            best_fitness = avg_cycle_fitness;
            best_overall = active.clone();
        }

        // Record cycle result
        meta_learner.record_cycle(avg_cycle_fitness);
        println!(" | Avg: {:.4} | {}", avg_cycle_fitness, meta_learner.summary());

        // Evolve: generate new strategies from top performers
        meta_learner.evolve();
        println!("  Evolved: {} strategies in pool", meta_learner.strategies().len());
    }

    println!("\n=== Meta-Evolution Complete ===");
    println!("  Best strategy: {} (fitness={:.4})", best_overall.name, best_fitness);
    println!("    mutation_rate={:.3}", best_overall.mutation_rate);
    println!("    selection_pressure={:.3}", best_overall.selection_pressure);
    println!("    population_size={}", best_overall.population_size);
    println!("    crossover_rate={:.3}", best_overall.crossover_rate);
    println!("    elitism={:.3}", best_overall.elitism);
    println!("  Final learner: {}", meta_learner.summary());

    Ok(best_overall)
}

/// Deterministic-ish pseudo-random [0, 1) for tests.
fn pseudo_random() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    (hasher.finish() % 10000) as f32 / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_strategy() {
        let s = MetaStrategy::default_strategy();
        assert_eq!(s.name, "default");
        assert!(s.mutation_rate > 0.0 && s.mutation_rate < 1.0);
        assert!(s.active);
    }

    #[test]
    fn test_mutate_creates_different() {
        let s = MetaStrategy::default_strategy();
        // Run mutate several times; at least one should differ
        let mut different = false;
        for _ in 0..20 {
            let m = s.mutate();
            if m.mutation_rate != s.mutation_rate
                || m.selection_pressure != s.selection_pressure
                || m.population_size != s.population_size
                || m.crossover_rate != s.crossover_rate
            {
                different = true;
                break;
            }
        }
        assert!(different, "mutate should eventually produce a different strategy");
    }

    #[test]
    fn test_crossover() {
        let mut a = MetaStrategy::default_strategy();
        a.mutation_rate = 0.2;
        let mut b = MetaStrategy::default_strategy();
        b.mutation_rate = 0.4;
        let child = MetaStrategy::crossover(&a, &b);
        assert!(child.name.contains("_x_"));
        assert!(!child.active);
    }

    #[test]
    fn test_metalearner_initialization() {
        let ml = MetaLearner::new();
        assert_eq!(ml.strategies().len(), 1);
        assert_eq!(ml.generation(), 0);
    }

    #[test]
    fn test_record_cycle_updates_fitness() {
        let mut ml = MetaLearner::new();
        ml.record_cycle(0.8);
        assert!(ml.active_strategy().fitness > 0.0);
    }

    #[test]
    fn test_evolve_increases_generation() {
        let mut ml = MetaLearner::new();
        ml.record_cycle(0.5);
        ml.evolve();
        assert_eq!(ml.generation(), 1);
        assert!(ml.strategies().len() > 1);
    }

    #[test]
    fn test_divergence_detection() {
        let mut ml = MetaLearner::new();
        ml.strategies.clear();
        // Create strategies with wildly different fitness
        ml.strategies.push(MetaStrategy { fitness: 0.9, name: "a".to_string(), ..MetaStrategy::default_strategy() });
        ml.strategies.push(MetaStrategy { fitness: 0.1, name: "b".to_string(), ..MetaStrategy::default_strategy() });
        ml.strategies.push(MetaStrategy { fitness: 0.9, name: "c".to_string(), ..MetaStrategy::default_strategy() });
        assert!(ml.divergence_detected());
    }

    #[test]
    fn test_rollback() {
        let mut ml = MetaLearner::new();
        ml.record_cycle(0.8);
        ml.evolve();
        ml.rollback();
        assert_eq!(ml.strategies().len(), 1);
        assert_eq!(ml.generation(), 0);
    }

    #[test]
    fn test_summary() {
        let ml = MetaLearner::new();
        let s = ml.summary();
        assert!(s.contains("gen="));
        assert!(s.contains("strategies="));
    }
}
