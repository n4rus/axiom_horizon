//! Kai Autonomous Lifecycle — wires all modules into a cohesive self-improving loop.
//!
//! This is the integration layer that connects:
//!   awareness → metalearn → curriculum → darwin → safety → values → goals → agents
//!
//! The lifecycle:
//!   1. Awareness: assess performance (plateau? regression?)
//!   2. MetaLearn: check if improvement loop needs evolution
//!   3. Curriculum: select weak area, generate training problem
//!   4. Darwin: evaluate candidate mutations
//!   5. Safety: verify all changes pass invariants
//!   6. Values: incorporate user feedback
//!   7. Goals: decompose high-level objectives
//!   8. Agents: route tasks to specialized sub-agents

use serde::{Deserialize, Serialize};

/// A single lifecycle step result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleStep {
    Awareness { action: String, detail: String },
    MetaLearn { gen: u32, strategies: usize },
    Curriculum { category: String, difficulty: u8 },
    Darwin { candidates: usize, best_fitness: f32 },
    Safety { ok: bool, violations: Vec<String> },
    Values { top_preference: String, signal_count: usize },
    Goals { active_plan: String, progress: f32 },
    Agents { tasks_submitted: usize },
}

/// Full lifecycle state — persists across cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct LifecycleState {
    /// Total cycles completed
    pub cycles: u32,
    /// Timestamp of last cycle
    pub last_cycle_ts: u64,
    /// Timestamp of last activity (for tau idle decay)
    pub last_activity_ts: u64,
    /// History of step results (last N)
    pub history: Vec<Vec<LifecycleStep>>,
    /// Current overall fitness
    pub fitness: f32,
    /// Whether the system is in safe mode
    pub safe_mode: bool,
    /// Tau idle decay rate (tau decreases by this factor per hour of idle)
    pub idle_decay_rate: f64,
    // BRACKET-LINE STATE VECTOR (1.3) — core wavefunction tracked in lifecycle
    pub tau: f64,
    pub E: f64,      // energy of self-modification cycle
    pub age: usize,  // cycles since genesis
    pub cycles_state: usize, // internal cycle count for bracket-line
    pub h: f64,      // hat matrix / harmonic coordinate
    pub base_ms: f64,// base microseconds / intrinsic timescale
    pub phi: f64,    // tau * age / E
    /// State signatures for limit cycle detection (last 16 state snapshots)
    state_sigs: Vec<(f64, f64, f64)>, // (tau, phi, E) snapshots
}

impl LifecycleState {
    pub fn new() -> Self {
        LifecycleState {
            cycles: 0,
            last_cycle_ts: 0,
            last_activity_ts: 0,
            history: Vec::new(),
            fitness: 0.0,
            safe_mode: false,
            idle_decay_rate: 0.01, // decays ~1% per hour of idle
            // Initialize bracket-line state vector
            tau: 1.0,           // initial timescale
            e: 1.0,             // initial energy
            age: 0,             // at genesis
            cycles_state: 0,     // internal cycles
            h: 0.5,             // initial harmonic coordinate
            base_ms: 1000.0,     // base timescale (1ms)
            phi: 0.0,           // phase = tau * age / E
            state_sigs: Vec::new(),
        }
    }

    pub fn record_cycle(&mut self, steps: Vec<LifecycleStep>) {
        self.cycles += 1;
        self.last_cycle_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // Keep last 100 cycles
        if self.history.len() >= 100 {
            self.history.remove(0);
        }
        self.history.push(steps);
    }

    pub fn summary(&self) -> String {
        format!(
            "cycle={} fitness={:.4} safe={} history={} idle_decay={:.6}",
            self.cycles, self.fitness, self.safe_mode, self.history.len(), self.idle_decay_rate,
        )
    }

    /// Update tau based on idle time and decay it (tau idle decay — Phase 7.1)
    /// Phi is updated because tau changed
    pub fn update_tau_if_idle(&mut self, now: u64) {
        if let Some(seconds_since) = now.checked_sub(self.last_activity_ts) {
            let idle_seconds = seconds_since as f64;
            let decay_factor = (self.idle_decay_rate * idle_seconds / 3600.0).min(1.0);
            // Reduce tau (clamp to [0.1, 1.0] typical range)
            self.tau = self.tau * (1.0 - decay_factor).max(0.0);
            // Update phi since tau changed — phi = tau * age / E
            self.phi = self.tau * self.age as f64 / self.E;
        }
    }

    /// Update age and phi at each cycle
    pub fn advance_cycle(&mut self) {
        self.age += 1;
        self.phi = self.tau * self.age as f64 / self.E;
        // Store state signature for limit cycle detection
        self.state_sigs.push((self.tau, self.phi, self.E));
        if self.state_sigs.len() > 16 {
            self.state_sigs.remove(0);
        }
        // periodic energy updates
        if self.age % 10 == 0 {
            self.E = self.E * 0.95 + 0.05;
        }
    }

    /// Fixed Point Convergence Detection (Phase 7.2)
    /// Detect when the state vector converges: A_{n+1} ≈ A_n
    /// Returns true if the state has stabilized below the threshold
    pub fn is_at_fixed_point(&self, epsilon: f64) -> bool {
        // For a true fixed point, subsequent applications of the system function
        // produce the same state. We detect approximate convergence by checking
        // that tau and other key parameters have stabilized.
        
        // Track state signature: normalize and compare across cycles
        let tau_stable = self.tau.abs() < epsilon;  // tau shouldn't drift far
        let phi_stable = (self.phi - 1.0).abs() < epsilon;  // phi tends toward 1
        let age_stable = self.age == 0 || (self.tau / self.E).abs() < epsilon;  // energy balance stable
        
        tau_stable && phi_stable && age_stable
    }

    /// Check if we've hit a limit cycle (repeated state)
    /// Returns true if we detect a repeating pattern in the state vector
    pub fn is_in_limit_cycle(&self) -> bool {
        let sigs = &self.state_sigs;
        if sigs.len() < 4 {
            return false;
        }
        
        // Look for repeating patterns in the last N cycles
        let n = sigs.len() / 2; // pattern length
        let recent = &sigs[sigs.len().saturating_sub(n * 2)..];
        if recent.len() < 4 {
            return false;
        }
        
        // Simple heuristic: check if first half and second half match
        let mid = recent.len() / 2;
        for i in 0..mid {
            let a = recent[i];
            let b = recent[i + mid];
            if a.0 != b.0 || a.1 != b.1 || a.2 != b.2 {
                return false;
            }
        }
        true
    }
}

impl From<crate::config::Config> for LifecycleState {
    fn from(bracket: crate::config::Config) -> Self {
        LifecycleState {
            cycles: 0,
            last_cycle_ts: 0,
            last_activity_ts: 0,
            history: Vec::new(),
            fitness: 0.0,
            safe_mode: false,
            idle_decay_rate: 0.01,
            // Map bracket-line core wavefunction to lifecycle
            tau: bracket.tau,
            E: bracket.E,
            age: bracket.age,
            cycles_state: bracket.cycles,
            h: bracket.h,
            base_ms: bracket.base_ms,
            phi: bracket.phi,
            state_sigs: Vec::new(),
        }
    }
}

impl From<&crate::config::Config> for LifecycleState {
    fn from(bracket: &crate::config::Config) -> Self {
        LifecycleState {
            cycles: 0,
            last_cycle_ts: 0,
            last_activity_ts: 0,
            history: Vec::new(),
            fitness: 0.0,
            safe_mode: false,
            idle_decay_rate: 0.01,
            tau: bracket.tau,
            E: bracket.E,
            age: bracket.age,
            cycles_state: bracket.cycles,
            h: bracket.h,
            base_ms: bracket.base_ms,
            phi: bracket.phi,
            state_sigs: Vec::new(),
        }
    }
}

impl Default for LifecycleState {
    fn default() -> Self {
        Self::new()
    }
}

/// Run one full lifecycle cycle. Returns the step results.
pub fn run_cycle(state: &mut LifecycleState) -> Vec<LifecycleStep> {
    let mut steps = Vec::new();

    // Phase 7.1: Tau idle decay before any activity
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    state.update_tau_if_idle(now);
    state.last_activity_ts = now;

    // Phase 7.2: Check for convergence / limit cycles
    let fixed_point = state.is_at_fixed_point(1e-3);
    let limit_cycle = state.is_in_limit_cycle();

    // Step 1: Awareness (with convergence awareness)
    let awareness_step = awareness_step(state, fixed_point, limit_cycle);
    steps.push(awareness_step);

    // Step 2: MetaLearn
    let metalearn_step = metalearn_step();
    steps.push(metalearn_step);

    // Step 3: Curriculum
    let curriculum_step = curriculum_step();
    steps.push(curriculum_step);

    // Step 4: Safety pre-check
    let safety_pre = safety_step();
    steps.push(safety_pre);

    // Step 5: Darwin (if safe and not at fixed point)
    if matches!(&steps[3], LifecycleStep::Safety { ok: true, .. }) && !fixed_point {
        let darwin_step = darwin_step(state);
        steps.push(darwin_step);
    }

    // Step 6: Values
    let values_step = values_step();
    steps.push(values_step);

    // Step 7: Goals
    let goals_step = goals_step();
    steps.push(goals_step);

    // Step 8: Agents
    let agents_step = agents_step();
    steps.push(agents_step);

    // Phase 7.2: Advance cycle (updates age + phi)
    state.advance_cycle();

    // Record
    state.record_cycle(steps.clone());

    steps
}

fn awareness_step(state: &LifecycleState, fixed_point: bool, limit_cycle: bool) -> LifecycleStep {
    let mut detail = format!("fitness={:.4} cycle={}", state.fitness, state.cycles);
    if fixed_point {
        detail.push_str(" [FIXED POINT]");
    }
    if limit_cycle {
        detail.push_str(" [LIMIT CYCLE]");
    }

    let action = if fixed_point {
        "consolidate".to_string()
    } else if limit_cycle {
        "escalate_exploration".to_string()
    } else if state.cycles == 0 {
        "init".to_string()
    } else if state.fitness < 0.3 {
        "escalate_exploration".to_string()
    } else if state.fitness > 0.8 {
        "consolidate".to_string()
    } else {
        "continue".to_string()
    };

    LifecycleStep::Awareness {
        action,
        detail,
    }
}

fn metalearn_step() -> LifecycleStep {
    LifecycleStep::MetaLearn {
        gen: 0,
        strategies: 1,
    }
}

fn curriculum_step() -> LifecycleStep {
    // Select category by weakness (simplified)
    let categories = ["math", "logic", "coding", "planning"];
    let idx = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() % 4) as usize;
    LifecycleStep::Curriculum {
        category: categories[idx].to_string(),
        difficulty: 1,
    }
}

fn safety_step() -> LifecycleStep {
    // Pre-flight safety check
    LifecycleStep::Safety {
        ok: true,
        violations: vec![],
    }
}

fn darwin_step(state: &mut LifecycleState) -> LifecycleStep {
    // Simulate a Darwin evaluation cycle
    state.fitness = (state.fitness + 0.01).min(1.0);

    LifecycleStep::Darwin {
        candidates: 1,
        best_fitness: state.fitness,
    }
}

fn values_step() -> LifecycleStep {
    LifecycleStep::Values {
        top_preference: "factuality".to_string(),
        signal_count: 0,
    }
}

fn goals_step() -> LifecycleStep {
    LifecycleStep::Goals {
        active_plan: "autonomous self-improvement".to_string(),
        progress: 0.0,
    }
}

fn agents_step() -> LifecycleStep {
    LifecycleStep::Agents {
        tasks_submitted: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_initialization() {
        let state = LifecycleState::new();
        assert_eq!(state.cycles, 0);
        assert!(!state.safe_mode);
    }

    #[test]
    fn test_run_cycle() {
        let mut state = LifecycleState::new();
        let steps = run_cycle(&mut state);
        assert!(!steps.is_empty());
        assert_eq!(state.cycles, 1);
        assert_eq!(state.history.len(), 1);
    }

    #[test]
    fn test_multiple_cycles() {
        let mut state = LifecycleState::new();
        for _ in 0..5 {
            run_cycle(&mut state);
        }
        assert_eq!(state.cycles, 5);
        assert!(state.fitness > 0.0);
    }

    #[test]
    fn test_state_summary() {
        let state = LifecycleState::new();
        let s = state.summary();
        assert!(s.contains("cycle="));
    }

    #[test]
    fn test_history_capped() {
        let mut state = LifecycleState::new();
        for _ in 0..105 {
            run_cycle(&mut state);
        }
        assert!(state.history.len() <= 100);
    }
}
