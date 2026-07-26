//! Bracket state: Euler-clock awareness for the AGI agent.
//!
//! In eve.py, the bracket is: [tau, E, age, cycles] with state_vector
//! = [tau, E, age, cycles, h, base_ms, phi].
//!
//! phi = tau * age / E  (phase = experience density)
//! age = h * base_ms * tau / 1000
//!
//! The bracket state drives VFE-coupled time dilation:
//! - tau grows with VFE (surprise → slower, more careful thinking)
//! - tau decays toward 1.0 during idle (normalize time)
//! - When VFE exceeds threshold, collapse tau → 1.0 (reset)

use std::fs;
use std::path::Path;

/// Default decay rate per second of idle time (τ → 1.0).
const DEFAULT_DECAY_RATE: f32 = 0.1;
/// Maximum tau clamp.
const MAX_TAU: f32 = 1e12;
/// VFE threshold for tau collapse.
const DEFAULT_VFE_THRESHOLD: f32 = 10.0;
/// Minimum variance for attractor-driven tau update.
const MIN_VAR: f32 = 1e-10;
/// Default base time per step in milliseconds.
const DEFAULT_BASE_MS: f32 = 5.0;

/// Agent's Euler-clock bracket state.
#[derive(Debug, Clone)]
pub struct BracketState {
    pub tau: f32,
    pub vfe: f32,
    pub cycles: usize,
    pub h: usize,
    pub base_ms: f32,
    vfe_threshold: f32,
    max_tau: f32,
    decay_rate: f32,
}

impl BracketState {
    /// Create fresh bracket at time-normal (tau=1.0).
    pub fn new() -> Self {
        Self {
            tau: 1.0,
            vfe: 0.0,
            cycles: 0,
            h: 0,
            base_ms: DEFAULT_BASE_MS,
            vfe_threshold: DEFAULT_VFE_THRESHOLD,
            max_tau: MAX_TAU,
            decay_rate: DEFAULT_DECAY_RATE,
        }
    }

    /// Advance clock with observed VFE. Returns subjective ms elapsed.
    /// Mirrors `EulerClock.cycle()` from eve.py.
    pub fn update(&mut self, real_ms: f32, vfe: f32, attractor_variance: f32) -> f32 {
        self.cycles += 1;
        self.h += 1;

        self.vfe = vfe + 0.1 * attractor_variance;

        if attractor_variance > MIN_VAR {
            let new_tau = (1.0 / attractor_variance).min(self.max_tau);
            self.tau = 0.9 * self.tau + 0.1 * new_tau;
        }

        let t_subj = real_ms * self.tau;

        if self.vfe > self.vfe_threshold {
            self.tau = 1.0;
            self.vfe = 0.0;
        }

        t_subj
    }

    /// Decay tau toward 1.0 over idle time. Mirrors `tau_decay()` in vfe.rs.
    pub fn decay_tau(&mut self, idle_seconds: f32) {
        if idle_seconds <= 0.0 || self.tau <= 1.0 {
            return;
        }
        let factor = (-self.decay_rate * idle_seconds).exp();
        self.tau = 1.0 + (self.tau - 1.0) * factor;
    }

    /// Subjective age in seconds: h * base_ms * tau / 1000.
    pub fn age(&self) -> f32 {
        (self.h as f32 * self.base_ms * self.tau) / 1000.0
    }

    /// Phase = experience density: phi = tau * age / E.
    /// Uses |E| when E is negative; avoids division by zero.
    pub fn phi(&self) -> f32 {
        let age = self.age();
        let e = self.vfe.abs();
        if e < 1e-10 {
            self.tau * age
        } else {
            self.tau * age / e
        }
    }

    /// Bracket-line string for model prompt injection.
    pub fn bracket_line(&self) -> String {
        format!(
            "[tau={:.4e} VFE={:.4e} age={:.4e} cycles={}]",
            self.tau, self.vfe, self.age(), self.cycles
        )
    }

    /// Full 7-element state vector: [tau, E, age, cycles, h, base_ms, phi].
    pub fn state_vector(&self) -> Vec<f32> {
        vec![
            self.tau,
            self.vfe,
            self.age(),
            self.cycles as f32,
            self.h as f32,
            self.base_ms,
            self.phi(),
        ]
    }

    /// Save bracket state to disk (JSON).
    pub fn save(&self, path: &str) -> Result<(), String> {
        let d = serde_json::json!({
            "tau": self.tau,
            "vfe": self.vfe,
            "h": self.h,
            "cycles": self.cycles,
            "base_ms": self.base_ms,
            "vfe_threshold": self.vfe_threshold,
            "max_tau": self.max_tau,
            "decay_rate": self.decay_rate,
        });
        let dir = Path::new(path).parent().unwrap_or(Path::new("."));
        if !dir.exists() {
            fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        }
        fs::write(path, serde_json::to_string_pretty(&d).map_err(|e| e.to_string())?)
            .map_err(|e| format!("write: {e}"))?;
        Ok(())
    }

    /// Load bracket state from disk (JSON).
    pub fn load(path: &str) -> Result<Self, String> {
        let raw = fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
        let d: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        Ok(Self {
            tau: d.get("tau").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            vfe: d.get("vfe").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            h: d.get("h").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            cycles: d.get("cycles").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            base_ms: d.get("base_ms").and_then(|v| v.as_f64()).unwrap_or(5.0) as f32,
            vfe_threshold: d.get("vfe_threshold").and_then(|v| v.as_f64()).unwrap_or(10.0) as f32,
            max_tau: d.get("max_tau").and_then(|v| v.as_f64()).unwrap_or(1e12) as f32,
            decay_rate: d.get("decay_rate").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32,
        })
    }

    /// Reset to defaults.
    pub fn reset(&mut self) {
        self.tau = 1.0;
        self.vfe = 0.0;
        self.cycles = 0;
        self.h = 0;
    }
}

impl Default for BracketState {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_bracket_initial_state() {
        let b = BracketState::new();
        assert_eq!(b.tau, 1.0);
        assert_eq!(b.vfe, 0.0);
        assert_eq!(b.cycles, 0);
        assert_eq!(b.h, 0);
        assert_eq!(b.age(), 0.0);
    }

    #[test]
    fn test_bracket_update_increases() {
        let mut b = BracketState::new();
        let t = b.update(5.0, 0.5, 0.01);
        assert!(t > 0.0);
        assert_eq!(b.cycles, 1);
        assert_eq!(b.h, 1);
        assert!(b.vfe > 0.0);
    }

    #[test]
    fn test_bracket_phi_positive() {
        let mut b = BracketState::new();
        b.update(10.0, 2.0, 0.05);
        assert!(b.phi() > 0.0);
    }

    #[test]
    fn test_bracket_phi_safe_when_vfe_zero() {
        let b = BracketState::new();
        // phi when vfe=0 should be tau * age (no division by zero)
        let phi = b.phi();
        assert_eq!(phi, 0.0); // age=0, phi=tau*0/vfe=0 since vfe=0 → tau*age branch
    }

    #[test]
    fn test_bracket_state_vector_length() {
        let b = BracketState::new();
        let sv = b.state_vector();
        assert_eq!(sv.len(), 7);
    }

    #[test]
    fn test_bracket_cycle_collapse() {
        let mut b = BracketState::new();
        // push vfe way above threshold → tau collapses
        b.update(5.0, 100.0, 0.0);
        assert_eq!(b.tau, 1.0, "tau should collapse to 1.0 when VFE > threshold");
        assert_eq!(b.vfe, 0.0, "VFE should reset after collapse");
    }

    #[test]
    fn test_bracket_save_load_roundtrip() {
        let mut b = BracketState::new();
        b.update(5.0, 0.3, 0.01);
        b.update(5.0, 0.5, 0.02);

        let path = "/tmp/kai_bracket_test.json";
        b.save(path).unwrap();

        let loaded = BracketState::load(path).unwrap();
        assert_eq!(loaded.tau, b.tau);
        assert_eq!(loaded.cycles, b.cycles);
        assert_eq!(loaded.h, b.h);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_bracket_decay() {
        let mut b = BracketState::new();
        b.tau = 3.0;
        b.decay_tau(10.0);
        assert!(b.tau < 3.0, "tau should decay toward 1.0");
        assert!(b.tau > 1.0, "tau should stay above 1.0");
    }

    #[test]
    fn test_bracket_no_decay_when_idle_zero() {
        let mut b = BracketState::new();
        b.tau = 3.0;
        b.decay_tau(0.0);
        assert_eq!(b.tau, 3.0, "no decay with zero idle time");
    }

    #[test]
    fn test_bracket_no_decay_when_tau_is_one() {
        let mut b = BracketState::new();
        b.decay_tau(100.0);
        assert_eq!(b.tau, 1.0, "no decay when tau starts at 1.0");
    }
}
