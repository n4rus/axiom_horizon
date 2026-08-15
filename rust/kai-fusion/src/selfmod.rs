//! Phase 5.4: Recursive Self-Modification — track agent-level code modifications.
//!
//! From AGI_PLAN.md §5.4:
//! "Level 0: Human writes code (current)
//! Level 1: Agent modifies code
//! Level 2: Agent modifies its modification code
//! Level 3: Agent modifies the modification-of-modification code
//! ...
//! Level N: System converges to a fixed point or diverges.
//!
//! The Tonal Collapse framework predicts:
//! - Fixed point: A_{n+1} = A_n (code stabilizes)
//! - Limit cycle: A_{n+k} = A_n (periodic behavior)
//! - Divergence: ||A_{n+1} - A_n|| grows (instability → reset).
//!
//! Safety: Human must approve crossing to L_{n+1}.
//! Rollback mechanism if divergence detected.
//! Kill switch: hardcoded, not modifiable by agent."

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Modification level — how deeply self-referential the modification is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModLevel {
    /// Level 0: Human writes code
    Human,
    /// Level 1: Agent modifies code
    AgentCode,
    /// Level 2: Agent modifies its modification code
    AgentMetaCode,
    /// Level 3: Agent modifies the modification-of-modification code
    AgentMetaMetaCode,
}

impl ModLevel {
    pub fn label(&self) -> &'static str {
        match self {
            ModLevel::Human => "L0:human",
            ModLevel::AgentCode => "L1:agent-code",
            ModLevel::AgentMetaCode => "L2:agent-meta",
            ModLevel::AgentMetaMetaCode => "L3:agent-meta-meta",
        }
    }

    pub fn numeric(&self) -> u8 {
        match self {
            ModLevel::Human => 0,
            ModLevel::AgentCode => 1,
            ModLevel::AgentMetaCode => 2,
            ModLevel::AgentMetaMetaCode => 3,
        }
    }
}

/// A single self-modification event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModEvent {
    /// Modification level
    pub level: ModLevel,
    /// Target file or module
    pub target: String,
    /// Description of the change
    pub description: String,
    /// Source hash before modification (for rollback verification)
    pub hash_before: String,
    /// Source hash after modification
    pub hash_after: String,
    /// Fitness before modification
    pub fitness_before: f32,
    /// Fitness after modification
    pub fitness_after: f32,
    /// Timestamp
    pub timestamp: u64,
    /// Whether this was approved by a human
    pub human_approved: bool,
}

impl ModEvent {
    /// Fitness delta (positive = improvement).
    pub fn fitness_delta(&self) -> f32 {
        self.fitness_after - self.fitness_before
    }

    /// Whether this modification improved fitness.
    pub fn improved(&self) -> bool {
        self.fitness_delta() > 0.0
    }
}

/// Self-Modification Tracker — monitors recursive self-modification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModTracker {
    /// Current level
    current_level: ModLevel,
    /// All modification events
    events: Vec<ModEvent>,
    /// Maximum events to keep
    max_events: usize,
    /// Kill switch: if true, self-modification is disabled
    kill_switch: bool,
    /// Fitness history for divergence detection
    fitness_history: Vec<f32>,
    /// Maximum fitness drop before triggering rollback
    divergence_threshold: f32,
}

impl SelfModTracker {
    pub fn new() -> Self {
        SelfModTracker {
            current_level: ModLevel::Human,
            events: Vec::new(),
            max_events: 10000,
            kill_switch: false,
            fitness_history: Vec::new(),
            divergence_threshold: 0.2,
        }
    }

    /// Record a self-modification event.
    pub fn record_event(&mut self, event: ModEvent) {
        if self.kill_switch {
            return; // Kill switch active, reject modification
        }

        self.fitness_history.push(event.fitness_after);
        if self.fitness_history.len() > 1000 {
            self.fitness_history.remove(0);
        }

        self.events.push(event);
        if self.events.len() > self.max_events {
            self.events.remove(0);
        }
    }

    /// Request level escalation. Returns true if allowed.
    pub fn request_escalation(&mut self, target_level: ModLevel) -> bool {
        if self.kill_switch { return false; }
        if target_level.numeric() <= self.current_level.numeric() {
            return true; // Downgrades always allowed
        }
        // Only allow one level at a time
        if target_level.numeric() == self.current_level.numeric() + 1 {
            // Level 1+ requires human approval
            if target_level.numeric() >= 1 {
                return false; // Must be explicitly approved
            }
            true
        } else {
            false // Cannot skip levels
        }
    }

    /// Approve level escalation (human action).
    pub fn approve_escalation(&mut self, target_level: ModLevel) {
        if target_level.numeric() == self.current_level.numeric() + 1 {
            self.current_level = target_level;
        }
    }

    /// Check for divergence: recent fitness dropping.
    pub fn divergence_detected(&self) -> bool {
        if self.fitness_history.len() < 10 { return false; }
        let recent: Vec<f32> = self.fitness_history.iter().rev().take(5).cloned().collect();
        let earlier: Vec<f32> = self.fitness_history.iter().rev().skip(5).take(5).cloned().collect();
        if earlier.is_empty() { return false; }
        let recent_avg = recent.iter().sum::<f32>() / recent.len() as f32;
        let earlier_avg = earlier.iter().sum::<f32>() / earlier.len() as f32;
        (earlier_avg - recent_avg) > self.divergence_threshold
    }

    /// Kill switch: permanently disable self-modification.
    pub fn activate_kill_switch(&mut self) {
        self.kill_switch = true;
    }

    /// Check if kill switch is active.
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch
    }

    /// Current modification level.
    pub fn current_level(&self) -> ModLevel {
        self.current_level
    }

    /// All events.
    pub fn events(&self) -> &[ModEvent] {
        &self.events
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let improved = self.events.iter().filter(|e| e.improved()).count();
        format!(
            "level={} events={} improved={}/{} div={} kill={}",
            self.current_level.label(),
            self.events.len(),
            improved,
            self.events.len(),
            if self.divergence_detected() { "YES" } else { "no" },
            if self.kill_switch { "ACTIVE" } else { "off" },
        )
    }
}

impl Default for SelfModTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple non-cryptographic hash for testing.
#[allow(dead_code)]
pub fn simple_hash(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_level() {
        let tracker = SelfModTracker::new();
        assert_eq!(tracker.current_level(), ModLevel::Human);
        assert_eq!(tracker.current_level().numeric(), 0);
    }

    #[test]
    fn test_record_event() {
        let mut tracker = SelfModTracker::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let event = ModEvent {
            level: ModLevel::AgentCode,
            target: "main.rs".to_string(),
            description: "Added test".to_string(),
            hash_before: "abc".to_string(),
            hash_after: "def".to_string(),
            fitness_before: 0.5,
            fitness_after: 0.6,
            timestamp: now,
            human_approved: true,
        };
        tracker.record_event(event);
        assert_eq!(tracker.events().len(), 1);
        assert!(tracker.events()[0].improved());
    }

    #[test]
    fn test_divergence_detection() {
        let mut tracker = SelfModTracker::new();
        // Simulate fitness dropping
        for f in [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25] {
            tracker.fitness_history.push(f);
        }
        assert!(tracker.divergence_detected());
    }

    #[test]
    fn test_kill_switch() {
        let mut tracker = SelfModTracker::new();
        assert!(!tracker.is_kill_switch_active());
        tracker.activate_kill_switch();
        assert!(tracker.is_kill_switch_active());
    }

    #[test]
    fn test_level_escalation_requires_approval() {
        let mut tracker = SelfModTracker::new();
        assert!(!tracker.request_escalation(ModLevel::AgentCode));
        // After approval
        tracker.approve_escalation(ModLevel::AgentCode);
        assert_eq!(tracker.current_level(), ModLevel::AgentCode);
    }

    #[test]
    fn test_no_level_skipping() {
        let mut tracker = SelfModTracker::new();
        assert!(!tracker.request_escalation(ModLevel::AgentMetaCode));
    }

    #[test]
    fn test_fitness_delta() {
        let event = ModEvent {
            level: ModLevel::AgentCode,
            target: "test.rs".to_string(),
            description: "test".to_string(),
            hash_before: "a".to_string(),
            hash_after: "b".to_string(),
            fitness_before: 0.3,
            fitness_after: 0.7,
            timestamp: 0,
            human_approved: true,
        };
        assert!((event.fitness_delta() - 0.4).abs() < 0.001);
        assert!(event.improved());
    }

    #[test]
    fn test_summary() {
        let tracker = SelfModTracker::new();
        let s = tracker.summary();
        assert!(s.contains("level="));
        assert!(s.contains("events="));
    }
}
