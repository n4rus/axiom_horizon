//! Phase 4.4: Value Learning — infer user preferences from interaction signals.
//!
//! From AGI_PLAN.md §4.4:
//! "Infer preferences from interaction:
//!   User praises factuality → increase factuality weight
//!   User praises speed → optimize for latency
//!   User corrects error → learn the correction
//!   User shows frustration → change approach
//! Implementation: reinforcement learning from human feedback (RLHF) but local.
//! Every user message is a reward signal."

use serde::{Deserialize, Serialize};

/// Value dimensions the agent tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum ValueDimension {
    /// Prefer factual, sourced answers
    Factuality,
    /// Prefer fast responses
    Speed,
    /// Prefer creative/novel responses
    Creativity,
    /// Prefer detailed, thorough responses
    Thoroughness,
    /// Prefer concise responses
    Conciseness,
    /// Prefer code examples
    CodeExamples,
    /// Prefer analogies and explanations
    Explanations,
}

impl ValueDimension {
    pub fn label(&self) -> &'static str {
        match self {
            ValueDimension::Factuality => "factuality",
            ValueDimension::Speed => "speed",
            ValueDimension::Creativity => "creativity",
            ValueDimension::Thoroughness => "thoroughness",
            ValueDimension::Conciseness => "conciseness",
            ValueDimension::CodeExamples => "code_examples",
            ValueDimension::Explanations => "explanations",
        }
    }

    pub fn all() -> &'static [ValueDimension] {
        &[
            ValueDimension::Factuality,
            ValueDimension::Speed,
            ValueDimension::Creativity,
            ValueDimension::Thoroughness,
            ValueDimension::Conciseness,
            ValueDimension::CodeExamples,
            ValueDimension::Explanations,
        ]
    }
}

/// A feedback signal detected from user interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct FeedbackSignal {
    /// Which value dimension this feedback relates to
    pub dimension: ValueDimension,
    /// Positive (+1) or negative (-1) feedback
    pub polarity: f32,
    /// The raw text that triggered the signal
    pub trigger: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Per-dimension weight with accumulated evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ValueWeight {
    pub dimension: ValueDimension,
    /// Accumulated weight (starts at 0.5, moves toward 0 or 1)
    pub weight: f32,
    /// Number of positive signals
    pub positive_count: u32,
    /// Number of negative signals
    pub negative_count: u32,
}

#[allow(dead_code)]
impl ValueWeight {
    fn new(dim: ValueDimension) -> Self {
        ValueWeight {
            dimension: dim,
            weight: 0.5,
            positive_count: 0,
            negative_count: 0,
        }
    }

    fn update(&mut self, polarity: f32) {
        if polarity > 0.0 {
            self.positive_count += 1;
        } else {
            self.negative_count += 1;
        }
        // Exponential moving average: weight shifts toward 1 on positive, 0 on negative
        let alpha = 0.15; // learning rate
        self.weight = self.weight * (1.0 - alpha) + ((polarity + 1.0) / 2.0) * alpha;
        self.weight = self.weight.clamp(0.05, 0.95);
    }
}

/// Value Learner — tracks user preferences across all dimensions.
#[allow(dead_code)]
pub struct ValueLearner {
    weights: Vec<ValueWeight>,
    signals: Vec<FeedbackSignal>,
    max_signals: usize,
}

#[allow(dead_code)]
impl ValueLearner {
    pub fn new() -> Self {
        ValueLearner {
            weights: ValueDimension::all().iter()
                .map(|&d| ValueWeight::new(d))
                .collect(),
            signals: Vec::new(),
            max_signals: 10000,
        }
    }

    /// Detect and record feedback signals from a user message.
    pub fn process_message(&mut self, message: &str) -> Vec<FeedbackSignal> {
        let lower = message.to_lowercase();
        let mut detected = Vec::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Positive signals
        let positive_patterns: &[(&[&str], ValueDimension)] = &[
            (&["correct", "right", "exactly", "perfect", "accurate", "true"], ValueDimension::Factuality),
            (&["fast", "quick", "speedy", "efficient", "instant"], ValueDimension::Speed),
            (&["creative", "clever", "innovative", "original", "novel"], ValueDimension::Creativity),
            (&["detailed", "thorough", "comprehensive", "in-depth"], ValueDimension::Thoroughness),
            (&["concise", "brief", "short", "to the point"], ValueDimension::Conciseness),
            (&["code", "function", "implement", "rust", "python"], ValueDimension::CodeExamples),
            (&["explain", "analogy", "like", "understand", "intuition"], ValueDimension::Explanations),
        ];

        let negative_patterns: &[(&[&str], ValueDimension)] = &[
            (&["wrong", "incorrect", "false", "mistake", "error", "nope"], ValueDimension::Factuality),
            (&["slow", "delay", "waiting", "took forever"], ValueDimension::Speed),
            (&["boring", "mundane", "generic", "cookie-cutter"], ValueDimension::Creativity),
            (&["too long", "too much", "overkill", "TMI", "excessive"], ValueDimension::Thoroughness),
            (&["too short", "incomplete", "missing", "need more"], ValueDimension::Conciseness),
            (&["pseudocode", "not real code", "just an example"], ValueDimension::CodeExamples),
            (&["confusing", "unclear", "doesn't make sense", "lost me"], ValueDimension::Explanations),
        ];

        for (keywords, dim) in positive_patterns {
            for kw in *keywords {
                if lower.contains(kw) {
                    let signal = FeedbackSignal {
                        dimension: *dim,
                        polarity: 1.0,
                        trigger: kw.to_string(),
                        timestamp: now,
                    };
                    detected.push(signal);
                    break; // One signal per dimension per message
                }
            }
        }

        for (keywords, dim) in negative_patterns {
            for kw in *keywords {
                if lower.contains(kw) {
                    let signal = FeedbackSignal {
                        dimension: *dim,
                        polarity: -1.0,
                        trigger: kw.to_string(),
                        timestamp: now,
                    };
                    detected.push(signal);
                    break;
                }
            }
        }

        // Record signals
        for s in &detected {
            self.signals.push(s.clone());
            if self.signals.len() > self.max_signals {
                self.signals.remove(0);
            }
            // Update weight
            if let Some(w) = self.weights.iter_mut().find(|w| w.dimension == s.dimension) {
                w.update(s.polarity);
            }
        }

        detected
    }

    /// Get the weight for a dimension.
    pub fn weight(&self, dim: ValueDimension) -> f32 {
        self.weights.iter()
            .find(|w| w.dimension == dim)
            .map(|w| w.weight)
            .unwrap_or(0.5)
    }

    /// Get the dominant preference (highest weight).
    pub fn dominant_preference(&self) -> &ValueWeight {
        self.weights.iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .unwrap()
    }

    /// Get all weights sorted by weight (highest first).
    pub fn ranked_preferences(&self) -> Vec<&ValueWeight> {
        let mut sorted: Vec<_> = self.weights.iter().collect();
        sorted.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        sorted
    }

    /// Total signals processed.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let ranked = self.ranked_preferences();
        let top = ranked.first().unwrap();
        format!(
            "top pref: {} ({:.2}) | {} signals | all: {}",
            top.dimension.label(),
            top.weight,
            self.signals.len(),
            ranked.iter()
                .map(|w| format!("{}={:.2}", w.dimension.label(), w.weight))
                .collect::<Vec<_>>()
                .join(" "),
        )
    }
}

impl Default for ValueLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_weights_balanced() {
        let learner = ValueLearner::new();
        for dim in ValueDimension::all() {
            assert!((learner.weight(*dim) - 0.5).abs() < 0.01, "{}", dim.label());
        }
    }

    #[test]
    fn test_positive_feedback_shifts_weight() {
        let mut learner = ValueLearner::new();
        let signals = learner.process_message("That was correct and accurate!");
        assert!(!signals.is_empty());
        assert!(learner.weight(ValueDimension::Factuality) > 0.5);
    }

    #[test]
    fn test_negative_feedback_shifts_weight() {
        let mut learner = ValueLearner::new();
        let _ = learner.process_message("That was wrong and incorrect.");
        assert!(learner.weight(ValueDimension::Factuality) < 0.5);
    }

    #[test]
    fn test_speed_feedback() {
        let mut learner = ValueLearner::new();
        let _ = learner.process_message("That was really fast, quick response!");
        assert!(learner.weight(ValueDimension::Speed) > 0.5);
    }

    #[test]
    fn test_multiple_signals_per_message() {
        let mut learner = ValueLearner::new();
        let signals = learner.process_message("That was fast and creative!");
        assert!(signals.len() >= 2);
    }

    #[test]
    fn test_dominant_preference() {
        let mut learner = ValueLearner::new();
        for _ in 0..5 {
            let _ = learner.process_message("That was very creative and clever!");
        }
        let dominant = learner.dominant_preference();
        assert_eq!(dominant.dimension, ValueDimension::Creativity);
    }

    #[test]
    fn test_signal_count() {
        let mut learner = ValueLearner::new();
        let _ = learner.process_message("correct");
        let _ = learner.process_message("wrong");
        assert_eq!(learner.signal_count(), 2);
    }

    #[test]
    fn test_summary() {
        let learner = ValueLearner::new();
        let s = learner.summary();
        assert!(s.contains("signals"));
    }

    #[test]
    fn test_ranked_preferences() {
        let mut learner = ValueLearner::new();
        let _ = learner.process_message("That was very creative!");
        let ranked = learner.ranked_preferences();
        assert_eq!(ranked.len(), ValueDimension::all().len());
        // First should be creativity (highest)
        assert_eq!(ranked[0].dimension, ValueDimension::Creativity);
    }
}
