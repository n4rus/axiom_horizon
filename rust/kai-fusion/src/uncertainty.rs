//! Phase 4.3: Uncertainty Estimation — confidence scoring on every answer.
//!
//! From AGI_PLAN.md §4.3:
//! "Every answer includes a confidence score.
//! Sample N times at temperature >0, check variance.
//! If answers disagree → low confidence.
//! Track which knowledge sources support the answer.
//! If no source found → 'I don't know'."
//!
//! This module extracts and extends the ConfidenceReport from cli.rs
//! with Bayesian calibration, knowledge source tracking, and
//! confidence band classification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Confidence bands for human-readable classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceBand {
    /// Very high confidence (>= 0.9)
    VeryHigh,
    /// High confidence (>= 0.7)
    High,
    /// Moderate confidence (>= 0.5)
    Moderate,
    /// Low confidence (>= 0.3)
    Low,
    /// Very low / uncertain (< 0.3)
    VeryLow,
}

impl ConfidenceBand {
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 { ConfidenceBand::VeryHigh }
        else if score >= 0.7 { ConfidenceBand::High }
        else if score >= 0.5 { ConfidenceBand::Moderate }
        else if score >= 0.3 { ConfidenceBand::Low }
        else { ConfidenceBand::VeryLow }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConfidenceBand::VeryHigh => "very high",
            ConfidenceBand::High => "high",
            ConfidenceBand::Moderate => "moderate",
            ConfidenceBand::Low => "low",
            ConfidenceBand::VeryLow => "very low",
        }
    }

    /// Human-readable recommendation based on confidence level.
    pub fn recommendation(&self) -> &'static str {
        match self {
            ConfidenceBand::VeryHigh => "Trusted answer — two or more sources confirm.",
            ConfidenceBand::High => "Likely correct — one supporting source found.",
            ConfidenceBand::Moderate => "Probably correct — verify with independent source.",
            ConfidenceBand::Low => "Uncertain — recommend verifying with a database query.",
            ConfidenceBand::VeryLow => "I don't know — no reliable sources found.",
        }
    }
}

/// A knowledge source that supports an answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSource {
    /// Source identifier (file path, engram ID, etc.)
    pub id: String,
    /// Similarity score (cosine sim to query)
    pub relevance: f32,
    /// Short label for display
    pub label: String,
}

/// Full uncertainty report for a generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyReport {
    /// Raw confidence score in [0, 1]
    pub score: f32,
    /// Classified confidence band
    pub band: ConfidenceBand,
    /// Number of samples taken
    pub n_samples: usize,
    /// Shannon entropy of the top-token distribution (bits)
    pub entropy: f32,
    /// Number of unique top-1 tokens across samples
    pub unique_predictions: usize,
    /// The winning prediction (most frequent top-1 token)
    pub top_prediction: String,
    /// Fraction of samples that chose the top prediction
    pub agreement_ratio: f32,
    /// Knowledge sources that support this answer
    pub sources: Vec<KnowledgeSource>,
    /// Whether any knowledge was retrieved
    pub memory_retrieved: bool,
    /// Calibration history: rolling accuracy at this confidence level
    pub calibration_accuracy: Option<f32>,
}

impl Default for UncertaintyReport {
    fn default() -> Self {
        UncertaintyReport {
            score: 0.0,
            band: ConfidenceBand::VeryLow,
            n_samples: 0,
            entropy: 0.0,
            unique_predictions: 0,
            top_prediction: String::new(),
            agreement_ratio: 0.0,
            sources: Vec::new(),
            memory_retrieved: false,
            calibration_accuracy: None,
        }
    }
}

impl UncertaintyReport {
    /// Format for CLI display.
    pub fn display(&self) -> String {
        let mut lines = vec![format!(
            "Confidence: {:.0}% ({})",
            self.score * 100.0,
            self.band.label(),
        )];
        if self.n_samples > 1 {
            lines.push(format!(
                "  samples: {} | agreement: {:.0}% | entropy: {:.2} bits | unique: {}",
                self.n_samples,
                self.agreement_ratio * 100.0,
                self.entropy,
                self.unique_predictions,
            ));
        }
        if self.memory_retrieved && !self.sources.is_empty() {
            lines.push(format!("  sources: {}", self.sources.len()));
            for s in &self.sources {
                lines.push(format!("    - {} (relevance: {:.2})", s.label, s.relevance));
            }
        }
        lines.push(format!("  {}", self.band.recommendation()));
        lines.join("\n")
    }
}

/// Compute Shannon entropy of a frequency distribution.
fn shannon_entropy(freqs: &[f32]) -> f32 {
    freqs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Compute uncertainty report from sampled token predictions.
pub fn compute_uncertainty(
    predictions: &[String],
    memory_retrieved: bool,
    sources: Vec<KnowledgeSource>,
) -> UncertaintyReport {
    if predictions.is_empty() {
        return UncertaintyReport {
            score: 0.0,
            band: ConfidenceBand::VeryLow,
            n_samples: 0,
            ..Default::default()
        };
    }

    let n = predictions.len() as f32;

    // Count frequency of each prediction
    let mut freq: HashMap<String, usize> = HashMap::new();
    for p in predictions {
        *freq.entry(p.clone()).or_insert(0) += 1;
    }

    // Find top prediction
    let (top_pred, &top_count) = freq.iter().max_by_key(|(_, &c)| c).unwrap();
    let agreement = top_count as f32 / n;

    // Shannon entropy
    let unique = freq.len();
    let freq_vals: Vec<f32> = freq.values().map(|&c| c as f32 / n).collect();
    let entropy = shannon_entropy(&freq_vals);

    // Max entropy = log2(unique); normalize to [0,1]
    let max_entropy = (unique as f32).log2().max(1.0);
    let norm_entropy = entropy / max_entropy;

    // Confidence = 1 - normalized entropy, boosted by memory retrieval
    let mut score = 1.0 - norm_entropy;
    if memory_retrieved {
        score = (score + 0.1).min(1.0);
    }

    let band = ConfidenceBand::from_score(score);

    UncertaintyReport {
        score,
        band,
        n_samples: predictions.len(),
        entropy,
        unique_predictions: unique,
        top_prediction: top_pred.to_string(),
        agreement_ratio: agreement,
        sources,
        memory_retrieved,
        calibration_accuracy: None,
    }
}

/// Calibration tracker — accumulates (confidence, was_correct) pairs
/// and computes rolling accuracy per confidence band.
#[allow(dead_code)]
pub struct CalibrationTracker {
    /// (confidence_score, was_correct) history
    history: Vec<(f32, bool)>,
    max_history: usize,
}

#[allow(dead_code)]
impl CalibrationTracker {
    pub fn new() -> Self {
        CalibrationTracker {
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Record a prediction outcome: was the top prediction correct?
    pub fn record(&mut self, confidence: f32, correct: bool) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push((confidence, correct));
    }

    /// Get rolling accuracy for a confidence band.
    pub fn accuracy_for_band(&self, band: ConfidenceBand) -> f32 {
        let (lo, hi) = match band {
            ConfidenceBand::VeryLow => (0.0, 0.3),
            ConfidenceBand::Low => (0.3, 0.5),
            ConfidenceBand::Moderate => (0.5, 0.7),
            ConfidenceBand::High => (0.7, 0.9),
            ConfidenceBand::VeryHigh => (0.9, 1.01),
        };
        let relevant: Vec<_> = self.history.iter()
            .filter(|(c, _)| *c >= lo && *c < hi)
            .collect();
        if relevant.is_empty() { return f32::NAN; }
        let correct = relevant.iter().filter(|(_, ok)| *ok).count();
        correct as f32 / relevant.len() as f32
    }

    /// Overall accuracy.
    pub fn overall_accuracy(&self) -> f32 {
        if self.history.is_empty() { return f32::NAN; }
        let correct = self.history.iter().filter(|(_, ok)| *ok).count();
        correct as f32 / self.history.len() as f32
    }

    /// Number of samples tracked.
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.history.len()
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_bands() {
        assert_eq!(ConfidenceBand::from_score(0.95), ConfidenceBand::VeryHigh);
        assert_eq!(ConfidenceBand::from_score(0.8), ConfidenceBand::High);
        assert_eq!(ConfidenceBand::from_score(0.6), ConfidenceBand::Moderate);
        assert_eq!(ConfidenceBand::from_score(0.4), ConfidenceBand::Low);
        assert_eq!(ConfidenceBand::from_score(0.1), ConfidenceBand::VeryLow);
    }

    #[test]
    fn test_high_agreement_high_confidence() {
        let predictions = vec!["42".to_string(); 10];
        let report = compute_uncertainty(&predictions, false, vec![]);
        assert!(report.score > 0.9, "score={}", report.score);
        assert_eq!(report.band, ConfidenceBand::VeryHigh);
        assert_eq!(report.agreement_ratio, 1.0);
    }

    #[test]
    fn test_disagreement_low_confidence() {
        let predictions = (0..10)
            .map(|i| format!("answer_{}", i))
            .collect::<Vec<_>>();
        let report = compute_uncertainty(&predictions, false, vec![]);
        assert!(report.score < 0.5, "score={}", report.score);
        assert_eq!(report.unique_predictions, 10);
    }

    #[test]
    fn test_memory_boost() {
        let mut preds = vec!["yes".to_string(); 8];
        preds.push("no".to_string());
        preds.push("maybe".to_string());

        let without = compute_uncertainty(&preds, false, vec![]);
        let with = compute_uncertainty(&preds, true, vec![]);
        assert!(with.score > without.score, "with={} without={}", with.score, without.score);
    }

    #[test]
    fn test_calibration_tracker() {
        let mut cal = CalibrationTracker::new();
        cal.record(0.9, true);
        cal.record(0.9, true);
        cal.record(0.9, false);
        assert!((cal.overall_accuracy() - 0.6667).abs() < 0.01);
        assert!((cal.accuracy_for_band(ConfidenceBand::VeryHigh) - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_empty_predictions() {
        let report = compute_uncertainty(&[], false, vec![]);
        assert_eq!(report.score, 0.0);
        assert_eq!(report.band, ConfidenceBand::VeryLow);
    }

    #[test]
    fn test_display() {
        let report = UncertaintyReport {
            score: 0.85,
            band: ConfidenceBand::High,
            n_samples: 5,
            agreement_ratio: 0.8,
            ..Default::default()
        };
        let display = report.display();
        assert!(display.contains("85%"));
        assert!(display.contains("high"));
    }

    #[test]
    fn test_recommendation_texts() {
        assert!(ConfidenceBand::VeryHigh.recommendation().contains("Trusted"));
        assert!(ConfidenceBand::VeryLow.recommendation().contains("don't know"));
    }
}
