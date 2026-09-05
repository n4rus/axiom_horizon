//! Criticality homeostat — holds generation at self-organized criticality.
//!
//! Physics: cortical networks operate at branching ratio σ ≈ 1 (neuronal
//! avalanches, Beggs & Plenz 2003). σ > 1 = supercritical cascade (seizure /
//! degenerate repetition); σ < 1 = subcritical decay (rigidity). Transformers
//! share the propagation structure: attention softmax(z/T) is the kernel, so T
//! is the actuator that moves σ.
//!
//! Kai mapping:
//!   - descendants per active unit ← attention mass a_ij above threshold
//!   - σ estimated per generation step from the attention/hidden spread
//!   - controller nudges temperature toward σ ≈ 1 (homeostat)
//!
//! Composes with `tau_update_epistemic` (vfe.rs): tau = subjective time,
//! σ = how far activity spreads during it. Both bench-gated upstream.

/// Estimate the branching ratio σ from one step's attention/spread weights.
///
/// `weights` = flattened attention scores (or responsibility values) for the
/// current step, already in [0,1]. A unit "fires forward" on edges whose
/// weight exceeds `threshold`; σ = mean out-degree over firing units.
///
/// Returns σ ≥ 0. Empty input → σ = 0 (subcritical by convention).
/// Public API + unit-test covered; binary profile uses `branching_ratio_n`.
#[allow(dead_code)]
pub fn branching_ratio(weights: &[f32], threshold: f32) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    // Group flat weights into rows of equal length when possible is not needed
    // for the estimate: mean firing degree = (#edges above threshold) / (#source units).
    // We approximate #source units as sqrt(n) for a square attention matrix,
    // or use explicit `n_sources` when provided via [`branching_ratio_n`].
    let n = weights.len() as f32;
    let sources = n.sqrt().max(1.0);
    let fired = weights.iter().filter(|&&w| w >= threshold).count() as f32;
    fired / sources
}

/// Branching ratio with an explicit source count (preferred API).
pub fn branching_ratio_n(weights: &[f32], threshold: f32, n_sources: usize) -> f32 {
    if weights.is_empty() || n_sources == 0 {
        return 0.0;
    }
    let fired = weights.iter().filter(|&&w| w >= threshold).count() as f32;
    fired / n_sources as f32
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CriticalRegime {
    /// σ < 1 − ε : activity dies out; open up temperature.
    SubCritical,
    /// |σ − 1| ≤ ε : critical band; hold temperature.
    Critical,
    /// σ > 1 + ε : runaway cascade; clamp temperature down.
    SuperCritical,
}

/// Classify regime and return the temperature multiplier to apply.
///
/// `sigma`      — measured branching ratio
/// `epsilon`    — half-width of the critical band (e.g. 0.05)
/// `gain`       — how hard the homeostat pushes (e.g. 0.15)
pub fn regulate(sigma: f32, epsilon: f32, gain: f32) -> (CriticalRegime, f32) {
    const MIN_MULT: f32 = 0.5;
    const MAX_MULT: f32 = 2.0;
    if sigma > 1.0 + epsilon {
        // supercritical: contract spread
        let mult = (1.0 - gain * (sigma - 1.0)).max(MIN_MULT);
        (CriticalRegime::SuperCritical, mult)
    } else if sigma < 1.0 - epsilon {
        // subcritical: open spread
        let mult = (1.0 + gain * (1.0 - sigma)).min(MAX_MULT);
        (CriticalRegime::SubCritical, mult)
    } else {
        (CriticalRegime::Critical, 1.0)
    }
}

/// One-shot helper: given base temperature and measured weights, produce the
/// regulated temperature. `n_sources` = number of query positions this step.
pub fn regulated_temperature(
    base_temp: f32,
    weights: &[f32],
    threshold: f32,
    n_sources: usize,
    epsilon: f32,
    gain: f32,
) -> (f32, CriticalRegime, f32) {
    let sigma = branching_ratio_n(weights, threshold, n_sources);
    let (regime, mult) = regulate(sigma, epsilon, gain);
    (base_temp * mult, regime, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_weights_is_subcritical_zero() {
        assert_eq!(branching_ratio(&[], 0.5), 0.0);
        assert_eq!(branching_ratio_n(&[], 0.5, 4), 0.0);
        assert_eq!(branching_ratio_n(&[0.9, 0.9], 0.5, 0), 0.0);
    }

    #[test]
    fn test_sigma_one_is_critical_holds_temperature() {
        // 8 sources, 8 fired edges → σ = 1.0 exactly
        let w = vec![0.9f32; 8];
        let (mult, regime, sigma) = regulated_temperature(1.0, &w, 0.5, 8, 0.05, 0.15);
        assert!((sigma - 1.0).abs() < 1e-6);
        assert_eq!(regime, CriticalRegime::Critical);
        assert!((mult - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_supercritical_clamps_temperature_down() {
        // 4 sources, 16 fired edges → σ = 4.0 (runaway)
        let w = vec![0.9f32; 16];
        let (mult, regime, sigma) = regulated_temperature(1.0, &w, 0.5, 4, 0.05, 0.15);
        assert!(sigma > 1.05);
        assert_eq!(regime, CriticalRegime::SuperCritical);
        assert!(mult < 1.0);
    }

    #[test]
    fn test_subcritical_opens_temperature_up() {
        // 8 sources, 1 fired edge → σ = 0.125 (activity dies)
        let mut w = vec![0.0f32; 8];
        w[0] = 0.9;
        let (mult, regime, sigma) = regulated_temperature(1.0, &w, 0.5, 8, 0.05, 0.15);
        assert!(sigma < 0.95);
        assert_eq!(regime, CriticalRegime::SubCritical);
        assert!(mult > 1.0);
    }

    #[test]
    fn test_regulate_respects_multiplier_bounds() {
        // extreme supercritical cannot push below MIN_MULT
        let (_, m_hi) = regulate(100.0, 0.05, 0.15);
        assert_eq!(m_hi, 0.5);
        // extreme subcritical opens up but stays bounded and gentle
        let (_, m_lo) = regulate(0.0001, 0.05, 0.15);
        assert!(m_lo > 1.0);
        assert!(m_lo <= 2.0);
        assert!((m_lo - (1.0 + 0.15)).abs() < 1e-4); // saturates at 1+gain
    }

    #[test]
    fn test_threshold_filters_weak_edges() {
        // same weights, high threshold → nothing fires → subcritical
        let w = vec![0.3f32; 16];
        let sigma_low_thr = branching_ratio_n(&w, 0.1, 4);
        let sigma_high_thr = branching_ratio_n(&w, 0.5, 4);
        assert!(sigma_low_thr > sigma_high_thr);
    }
}
