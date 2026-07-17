//! Kai free-energy (VFE) controller — the meta-layer that governs the backbone.
//! Mirrors physics-dialect's `calculate_vfe` math (Kai's core): prediction
//! surprisal + epistemic uncertainty + novelty. Later slices route this through
//! the IR; here it is the same formula, self-contained for the engine build.

/// VFE = surprisal(-Σ actual·log pred) + novelty + variance(uncertainty).
pub fn calculate_vfe(pred: &[f32], actual: &[f32], variance: f32, nov: f32) -> f32 {
    let mut surp = 0.0;
    for i in 0..pred.len() {
        surp += actual[i] * (pred[i] + 1e-8).ln();
    }
    let surp = -surp;
    surp + nov + variance
}
