//! Kai free-energy (VFE) controller — the meta-layer that governs the backbone.
//! Mirrors physics-dialect's `calculate_vfe` math (Kai's core): prediction
//! surprisal + epistemic uncertainty + novelty + thermodynamic cost. This module
//! is the canonical home of the VFE controller for Phase C (see
//! Kai_FUSION_ARCHITECTURE.md §6/§9); later slices route this through the
//! kai-mlir IR, here it is the same formula.

/// VFE as defined in Kai_FUSION_ARCHITECTURE.md §6.2:
/// `VFE = next-token surprisal + epistemic uncertainty`.
/// The epistemic term is the KL-vs-attractor-prior distance (how far the current
/// hidden state sits from what Kai has already assimilated). Weighted lightly so
/// the controller explores when far from assimilated knowledge and consolidates
/// near it. `epistemic` may also be a curvature (attention-dispersion) proxy when
/// no attractor prior is active.
#[allow(dead_code)]
pub fn calculate_vfe(surprisal: f64, epistemic: f64) -> f64 {
    surprisal + 0.3 * epistemic
}

/// Extended VFE with Landauer thermodynamic cost term.
///
/// `VFE = surprisal + 0.3·epistemic + β·‖∇‖²`
///
/// The thermodynamic term `β·‖∇‖²` models the Landauer cost of weight updates:
/// erasing information (changing weights) dissipates energy proportional to the
/// squared gradient norm. `beta` is the Landauer coefficient (default 0.001).
/// `delta_e` is the sum of squared gradient magnitudes across all parameters.
///
/// When `delta_e` is large (steep gradients), the thermodynamic cost penalizes
/// large weight changes, acting as a natural regularizer. When the model is
/// near convergence (small gradients), the cost vanishes.
pub fn calculate_vfe_thermo(surprisal: f64, epistemic: f64, delta_e: f64) -> f64 {
    const BETA: f64 = 0.001; // Landauer coefficient
    let thermo_cost = BETA * delta_e;
    surprisal + 0.3 * epistemic + thermo_cost
}

/// Legacy VFE form used by the assimilate/train REPL (surprisal over an emitted
/// one-hot target + novelty + variance). Kept for the interactive assimilate
/// loop; new generation paths use [`calculate_vfe`].
pub fn calculate_vfe_legacy(pred: &[f32], actual: &[f32], variance: f32, nov: f32) -> f32 {
    let mut surp = 0.0;
    for i in 0..pred.len() {
        surp += actual[i] * (pred[i] + 1e-8).ln();
    }
    let surp = -surp;
    surp + nov + variance
}

/// Compute the squared gradient norm `‖∇‖²` = sum of all parameter gradient
/// squares. Used as `delta_e` in the thermodynamic VFE cost.
pub fn grad_norm_sq(grads: &crate::model::Gradients) -> f64 {
    let mut sum = 0.0_f64;
    if let Some(ref g) = grads.embed {
        sum += g.mapv(|x| (x as f64) * (x as f64)).sum();
    }
    if let Some(ref g) = grads.output {
        sum += g.mapv(|x| (x as f64) * (x as f64)).sum();
    }
    sum += grads
        .final_norm
        .mapv(|x| (x as f64) * (x as f64))
        .sum();
    for lg in grads.layers.iter().flatten() {
        sum += lg.wq.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.wk.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.wv.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.wo.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.w1.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.w2.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.w3.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.attn_norm.mapv(|x| (x as f64) * (x as f64)).sum();
        sum += lg.ffn_norm.mapv(|x| (x as f64) * (x as f64)).sum();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_vfe_basic() {
        let vfe = calculate_vfe(5.0, 2.0);
        assert!((vfe - 5.6).abs() < 1e-12); // 5.0 + 0.3*2.0 = 5.6
    }

    #[test]
    fn test_calculate_vfe_thermo_zero_grad() {
        // Zero gradients → no thermodynamic cost
        let vfe = calculate_vfe_thermo(5.0, 2.0, 0.0);
        assert!((vfe - 5.6).abs() < 1e-12); // same as basic
    }

    #[test]
    fn test_calculate_vfe_thermo_nonzero_grad() {
        // With large gradients, thermodynamic cost adds β·‖∇‖² = 0.001 * 100 = 0.1
        let vfe = calculate_vfe_thermo(5.0, 2.0, 100.0);
        assert!((vfe - 5.7).abs() < 1e-12); // 5.6 + 0.1 = 5.7
    }

    #[test]
    fn test_calculate_vfe_thermo_dominates_at_high_grad() {
        // With enormous gradients, thermodynamic cost dominates
        let vfe = calculate_vfe_thermo(1.0, 0.0, 1_000_000.0);
        assert!((vfe - 1001.0).abs() < 1.0); // 1.0 + 0.001*1e6 = 1001.0
    }

    #[test]
    fn test_grad_norm_sq_empty() {
        // Empty/minimal Gradients should not panic
        let grads = crate::model::Gradients {
            embed: None,
            output: None,
            final_norm: ndarray::Array1::zeros(0),
            layers: vec![],
        };
        let n = grad_norm_sq(&grads);
        assert!(n == 0.0);
    }
}
