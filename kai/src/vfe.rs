//! Variational Free Energy Controller
//!
//! Implements: F = E_q[log q(z|x) - log p(x,z)] = surprisal + KL(q||p)
//! - Surprisal: -log p(x) under current belief
//! - Epistemic KL: KL(q(z|x) || p(z|attractor))  -- information gain toward attractor
//! - Adaptive temperature: novelty + curvature -> temp
//! - Time dilation τ: VFE -> τ (high VFE -> low τ -> faster sampling)

use crate::config::PhysicsParams;
use ndarray::{Array1, ArrayView1};

#[derive(Debug, Clone, Default)]
pub struct VFEState {
    pub tau: f32,
    pub vfe: f32,
    pub novelty: f32,
    pub curvature: f32,
}

pub fn compute_vfe(
    logits: &ArrayView1<f32>,
    target: usize,
    prior: &ArrayView1<f32>,
    state: &mut VFEState,
    phys: &PhysicsParams,
) -> f32 {
    // Softmax probabilities
    let max_logit = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Array1<f32> = logits.mapv(|l| (l - max_logit).exp());
    let sum_exp = exp_logits.sum();
    let probs = exp_logits / sum_exp;

    // Surprisal: -log p(target)
    let p_target = probs[target].max(1e-10);
    let surprisal = -p_target.ln();

    // Epistemic KL: KL(q||p) where q=posterior, p=attractor prior
    let kl = probs.iter().zip(prior.iter())
        .map(|(q, p)| {
            let q = q.max(1e-10);
            let p = p.max(1e-10);
            q * (q / p).ln()
        })
        .sum::<f32>();

    // VFE = surprisal + epistemic KL
    let vfe = surprisal + kl;
    state.vfe = vfe;

    // Novelty = attention entropy
    let entropy = -probs.iter().map(|p| {
        let p = p.max(1e-10);
        p * p.ln()
    }).sum::<f32>();
    let max_entropy = (probs.len() as f32).ln();
    let novelty = entropy / max_entropy;
    state.novelty = novelty;

    // Curvature = 1 - max(probs)
    let max_prob = probs.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let curvature = 1.0 - max_prob;
    state.curvature = curvature;

    // Adaptive temperature
    let physics_boost = phys.novelty_scale * ((novelty + curvature) * 0.5 - 0.5);
    let temp_eff = if phys.base_temperature > 0.0 {
        (phys.base_temperature * (1.0 + physics_boost) / state.tau.max(0.1)).max(0.01)
    } else { 0.0 };

    // τ update: high VFE -> reduce τ (dilate time)
    // Clamp to avoid NaN from negative sqrt
    let vfe_factor = (1.0 - phys.vfe_tau_rate * vfe).max(0.0).min(1.0);
    let new_tau = state.tau * vfe_factor.sqrt();
    state.tau = new_tau.clamp(phys.tau_min, phys.tau_max);

    temp_eff
}

pub fn sample_top_p(logits: &[f32], temp: f32, top_p: f32) -> usize {
    if temp <= 0.0 {
        return argmax(logits);
    }
    // Correct temperature scaling: softmax(logits / temp)
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits.iter().map(|l| ((l - max_logit) / temp).exp()).collect();
    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|p| p / sum).collect();

    // Top-p filtering
    let mut sorted: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    let mut cumsum = 0.0;
    let mut candidates = Vec::new();
    for (i, p) in sorted {
        cumsum += p;
        candidates.push((i, p));
        if cumsum >= top_p { break; }
    }

    let sum_cand: f32 = candidates.iter().map(|(_, p)| p).sum();
    let r: f32 = rand::random::<f32>() * sum_cand;
    let mut cum = 0.0;
    for &(i, p) in &candidates {
        cum += p;
        if r <= cum { return i; }
    }
    candidates.last().map(|&(i, _)| i).unwrap_or(0)
}

pub fn argmax(probs: &[f32]) -> usize {
    probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
}