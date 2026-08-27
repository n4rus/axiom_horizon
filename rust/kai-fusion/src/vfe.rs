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

// ---------------------------------------------------------------------------
// Tau horizon: subjective time dilation and decay
// ---------------------------------------------------------------------------

/// Default tau decay rate per second of idle time.
pub const TAU_DECAY_RATE: f32 = 0.1;

/// Default VFE → tau learning rate.
pub const TAU_LEARNING_RATE: f32 = 0.05;

/// Update tau based on VFE (novelty/surprisal).
/// High VFE → tau increases (more thinking time needed).
/// Returns the updated tau.
pub fn tau_update(tau: f32, vfe: f32, learning_rate: f32) -> f32 {
    let delta = learning_rate * vfe.max(0.0);
    let new_tau = tau + delta;
    // Clamp to prevent runaway: tau ∈ [0.1, 1e12]
    new_tau.clamp(0.1, 1e12)
}

/// Decay tau over idle time.
/// When no input arrives, tau should relax toward 1.0 (normal time).
/// `idle_seconds`: how long since last input.
/// `decay_rate`: how fast tau decays per second (default 0.1 → 10%/s).
/// Returns the decayed tau.
pub fn tau_decay(tau: f32, idle_seconds: f32, decay_rate: f32) -> f32 {
    if idle_seconds <= 0.0 || tau <= 1.0 {
        return tau;
    }
    // Exponential decay toward 1.0: tau_new = 1.0 + (tau - 1.0) * exp(-rate * time)
    let factor = (-decay_rate * idle_seconds).exp();
    1.0 + (tau - 1.0) * factor
}

/// Compute the subjective time ratio: how many seconds of thought per wall-second.
/// tau=1.0 → normal time. tau=1000 → 1000 subjective seconds per real second.
#[allow(dead_code)]
pub fn subjective_ratio(tau: f32) -> f32 {
    tau.max(0.1)
}

/// Estimate how many seconds of subjective time have passed given a tau history.
/// `tau_history`: list of (wall_time_delta, tau_value) pairs.
pub fn subjective_seconds(history: &[(f32, f32)]) -> f32 {
    let mut total = 0.0f32;
    for &(wall_dt, tau) in history {
        total += wall_dt * tau.max(0.1);
    }
    total
}

// ---------------------------------------------------------------------------
// Multi-prior VFE and Kalman fusion (LAYER 2)
// ---------------------------------------------------------------------------
// Multi-prior VFE: predictive surprisal + mixture-novelty + epistemic KL of the
// responsibility distribution. The KL term is the key upgrade: it measures how
// much the input spreads across domain priors (genuinely novel) vs cleanly
// activating one memory.
//
// Kalman fusion: weight the N source architectures (the 10 ingested GGUFs) not
// by the number of votes but by inverse prediction-covariance — a true state
// estimate, not a naive ensemble average.

/// A single predicted estimate with an uncertainty (variance) attached.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // L2 fusion API (consumed by tests, wired in L3)
pub struct Estimate {
    pub value: f32,
    pub variance: f32,
}

/// Kalman gain K = P / (P + R). The bridge between a prior prediction's
/// uncertainty P and a measurement's uncertainty R. High K = trust the
/// measurement; low K = trust the prior.
#[allow(dead_code)] // L2 fusion API (consumed by tests, wired in L3)
pub fn kalman_gain(pred_var: f32, obs_var: f32) -> f32 {
    let denom = pred_var + obs_var;
    if denom < 1e-12 {
        return 0.5;
    }
    pred_var / denom
}

/// One Kalman update: fuse a prior estimate with an observation.
/// Returns the fused estimate and its updated variance.
#[allow(dead_code)] // L2 fusion API (consumed by tests, wired in L3)
pub fn kalman_fuse(pred: f32, pred_var: f32, obs: f32, obs_var: f32) -> Estimate {
    let k = kalman_gain(pred_var, obs_var);
    let value = pred + k * (obs - pred);
    let variance = (1.0 - k) * pred_var;
    Estimate { value, variance }
}

/// Sequential Kalman fusion of the N source architectures.
/// `sources`: the per-source predictions (each an Estimate: value + variance).
/// Fuses left-to-right; each step's fused variance becomes the next prior.
#[allow(dead_code)] // L2 fusion API (consumed by tests, wired in L3)
pub fn kalman_fuse_sources(sources: &[Estimate]) -> Estimate {
    if sources.is_empty() {
        return Estimate { value: 0.0, variance: 1.0 };
    }
    let mut fused = sources[0];
    for src in &sources[1..] {
        fused = fuse_2(fused, *src);
    }
    fused
}

/// Fuse two inverse-variance-weighted estimates. The more certain source
/// (lower variance) dominates the fused value. (fusion-D; L3 swaps the vector
/// curvature form.)
#[allow(dead_code)] // L2 fusion API (consumed by tests, wired in L3)
fn fuse_2(prev: Estimate, cur: Estimate) -> Estimate {
    let w_prev = 1.0 / prev.variance.max(1e-12);
    let w_cur = 1.0 / cur.variance.max(1e-12);
    let tw = w_prev + w_cur;
    Estimate {
        value: (prev.value * w_prev + cur.value * w_cur) / tw,
        variance: 1.0 / tw,
    }
}

/// Multi-prior VFE: predictive surprisal + mixture novelty + epistemic
/// entropy, the L2 upgrade over the single-centroid `calculate_vfe`.
///
/// `priors` = the N domain attractor centroids. Surprisal is the base
/// predictive term; novelty is distance from the *softmax mixture* prior
/// (which memory the input pulls); epistemic is the entropy of the
/// responsibility distribution (how spread the input is across domains).
/// The epistemic term is what raises tau on genuinely novel input.
#[allow(dead_code)] // L2 fusion API (consumed by tests, wired in L3)
pub fn multi_prior_vfe(
    pred: &[f32],
    actual: &[f32],
    variance: f32,
    hidden: &[f32],
    priors: &[Vec<f32>],
    temp: f32,
) -> f32 {
    let base = calculate_vfe(pred, actual, variance, 0.0);
    if priors.is_empty() || hidden.is_empty() {
        return base;
    }
    let r = crate::attractor::responsibilities(hidden, priors, temp);
    let mix_nov = crate::attractor::novelty_vs_mixture(hidden, priors, temp);
    let eps = crate::attractor::epistemic_kl(&r);
    base + mix_nov + eps
}

// ---------------------------------------------------------------------------
// Curvature-wired VFE (LAYER 3a)
// ---------------------------------------------------------------------------
// The metric tensor step: the hidden-state cloud around the current token has a
// local scalar curvature R. High-curvature regions are where the representation
// space is bent — genuinely uncertain / spread — so the controller (1) adds
// curvature to VFE (more thinking time), (2) collapses the sampling temperature
// (decisive local choice), and (3) raises tau (subjective time dilation).

/// Curvature-regularized VFE: base (multi-prior) VFE + scalar curvature of the
/// hidden-state cloud. A flat cloud (all points ~equal distance from centroid)
/// adds nothing; a curved cloud signals unexplored geometry -> +VFE.
/// `k_curve` scales the contribution.
/// Wired into live inference via `advance_physics` in main.rs.
pub fn vfe_with_curvature(
    base_vfe: f32,
    cloud: &[Vec<f32>],
    k_curve: f32,
) -> f32 {
    if cloud.is_empty() {
        return base_vfe;
    }
    let r = crate::curvature::scalar_curvature(cloud);
    base_vfe + k_curve * r
}

/// Curvature-driven sampling temperature. In a curved region the sampler
/// collapses: T' = T / (1 + k * R). Returns the effective temperature to use
/// for the next token draw.
/// Wired into live inference via `advance_physics` in main.rs.
pub fn temperature_with_curvature(
    base_temp: f32,
    cloud: &[Vec<f32>],
    k: f32,
) -> f32 {
    if cloud.is_empty() {
        return base_temp;
    }
    let r = crate::curvature::scalar_curvature(cloud);
    crate::curvature::temperature_from_curvature(base_temp, r, k)
}

/// Curvature-modulated tau update: high curvature dilates subjective time on
/// top of the base VFE signal. Equivalent to `tau_update` with an added
/// curvature term, keeping the same clamps.
/// Wired into live inference via `advance_physics` in main.rs.
pub fn tau_update_curvature(
    tau: f32,
    vfe: f32,
    cloud: &[Vec<f32>],
    learning_rate: f32,
    k_curve: f32,
) -> f32 {
    let extra = if cloud.is_empty() {
        0.0
    } else {
        k_curve * crate::curvature::scalar_curvature(cloud)
    };
    tau_update(tau, vfe + extra, learning_rate)
}

// ---------------------------------------------------------------------------
// Multi-prior attractor tau (LAYER 3c): WHICH memory is being pulled
// ---------------------------------------------------------------------------
// The single biggest conceptual upgrade from plan item #2: tau must become a
// measure of *which* attractor the input activates, not a scalar heuristic.
// For the hidden state, compute the responsibility KL (epistemic entropy of
// the softmax mix over the N named domain priors). Engagement (one attractor
// cleanly responsible, low normalized KL) CONTRACTS tau — the system knows
// exactly which memory it is in, so it acts fast. Novelty (responsibility
// spread across all domans, high normalized KL) DILATES tau — it must think
// harder. This replaces the monotone "high VFE → high tau" with a directed
// tau that falls where the input is canonical and rises where it is new.

/// Normalized epistemic KL ∈ [0,1]: 0 = one prior fully responsible (engaged),
/// 1 = uniform across all N priors (maximally novel). `kl` is the raw
/// Shannon entropy from `attractor::epistemic_kl`.
pub fn kl_normalized(kl: f32, n_priors: usize) -> f32 {
    if n_priors <= 1 || kl <= 0.0 {
        return 0.0;
    }
    (kl / (n_priors as f32).ln()).clamp(0.0, 1.0)
}

/// Cap on the number of priors evaluated per token (dynamic active-prior
/// prefilter). The domain-prior bank can grow arbitrarily; per-token
/// responsibility cost never exceeds `ACTIVE_PRIOR_CAP` centroids (plan #2
/// risk mitigation: embedding cost / capping).
pub const ACTIVE_PRIOR_CAP: usize = 4;

/// Responsibilities restricted to the top-K *active* priors (prefiltered by
/// nearest centroid), for bounded per-token cost.
///
/// Returns `(active_indices, responsibilities)` where `responsibilities[i]`
/// is the softmax weight of `priors[active_indices[i]]` — the mixture is
/// renormalized over the active set only, so a huge bank still yields a
/// meaningful engagement signal without scaling cost.
pub fn responsibilities_capped(
    hidden: &[f32],
    priors: &[Vec<f32>],
    cap: usize,
    temp: f32,
) -> (Vec<usize>, Vec<f32>) {
    if hidden.is_empty() || priors.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let idx = crate::attractor::active_prior_indices(hidden, priors, cap.max(1));
    if idx.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Renormalize the softmax over the active subset (indexes into `priors`).
    let active: Vec<Vec<f32>> = idx.iter().map(|&i| priors[i].clone()).collect();
    let r = crate::attractor::responsibilities(hidden, &active, temp);
    (idx, r)
}

/// Engage-modulated tau: target-seeking contraction on canonical input,
/// dilation on novelty. This is the correct physics of "which memory is
/// being pulled" — tau is a measure of subjective time dilation, and the
/// normal-time baseline is `tau = 1.0`.
///
/// `raw_vfe` is the VFE that would otherwise raise tau. Two regimes:
///   - `kn == 0` (one domain fully responsible): input is *canonical* to the
///     pulled attractor. The `- k_engage * (1-kn) * (tau - 1.0)` term pulls
///     tau toward the normal-time baseline (1.0). When tau > 1.0, tau
///     contracts; when tau < 1.0, tau relaxes upward. This is target-seeking
///     dynamics, not decay — tau=1.0 is a stable fixed point.
///   - `kn == 1` (spread across all priors): no contraction, plus a novelty
///     factor `(1 + k_engage)` multiplies the VFE signal -> tau dilates.
///
/// The result is clamped at the floor (callers clamp to tau_min).
pub fn tau_update_epistemic(
    tau: f32,
    raw_vfe: f32,
    kn: f32,
    learning_rate: f32,
    k_engage: f32,
) -> f32 {
    let vfe_sig = raw_vfe.max(0.0) * (1.0 + k_engage * kn);
    // Target-seeking: pull toward tau=1.0 (normal time) when engaged.
    let contract = k_engage * (1.0 - kn) * (tau - 1.0);
    let delta = learning_rate * (vfe_sig - contract);
    (tau + delta).clamp(0.1, 1e12)
}

// ---------------------------------------------------------------------------
// Tau-seeking actuator (Phase B item 4 — inverted tau_update_epistemic)
// ---------------------------------------------------------------------------
// Proposal distribution biased toward high predicted τ′ (novelty-seeking),
// BUT bench-gated: fitness stays the judge so tau cannot be gamed. Safe by
// construction — only candidates that also improve bench fitness promote.

/// Predict VFE proxy from patch text alone (offline hash, no model).
/// Longer, more entropic patches → higher predicted novelty.
fn predicted_vfe_proxy(patch: &str) -> f32 {
    if patch.trim().is_empty() {
        return 0.0;
    }
    // length-normalized character entropy
    let mut counts = [0usize; 256];
    for b in patch.bytes() {
        counts[b as usize] += 1;
    }
    let len = patch.len() as f32;
    let mut entropy = 0.0f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / len;
            entropy -= p * p.ln();
        }
    }
    // 0..~5, normalize to 0..1 via tanh-ish
    let ent_norm = (entropy / 4.0).clamp(0.0, 1.0);
    let len_norm = ((patch.len() as f32).ln() / 8.0).clamp(0.0, 1.0);
    0.6 * ent_norm + 0.4 * len_norm
}

fn predicted_kn_proxy(patch: &str) -> f32 {
    // lexical diversity: distinct tokens / total tokens, 0..1
    let toks: Vec<&str> = patch.split_whitespace().collect();
    if toks.is_empty() {
        return 0.0;
    }
    let uniq: std::collections::HashSet<&str> = toks.iter().cloned().collect();
    uniq.len() as f32 / toks.len() as f32
}

/// Predicted τ′ for a patch without running the model (hash proxy).
/// Uses the same dynamics as `tau_update_epistemic` so ranking by this
/// predicts the real tau movement.
pub fn tau_seeking_score(patch: &str, tau: f32, learning_rate: f32, k_engage: f32) -> f32 {
    let vfe_proxy = predicted_vfe_proxy(patch);
    let kn_proxy = predicted_kn_proxy(patch);
    tau_update_epistemic(tau, vfe_proxy, kn_proxy, learning_rate, k_engage)
}

// ---------------------------------------------------------------------------
// Layered physics controller (WIRING: L2 + L3 into live inference)
// ---------------------------------------------------------------------------
// One per-token call that consumes the ENTIRE layered stack in the real
// generation loop (not just tests):
//   - multi_prior_vfe       : surprisal + mixture novelty + epistemic entropy
//   - vfe_with_curvature    : + scalar curvature of the hidden-state cloud
//   - temperature_with_curvature : collapse sampling T in curved regions
//   - tau_update_curvature  : dilate tau on curved/novel geometry
//   - kalman_fuse_sources   : fusion across the source architectures
//
// `priors` = domain attractor centroids (from make_domain_priors over prompt
// hidden states, or the persisted attractor). `cloud` = recent hidden states
// (the metric-tensor sample). Returns (effective_temperature, new_tau, vfe).

/// Aggregate output of the layered physics controller for one token.
#[derive(Clone, Debug)]
pub struct PhysicsAdvance {
    pub temperature: f32,
    pub tau: f32,
    pub vfe: f32,
    /// VFE after curvature regularization (>= raw VFE).
    pub vfe_curved: f32,
    /// Normalized epistemic KL ∈ [0,1] of the responsibility mix over the
    /// named domain priors. 0 = canonical (one attractor pulled), 1 = novel.
    pub kl_norm: f32,
    /// Index of the dominant (max-responsibility) prior; None when no priors.
    pub dominant_prior: Option<usize>,
    /// Responsibility of the dominant prior ∈ [0,1].
    pub dominant_confidence: f32,
}

/// Default per-token curvature coupling (matched to the gguf stream scale).
pub const CURVATURE_COUPLING: f32 = 1.0;

/// Default engagement coupling for the multi-prior tau contraction.
pub const ENGAGE_COUPLING: f32 = 1.0;

/// Fixed temperature for responsibility computation in multi-prior VFE.
/// This is decoupled from the sampling temperature to allow sharp responsibility
/// computation regardless of the sampling regime (greedy or high temperature).
pub const RESPONSIBILITY_TEMP: f32 = 1.0;

/// Advance the layered physics state for one generated token.
///
/// * `base_temp` — the configured base temperature (may be 0 for greedy).
/// * `tau` — current subjective-time dilation.
/// * `vfe` — raw per-token surprisal VFE (from the sampler / kai-mlir).
/// * `pred`, `actual` — model distribution slices for multi-prior VFE
///   (pred = model logits-as-probs, actual = one-hot chosen token).
/// * `hidden` — the model's hidden-state vector at this token (used as the
///   multi-prior responsibility input).
/// * `cloud` — recent hidden-state vectors (metric sample); empty = no curve.
/// * `priors` — domain attractor centroids; empty = fall back to base VFE.
/// * `sources` — per-source predictions for Kalman fusion; empty = skip.
/// * `lr` — VFE→tau learning rate.
///
/// Temperature semantics: if `base_temp <= 0` the caller wants greedy; we
/// still return the physics-computed temperature for diagnostics but never
/// force sampling. Callers should treat `temperature <= 0` as argmax.
/// Wired into live inference (generate_streaming in main.rs, line ~1240).
pub fn advance_physics(
    base_temp: f32,
    tau: f32,
    vfe: f32,
    pred: &[f32],
    actual: &[f32],
    hidden: &[f32],
    variance: f32,
    cloud: &[Vec<f32>],
    priors: &[Vec<f32>],
    sources: &[Estimate],
    lr: f32,
) -> PhysicsAdvance {
    // L2: multi-prior VFE (surprisal + mixture novelty + epistemic entropy).
    // softmax temperature for responsibilities: fixed RESPONSIBILITY_TEMP.
    // Dynamic active-prior capping (plan #2): only the top-ACTIVE_PRIOR_CAP
    // nearest centroids are scored, so per-token cost is bounded however
    // large the domain-prior bank grows. `active_idx` maps the capped result
    // back onto the full priors list so dominant_prior stays the true index.
    let resp_temp = RESPONSIBILITY_TEMP;
    let (active_idx, r): (Vec<usize>, Vec<f32>) = if hidden.is_empty() || priors.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        responsibilities_capped(hidden, priors, ACTIVE_PRIOR_CAP, resp_temp)
    };
    let kl_norm = if r.iter().sum::<f32>() > 0.0 {
        kl_normalized(crate::attractor::epistemic_kl(&r), r.len())
    } else {
        0.0
    };
    let (dominant_prior, dominant_confidence) = if r.is_empty() {
        (None, 0.0)
    } else {
        let bi = r
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Map back to the full-bank index: r[i] is the weight of
        // priors[active_idx[i]].
        (Some(active_idx[bi]), r[bi])
    };
    let mp_vfe = if pred.is_empty() || actual.is_empty() || priors.is_empty() || hidden.is_empty() {
        vfe
    } else if active_idx.len() < priors.len() {
        // Capped: score only the active subset so the mixture-novelty and
        // epistemic terms are computed against the same bounded set, keeping
        // the dominant-attractor signal while limiting per-token cost.
        let active: Vec<Vec<f32>> = active_idx.iter().map(|&i| priors[i].clone()).collect();
        multi_prior_vfe(pred, actual, variance, hidden, &active, resp_temp)
    } else {
        multi_prior_vfe(pred, actual, variance, hidden, priors, resp_temp)
    };

    // L3a: curvature regularization (metric-tensor step).
    let vfe_curved = vfe_with_curvature(mp_vfe, cloud, CURVATURE_COUPLING);

    // L3a: collapse sampling temperature in curved regions.
    let t_eff = if base_temp > 0.0 {
        temperature_with_curvature(base_temp, cloud, CURVATURE_COUPLING)
    } else {
        base_temp
    };

    // L3c: multi-prior attractor tau (WHICH memory is being pulled).
    // Engage-modulated tau replaces the monotone curvature dilation: input
    // canonical to one attractor (kl_norm → 0) contracts tau, novelty spread
    // across the domain mixture (kl_norm → 1) dilates it. The curvature term
    // still rides on top, both clamped by the caller to [tau_min, tau_max].
    let tau_epi = tau_update_epistemic(tau, mp_vfe, kl_norm, lr, ENGAGE_COUPLING);
    let tau_new = tau_update_curvature(tau_epi, 0.0, cloud, lr, CURVATURE_COUPLING);

    // L2: Kalman fusion across source architectures (naive-ensemble upgrade).
    let fused = if sources.is_empty() {
        None
    } else {
        let f = kalman_fuse_sources(sources);
        Some((f.value, f.variance))
    };
    // Fold the fused estimate into the final VFE if present (variance-weighted
    // confidence bonus: a low-variance consensus raises the effective signal).
    let final_vfe = match fused {
        Some((value, var)) if var < 1.0 => vfe_curved + value * (1.0 - var),
        _ => vfe_curved,
    };

    PhysicsAdvance {
        temperature: t_eff,
        tau: tau_new,
        vfe,
        vfe_curved: final_vfe,
        kl_norm,
        dominant_prior,
        dominant_confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_update_increases_with_vfe() {
        let tau = 1.0;
        let tau2 = tau_update(tau, 5.0, 0.1);
        assert!(tau2 > tau, "tau should increase with positive VFE");
        assert!(tau2 < 2.0, "should not jump too much: {tau2}");
    }

    #[test]
    fn test_tau_update_clamps() {
        let tau = tau_update(1.0, 1e15, 1.0);
        assert!(tau <= 1e12, "tau should be clamped to 1e12, got {tau}");
        let tau = tau_update(0.01, 0.0, 0.1);
        assert!(tau >= 0.1, "tau should be clamped to 0.1, got {tau}");
    }

    #[test]
    fn test_tau_decay_reduces_tau() {
        let tau = 100.0;
        let decayed = tau_decay(tau, 10.0, 0.1);
        assert!(decayed < tau, "tau should decay: {decayed} < {tau}");
        assert!(decayed > 1.0, "tau should not decay below 1: {decayed}");
    }

    #[test]
    fn test_tau_decay_converges_to_one() {
        let tau = 10.0;
        let decayed = tau_decay(tau, 9999.0, 1.0);
        assert!((decayed - 1.0).abs() < 0.01, "tau should approach 1: {decayed}");
    }

    #[test]
    fn test_subjective_seconds() {
        let history = vec![(1.0, 2.0), (1.0, 4.0), (0.5, 8.0)];
        let subj = subjective_seconds(&history);
        assert!((subj - 10.0).abs() < 0.01, "2 + 4 + 4 = 10, got {subj}");
    }

    #[test]
    fn test_tau_no_decay_when_idle_zero() {
        let tau = 5.0;
        assert_eq!(tau_decay(tau, 0.0, 0.1), tau);
    }

    // ── LAYER 2: multi-prior / Kalman ────────────────────────────────
    #[test]
    fn test_kalman_gain_clamps() {
        assert!((kalman_gain(1.0, 1.0) - 0.5).abs() < 1e-6, "equal vars -> K=0.5");
        let k_trust_meas = kalman_gain(1.0, 0.1);
        assert!(k_trust_meas > 0.7, "small obs variance -> trust measurement, got {k_trust_meas}");
        let k_trust_prior = kalman_gain(0.1, 1.0);
        assert!(k_trust_prior < 0.3, "small prior variance -> trust prior, got {k_trust_prior}");
    }

    #[test]
    fn test_kalman_fuse_pulls_toward_measurement() {
        let f = kalman_fuse(0.0, 1.0, 10.0, 1.0);
        assert!((f.value - 5.0).abs() < 1e-6, "equal vars -> midpoint 5, got {}", f.value);
        // low measurement variance -> strongly pulled to the measurement (10)
        let f2 = kalman_fuse(0.0, 1.0, 10.0, 0.1);
        assert!(f2.value > 8.0, "should be pulled toward measurement, got {}", f2.value);
        assert!(f2.value < 10.0, "still not past measurement, got {}", f2.value);
    }

    #[test]
    fn test_kalman_fuse_sources_inverse_variance_weighting() {
        // two sources disagree; the certain one (var 0.1) should dominate
        let sources = [
            Estimate { value: 1.0, variance: 0.1 },
            Estimate { value: 9.0, variance: 10.0 },
        ];
        let fused = kalman_fuse_sources(&sources);
        assert!(fused.value < 3.0, "certain source (1.0) should dominate, got {}", fused.value);
        assert!(fused.variance < 0.2, "fused variance should shrink, got {}", fused.variance);
    }

    #[test]
    fn test_kalman_fuse_sources_empty() {
        let fused = kalman_fuse_sources(&[]);
        assert_eq!(fused.value, 0.0);
    }

    #[test]
    fn test_multi_prior_vfe_larger_than_base() {
        // identical prediction/actual -> base surprisal ~0; the multi-prior
        // terms (mixture novelty + epistemic) should add genuine content.
        let pred = vec![0.5, 0.5];
        let actual = vec![0.5, 0.5];
        // uniform-ish responsibilities -> high epistemic entropy + mixture novelty
        let priors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let hidden = vec![0.5, 0.5];
        let base = calculate_vfe(&pred, &actual, 0.0, 0.0);
        let mp = multi_prior_vfe(&pred, &actual, 0.0, &hidden, &priors, 1.0);
        assert!(mp > base, "multi-prior VFE must be >= base surprisal, {mp} !> {base}");
        assert!(mp > 0.0, "novel spread across priors should give positive VFE");
    }

    #[test]
    fn test_multi_prior_vfe_single_prior_no_epistemic() {
        // with a single prior the input either matches it (low nov) or not;
        // an already-known input aligned to its prior stays low.
        let pred = vec![0.7, 0.3];
        let actual = vec![0.7, 0.3];
        let priors = vec![vec![0.8, 0.2]];
        let hidden = vec![0.75, 0.25]; // close to prior -> low mixture novelty
        let base = calculate_vfe(&pred, &actual, 0.0, 0.0);
        let mp = multi_prior_vfe(&pred, &actual, 0.0, &hidden, &priors, 1.0);
        // N=1: responsibilities entropy = 0, novelty small -> mp ~ base
        assert!((mp - base).abs() < 0.5, "single prior should add only novelty, got diff {}", mp - base);
    }

    // ── LAYER 3a: curvature-wired VFE ───────────────────────────────
    #[test]
    fn test_vfe_curved_cloud_adds_vfe() {
        // flat/even cloud -> R~0 -> no addition
        let flat: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        let v_flat = vfe_with_curvature(1.0, &flat, 1.0);
        assert!((v_flat - 1.0).abs() < 1e-4, "flat cloud adds ~0, got {v_flat}");
        // uneven / spread cloud -> positive curvature -> VFE grows
        let curved: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 0.1], vec![0.0, 3.0]];
        let v_curved = vfe_with_curvature(1.0, &curved, 1.0);
        assert!(v_curved > 1.0, "curved cloud should add VFE: {v_curved}");
    }

    #[test]
    fn test_vcurv_empty_cloud_passthrough() {
        assert_eq!(vfe_with_curvature(2.5, &[], 5.0), 2.5);
    }

    #[test]
    fn test_temp_curved_collapses_temperature() {
        let base = 1.0;
        let flat: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        // flat cloud (2 even points) -> low/zero curvature -> keep temp
        let t_flat = temperature_with_curvature(base, &flat, 1.0);
        let curved: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 0.1], vec![0.0, 3.0]];
        let t_curved = temperature_with_curvature(base, &curved, 1.0);
        assert!(t_curved < t_flat, "curvature must lower temperature: {t_curved} !< {t_flat}");
        assert!(t_curved < base, "must drop below base temp");
    }

    #[test]
    fn test_tau_curvature_raises_tau() {
        let flat: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        let curved: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 0.1], vec![0.0, 3.0]];
        let t_flat = tau_update_curvature(1.0, 0.0, &flat, 0.05, 1.0);
        let t_curved = tau_update_curvature(1.0, 0.0, &curved, 0.05, 1.0);
        assert!(t_curved > t_flat, "curvature should dilate tau: {t_curved} !> {t_flat}");
    }

    // ── LAYER WIRING: advance_physics — the live per-token controller ────
    #[test]
    fn test_advance_physics_greedy_passthrough() {
        // base_temp 0 (greedy) with no priors/cloud/sources -> temp stays 0
        let a = advance_physics(0.0, 1.0, 0.3, &[], &[], &[], 0.0, &[], &[], &[], 0.05);
        assert_eq!(a.temperature, 0.0, "greedy base temp must pass through");
        assert!(a.tau >= 0.1, "tau must stay in clamp range");
        assert_eq!(a.vfe_curved, a.vfe, "no curvature/priors -> no change");
    }

    #[test]
    fn test_advance_physics_curvature_modulates() {
        let pred = vec![0.5, 0.5];
        let actual = vec![0.5, 0.5];
        let hidden = vec![0.5, 0.5];
        let priors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let flat = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        let curved = vec![vec![1.0, 0.0], vec![0.0, 0.1], vec![0.0, 3.0]];
        let a_flat = advance_physics(0.8, 1.0, 0.1, &pred, &actual, &hidden, 0.05, &flat, &priors, &[], 0.05);
        let a_curved = advance_physics(0.8, 1.0, 0.1, &pred, &actual, &hidden, 0.05, &curved, &priors, &[], 0.05);
        assert!(a_curved.vfe_curved > a_flat.vfe_curved, "curved cloud must raise VFE");
        assert!(a_curved.temperature < a_flat.temperature, "curved cloud must lower temperature");
        assert!(a_curved.tau > a_flat.tau, "curved cloud must dilate tau");
    }

    #[test]
    fn test_advance_physics_kalman_consensus_bonus() {
        let srcs = [Estimate { value: 3.0, variance: 0.1 }, Estimate { value: 4.0, variance: 0.1 }];
        let a_none = advance_physics(0.7, 1.0, 0.2, &[], &[], &[], 0.0, &[], &[], &[], 0.05);
        let a_fused = advance_physics(0.7, 1.0, 0.2, &[], &[], &[], 0.0, &[], &[], &srcs, 0.05);
        assert!(
            a_fused.vfe_curved > a_none.vfe_curved,
            "low-variance consensus should boost signal"
        );
    }

    // ── LAYER 3c: multi-prior attractor tau (WHICH memory) ─────────────
    #[test]
    fn test_kl_normalized_bounds() {
        // single prior -> floor 0
        assert_eq!(kl_normalized(0.9, 1), 0.0);
        // uniform over 2 priors -> ln(2)/ln(2) = 1
        let u = kl_normalized(0.693147, 2);
        assert!((u - 1.0).abs() < 1e-3, "uniform N=2 KL should normalize to 1, got {u}");
        // confident single-responsibility -> 0
        assert_eq!(kl_normalized(0.0, 4), 0.0);
    }

    #[test]
    fn test_responsibilities_capped_selects_and_renormalizes() {
        let hidden = vec![0.02, 1.0];
        let priors = vec![
            vec![0.3, 0.3],
            vec![0.9, 0.05],
            vec![0.05, 0.95], // nearest
            vec![0.4, 0.6],
        ];
        let (idx, r) = responsibilities_capped(&hidden, &priors, 2, 0.3);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx[0], 2, "nearest prior must be the active set leader");
        // renormalized over the active set: r sums to 1
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "active responsibilities must sum to 1, got {s}");
        // the leader holds the dominant share
        assert!(r[0] > 0.5, "nearest prior must dominate the active mix, got {}", r[0]);
        // empty hidden -> no responsibilities
        assert!(responsibilities_capped(&[], &priors, 2, 1.0).0.is_empty());
        assert!(responsibilities_capped(&hidden, &[], 2, 1.0).0.is_empty());
    }

    #[test]
    fn test_advance_physics_capped_cost_and_dominant_mapping() {
        // A large bank (8 priors) must only consult ACTIVE_PRIOR_CAP centroids,
        // and dominant_prior must report the TRUE index into the full bank.
        let mut priors: Vec<Vec<f32>> = vec![vec![-1.0, -1.0]; 8];
        // hidden is EXACTLY prior #5 (cosine sim = 1.0) and all other priors
        // are far away (antipodal), so the top-1 share is unambiguous no
        // matter how the capping prefilter ranks the rest.
        priors[5] = vec![0.72, 0.28];
        priors[0] = vec![-0.9, -0.9];
        priors[2] = vec![-0.9, 0.9];
        priors[6] = vec![0.9, -0.9];
        let hidden = vec![0.72, 0.28];
        let pred = vec![0.5, 0.5];
        let actual = vec![0.5, 0.5];
        let a = advance_physics(
            0.8, 1.0, 0.1, &pred, &actual, &hidden, 0.05, &[], &priors, &[], 0.05,
        );
        assert_eq!(a.dominant_prior, Some(5), "true bank index must survive capping");
        assert!(a.dominant_confidence > 0.5, "dominant share must be clear, got {}", a.dominant_confidence);
        // kl_norm stays a bounded probability-spread measure even under a huge
        // bank (the active cap keeps the mix normalizable).
        assert!((0.0..=1.0).contains(&a.kl_norm), "kl_norm must stay in [0,1], got {}", a.kl_norm);
    }

    #[test]
    fn test_tau_epistemic_target_seeking() {
        // tau above 1.0 + engaged (kn=0) -> tau relaxes toward 1.0
        let tau = 10.0;
        let new = tau_update_epistemic(tau, 0.1, 0.0, 0.1, 1.0);
        assert!(new < tau, "engaged + tau>1 must contract");
        assert!(new > 1.0, "must not overshoot below 1.0");
        
        // tau at 1.0 + engaged -> stable (no net change)
        let stable = tau_update_epistemic(1.0, 0.1, 0.0, 0.1, 1.0);
        assert!((stable - 1.0).abs() < 0.01, "tau=1.0 is the fixed point, got {stable}");
        
        // tau below 1.0 + engaged -> tau relaxes toward 1.0 (upward)
        let tau_below = 0.5;
        let new_below = tau_update_epistemic(tau_below, 0.1, 0.0, 0.1, 1.0);
        assert!(new_below > tau_below, "engaged + tau<1 must dilate toward 1.0");
        
        // Novelty (kn=1) always dilates tau
        let novel_high = tau_update_epistemic(tau, 1.0, 1.0, 0.1, 1.0);
        assert!(novel_high > tau, "novelty must dilate tau");
    }

    // ── Edge cases from REFLECTION_KAI.md §7/§8 (boundary conditions) ──
    // These were the missing coverage: extreme tau, degenerate clouds,
    // curvature-only modes, timeout safety, and multi-prior degenerate
    // distributions.

    #[test]
    fn test_tau_update_extreme_values() {
        // Huge VFE must clamp tau to the upper bound, not run away to inf.
        let huge = tau_update(10.0, 1e12, 1.0);
        assert!(huge.is_finite(), "tau_update must be finite for extreme VFE");

        // Zero VFE -> no change.
        assert!((tau_update(3.0, 0.0, 0.05) - 3.0).abs() < 1e-6);

        // Negative VFE is floored (>= 0) -> tau does not shrink below prior.
        let no_shrink = tau_update(2.0, -999.0, 1.0);
        assert!((no_shrink - 2.0).abs() < 1e-6, "negative VFE must not move tau");

        // Lower clamp at 0.1: tau already at floor, huge negative vfe stays floored.
        let floored = tau_update(0.0, 0.0, 1.0);
        assert!((floored - 0.1).abs() < 1e-6);

        // Extreme decay: tau dilated to 1e11 collapses toward 1.0 under a
        // large idle window but stays finite. exp(-0.1*1e6) underflows to 0,
        // so the result is exactly 1.0 (full relaxation), never above.
        let decayed = tau_decay(1.0e11, 1.0e6, TAU_DECAY_RATE);
        assert!(decayed.is_finite() && decayed >= 1.0);
        // Near-infinite idle fully relaxes to ~1.0.
        let relaxed = tau_decay(1.0e6, 1.0e9, TAU_DECAY_RATE);
        assert!((relaxed - 1.0).abs() < 1e-2, "tau must relax to ~1.0: got {}", relaxed);
    }

    #[test]
    fn test_curvature_only_mode_degenerate_clouds() {
        // Empty cloud: curvature is 0, no VFE added, no division-by-zero.
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(vfe_with_curvature(0.5, &empty, 1.0), 0.5,
                   "empty cloud adds no curvature VFE");

        // Single-point cloud is degenerate: zero variance/curvature.
        let single = vec![vec![1.0_f32, 2.0, 3.0]];
        let v = vfe_with_curvature(0.4, &single, 1.0);
        assert!(v.is_finite(), "single-point cloud must not NaN");

        // Curvature-only: base_vfe = 0 but a curved (spread) cloud still
        // produces a finite, non-negative curvature contribution.
        let spread = vec![vec![0.0_f32, 0.0, 0.0], vec![3.0, 4.0, 0.0], vec![0.0, 5.0, 0.0]];
        let v_only = vfe_with_curvature(0.0, &spread, 1.0);
        assert!(v_only >= 0.0, "curvature-only VFE must be non-negative");

        // temperature_with_curvature with degenerate inputs must be finite
        // and not exceed base when curvature is nil.
        let t_empty = temperature_with_curvature(0.3, &empty, 1.0);
        assert!((t_empty - 0.3).abs() < 1e-5,
                "empty cloud temperature should pass base through: got {}", t_empty);
        let t_spread = temperature_with_curvature(0.3, &spread, CURVATURE_COUPLING);
        assert!(t_spread < 0.3, "curved region should collapse temperature below base");

        // tau_update_curvature must not NaN on degenerate input (empty cloud)
        // or on a spread cloud.
        let tau_empty = tau_update_curvature(1.0, 0.0, &empty, TAU_LEARNING_RATE, CURVATURE_COUPLING);
        assert!(tau_empty.is_finite(), "tau_update_curvature must not NaN on empty cloud");
        let tau_spread = tau_update_curvature(1.0, 0.0, &spread, TAU_LEARNING_RATE, CURVATURE_COUPLING);
        assert!(tau_spread.is_finite() && tau_spread >= 0.1,
                "tau_update_curvature on spread cloud must be finite & floored");
    }

    #[test]
    fn test_tau_update_does_not_hang_timeout_guard() {
        // Safety/timeout assertion: tau_update and subjective_seconds must
        // complete deterministically in negligible time even for pathological
        // inputs (no infinite loops / no sqrt-of-negative blowups).
        let start = std::time::Instant::now();
        let huge_hist: Vec<(f32, f32)> = (0..100_000).map(|i| (1.0, i as f32)).collect();
        let _ = subjective_seconds(&huge_hist);
        let _ = tau_update_f32_nan_guard(1.0, f32::NAN);
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 500,
                "tau/subjective update must be fast, took {:?}", elapsed);
    }

    #[test]
    fn test_multi_prior_degenerate_and_bounds() {
        // Degenerate responsibilities: all mass on one prior -> kl_norm = 0.
        let pred = vec![0.5_f32, 0.5];
        let actual = vec![1.0_f32, 0.0];
        let hidden = vec![1.0_f32, 0.0, 0.0];
        let priors = vec![vec![1.0_f32, 0.0, 0.0], vec![1.0_f32, 0.0, 0.0]];
        let v = multi_prior_vfe(&pred, &actual, 0.0, &hidden, &priors, 1.0);
        assert!(v.is_finite(), "degenerate single-prior case must be finite");

        // Empty priors/hidden must pass through base VFE without panic.
        let empty_p: Vec<Vec<f32>> = vec![];
        let empty_h: Vec<f32> = vec![];
        let base = calculate_vfe(&pred, &actual, 0.0, 0.0);
        let vp = multi_prior_vfe(&pred, &actual, 0.0, &empty_h, &empty_p, 1.0);
        assert!((vp - base).abs() < 1e-5, "empty priors must return base VFE");

        // kl_normalized bounds: always in [0, 1].
        for kl in [-5.0_f32, 0.0, 0.5, 1.0, 42.0, 1e9] {
            for n in [1, 2, 5, 100] {
                let kn = kl_normalized(kl, n);
                assert!(kn >= 0.0 && kn <= 1.0, "kl_normalized({},{})={} out of [0,1]", kl, n, kn);
            }
        }

        // responsibilities_capped over a degenerate empty hidden must return
        // empty (early-return guard), no panic. A zero-vector (non-empty but
        // all-zero) also must stay finite rather than NaN.
        let empty_h: Vec<f32> = vec![];
        let caps = responsibilities_capped(&empty_h, &priors, 4, 1.0);
        assert!(caps.1.is_empty(), "truly-empty hidden must yield empty responsibilities");
        let zero = vec![0.0_f32; 3];
        let caps_z = responsibilities_capped(&zero, &priors, 4, 1.0);
        let sum_z: f32 = caps_z.1.iter().sum();
        assert!(sum_z.is_finite(), "all-zero hidden must not produce NaN responsibilities");
        let nonzero = vec![1.0_f32, 0.0, 0.0];
        let caps2 = responsibilities_capped(&nonzero, &priors, 4, 1.0);
        let sum: f32 = caps2.1.iter().sum();
        assert!(sum.is_finite() && sum >= 0.0,
                "capped responsibilities on valid hidden must be finite & >=0");
    }

    // Helper: NaN-safe tau update guard (no NaN should ever leave the controller).
    fn tau_update_f32_nan_guard(tau: f32, vfe: f32) -> f32 {
        let r = tau_update(tau, vfe, TAU_LEARNING_RATE);
        debug_assert!(r.is_finite(), "tau_update produced non-finite value");
        r
    }

    #[test]
    fn test_tau_epistemic_novelty_dilates_more_when_engaged_less() {
        // For fixed vfe, novelty (kn=1) gives more dilation than engagement (kn=0)
        let tau = 5.0;
        let engaged = tau_update_epistemic(tau, 1.0, 0.0, 0.1, 1.0);
        let novel = tau_update_epistemic(tau, 1.0, 1.0, 0.1, 1.0);
        assert!(novel > engaged, "novelty should dilate more than engagement");
    }
}
