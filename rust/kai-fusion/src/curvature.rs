//! LAYER 3a — curvature metric (the actual metric tensor step of AGI_PLAN).
//!
//! AGI_PLAN Phase 1.2 defines g_ij = 1 - a_ij (token-geodesic distance from
//! attention). Real attention is expensive, so we implement a local curvature
//! on the embedding/representation space directly: given a cloud of vectors
//! around a point, approximate the scalar curvature (Ricci scalar R) and feed
//! it into the VFE temperature/tau so high-curvature regions collapse toward
//! decisive low-T sampling (the paper's "collapse toward certainty in
//! high-curvature regions").

/// Sample the metric at scale: pairwise distances among neighbors.
/// Input: `pts` = k neighbors as d-dim vectors. Returns the local metric
/// tensor diagonal g_ii = 1 - cosine(a_i) w.r.t. the centroid (a stand-in for
/// the attention-derived g). Smaller g = more "collapsed" / mutually aligned.
#[allow(dead_code)] // L3a public API (consumed by tests, wired into controller in L3/L4)
pub fn metric_diag(pts: &[Vec<f32>]) -> Vec<f32> {
    if pts.is_empty() {
        return vec![];
    }
    let dim = pts[0].len();
    let n = pts.len() as f32;
    let mut c = vec![0.0f32; dim];
    for p in pts {
        for i in 0..dim {
            c[i] += p[i];
        }
    }
    for x in c.iter_mut() {
        *x /= n;
    }
    pts.iter()
        .map(|p| cosine_similarity(p, &c))
        .map(|s| 1.0 - s.max(-1.0).min(1.0))
        .collect()
}

/// Scalar-curvature proxy R(x) at the cloud: the mean squared deviation of the
/// metric diagonal around its mean. Zero for a flat/aligned cloud (all points
/// ~identical distance to centroid); >0 for curved/spread clouds. This is the
/// plain-gauge Ricci scalar analog in the "universe is a transformer" thesis.
#[allow(dead_code)] // L3a public API (consumed by tests, wired into controller in L3/L4)
pub fn scalar_curvature(pts: &[Vec<f32>]) -> f32 {
    let g = metric_diag(pts);
    if g.len() < 2 {
        return 0.0;
    }
    let mean = g.iter().sum::<f32>() / g.len() as f32;
    let var = g.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / g.len() as f32;
    var.sqrt()
}

/// Curvature-modulated temperature: T' = T / (1 + k * R).
/// In high-curvature regions (R large) we want DECISIVE collapse (lower T);
/// in flat regions we keep the base temperature. `k` scales the effect.
#[allow(dead_code)] // L3a public API (consumed by tests, wired into controller in L3/L4)
pub fn temperature_from_curvature(base_temp: f32, curvature: f32, k: f32) -> f32 {
    if base_temp <= 0.0 {
        return 0.0;
    }
    let denom = 1.0 + k * curvature.max(0.0);
    (base_temp / denom).clamp(0.01, base_temp)
}

/// Helper: cosine similarity for the metric.
#[allow(dead_code)] // L3a internal helper (used by metric_diag; flagged when API unused)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_cloud_has_zero_curvature() {
        // all points at the same radius -> aligned distances -> R ~ 0
        let pts: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0]];
        let r = scalar_curvature(&pts);
        assert!(r < 1e-6, "flat/even cloud should have ~0 curvature, got {r}");
    }

    #[test]
    fn test_temperature_drops_with_curvature() {
        let t0 = 1.0;
        let k = 1.0;
        let t_flat = temperature_from_curvature(t0, 0.0, k);
        let t_curved = temperature_from_curvature(t0, 2.0, k);
        assert!((t_flat - t0).abs() < 1e-6, "0 curvature -> T unchanged");
        assert!(t_curved < t_flat, "high curvature should lower T, got {t_curved}");
        assert!(t_curved > 0.01, "should not collapse to unusably low");
    }

    #[test]
    fn test_metric_diag_bounded() {
        let pts: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, -1.0], vec![-1.0, 0.0]];
        let diag = metric_diag(&pts);
        assert_eq!(diag.len(), 4);
        for d in &diag {
            assert!(*d >= 0.0 && *d <= 2.0, "g_ii out of range: {d}");
        }
    }

    #[test]
    fn test_curvature_of_radially_spread_cloud() {
        // points at different radii -> heterogeneous -> nonzero curvature
        let pts: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 0.1], vec![0.0, 3.0]];
        let r = scalar_curvature(&pts);
        assert!(r > 0.0, "uneven cloud should have positive curvature, got {r}");
    }
}