//! Bridge between Kai's Phase-A attractor (the 10 ingested architectures) and
//! the live decoder. The attractor is a set of 768-dim vectors; we compress it
//! to a centroid and project that centroid into the model's hidden dimension to
//! form a *prior* the decoder is seeded with. Novelty in Kai's VFE is then
//! measured as distance of a context's hidden state from that assimilated
//! prior — i.e. how far new input is from everything Kai has already absorbed.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Load the Phase-A attractor vectors (each 768-dim) from a JSON `{vecs:[...]}`.
pub fn load(path: &str) -> Result<Vec<Vec<f32>>, String> {
    // Debug: check if file exists
    if !std::path::Path::new(path).exists() {
        return Err(format!("attractor file does not exist: {}", path));
    }
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("json parse error: {}", e))?;
    let vecs = v
        .get("vecs")
        .and_then(|x| x.as_array())
        .ok_or("attractor missing 'vecs'")?;
    let mut out = Vec::with_capacity(vecs.len());
    for row in vecs {
        let row = row
            .as_array()
            .ok_or("attractor row not an array")?
            .iter()
            .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(0.0))
            .collect::<Vec<_>>();
        out.push(row);
    }
    Ok(out)
}

/// Mean vector of the attractor (768-dim).
pub fn centroid(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![];
    }
    let n = vecs.len() as f32;
    let dim = vecs[0].len();
    let mut c = vec![0.0f32; dim];
    for v in vecs {
        for (i, x) in v.iter().enumerate() {
            c[i] += x;
        }
    }
    for x in c.iter_mut() {
        *x /= n;
    }
    c
}

/// Deterministic seeded projection of a 768-dim centroid into `dim` hidden
/// space. Uses a fixed (seed-derived) random projection matrix, so the same
/// attractor always yields the same prior — Kai's assimilated "world model"
/// seed is stable across runs.
pub fn project_to_dim(centroid768: &[f32], dim: usize, seed: u64) -> Vec<f32> {
    let src = centroid768.len();
    let mut rng = Xor(seed);
    let mut proj = vec![0.0f32; dim];
    for d in 0..dim {
        let mut acc = 0.0f32;
        for s in 0..src {
            // random projection weight ~ N(0,1) approximated by summed uniforms
            let w = (rng.next() + rng.next() + rng.next() + rng.next() - 2.0) * 0.5;
            acc += w * centroid768[s];
        }
        proj[d] = acc / (src as f32).sqrt();
    }
    proj
}

/// Kai novelty: distance of a context hidden state from the assimilated prior.
/// High => the input is surprising relative to everything already absorbed;
/// low => the input is already "known" to Kai. This is the novelty term of VFE.
pub fn novelty(hidden: &[f32], prior: &[f32]) -> f32 {
    if hidden.len() != prior.len() || hidden.is_empty() {
        return 0.0;
    }
    let mut d = 0.0f32;
    for i in 0..hidden.len() {
        let diff = hidden[i] - prior[i];
        d += diff * diff;
    }
    (d / hidden.len() as f32).sqrt()
}

/// Add the assimilated prior into the decoder's token-embedding table so the
/// model literally starts biased by the 10 architectures Kai ingested.
#[allow(dead_code)]
pub fn seed_embedding(embed: &mut Array1<f32>, prior: &[f32], scale: f32) {
    let dim = prior.len().min(embed.len());
    for i in 0..dim {
        embed[i] += scale * prior[i];
    }
}

/// Tiny deterministic RNG (xorshift64) — keeps the projection reproducible.
struct Xor(u64);
impl Xor {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32) / (1u64 << 24) as f32 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Attractor persistence and convergence utilities (multi-agent coordination)
// ---------------------------------------------------------------------------

/// Default path for the shared attractor file.
pub const DEFAULT_ATTRACTOR_PATH: &str = ".axiom_state/kai_fusion_attractor.json";

/// Default path for the NAMED domain-prior registry (multi-prior VFE targets).
pub const DEFAULT_DOMAIN_PRIORS_PATH: &str = ".axiom_state/domain_priors.json";

/// A named domain prior: a persisted centroid with a human-readable domain
/// (e.g. "physics", "code", "wiki", "conversation"). The registry of N such
/// priors is what multi-prior VFE mixes against — tau becomes a measure of
/// WHICH memory the input is pulling, not a scalar heuristic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainPrior {
    pub name: String,
    pub centroid: Vec<f32>,
}

/// Load the named domain-prior registry. Format:
/// `{"domains":[{"name":"physics","centroid":[...]}, ...]}`.
pub fn load_named_domains(path: &str) -> Result<Vec<DomainPrior>, String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("domain-prior registry does not exist: {path}"));
    }
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("json parse error: {e}"))?;
    let arr = v
        .get("domains")
        .and_then(|x| x.as_array())
        .ok_or("registry missing 'domains' array")?;
    let mut out = Vec::with_capacity(arr.len());
    for row in arr {
        let name = row
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("domain")
            .to_string();
        let centroid = row
            .get("centroid")
            .and_then(|x| x.as_array())
            .ok_or("domain missing 'centroid'")?
            .iter()
            .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(0.0))
            .collect::<Vec<_>>();
        out.push(DomainPrior { name, centroid });
    }
    Ok(out)
}

/// Save the named domain-prior registry.
pub fn save_named_domains(path: &str, domains: &[DomainPrior]) -> Result<(), String> {
    let doms: Vec<serde_json::Value> = domains
        .iter()
        .map(|d| {
            serde_json::json!({
                "name": d.name,
                "centroid": d.centroid,
            })
        })
        .collect();
    let json = serde_json::json!({ "domains": doms });
    let dir = std::path::Path::new(path).parent().unwrap();
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let raw =
        serde_json::to_string_pretty(&json).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(path, &raw).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

/// Build a named domain-prior registry from raw attractor vectors by k-means,
/// then give each centroid the name of its best-aligned vector's label. If
/// `labels` is shorter than the vector set, falls back to `domain_{i}`.
pub fn build_named_domains(
    vecs: &[Vec<f32>],
    labels: &[String],
    n_domains: usize,
    iters: usize,
    seed: u64,
) -> Vec<DomainPrior> {
    let priors = make_domain_priors(vecs, n_domains, iters, seed);
    priors
        .into_iter()
        .enumerate()
        .map(|(i, centroid)| {
            // name = label of the vector most similar to this centroid
            let mut best_name = format!("domain_{i}");
            let mut best_sim = f32::NEG_INFINITY;
            for (vi, v) in vecs.iter().enumerate() {
                let s = cosine_similarity(&centroid, v);
                if s > best_sim {
                    best_sim = s;
                    if let Some(lbl) = labels.get(vi) {
                        best_name = lbl.clone();
                    }
                }
            }
            DomainPrior { name: best_name, centroid }
        })
        .collect()
}

/// Responsibility of each NAMED domain prior for a hidden state — but returns
/// the (name, responsibility) pairs so the caller can report WHICH memory is
/// being pulled.
pub fn named_responsibilities(
    hidden: &[f32],
    domains: &[DomainPrior],
    temp: f32,
) -> Vec<(String, f32)> {
    let priors: Vec<Vec<f32>> = domains.iter().map(|d| d.centroid.clone()).collect();
    let r = responsibilities(hidden, &priors, temp);
    domains
        .iter()
        .zip(r.iter())
        .map(|(d, &ri)| (d.name.clone(), ri))
        .collect()
}

/// Push a new vector into the attractor file.
/// Creates the file if it doesn't exist. Appends the vector.
/// Uses a lock file (path + ".lock") for multi-agent safety.
pub fn push(path: &str, vec: &[f32]) -> Result<(), String> {
    // Acquire lock
    let lock_path = format!("{path}.lock");
    let lock_file = lock_acquire(&lock_path)?;

    let mut vecs = if std::path::Path::new(path).exists() {
        load(path)?
    } else {
        Vec::new()
    };
    vecs.push(vec.to_vec());
    let result = save(path, &vecs);

    // Release lock
    lock_release(lock_file, &lock_path);

    result
}

/// Acquire a lock file, retrying with exponential backoff.
/// Returns the lock file handle (kept open during critical section).
fn lock_acquire(lock_path: &str) -> Result<std::fs::File, String> {
    let mut wait_ms = 1;
    for _ in 0..20 {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock_path)
        {
            Ok(f) => return Ok(f),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                if wait_ms < 256 { wait_ms *= 2; }
            }
            Err(e) => return Err(format!("lock acquire: {e}")),
        }
    }
    Err(format!("could not acquire lock after 20 retries: {lock_path}"))
}

/// Release a lock file by closing and deleting it.
fn lock_release(_file: std::fs::File, lock_path: &str) {
    drop(_file);
    let _ = std::fs::remove_file(lock_path);
}

/// Save attractor vectors to a JSON file.
pub fn save(path: &str, vecs: &[Vec<f32>]) -> Result<(), String> {
    let json = serde_json::json!({ "vecs": vecs });
    let dir = std::path::Path::new(path).parent().unwrap();
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let raw = serde_json::to_string_pretty(&json).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(path, &raw).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

/// Compute convergence score: mean pairwise cosine similarity of the last `window` vectors.
/// Returns (convergence, n) where convergence in [0,1], 1 = perfectly aligned.
/// If there are <2 vectors, returns (1.0, n) (trivially converged).
pub fn convergence(path: &str, window: usize) -> Result<(f32, usize), String> {
    let vecs = load(path)?;
    let n = vecs.len();
    if n < 2 {
        return Ok((1.0, n));
    }
    let window = window.min(n);
    let recent = &vecs[n - window..];
    let mut total_sim = 0.0f32;
    let mut pairs = 0;
    for i in 0..window {
        for j in (i + 1)..window {
            let sim = cosine_similarity(&recent[i], &recent[j]);
            total_sim += sim;
            pairs += 1;
        }
    }
    let conv = if pairs > 0 { total_sim / pairs as f32 } else { 1.0 };
    Ok((conv, n))
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Multi-prior attractor (LAYER 2)
// ---------------------------------------------------------------------------
// Instead of a single centroid prior, Kai maintains N domain priors (physics,
// code, wiki, conversation, ...). VFE against the mixture: the responsibility
// r_i of each prior for an incoming hidden state is a softmax over similarity,
// and the *epistemic KL* of that responsibility distribution tells us whether
// the input clearly activates one memory (low KL) or is novel across all
// domains (high KL). This is what turns tau from a scalar heuristic into a
// measure of WHICH memory is being pulled.

/// Indices of the `cap` centroids nearest to `hidden` (by cosine similarity),
/// best first. This is the dynamic active-prior prefilter (plan #2 risk
/// mitigation): per-token responsibility cost is bounded by evaluating only
/// the top-K nearest centroids, so the domain-prior bank can grow without
/// scaling embedding cost per token. Returns `[]` on empty priors.
pub fn active_prior_indices(hidden: &[f32], priors: &[Vec<f32>], cap: usize) -> Vec<usize> {
    if priors.is_empty() || hidden.is_empty() || cap == 0 {
        return vec![];
    }
    let cap = cap.min(priors.len());
    let mut ranked: Vec<(usize, f32)> = priors
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cosine_similarity(hidden, p)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(cap);
    ranked.into_iter().map(|(i, _)| i).collect()
}

/// Softmax responsibility of each prior centroid for a hidden state.
/// Returns r in [0,1]^N summing to 1. `temp` scales the sharpness
/// (lower temp = sharper assignment to the best-matching prior).

pub fn responsibilities(hidden: &[f32], priors: &[Vec<f32>], temp: f32) -> Vec<f32> {
    if priors.is_empty() {
        return vec![];
    }
    let t = temp.max(1e-3);
    let mut scores = Vec::with_capacity(priors.len());
    let mut max_s = f32::NEG_INFINITY;
    for p in priors {
        let s = cosine_similarity(hidden, p) / t;
        max_s = max_s.max(s);
        scores.push(s);
    }
    let mut exps = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for s in scores {
        let e = (s - max_s).exp(); // subtract max for numeric stability
        exps.push(e);
        sum += e;
    }
    if sum <= 1e-12 {
        return vec![1.0 / exps.len() as f32; exps.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Weighted mixture prior: sum of r_i * prior_i.

pub fn mixture_prior(hidden: &[f32], priors: &[Vec<f32>], temp: f32) -> Vec<f32> {
    if priors.is_empty() {
        return vec![];
    }
    let r = responsibilities(hidden, priors, temp);
    let dim = priors[0].len();
    let mut mix = vec![0.0f32; dim];
    for (ri, p) in r.iter().zip(priors.iter()) {
        for i in 0..dim {
            mix[i] += ri * p[i];
        }
    }
    mix
}

/// Epistemic uncertainty of the responsibility spread: the Shannon entropy
/// H(r) = -Σ r_i ln r_i of the responsibility distribution.
/// 0 = one prior fully responsible (confident, the input cleanly activates a
/// known memory); ln(N) = all priors equally responsible (maximally uncertain —
/// the input is novel / not pinned to any single domain). This is the term
/// that should drive tau UP on genuine novelty.

pub fn epistemic_kl(r: &[f32]) -> f32 {
    if r.len() <= 1 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &x in r {
        if x > 1e-12 {
            h -= x * x.ln();
        }
    }
    h
}

/// Novely of a hidden state against the MIXTURE prior (multi-prior novelty).
/// Reuses the base distance but against the blended prior instead of the
/// single centroid.

pub fn novelty_vs_mixture(hidden: &[f32], priors: &[Vec<f32>], temp: f32) -> f32 {
    if priors.is_empty() || hidden.is_empty() {
        return 0.0;
    }
    let mix = mixture_prior(hidden, priors, temp);
    novelty(hidden, &mix)
}

/// Cluster attractor vectors into N domain priors via a light k-means-ish
/// run (deterministic seed) — the bootstrapping step that creates the domain
/// priors from the raw attractor set before VFE mixing starts.

pub fn make_domain_priors(vecs: &[Vec<f32>], n_clusters: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    if vecs.is_empty() || n_clusters == 0 {
        return vec![];
    }
    let n_clusters = n_clusters.min(vecs.len());
    let dim = vecs[0].len();
    let mut rng = Xor(seed);
    // init: pick n_clusters distinct seed vectors deterministically
    let step = vecs.len() / n_clusters;
    let mut centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|c| vecs[(c * step) % vecs.len()].clone())
        .collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0.0f32; dim]; n_clusters];
        let mut counts = vec![0usize; n_clusters];
        for v in vecs {
            let mut bi = 0;
            let mut bd = f32::INFINITY;
            for (i, c) in centroids.iter().enumerate() {
                let d = cosine_similarity(v, c);
                // treat distance as 1 - sim (cosine distance)
                let d = 1.0 - d;
                if d < bd {
                    bd = d;
                    bi = i;
                }
            }
            for i in 0..dim {
                sums[bi][i] += v[i];
            }
            counts[bi] += 1;
        }
        for i in 0..n_clusters {
            if counts[i] > 0 {
                for j in 0..dim {
                    centroids[i][j] = sums[i][j] / counts[i] as f32;
                }
            }
        }
        rng.next(); // keep deterministic across iters even if unused
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("kai_test_attractor.json");
        let p = path.to_str().unwrap().to_string();

        // Push first vector
        push(&p, &[1.0, 2.0, 3.0]).unwrap();
        let vecs = load(&p).unwrap();
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0], vec![1.0, 2.0, 3.0]);

        // Push second
        push(&p, &[4.0, 5.0, 6.0]).unwrap();
        let vecs = load(&p).unwrap();
        assert_eq!(vecs.len(), 2);

        // Clean up
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_convergence_identical_vectors() {
        let dir = std::env::temp_dir();
        let path = dir.join("kai_test_conv.json");
        let p = path.to_str().unwrap().to_string();

        // Remove any leftovers from an interrupted earlier run: push() is
        // append-only, so a stale file would corrupt the count assertion.
        let _ = std::fs::remove_file(&p);
        for _ in 0..5 {
            push(&p, &[1.0, 0.0, 0.0]).unwrap();
        }
        let (conv, n) = convergence(&p, 10).unwrap();
        assert_eq!(n, 5);
        assert!((conv - 1.0).abs() < 1e-6, "identical vectors should have conv=1, got {conv}");

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_convergence_orthogonal_vectors() {
        let dir = std::env::temp_dir();
        let path = dir.join("kai_test_conv2.json");
        let p = path.to_str().unwrap().to_string();

        // Same defensive cleanup as test_convergence_identical_vectors.
        let _ = std::fs::remove_file(&p);
        push(&p, &[1.0, 0.0]).unwrap();
        push(&p, &[0.0, 1.0]).unwrap();
        let (conv, n) = convergence(&p, 10).unwrap();
        assert_eq!(n, 2);
        assert!(conv.abs() < 1e-6, "orthogonal vectors should have conv=0, got {conv}");

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_responsibilities_sum_to_one() {
        let hidden = vec![1.0, 0.0];
        let priors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = responsibilities(&hidden, &priors, 0.5);
        assert_eq!(r.len(), 2);
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "responsibilities must sum to 1, got {sum}");
        // hidden is aligned with prior0, so it should dominate at low temp
        assert!(r[0] > r[1], "hidden matches prior0 so r[0] should dominate");
    }

    #[test]
    fn test_epistemic_kl_uniform_is_max() {
        // uniform responsibility over 2 priors -> KL = ln(2) ~ 0.693
        let r_uniform = vec![0.5, 0.5];
        let r_confident = vec![1.0, 0.0];
        let kl_u = epistemic_kl(&r_uniform);
        let kl_c = epistemic_kl(&r_confident);
        assert!((kl_u - 0.693147).abs() < 1e-3, "uniform KL should be ln(2), got {kl_u}");
        assert!(kl_c.abs() < 1e-6, "confident KL should be 0, got {kl_c}");
        assert!(kl_u > kl_c, "more uncertain assignment should have higher epistemic KL");
    }

    #[test]
    fn test_make_domain_priors_reproduces_separation() {
        // two well-separated clouds -> kmeans with n=2 should find both
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![ -1.0 + i as f32 * 0.05, 0.0 ])
            .chain((0..10).map(|i| vec![ 1.0 + i as f32 * 0.05, 0.0 ]))
            .collect();
        let priors = make_domain_priors(&vecs, 2, 10, 42);
        assert_eq!(priors.len(), 2);
        // the two centroids should be far apart in cosine distance
        let d = 1.0 - cosine_similarity(&priors[0], &priors[1]);
        assert!(d > 0.2, "two separated clouds should give separated priors, dist={d}");
    }

    #[test]
    fn test_mixture_prior_blends() {
        let hidden = vec![0.5, 0.5];
        let priors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // at very high temp ~ uniform -> mixture is average of the two priors
        let mix = mixture_prior(&hidden, &priors, 50.0);
        assert!((mix[0] - 0.5).abs() < 0.1, "uniform blend should be ~0.5, got {}", mix[0]);
    }

    #[test]
    fn test_named_domains_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("kai_test_named_domains.json");
        let p = path.to_str().unwrap().to_string();
        let doms = vec![
            DomainPrior { name: "physics".into(), centroid: vec![1.0, 0.0, 0.5] },
            DomainPrior { name: "code".into(), centroid: vec![0.0, 1.0, -0.5] },
        ];
        save_named_domains(&p, &doms).unwrap();
        let loaded = load_named_domains(&p).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "physics");
        assert_eq!(loaded[0].centroid, vec![1.0, 0.0, 0.5]);
        assert_eq!(loaded[1].name, "code");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn test_build_named_domains_labels_by_similarity() {
        // two well-separated clouds, each labeled -> each centroid picks up a
        // label from one of its own members.
        let mut vecs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..5 {
            vecs.push(vec![-1.0 + i as f32 * 0.05, 0.0]);
            labels.push(format!("neg_{i}"));
        }
        for i in 0..5 {
            vecs.push(vec![1.0 + i as f32 * 0.05, 0.0]);
            labels.push(format!("pos_{i}"));
        }
        let domains = build_named_domains(&vecs, &labels, 2, 8, 7);
        assert_eq!(domains.len(), 2);
        // the two centroids separate along x
        let xs: Vec<f32> = domains.iter().map(|d| d.centroid[0]).collect();
        let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max_x - min_x > 1.5, "negative/positive clouds should separate, span={}", max_x - min_x);
        // each centroid's label should be aligned with its cluster sign
        for d in &domains {
            assert!(
                (d.centroid[0] < 0.0 && d.name.starts_with("neg"))
                    || (d.centroid[0] > 0.0 && d.name.starts_with("pos")),
                "centroid label should match cluster: {} @ x={}",
                d.name, d.centroid[0]
            );
        }
    }

    #[test]
    fn test_named_responsibilities_reports_pulled_domain() {
        let hidden = vec![0.95, 0.05];
        let doms = vec![
            DomainPrior { name: "code".into(), centroid: vec![1.0, 0.0] },
            DomainPrior { name: "conversation".into(), centroid: vec![0.0, 1.0] },
        ];
        // sharp temp: canonical input cleanly activates one attractor
        let nr = named_responsibilities(&hidden, &doms, 0.2);
        assert_eq!(nr.len(), 2);
        // hidden is aligned with "code" -> it must dominate the report
        let (name, r) = &nr[0];
        assert_eq!(name, "code");
        assert!(*r > 0.85, "code must be pulled, r={r}");
    }

    #[test]
    fn test_active_prior_indices_picks_nearest() {
        // hidden is closest to prior #2, then #0 — capping must preserve order.
        let hidden = vec![0.02, 1.0];
        let priors = vec![
            vec![0.3, 0.3], // #0 moderately close
            vec![0.9, 0.05], // #1 farthest
            vec![0.05, 0.95], // #2 nearest
            vec![0.4, 0.6], // #3 second
        ];
        let idx = active_prior_indices(&hidden, &priors, 3);
        assert_eq!(idx.len(), 3, "cap 3 of 4 priors");
        assert_eq!(idx[0], 2, "nearest centroid must rank first, got {idx:?}");
        // cap >= len -> all priors
        assert_eq!(active_prior_indices(&hidden, &priors, 9).len(), 4);
        // cap 0 / empty -> no priors
        assert!(active_prior_indices(&hidden, &[], 3).is_empty());
        assert!(active_prior_indices(&hidden, &priors, 0).is_empty());
    }
}
