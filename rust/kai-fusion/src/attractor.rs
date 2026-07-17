//! Bridge between Kai's Phase-A attractor (the 10 ingested architectures) and
//! the live decoder. The attractor is a set of 768-dim vectors; we compress it
//! to a centroid and project that centroid into the model's hidden dimension to
//! form a *prior* the decoder is seeded with. Novelty in Kai's VFE is then
//! measured as distance of a context's hidden state from that assimilated
//! prior — i.e. how far new input is from everything Kai has already absorbed.

use ndarray::Array1;

/// Load the Phase-A attractor vectors (each 768-dim) from a JSON `{vecs:[...]}`.
pub fn load(path: &str) -> Result<Vec<Vec<f32>>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
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
