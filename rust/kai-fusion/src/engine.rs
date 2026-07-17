//! Core math kernels (assimilated patterns: RMSNorm, RoPE, SwiGLU, attention).
//! Pure `ndarray` f32 — consistent with kai-core / physics-dialect.

use ndarray::{Array1, Array2};

pub const EPS: f32 = 1e-5;

/// RMSNorm over the last dim, then affine scale `w`.
pub fn rmsnorm_rows(x: &Array2<f32>, w: &Array1<f32>, eps: f32) -> Array2<f32> {
    let mut out = Array2::zeros(x.raw_dim());
    for i in 0..x.nrows() {
        let row = x.row(i);
        let mean_sq: f32 = row.mapv(|v| v * v).sum() / row.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for j in 0..x.ncols() {
            out[[i, j]] = x[[i, j]] * inv * w[j];
        }
    }
    out
}

/// Linear: w is stored (out, in); y = x · wᵀ.
/// y = x @ W^T where W is stored in (output, input) format.
pub fn linear(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

pub fn silu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

pub fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

pub fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for x in v.iter_mut() {
        *x = (*x - max).exp();
    }
    let s: f32 = v.iter().sum();
    for x in v.iter_mut() {
        *x /= s;
    }
}

pub fn variance(v: &[f32]) -> f32 {
    let m: f32 = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
}

/// RoPE on one head slice (rotate consecutive pairs by position angle).
pub fn rope_slice(s: &mut [f32], pos: usize, theta: f32) {
    let hd = s.len();
    let half = hd / 2;
    for i in 0..half {
        let angle = pos as f32 / theta.powf(2.0 * i as f32 / hd as f32);
        let (c, sn) = (angle.cos(), angle.sin());
        let a = s[2 * i];
        let b = s[2 * i + 1];
        s[2 * i] = a * c - b * sn;
        s[2 * i + 1] = a * sn + b * c;
    }
}

/// KL divergence: KL(student || teacher) = sum(p * log(p/q))
pub fn kl_divergence(student_logits: &[f32], teacher_logits: &[f32]) -> f32 {
    let max_s = student_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_t = teacher_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let sum_s: f32 = student_logits.iter().map(|&x| (x - max_s).exp()).sum();
    let sum_t: f32 = teacher_logits.iter().map(|&x| (x - max_t).exp()).sum();
    
    let mut kl = 0.0;
    for i in 0..student_logits.len() {
        let p = (student_logits[i] - max_s).exp() / sum_s;
        let q = (teacher_logits[i] - max_t).exp() / sum_t;
        if p > 1e-9 && q > 1e-9 {
            kl += p * (p / q).ln();
        }
    }
    kl
}

pub fn apply_rope_all(q: &mut Array2<f32>, n_heads: usize, head_dim: usize, theta: f32) {
    let t = q.nrows();
    for i in 0..t {
        for head in 0..n_heads {
            let base = head * head_dim;
            let mut buf: Vec<f32> = (0..head_dim).take(head_dim).map(|d| q[[i, base + d]]).collect();
            rope_slice(&mut buf, i, theta);
            for d in 0..head_dim {
                q[[i, base + d]] = buf[d];
            }
        }
    }
}

/// Apply RoPE to a single-row tensor (one token) at an explicit absolute
/// `pos`. Used by incremental (KV-cached) generation where the token is
/// always row 0 but must be rotated by its true sequence position.
pub fn apply_rope_at(q: &mut Array2<f32>, n_heads: usize, head_dim: usize, theta: f32, pos: usize) {
    for head in 0..n_heads {
        let base = head * head_dim;
        let mut buf: Vec<f32> = (0..head_dim).take(head_dim).map(|d| q[[0, base + d]]).collect();
        rope_slice(&mut buf, pos, theta);
        for d in 0..head_dim {
            q[[0, base + d]] = buf[d];
        }
    }
}
