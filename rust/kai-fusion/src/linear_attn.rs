//! Linear / lightning attention (MiniMax-M3 style).
//! Replaces softmax O(T^2) with recurrent O(T*d²) via feature-map cumulative sums.
//!
//! Reference: "Efficient Attention via Linear Transformers" (Katharopoulos+ 2020)
//! and MiniMax-M3 MSA (two-stage) architecture.

use ndarray::{Array1, Array2, ArrayView2, Axis, s};

// ---------------------------------------------------------------------------
// Feature function selection
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum FeatureFn {
    /// `elu(x) + 1` — standard linear attention (Katharopoulos+)
    Elu1,
    /// `relu(x)` — simpler, used in FLASH / TransNormer
    Relu,
}

impl Default for FeatureFn {
    fn default() -> Self {
        Self::Elu1
    }
}

// ---------------------------------------------------------------------------
// Cumulative state for streaming inference (replaces KV cache)
// ---------------------------------------------------------------------------
#[derive(Clone)]
#[allow(dead_code)]
pub struct LinearAttnState {
    /// Per-head cumulative S matrices: [n_kv_heads, head_dim, head_dim]
    pub s: Vec<Array2<f32>>,
    /// Per-head cumulative z vectors: [n_kv_heads, head_dim]
    pub z: Vec<Array1<f32>>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

#[allow(dead_code)]
impl LinearAttnState {
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        let s: Vec<Array2<f32>> = (0..n_kv_heads)
            .map(|_| Array2::zeros((head_dim, head_dim)))
            .collect();
        let z: Vec<Array1<f32>> = (0..n_kv_heads)
            .map(|_| Array1::zeros(head_dim))
            .collect();
        Self {
            s,
            z,
            n_kv_heads,
            head_dim,
        }
    }

    /// Reset state to zero (for new sequence).
    pub fn reset(&mut self) {
        for mat in &mut self.s {
            mat.fill(0.0);
        }
        for vec in &mut self.z {
            vec.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Feature-map function
// ---------------------------------------------------------------------------
#[allow(dead_code)]
fn feature_map(x: &Array2<f32>, fn_type: &FeatureFn) -> Array2<f32> {
    match fn_type {
        FeatureFn::Elu1 => x.mapv(|v| (if v >= 0.0 { v } else { v.exp() - 1.0 }) + 1.0),
        FeatureFn::Relu => x.mapv(|v| v.max(0.0)),
    }
}

// ---------------------------------------------------------------------------
// Linear attention forward for a single token (streaming mode with state)
// ---------------------------------------------------------------------------
#[allow(dead_code)]
pub fn linear_attn_step(
    q: &ArrayView2<f32>,   // [1, dim]
    k: &ArrayView2<f32>,   // [1, dim_kv]
    v: &ArrayView2<f32>,   // [1, dim_kv]
    state: &mut LinearAttnState,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    feature_fn: &FeatureFn,
    eps: f32,
) -> Array1<f32> {
    let dim = q.ncols();
    let group = n_heads / n_kv_heads;

    // Reshape Q/K/V into heads
    // q: [1, n_heads * head_dim] -> [1, n_heads, head_dim]
    let q_heads = q
        .to_shape((1, n_heads, head_dim))
        .unwrap()
        .to_owned();
    let k_heads = k
        .to_shape((1, n_kv_heads, head_dim))
        .unwrap()
        .to_owned();
    let v_heads = v
        .to_shape((1, n_kv_heads, head_dim))
        .unwrap()
        .to_owned();

    // Apply feature map to Q and K (per-token)
    // Feature map on each KV head
    let mut phi_k_all = Array2::<f32>::zeros((n_kv_heads, head_dim));
    for kvh in 0..n_kv_heads {
        let k_view = k_heads.slice(s![0, kvh, ..]).to_owned().insert_axis(Axis(0));
        let phi = feature_map(&k_view, feature_fn);
        phi_k_all.row_mut(kvh).assign(&phi.row(0));
    }

    let mut out = Array1::<f32>::zeros(dim);

    for head in 0..n_heads {
        let kvh = head / group;
        let hd = head_dim;

        // q feature: [1, hd] (q_h already 1D from the slice)
        let q_h = q_heads.slice(s![0, head, ..]);
        let q_view = q_h.to_owned().insert_axis(Axis(0));
        let phi_q = feature_map(&q_view, feature_fn).row(0).to_owned();

        // S: [hd, hd], z: [hd]
        let s_mat = &state.s[kvh];
        let z_vec = &state.z[kvh];

        // numerator = phi_q @ S  -> [hd]
        let mut num = Array1::<f32>::zeros(hd);
        for i in 0..hd {
            for j in 0..hd {
                num[i] += phi_q[j] * s_mat[[j, i]];
            }
        }

        // denominator = phi_q @ z  -> scalar
        let den: f32 = phi_q.iter().zip(z_vec.iter()).map(|(a, b)| a * b).sum();
        let scale = 1.0 / (den.abs() + eps);

        // o = num / (den + eps)
        let out_start = head * hd;
        for d in 0..hd {
            out[out_start + d] = num[d] * scale;
        }
    }

    // Update state with current K, V
    for kvh in 0..n_kv_heads {
        let phi_k = phi_k_all.row(kvh);
        let v_h = v_heads.slice(s![0, kvh, ..]);
        let hd = head_dim;

        // S += outer(phi_k, v_h)
        for i in 0..hd {
            for j in 0..hd {
                state.s[kvh][[i, j]] += phi_k[i] * v_h[j];
            }
        }

        // z += phi_k
        for d in 0..hd {
            state.z[kvh][d] += phi_k[d];
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Full-sequence linear attention (training / non-streaming)
// ---------------------------------------------------------------------------
#[allow(dead_code)]
pub fn linear_attn_forward(
    q: &Array2<f32>,   // [T, dim]
    k: &Array2<f32>,   // [T, dim_kv]
    v: &Array2<f32>,   // [T, dim_kv]
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    feature_fn: &FeatureFn,
    eps: f32,
) -> Array2<f32> {
    let t = q.nrows();
    let dim = q.ncols();
    let group = n_heads / n_kv_heads;

    // Split into heads: [T, n_heads, hd] and [T, n_kv_heads, hd]
    let q_heads = q.to_shape((t, n_heads, head_dim)).unwrap().to_owned();
    let k_heads = k.to_shape((t, n_kv_heads, head_dim)).unwrap().to_owned();
    let v_heads = v.to_shape((t, n_kv_heads, head_dim)).unwrap().to_owned();

        let mut out = Array2::<f32>::zeros((t, dim));

        for h in 0..n_heads {
            let kvh = h / group;
            let hd = head_dim;

            // Apply feature map to Q and K for this head
            // Q_h: [T, hd], K_h: [T, hd]
            let q_h = q_heads.slice(s![.., h, ..]);
            let k_h = k_heads.slice(s![.., kvh, ..]);
            let v_h = v_heads.slice(s![.., kvh, ..]);

            // Per-head feature maps
            let mut phi_q = Array2::<f32>::zeros((t, hd));
            let mut phi_k = Array2::<f32>::zeros((t, hd));
        for i in 0..t {
            let q_row = q_h.row(i).to_owned().insert_axis(Axis(0));
            let k_row = k_h.row(i).to_owned().insert_axis(Axis(0));
            phi_q.row_mut(i).assign(&feature_map(&q_row, feature_fn).row(0));
            phi_k.row_mut(i).assign(&feature_map(&k_row, feature_fn).row(0));
        }

            // Cumulative state
            let mut s = Array2::<f32>::zeros((hd, hd));
            let mut z = Array1::<f32>::zeros(hd);

            for i in 0..t {
                // numerator = phi_q[i] @ S  -> [hd]
                let mut num = Array1::<f32>::zeros(hd);
            for a in 0..hd {
                for b in 0..hd {
                    num[a] += phi_q[[i, b]] * s[[b, a]];
                }
            }

            // denominator = phi_q[i] @ z
            let den: f32 = (0..hd).map(|d| phi_q[[i, d]] * z[d]).sum();
            let scale = 1.0 / (den.abs() + eps);

            // Write output
            for d in 0..hd {
                out[[i, h * hd + d]] = num[d] * scale;
            }

            // Update state with position i's K, V
            for a in 0..hd {
                for b in 0..hd {
                    s[[a, b]] += phi_k[[i, a]] * v_h[[i, b]];
                }
            }
            for d in 0..hd {
                z[d] += phi_k[[i, d]];
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature_input(t: usize, d: usize, seed: u64) -> Array2<f32> {
        let mut s = seed;
        let mut a = Array2::zeros((t, d));
        for i in 0..t {
            for j in 0..d {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                a[[i, j]] = ((s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
            }
        }
        a
    }

    #[test]
    fn linear_attn_full_sequence_produces_valid_output() {
        let t = 8;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 4;
        let dim = n_heads * head_dim;
        let dim_kv = n_kv_heads * head_dim;

        let q = make_feature_input(t, dim, 42);
        let k = make_feature_input(t, dim_kv, 99);
        let v = make_feature_input(t, dim_kv, 77);

        let out = linear_attn_forward(
            &q, &k, &v,
            n_heads, n_kv_heads, head_dim,
            &FeatureFn::Elu1,
            1e-6,
        );

        assert_eq!(out.shape(), &[t, dim], "linear attn output shape mismatch");
        for &val in out.iter() {
            assert!(val.is_finite(), "linear attn output contains non-finite: {val}");
        }
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 1e-6, "linear attn output is all zeros (max_abs={max_abs})");
    }

    #[test]
    fn linear_attn_step_matches_full_forward() {
        let t = 6;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let dim = n_heads * head_dim;
        let dim_kv = n_kv_heads * head_dim;
        let eps = 1e-6;

        let q = make_feature_input(t, dim, 42);
        let k = make_feature_input(t, dim_kv, 99);
        let v = make_feature_input(t, dim_kv, 77);

        let full_out = linear_attn_forward(
            &q, &k, &v,
            n_heads, n_kv_heads, head_dim,
            &FeatureFn::Elu1,
            eps,
        );

        // Step through one token at a time
        let mut state = LinearAttnState::new(n_kv_heads, head_dim);
        let mut step_out = Array2::zeros((t, dim));
        for i in 0..t {
            let q_i = q.slice(s![i..i + 1, ..]);
            let k_i = k.slice(s![i..i + 1, ..]);
            let v_i = v.slice(s![i..i + 1, ..]);
            let o_i = linear_attn_step(
                &q_i, &k_i, &v_i,
                &mut state,
                n_heads, n_kv_heads, head_dim,
                &FeatureFn::Elu1,
                eps,
            );
            step_out.row_mut(i).assign(&o_i);
        }

        // Step-by-step should match full forward
        for i in 0..t {
            for d in 0..dim {
                let diff = (full_out[[i, d]] - step_out[[i, d]]).abs();
                assert!(
                    diff < 1e-4,
                    "step vs full mismatch at [{i}, {d}]: full={} step={}",
                    full_out[[i, d]], step_out[[i, d]]
                );
            }
        }
    }

    #[test]
    fn linear_attn_relu_feature_fn() {
        let t = 4;
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let dim = n_heads * head_dim;
        let dim_kv = n_kv_heads * head_dim;

        // Use positive inputs so ReLU doesn't kill everything
        let mut q = make_feature_input(t, dim, 55);
        let mut k = make_feature_input(t, dim_kv, 66);
        let v = make_feature_input(t, dim_kv, 44);
        q.mapv_inplace(|x| x.abs() + 0.1);
        k.mapv_inplace(|x| x.abs() + 0.1);

        let out = linear_attn_forward(
            &q, &k, &v,
            n_heads, n_kv_heads, head_dim,
            &FeatureFn::Relu,
            1e-6,
        );

        assert_eq!(out.shape(), &[t, dim]);
        for &val in out.iter() {
            assert!(val.is_finite(), "relu linear attn has non-finite: {val}");
        }
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 1e-6, "relu linear attn is all zeros (max_abs={max_abs})");
    }
}
