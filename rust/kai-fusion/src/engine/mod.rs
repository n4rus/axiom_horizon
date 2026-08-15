//! Core math kernels (assimilated patterns: RMSNorm, RoPE, SwiGLU, attention).
//! Pure `ndarray` f32 by default — consistent with kai-core / physics-dialect.
//! With the `gpu` feature the heavy matmuls (linear/rmsnorm/silu) run on CUDA via
//! candle, so Kai reuses the GGUF weight library on the local GPU instead of
//! slow single-threaded f32 on CPU. The public `Array2<f32>` boundary is kept so
//! callers (model.rs, generate_streaming) are unchanged.
//!
//! ## GPU weight cache
//! With the `gpu` feature, `linear()` caches uploaded weight tensors in GPU VRAM
//! keyed by the weight Array2's memory address. Weights are immutable between
//! gradient updates, so re-uploading every call wastes 100x bandwidth.
//! Call `invalidate_weight_cache()` after `apply_gradients` to drop stale entries.

use ndarray::{Array1, Array2};

pub const EPS: f32 = 1e-5;

#[cfg(feature = "gpu")]
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use std::sync::Mutex;

#[cfg(feature = "gpu")]
thread_local! {
    /// GPU weight tensor cache: (addr, rows, cols) → cached transposed Tensor on CUDA.
    /// Key includes shape to avoid collisions between different weights that share
    /// the same base pointer (e.g. from memory-mapped GGUF loading).
    /// Cleared when weights are modified (apply_gradients / assimilate_step).
    static WEIGHT_CACHE: Mutex<HashMap<(usize, usize, usize), candle_core::Tensor>> = Mutex::new(HashMap::new());
}

/// Invalidate the GPU weight cache after modifying model weights.
#[cfg(feature = "gpu")]
pub fn invalidate_weight_cache() {
    WEIGHT_CACHE.with(|c| c.lock().unwrap().clear());
}

/// Select the compute device. With `gpu`: CUDA if available else CPU. Without the
/// feature this is a no-op (CPU ndarray path is used directly).
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn device() -> GpuDevice {
    GpuDevice::select()
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use candle_core::{Device, Tensor};
    use ndarray::Array2;
    use std::sync::OnceLock;

    static DEVICE: OnceLock<Device> = OnceLock::new();

    pub fn select() -> &'static Device {
        DEVICE.get_or_init(|| Device::cuda_if_available(0).unwrap_or(Device::Cpu))
    }

    /// f32 Array2 -> CUDA/CPU Tensor.
    pub fn to_tensor(a: &Array2<f32>) -> Tensor {
        let dev = select();
        let shape = [a.nrows(), a.ncols()];
        // Force standard layout (row-major contiguous) for CUDA transfer
        let standard = a.as_standard_layout();
        Tensor::from_slice(standard.as_slice().unwrap(), &shape, &dev)
            .expect("tensor from ndarray")
    }

    /// Tensor -> f32 Array2.
    pub fn from_tensor(t: &Tensor) -> Array2<f32> {
        let v: Vec<Vec<f32>> = t.to_vec2().expect("tensor to vec2");
        let rows = v.len();
        let cols = if rows > 0 { v[0].len() } else { 0 };
        let flat: Vec<f32> = v.into_iter().flatten().collect();
        Array2::from_shape_vec((rows, cols), flat).expect("reshape")
    }

    /// RMSNorm using basic ops (works on CUDA).
    #[allow(dead_code)]
    pub fn gpu_rmsnorm(x: &Tensor, w: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
        let x2 = x.sqr()?;
        let sum = x2.sum_keepdim(1)?;
        let ncols = x.dims()[x.dims().len() - 1] as f64;
        let mean = sum / ncols;
        let mean = mean?;
        // Add scalar: Tensor + f64 returns Result<Tensor>
        let mean_eps = (mean + eps as f64)?;
        let rms = mean_eps.sqrt()?;
        let divided = x.broadcast_div(&rms)?;
        divided.broadcast_mul(w)
    }
}

/// Opaque device handle (CUDA when `gpu` is on, otherwise unit).
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub enum GpuDevice {
    Cuda(&'static candle_core::Device),
    Cpu,
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
impl GpuDevice {
    pub fn select() -> Self {
        GpuDevice::Cuda(gpu::select())
    }
    pub fn is_gpu(&self) -> bool {
        matches!(self, GpuDevice::Cuda(candle_core::Device::Cuda(_)))
    }
}


/// RMSNorm over the last dim, then affine scale `w`.
/// Caches the `w` tensor on GPU by address+len (like linear).
pub fn rmsnorm_rows(x: &Array2<f32>, w: &Array1<f32>, eps: f32) -> Array2<f32> {
    #[cfg(feature = "gpu")]
    {
        use candle_core::Tensor;
        let dev = gpu::select();
        let xt = gpu::to_tensor(x);
        let wt = WEIGHT_CACHE.with(|cache| {
            let mut cache = cache.lock().unwrap();
            let addr = w.as_ptr() as usize;
            let len = w.len();
            let key = (addr, len, 0);
            if let Some(t) = cache.get(&key) {
                t.clone()
            } else {
                let wshape = [len];
                let t = Tensor::from_slice(w.as_slice().unwrap(), &wshape, dev)
                    .expect("rmsnorm weight tensor");
                cache.insert(key, t.clone());
                t
            }
        });
        // Manual RMSNorm using basic CUDA-compatible ops
        // rms = sqrt(mean(x^2) + eps), y = x / rms * w
        let ncols_f = x.ncols() as f64;
        let x2 = xt.sqr().unwrap();
        let sum = x2.sum_keepdim(1).unwrap();
        let mean = (sum / ncols_f).unwrap();  // Tensor / f64
        let mean_eps = (mean + eps as f64).unwrap();  // Tensor + f64
        let rms = mean_eps.sqrt().unwrap();
        let divided = xt.broadcast_div(&rms).unwrap();
        // wt is [dim] — reshape to [1, dim] for broadcast mul
        let wt2d = wt.reshape((1, w.len())).unwrap();
        let rn = divided.broadcast_mul(&wt2d).unwrap();
        return gpu::from_tensor(&rn);
    }
    #[cfg(not(feature = "gpu"))]
    {
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
}

/// RMSNorm for a single row (1D array).
pub fn rmsnorm_row(x: &Array1<f32>, w: &Array1<f32>, eps: f32) -> Array1<f32> {
    let mean_sq: f32 = x.mapv(|v| v * v).sum() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x * inv * w
}

/// Fused FFN block on GPU: rmsnorm → silu(x·w1ᵀ) * (x·w3ᵀ) → ·w2ᵀ.
/// Single upload, all compute on GPU, single download. ~6x fewer roundtrips.
#[cfg(feature = "gpu")]
pub fn fused_ffn(
    x: &Array2<f32>, norm_w: &[f32],
    w1: &Array2<f32>, w3: &Array2<f32>, w2: &Array2<f32>,
    eps: f32,
) -> Array2<f32> {
    use candle_core::Tensor;
    let dev = gpu::select();
    let xt = gpu::to_tensor(x);
    let w1_t = gpu::to_tensor(w1).t().unwrap();
    let w3_t = gpu::to_tensor(w3).t().unwrap();
    let w2_t = gpu::to_tensor(w2).t().unwrap();

    // rmsnorm on GPU (using basic ops that work on CUDA)
    let h = if norm_w.is_empty() {
        xt
    } else {
        let norm_t = Tensor::from_slice(norm_w, &[norm_w.len()], dev).unwrap();
        gpu::gpu_rmsnorm(&xt, &norm_t, eps).unwrap()
    };

    // gate = silu(h @ w1^T), up = h @ w3^T, out = (gate * up) @ w2^T
    let gate = h.matmul(&w1_t).unwrap().silu().unwrap();
    let up = h.matmul(&w3_t).unwrap();
    let out = (gate * up).unwrap().matmul(&w2_t).unwrap();

    gpu::from_tensor(&out)
}

/// Fused FFN forward that keeps intermediates on GPU for backward.
/// Returns (out_cpu, h_ffn_gpu, gate_gpu, gate_silu_gpu, up_gpu).
#[cfg(feature = "gpu")]
pub fn fused_ffn_with_cache(
    x: &Array2<f32>, norm_w: &[f32],
    w1: &Array2<f32>, w3: &Array2<f32>, w2: &Array2<f32>,
    eps: f32,
) -> (Array2<f32>, candle_core::Tensor, candle_core::Tensor, candle_core::Tensor, candle_core::Tensor) {
    use candle_core::Tensor;
    let dev = gpu::select();
    let xt = gpu::to_tensor(x);
    let w1_t = gpu::to_tensor(w1).t().unwrap();
    let w3_t = gpu::to_tensor(w3).t().unwrap();
    let w2_t = gpu::to_tensor(w2).t().unwrap();

    // rmsnorm on GPU (using basic ops that work on CUDA)
    let h = if norm_w.is_empty() {
        xt
    } else {
        let norm_t = Tensor::from_slice(norm_w, &[norm_w.len()], dev).unwrap();
        gpu::gpu_rmsnorm(&xt, &norm_t, eps).unwrap()
    };

    // gate = h @ w1^T (pre-silu), gate_silu = silu(gate), up = h @ w3^T
    let gate_t = h.matmul(&w1_t).unwrap();
    let gate_silu_t = gate_t.silu().unwrap();
    let up_t = h.matmul(&w3_t).unwrap();

    // out = (gate_silu * up) @ w2^T
    let up_clone = up_t.clone();
    let out = (gate_silu_t.clone() * up_clone).unwrap().matmul(&w2_t).unwrap();
    let out_cpu = gpu::from_tensor(&out);

    (out_cpu, h, gate_t, gate_silu_t, up_t)
}

/// Fused attention block on GPU: rmsnorm → QKV proj → RoPE → causal attn → wo.
/// Single upload, all compute on GPU, single download.
#[cfg(feature = "gpu")]
pub fn fused_attention(
    x: &Array2<f32>, norm_w: &[f32],
    wq: &Array2<f32>, wk: &Array2<f32>, wv: &Array2<f32>, wo: &Array2<f32>,
    n_heads: usize, n_kv_heads: usize, head_dim: usize, rope_theta: f32, eps: f32,
) -> Array2<f32> {
    let (proj, _, _, _, _, _) = fused_attention_with_cache(x, norm_w, wq, wk, wv, wo,
        n_heads, n_kv_heads, head_dim, rope_theta, eps);
    proj
}

/// Fused attention forward that keeps intermediates on GPU for backward.
/// Returns (attn_proj_cpu, h_attn_gpu, q_rope_gpu, k_rope_gpu, v_gpu, attn_out_gpu).
/// The _gpu tensors are kept on device to avoid re-upload in backward.
#[cfg(feature = "gpu")]
pub fn fused_attention_with_cache(
    x: &Array2<f32>, norm_w: &[f32],
    wq: &Array2<f32>, wk: &Array2<f32>, wv: &Array2<f32>, wo: &Array2<f32>,
    n_heads: usize, n_kv_heads: usize, head_dim: usize, rope_theta: f32, eps: f32,
) -> (Array2<f32>, candle_core::Tensor, candle_core::Tensor, candle_core::Tensor, candle_core::Tensor, candle_core::Tensor) {
    use candle_core::Tensor;
    let dev = gpu::select();
    let t = x.nrows();

    let xt = gpu::to_tensor(x);

    // rmsnorm on GPU (using basic ops that work on CUDA)
    let h = if norm_w.is_empty() {
        xt
    } else {
        let norm_t = Tensor::from_slice(norm_w, &[norm_w.len()], dev).unwrap();
        gpu::gpu_rmsnorm(&xt, &norm_t, eps).unwrap()
    };

    // QKV projections on GPU
    let wq_t = gpu::to_tensor(wq).t().unwrap();
    let wk_t = gpu::to_tensor(wk).t().unwrap();
    let wv_t = gpu::to_tensor(wv).t().unwrap();
    let q = h.matmul(&wq_t).unwrap(); // [T, n_heads*hd]
    let k = h.matmul(&wk_t).unwrap(); // [T, n_kv_heads*hd]
    let v_t = h.matmul(&wv_t).unwrap(); // [T, n_kv_heads*hd]

    // RoPE on GPU
    let q = rope_tensor(&q, n_heads, head_dim, rope_theta, dev);
    let k = rope_tensor(&k, n_kv_heads, head_dim, rope_theta, dev);

    // Batched multi-head causal attention on GPU (all heads in one call)
    let group = n_heads / n_kv_heads.max(1);
    let q_3d = q.reshape((n_heads, t, head_dim)).unwrap();
    let k_3d = k.reshape((n_kv_heads, t, head_dim)).unwrap();
    let v_3d = v_t.reshape((n_kv_heads, t, head_dim)).unwrap();
    // GQA: expand K,V from [n_kv_heads, T, hd] → [n_heads, T, hd]
    let k_3d_exp = if group > 1 {
        k_3d.unsqueeze(1).unwrap()
           .expand(&[n_kv_heads, group, t, head_dim]).unwrap()
           .reshape((n_heads, t, head_dim)).unwrap()
    } else { k_3d };
    let v_3d_exp = if group > 1 {
        v_3d.unsqueeze(1).unwrap()
           .expand(&[n_kv_heads, group, t, head_dim]).unwrap()
           .reshape((n_heads, t, head_dim)).unwrap()
    } else { v_3d };
    // Batched scores: Q [n_heads, T, hd] @ K^T [n_heads, hd, T] → [n_heads, T, T]
    let scale = (head_dim as f64).sqrt();
    let k_t = k_3d_exp.transpose(1, 2).unwrap();
    let scores = (q_3d.matmul(&k_t).unwrap() * scale).unwrap();
    // Causal mask
    let mut mask = Array2::zeros((t, t));
    for i in 0..t { for j in (i + 1)..t { mask[[i, j]] = f32::NEG_INFINITY; } }
    let mask_t = gpu::to_tensor(&mask).unsqueeze(0).unwrap()
        .expand(&[n_heads, t, t]).unwrap();
    let scores_masked = (scores + mask_t).unwrap();
    let probs = candle_nn::ops::softmax(&scores_masked, 2).unwrap();
    let attn_3d = probs.matmul(&v_3d_exp).unwrap();
    let attn_out = attn_3d.reshape((t, n_heads * head_dim)).unwrap();

    // Output projection on GPU
    let wo_t = gpu::to_tensor(wo).t().unwrap();
    let attn_proj = attn_out.matmul(&wo_t).unwrap();
    let attn_proj_cpu = gpu::from_tensor(&attn_proj);

    // Return GPU tensors (for backward) + CPU attn_proj (for residual add)
    (attn_proj_cpu, h, q, k, v_t, attn_out)
}

/// Apply RoPE to a tensor on GPU: (T, dim) → (T, dim) with rotary embeddings.
#[cfg(feature = "gpu")]
pub fn rope_tensor(t: &candle_core::Tensor, n_heads: usize, head_dim: usize, theta: f32, dev: &candle_core::Device) -> candle_core::Tensor {
    // Flatten 2D (T, n_heads*head_dim) to 1D for rope manipulation
    let t_flat = t.flatten_all().unwrap();
    let t_vec: Vec<f32> = t_flat.to_vec1().unwrap();
    let n = t_vec.len() / (n_heads * head_dim);
    let mut out = t_vec.clone();
    for i in 0..n {
        for head in 0..n_heads {
            let base = i * n_heads * head_dim + head * head_dim;
            let mut buf: Vec<f32> = (0..head_dim).map(|d| t_vec[base + d]).collect();
            crate::engine::rope_slice(&mut buf, i, theta);
            for d in 0..head_dim {
                out[base + d] = buf[d];
            }
        }
    }
    candle_core::Tensor::from_slice(&out, (n, n_heads * head_dim), dev).unwrap()
}


/// Linear: w is stored (out, in); y = x · wᵀ.
/// y = x @ W^T where W is stored in (output, input) format.
/// Uploads `w` to GPU once and caches it by address — subsequent calls
/// with the same weight matrix reuse the cached GPU tensor (~100x faster).
pub fn linear(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    #[cfg(feature = "gpu")]
    {
        linear_impl(x, w, true)  // cached
    }
    #[cfg(not(feature = "gpu"))]
    x.dot(&w.t())
}

/// Same as `linear`, but uses a fresh upload every call (no GPU weight cache).
/// Slower (PCIe transfer each time) but uses less VRAM — useful for large models.
#[allow(dead_code)]
pub fn linear_nocache(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    #[cfg(feature = "gpu")]
    {
        linear_impl(x, w, false)
    }
    #[cfg(not(feature = "gpu"))]
    x.dot(&w.t())
}

#[cfg(feature = "gpu")]
fn linear_impl(x: &Array2<f32>, w: &Array2<f32>, use_cache: bool) -> Array2<f32> {
    let xt = gpu::to_tensor(x);
    let wt = if use_cache {
        WEIGHT_CACHE.with(|cache| {
            let mut cache = cache.lock().unwrap();
            let addr = w.as_ptr() as usize;
            let rows = w.nrows();
            let cols = w.ncols();
            let key = (addr, rows, cols);
            if let Some(t) = cache.get(&key) {
                t.clone()
            } else {
                let shape = [rows, cols];
                let standard = w.as_standard_layout();
                let t = candle_core::Tensor::from_slice(
                    standard.as_slice().unwrap(), &shape, gpu::select(),
                ).expect("weight tensor");
                let tt = t.t().expect("transpose");
                cache.insert(key, tt.clone());
                tt
            }
        })
    } else {
        let shape = [w.nrows(), w.ncols()];
        let standard = w.as_standard_layout();
        let t = candle_core::Tensor::from_slice(
            standard.as_slice().unwrap(), &shape, gpu::select(),
        ).expect("weight tensor");
        t.t().expect("transpose")
    };
    let yt = xt.matmul(&wt).expect("gpu matmul");
    gpu::from_tensor(&yt)
}

pub fn silu(x: &Array2<f32>) -> Array2<f32> {
    #[cfg(feature = "gpu")]
    {
        let t = gpu::to_tensor(x);
        let s = t.silu().expect("gpu silu");
        return gpu::from_tensor(&s);
    }
    #[cfg(not(feature = "gpu"))]
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

/// Compute a.T @ b (weight gradient) on GPU if available, else CPU ndarray.
/// Generic: a is [T, da], b is [T, db] → result is [da, db].
pub fn grad_weight(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    #[cfg(feature = "gpu")]
    {
        let a_t = gpu::to_tensor(&a.t().to_owned());
        let b_t = gpu::to_tensor(b);
        let r = a_t.matmul(&b_t).expect("gpu grad_weight");
        gpu::from_tensor(&r)
    }
    #[cfg(not(feature = "gpu"))]
    a.t().dot(b)
}

/// Compute a @ b (upstream gradient / forward matmul) on GPU if available.
/// a is [M, K], b is [K, N] → result is [M, N].
pub fn gpu_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    #[cfg(feature = "gpu")]
    {
        let a_t = gpu::to_tensor(a);
        let b_t = gpu::to_tensor(b);
        let r = a_t.matmul(&b_t).expect("gpu matmul");
        gpu::from_tensor(&r)
    }
    #[cfg(not(feature = "gpu"))]
    a.dot(b)
}

/// Per-head GPU attention backward: compute dQ, dK, dV for one head.
/// All tensors are [T, head_dim] CPU arrays.
/// Returns (d_q, d_k, d_v) as CPU arrays [T, head_dim].
/// On CPU (no gpu feature), falls back to the full nested-loop computation.
#[cfg(feature = "gpu")]
pub fn gpu_head_attn_backward(
    q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, d_o: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    use candle_core::Tensor;
    let dev = gpu::select();
    let t = q.nrows();
    let head_dim = q.ncols();
    let scale = (head_dim as f64).sqrt() as f64;

    // Upload
    let q_t = gpu::to_tensor(q);
    let k_t = gpu::to_tensor(k);
    let v_t = gpu::to_tensor(v);
    let do_t = gpu::to_tensor(d_o);

    // Causal mask
    let mut mask = vec![0.0f32; t * t];
    for i in 0..t { for j in (i + 1)..t { mask[i * t + j] = f32::NEG_INFINITY; } }
    let mask_t = Tensor::from_slice(&mask, &[t, t], dev).unwrap();

    // Forward: scores = Q @ K^T / sqrt(hd) + mask → P = softmax
    let scores = (q_t.matmul(&k_t.t().unwrap()).unwrap() * scale).unwrap();
    let s_masked = (scores + &mask_t).unwrap();
    let p = candle_nn::ops::softmax(&s_masked, 1).unwrap();

    // dV = P^T @ dO  [T, hd]
    let dv = p.t().unwrap().matmul(&do_t).unwrap();

    // dP = dO @ V^T  [T, T]
    let dp = do_t.matmul(&v_t.t().unwrap()).unwrap();

    // Softmax backward: dS = P * (dP - sum(P * dP, dim=-1, keepdim))
    // Note: candle doesn't auto-broadcast [T,1] vs [T,T], so expand explicitly
    let p_clone = p.clone();
    let p_dp = (p * &dp).unwrap();
    let sum_p_dp = p_dp.sum_keepdim(1).unwrap().expand(&[t, t]).unwrap();
    let ds = (p_clone * (dp - &sum_p_dp).unwrap()).unwrap();

    // dQ = dS @ K / sqrt(hd)  [T, hd]
    let dq = (ds.matmul(&k_t).unwrap() * scale).unwrap();
    // dK = dS^T @ Q / sqrt(hd)  [T, hd]
    let dk = (ds.t().unwrap().matmul(&q_t).unwrap() * scale).unwrap();

    (gpu::from_tensor(&dq), gpu::from_tensor(&dk), gpu::from_tensor(&dv))
}

/// Batched GPU attention backward — ALL heads in ONE GPU call.
/// Accepts full [T, n_heads*hd], [T, n_kv_heads*hd] CPU arrays, runs
/// batched matmuls on GPU with zero per-head data copies.
/// Returns (dq, dk, dv) as CPU arrays.
#[cfg(feature = "gpu")]
pub fn gpu_attn_backward_batched(
    q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, d_o: &Array2<f32>,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    use candle_core::Tensor;
    let dev = gpu::select();
    let t = q.nrows();
    let scale = 1.0 / (head_dim as f64).sqrt();
    let group = n_heads / n_kv_heads;

    // Upload full tensors (single upload, no per-head copies)
    let q_t = gpu::to_tensor(q);   // [T, n_heads*hd]
    let k_t = gpu::to_tensor(k);   // [T, n_kv_heads*hd]
    let v_t = gpu::to_tensor(v);   // [T, n_kv_heads*hd]
    let do_t = gpu::to_tensor(d_o);

    // Reshape to 3D: [n_heads, T, hd] and [n_kv_heads, T, hd]
    let q_3d = q_t.reshape((n_heads, t, head_dim)).unwrap();
    let k_3d = k_t.reshape((n_kv_heads, t, head_dim)).unwrap();
    let v_3d = v_t.reshape((n_kv_heads, t, head_dim)).unwrap();
    let d_o_3d = do_t.reshape((n_heads, t, head_dim)).unwrap();

    // GQA: expand K,V from [n_kv_heads, T, hd] → [n_heads, T, hd]
    // Using unsqueeze + expand + reshape (no repeat_interleave in candle 0.8)
    let k_exp = if group > 1 {
        k_3d.unsqueeze(1).unwrap()
           .expand(&[n_kv_heads, group, t, head_dim]).unwrap()
           .reshape((n_heads, t, head_dim)).unwrap()
    } else { k_3d };
    let v_exp = if group > 1 {
        v_3d.unsqueeze(1).unwrap()
           .expand(&[n_kv_heads, group, t, head_dim]).unwrap()
           .reshape((n_heads, t, head_dim)).unwrap()
    } else { v_3d };

    // Causal mask: [T, T]
    let mut mask = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t { mask[i * t + j] = f32::NEG_INFINITY; }
    }
    let mask_t = Tensor::from_slice(&mask, &[t, t], dev).unwrap();

    // Batched scores: [n_heads, T, T] = Q @ K^T
    let k_t_3d = k_exp.transpose(1, 2).unwrap(); // [n_heads, hd, T]
    let scores = q_3d.matmul(&k_t_3d).unwrap();  // [n_heads, T, T]
    let scores_scaled = (scores * scale).unwrap();
    let mask_3d = mask_t.unsqueeze(0).unwrap()  // [1, T, T]
        .expand(&[n_heads, t, t]).unwrap();  // [n_heads, T, T]
    let s_masked = (scores_scaled + mask_3d).unwrap();
    let p = candle_nn::ops::softmax(&s_masked, 2).unwrap(); // [n_heads, T, T]

    // dV = P^T @ dO: [n_heads, T, hd]
    let p_t = p.transpose(1, 2).unwrap();
    let dv_3d = p_t.matmul(&d_o_3d).unwrap();

    // dP = dO @ V^T: [n_heads, T, T]
    let v_t_3d = v_exp.transpose(1, 2).unwrap();
    let dp = d_o_3d.matmul(&v_t_3d).unwrap();

    // Softmax backward: dS = P * (dP - sum(P*dP, keepdim))
    let p_clone = p.clone();
    let p_dp = (p_clone * &dp).unwrap();
    let sum_p_dp = p_dp.sum_keepdim(2).unwrap(); // [n_heads, T, 1]
    // Candle doesn't broadcast — expand explicitly
    let sum_p_dp_exp = sum_p_dp.expand(&[n_heads, t, t]).unwrap();
    let ds = (p * (&dp - sum_p_dp_exp).unwrap()).unwrap();

    // dQ = dS @ K: [n_heads, T, hd]
    let dq_3d = (ds.matmul(&k_exp).unwrap() * scale).unwrap();

    // dK = dS^T @ Q: [n_heads, T, hd]
    let ds_t = ds.transpose(1, 2).unwrap();
    let dk_3d = (ds_t.matmul(&q_3d).unwrap() * scale).unwrap();

    // GQA reduction: sum dK,dV over grouped heads
    let (dk_out, dv_out) = if group > 1 {
        let dk_4d = dk_3d.reshape((n_kv_heads, group, t, head_dim)).unwrap();
        let dv_4d = dv_3d.reshape((n_kv_heads, group, t, head_dim)).unwrap();
        (dk_4d.sum(1).unwrap(), dv_4d.sum(1).unwrap())
    } else { (dk_3d, dv_3d) };

    // Reshape back to [T, n_heads*hd] / [T, n_kv_heads*hd]
    let dq = dq_3d.reshape((n_heads * t, head_dim)).unwrap()
        .reshape((t, n_heads * head_dim)).unwrap();
    let dk = dk_out.reshape((n_kv_heads * t, head_dim)).unwrap()
        .reshape((t, n_kv_heads * head_dim)).unwrap();
    let dv = dv_out.reshape((n_kv_heads * t, head_dim)).unwrap()
        .reshape((t, n_kv_heads * head_dim)).unwrap();

    (gpu::from_tensor(&dq), gpu::from_tensor(&dk), gpu::from_tensor(&dv))
}

/// Batched GPU attention backward using GPU tensors for Q,K,V (no CPU re-upload).
/// Only d_attn_proj (upstream gradient, CPU-computed) is uploaded.
/// Returns (dq, dk, dv) as CPU arrays.
#[cfg(feature = "gpu")]
pub fn gpu_attn_backward_batched_from_cache(
    q_gpu: &candle_core::Tensor, k_gpu: &candle_core::Tensor, v_gpu: &candle_core::Tensor,
    d_attn_proj: &Array2<f32>,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    use candle_core::Tensor;
    let dev = gpu::select();
    let t = q_gpu.dims()[0];
    let scale = 1.0 / (head_dim as f64).sqrt();
    let group = n_heads / n_kv_heads;

    // Upload d_attn_proj (computed on CPU as part of layer backward chain).
    let do_t = gpu::to_tensor(d_attn_proj);

    // Reshape to 3D: Q already on GPU in cache; K,V on GPU in cache.
    let q_3d = q_gpu.reshape((n_heads, t, head_dim)).unwrap();
    let k_3d = k_gpu.reshape((n_kv_heads, t, head_dim)).unwrap();
    let v_3d = v_gpu.reshape((n_kv_heads, t, head_dim)).unwrap();
    let d_o_3d = do_t.reshape((n_heads, t, head_dim)).unwrap();

    // GQA: expand K,V from [n_kv_heads, T, hd] → [n_heads, T, hd]
    let k_exp = if group > 1 {
        k_3d.unsqueeze(1).unwrap()
           .expand(&[n_kv_heads, group, t, head_dim]).unwrap()
           .reshape((n_heads, t, head_dim)).unwrap()
    } else { k_3d };
    let v_exp = if group > 1 {
        v_3d.unsqueeze(1).unwrap()
           .expand(&[n_kv_heads, group, t, head_dim]).unwrap()
           .reshape((n_heads, t, head_dim)).unwrap()
    } else { v_3d };

    // Causal mask: [T, T]
    let mut mask = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t { mask[i * t + j] = f32::NEG_INFINITY; }
    }
    let mask_t = Tensor::from_slice(&mask, &[t, t], dev).unwrap();

    // Batched scores: [n_heads, T, T] = Q @ K^T
    let k_t_3d = k_exp.transpose(1, 2).unwrap();
    let scores = q_3d.matmul(&k_t_3d).unwrap();
    let scores_scaled = (scores * scale).unwrap();
    let mask_3d = mask_t.unsqueeze(0).unwrap()
        .expand(&[n_heads, t, t]).unwrap();
    let s_masked = (scores_scaled + mask_3d).unwrap();
    let p = candle_nn::ops::softmax(&s_masked, 2).unwrap();

    // dV = P^T @ dO
    let p_t = p.transpose(1, 2).unwrap();
    let dv_3d = p_t.matmul(&d_o_3d).unwrap();

    // dP = dO @ V^T
    let v_t_3d = v_exp.transpose(1, 2).unwrap();
    let dp = d_o_3d.matmul(&v_t_3d).unwrap();

    // Softmax backward
    let p_clone = p.clone();
    let p_dp = (p_clone * &dp).unwrap();
    let sum_p_dp = p_dp.sum_keepdim(2).unwrap();
    let sum_p_dp_exp = sum_p_dp.expand(&[n_heads, t, t]).unwrap();
    let ds = (p * (&dp - sum_p_dp_exp).unwrap()).unwrap();

    // dQ = dS @ K
    let dq_3d = (ds.matmul(&k_exp).unwrap() * scale).unwrap();

    // dK = dS^T @ Q
    let ds_t = ds.transpose(1, 2).unwrap();
    let dk_3d = (ds_t.matmul(&q_3d).unwrap() * scale).unwrap();

    // GQA reduction
    let (dk_out, dv_out) = if group > 1 {
        let dk_4d = dk_3d.reshape((n_kv_heads, group, t, head_dim)).unwrap();
        let dv_4d = dv_3d.reshape((n_kv_heads, group, t, head_dim)).unwrap();
        (dk_4d.sum(1).unwrap(), dv_4d.sum(1).unwrap())
    } else { (dk_3d, dv_3d) };

    // Reshape back to 2D
    let dq = dq_3d.reshape((n_heads * t, head_dim)).unwrap()
        .reshape((t, n_heads * head_dim)).unwrap();
    let dk = dk_out.reshape((n_kv_heads * t, head_dim)).unwrap()
        .reshape((t, n_kv_heads * head_dim)).unwrap();
    let dv = dv_out.reshape((n_kv_heads * t, head_dim)).unwrap()
        .reshape((t, n_kv_heads * head_dim)).unwrap();

    (gpu::from_tensor(&dq), gpu::from_tensor(&dk), gpu::from_tensor(&dv))
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

#[cfg(test)]
#[cfg(feature = "gpu")]
mod gpu_tests {
    use ndarray::Array2;
    use super::*;
    #[test]
    fn gpu_linear_runs_on_cuda() {
        let x = Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
        let w = Array2::from_shape_vec((3, 4), vec![0.5; 12]).unwrap();
        let y = linear(&x, &w);
        assert_eq!(y.dim(), (2, 3));
        // x@wT = 4*0.5 = 2.0 per element
        for v in y.iter() { assert!((v - 2.0).abs() < 1e-3); }
        let d = device();
        assert!(d.is_gpu(), "expected CUDA device, got CPU");
    }
}
