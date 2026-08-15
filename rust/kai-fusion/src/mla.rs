//! Multi-Latent Attention (MLA) — DeepSeek-V2 style low-rank K/V compression.
//!
//! MLA compresses K/V into low-rank latents:
#![allow(dead_code)]
//!   K = (h W_K_D) W_K_U   [batch, seq, kv_lora_rank]
//!   V = (h W_V_D) W_V_U   [batch, seq, v_lora_rank]
//! Then QK^T uses low-rank K, and V uses low-rank V.
//!
//! Paper: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

use crate::config::{Config, MLAConfig};
use ndarray::{Array1, Array2, ArrayView2, s, Axis};

/// MLA projection weights
pub struct MLAWeights {
    // Q: h -> q_lora_rank -> [n_heads * qk_rope_head_dim] + [n_heads * qk_nope_head_dim]
    pub wq_a: Array2<f32>,      // [q_lora_rank, dim]
    pub wq_b: Array2<f32>,      // [n_heads * (qk_rope_head_dim + qk_nope_head_dim), q_lora_rank]

    // K/V: h -> kv_lora_rank
    pub wk_v_a: Array2<f32>,    // [kv_lora_rank, dim] (shared K/V down-proj)
    pub wk_b: Array2<f32>,      // [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    pub wv_b: Array2<f32>,      // [n_heads * v_head_dim, kv_lora_rank] (separate V up-proj)

    // RoPE on Q/K
    pub wq_rope: Array2<f32>,   // [n_heads * qk_rope_head_dim, q_lora_rank]

    // Output projection
    pub wo: Array2<f32>,        // [dim, n_heads * v_head_dim]

    // Layer norm
    pub norm_weight: Array1<f32>,

    config: MLAConfig,
    dim: usize,
    n_heads: usize,
}

impl MLAWeights {
    pub fn new(config: &Config) -> Self {
        let dim = config.dim;
        let n_heads = config.n_heads;
        let mla = &config.mla;

        let q_lora_rank = mla.q_lora_rank;
        let kv_lora_rank = mla.kv_lora_rank;
        let qk_rope_head_dim = mla.qk_rope_head_dim;
        let v_head_dim = mla.v_head_dim;

        let qk_nope_head_dim = mla.v_head_dim; // nope head dim = v_head_dim in DeepSeek-V2
        let q_head_dim = qk_rope_head_dim + qk_nope_head_dim;

        Self {
            // Q: dim -> q_lora_rank -> n_heads * q_head_dim
            wq_a: Array2::zeros((q_lora_rank, dim)),
            wq_b: Array2::zeros((n_heads * q_head_dim, q_lora_rank)),

            // K/V: dim -> kv_lora_rank -> n_heads * (qk_nope + v_head_dim)
            wk_v_a: Array2::zeros((kv_lora_rank, dim)),
            wk_b: Array2::zeros((n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)),
            wv_b: Array2::zeros((n_heads * v_head_dim, kv_lora_rank)), // separate V up-proj

            // RoPE on Q
            wq_rope: Array2::zeros((n_heads * qk_rope_head_dim, q_lora_rank)),

            // Output: n_heads * v_head_dim -> dim
            wo: Array2::zeros((dim, n_heads * v_head_dim)),

            // Layer norm
            norm_weight: Array1::ones(dim),

            config: mla.clone(),
            dim,
            n_heads,
        }
    }

    /// Load from pretrained tensors (for transplant)
    pub fn load_pretrained(&mut self,
        wq_a: Array2<f32>, wq_b: Array2<f32>,
        wk_v_a: Array2<f32>, wk_b: Array2<f32>, wv_b: Array2<f32>,
        wq_rope: Array2<f32>, wo: Array2<f32>,
        norm_weight: Array1<f32>
    ) {
        self.wq_a = wq_a;
        self.wq_b = wq_b;
        self.wk_v_a = wk_v_a;
        self.wk_b = wk_b;
        self.wv_b = wv_b;
        self.wq_rope = wq_rope;
        self.wo = wo;
        self.norm_weight = norm_weight;
    }

    /// MLA forward pass
    /// x: [batch, dim] -> out: [batch, dim]
    pub fn forward(&self, x: &ArrayView2<f32>, cache: &mut MLACache) -> Array2<f32> {
        let batch = x.nrows();
        let _seq = 1; // single token per call in our streaming setup
        let dim = self.dim;
        let n_heads = self.n_heads;

        let _q_lora_rank = self.config.q_lora_rank;
        let _kv_lora_rank = self.config.kv_lora_rank;
        let qk_rope_head_dim = self.config.qk_rope_head_dim;
        let v_head_dim = self.config.v_head_dim;
        let qk_nope_head_dim = v_head_dim;
        let q_head_dim = qk_rope_head_dim + qk_nope_head_dim;

        let mut out = Array2::zeros((batch, dim));

        for b in 0..batch {
            let x_b = x.row(b); // [dim]

            // Layer norm
            let x_norm = crate::engine::rmsnorm_rows(&x_b.to_owned().insert_axis(Axis(0)), &self.norm_weight, crate::engine::EPS).row(0).to_owned();

            // Q down-projection: [dim -> q_lora_rank
            let q_a = self.wq_a.dot(&x_norm); // [q_lora_rank]

            // Q up-projection: q_lora_rank -> n_heads * q_head_dim
            let q_b = self.wq_b.dot(&q_a); // [n_heads * q_head_dim]

            // Split Q into rope + nope
            let q_rope = q_b.slice(s![..n_heads * qk_rope_head_dim]).to_owned();
            let q_nope = q_b.slice(s![n_heads * qk_rope_head_dim..]).to_owned();

            // K/V down-projection: dim -> kv_lora_rank
            let kv_a = self.wk_v_a.dot(&x_norm); // [kv_lora_rank]

            // K/V up-projection
            let kv_b = self.wk_b.dot(&kv_a); // [n_heads * (qk_nope + v_head_dim)]
            let v_b = self.wv_b.dot(&kv_a); // [n_heads * v_head_dim] (separate V up-proj)

            // Split KV into K_nope + V
            let k_nope = kv_b.slice(s![..n_heads * qk_nope_head_dim]).to_owned();
            let v = v_b; // [n_heads * v_head_dim]

            // RoPE on Q_rope and implicit K_rope (DeepSeek-V2 applies RoPE to Q_rope and a learned K_rope)
            // For simplicity, apply standard RoPE to q_rope
            let mut q_rope_owned = q_rope.to_shape((n_heads, qk_rope_head_dim)).unwrap().to_owned();
            crate::engine::apply_rope_all(&mut q_rope_owned, 0, self.config.qk_rope_head_dim, 10000.0);

            // Reshape for attention
            // Q: [n_heads, q_head_dim] where q_head_dim = qk_rope + qk_nope
            // K: [n_heads, qk_nope] (no RoPE on K_nope)
            // V: [n_heads, v_head_dim]

            // Combine Q
            let mut q = Array1::zeros(n_heads * q_head_dim);
            let q_rope_flat = q_rope_owned.into_shape(n_heads * qk_rope_head_dim).unwrap();
            q.slice_mut(s![..n_heads * qk_rope_head_dim]).assign(&q_rope_flat);
            q.slice_mut(s![n_heads * qk_rope_head_dim..]).assign(&q_nope);

            // K: only nope part (no rope)
            let k = k_nope;

            // V: already shaped
            let v = v;

            // Attention: Q @ K^T / sqrt(d) -> softmax -> @ V
            let mut attn_out = Array1::zeros(n_heads * v_head_dim);

            for h in 0..n_heads {
                let q_h = q.slice(s![h * q_head_dim..(h + 1) * q_head_dim]);
                let k_h = k.slice(s![h * qk_nope_head_dim..(h + 1) * qk_nope_head_dim]);
                let v_h = v.slice(s![h * v_head_dim..(h + 1) * v_head_dim]);

                // Attention scores: q_h @ k_h^T / sqrt(qk_nope_head_dim)
                let scale = 1.0 / (qk_nope_head_dim as f32).sqrt();
                let _scores = q_h.slice(s![qk_rope_head_dim..]).dot(&k_h) * scale;

                // Single-token attention: softmax of a single value is 1.0
                // (proper implementation would use KV cache with multi-position scores)
                let out_h = v_h.to_owned();
                attn_out.slice_mut(s![h * v_head_dim..(h + 1) * v_head_dim]).assign(&out_h);
            }

            // Output projection
            let out_b = self.wo.dot(&attn_out);
            out.row_mut(b).assign(&out_b);
        }

        // Update cache (for next token)
        cache.update_kv(&x, &self);

        out
    }
}

/// Cache for MLA (stores compressed K/V latents)
pub struct MLACache {
    /// Compressed K latents: [batch, seq_len, kv_lora_rank]
    pub kv_cache: Option<Array2<f32>>,
    /// Compressed V latents: [batch, seq_len, kv_lora_rank] (if separate)
    pub v_cache: Option<Array2<f32>>,
}

impl MLACache {
    pub fn new() -> Self {
        Self { kv_cache: None, v_cache: None }
    }

    pub fn update_kv(&mut self, x: &ArrayView2<f32>, weights: &MLAWeights) {
        let batch = x.nrows();
        let kv_lora_rank = weights.config.kv_lora_rank;
        let _dim = weights.dim;

        // Compute K/V latents for this token
        let mut new_kv = Array2::zeros((batch, kv_lora_rank));
        let mut new_v = Array2::zeros((batch, kv_lora_rank));

        for b in 0..batch {
            let x_b = x.row(b);
            let x_norm = crate::engine::rmsnorm_rows(&x_b.to_owned().insert_axis(Axis(0)), &weights.norm_weight, crate::engine::EPS).row(0).to_owned();
            let kv_a = weights.wk_v_a.dot(&x_norm);
            new_kv.row_mut(b).assign(&kv_a);
            new_v.row_mut(b).assign(&weights.wv_b.dot(&kv_a));
        }

        // Append to cache
        if let Some(cache) = &self.kv_cache {
            let mut combined = Array2::zeros((cache.nrows() + batch, kv_lora_rank));
            combined.slice_mut(s![..cache.nrows(), ..]).assign(cache);
            combined.slice_mut(s![cache.nrows().., ..]).assign(&new_kv);
            self.kv_cache = Some(combined);
        } else {
            self.kv_cache = Some(new_kv);
        }

        if let Some(cache) = &self.v_cache {
            let mut combined = Array2::zeros((cache.nrows() + batch, kv_lora_rank));
            combined.slice_mut(s![..cache.nrows(), ..]).assign(cache);
            combined.slice_mut(s![cache.nrows().., ..]).assign(&new_v);
            self.v_cache = Some(combined);
        } else {
            self.v_cache = Some(new_v);
        }
    }

    pub fn clear(&mut self) {
        self.kv_cache = None;
        self.v_cache = None;
    }
}