//! Model configuration with correct f32 memory estimation

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnKind { MHA, MLA, Linear }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnPolicy {
    Global(AttnKind),
    PerLayer(Vec<AttnKind>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlpKind { Dense, MoE }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    pub enabled: bool,
    pub n_experts: usize,
    pub n_shared: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub router_bias: bool,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self { enabled: false, n_experts: 64, n_shared: 2, top_k: 6, capacity_factor: 1.25, router_bias: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLAConfig {
    pub enabled: bool,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl Default for MLAConfig {
    fn default() -> Self {
        Self { enabled: false, q_lora_rank: 0, kv_lora_rank: 512, qk_rope_head_dim: 64, v_head_dim: 128 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub enabled: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    pub base_temperature: f32,
    pub top_p: f32,
    pub novelty_scale: f32,
    pub vfe_tau_rate: f32,
    pub tau_min: f32,
    pub tau_max: f32,
    pub assim_iters: usize,
    pub assim_lr: f32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            base_temperature: 0.7,
            top_p: 0.9,
            novelty_scale: 0.5,
            vfe_tau_rate: 0.1,
            tau_min: 0.5,
            tau_max: 2.0,
            assim_iters: 0,
            assim_lr: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub rope_dim: usize,
    pub max_seq: usize,
    pub tau: f32,
    pub e: f32,
    pub age: u64,
    pub cycles: u64,
    pub h: f32,
    pub base_ms: f32,
    pub phi: f32,
    pub attn_policy: AttnPolicy,
    pub mlp_kind: MlpKind,
    pub moe: MoEConfig,
    pub mla: MLAConfig,
    pub vision: VisionConfig,
    pub leading_dense_blocks: usize,
    pub expert_intermediate: usize,
}

impl Config {
    pub fn is_mla(&self) -> bool { self.mla.enabled }
    pub fn is_moe(&self) -> bool { self.mlp_kind == MlpKind::MoE && self.moe.enabled }
    pub fn head_dim(&self) -> usize { self.dim / self.n_heads }
    pub fn dim_kv(&self) -> usize { self.n_kv_heads * self.head_dim() }

    /// Estimated f32 memory in GB for all weights (embed + layers + output + norms)
    /// FIX: For non-MoE models, ALL layers are dense. Only DeepSeek-V2 uses leading_dense_blocks.
    pub fn estimated_f32_gb(&self) -> f64 {
        let attn_params = if self.is_mla() {
            let q = self.dim * self.mla.q_lora_rank;
            let kv_a = self.dim * (self.mla.kv_lora_rank + self.mla.qk_rope_head_dim);
            let kv_b = self.mla.kv_lora_rank * self.n_heads * self.mla.v_head_dim;
            let o = self.dim * self.dim;
            q + kv_a + kv_b + o
        } else {
            let dk = self.dim_kv();
            self.dim * self.dim       // wq
                + dk * self.dim       // wk
                + dk * self.dim       // wv
                + self.dim * self.dim // wo
        };

        let ffn_dense = self.intermediate * self.dim   // w1 (gate)
            + self.dim * self.intermediate   // w2 (down)
            + self.intermediate * self.dim;  // w3 (up)

        let (ffn_moe, ffn_shared) = if self.is_moe() {
            let e_inter = if self.expert_intermediate > 0 { self.expert_intermediate } else { self.intermediate };
            let routed = e_inter * self.dim * self.moe.n_experts * 3;
            let shared_inter = self.moe.n_shared * e_inter;
            let shared = shared_inter * self.dim * 3;
            (routed, shared)
        } else {
            (0, 0)
        };

        let norms_per_layer = self.dim + self.dim; // attn_norm + ffn_norm

        // FIX: For non-MoE models, ALL layers are dense. Only DeepSeek-V2 (is_moe=true) uses leading_dense_blocks.
        let (dense_layers, moe_layers) = if self.is_moe() {
            let d = self.leading_dense_blocks.min(self.n_layers);
            (d, self.n_layers.saturating_sub(d))
        } else {
            (self.n_layers, 0)
        };

        let per_dense = attn_params + ffn_dense + norms_per_layer;
        let per_moe = attn_params + ffn_moe + ffn_shared + norms_per_layer;

        let total = self.vocab_size * self.dim       // embed
            + self.vocab_size * self.dim             // output
            + self.dim                               // final_norm
            + dense_layers * per_dense
            + moe_layers * per_moe;

        total as f64 * 4.0 / 1e9  // f32 = 4 bytes
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dim={} layers={} heads={} kv_heads={} vocab={} inter={} est_gb={:.1}",
            self.dim, self.n_layers, self.n_heads, self.n_kv_heads, 
            self.vocab_size, self.intermediate, self.estimated_f32_gb())
    }
}
