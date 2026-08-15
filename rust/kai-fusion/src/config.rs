//! Kai-Fusion model configuration (Phase C, Slice 1+: dense RoPE-MHA + MoE + MLA).
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnKind {
    MHA,      // dense multi-head (Llama/Qwen/Gemma default)
    MLA,      // Multi-Latent Attention (DeepSeek-V2)
    Linear,   // lightning/linear attention (MiniMax-M3)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlpKind {
    Dense,    // SwiGLU (Llama/Qwen/Gemma)
    MoE,      // Mixture of Experts (DeepSeek-V2)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoEConfig {
    pub enabled: bool,
    pub n_experts: usize,
    pub n_shared: usize,
    pub top_k: usize,
    pub capacity_factor: f32, // expert capacity multiplier
    #[serde(default)]
    pub router_bias: bool, // learnable router bias (DeepSeek-V2 style)
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            n_experts: 64,
            n_shared: 2,
            top_k: 6,
            capacity_factor: 1.25,
            router_bias: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnPolicy {
    Global(AttnKind),                    // same kind for all layers
    PerLayer(Vec<AttnKind>),            // per-layer override
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MLAConfig {
    pub enabled: bool,
    pub q_lora_rank: usize,    // Q low-rank projection dim
    pub kv_lora_rank: usize,   // K/V low-rank projection dim
    pub qk_rope_head_dim: usize, // RoPE head dim for Q/K
    pub v_head_dim: usize,     // V head dim
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisionTowerKind {
    SigLIP,
    CLIP,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    pub enabled: bool,
    pub tower: VisionTowerKind,
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_dim: usize,
    pub n_vision_layers: usize,
    pub n_vision_heads: usize,
    pub vision_intermediate: usize,
    /// Text-space dimension after projector
    pub proj_dim: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tower: VisionTowerKind::SigLIP,
            image_size: 224,
            patch_size: 16,
            vision_dim: 768,
            n_vision_layers: 12,
            n_vision_heads: 12,
            vision_intermediate: 3072,
            proj_dim: 768,
        }
    }
}

impl Default for MLAConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        }
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct Config {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize, // grouped-query; = n_heads for MHA
    pub vocab_size: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub max_seq: usize,
    // Architecture policy
    pub attn_policy: AttnPolicy,
    pub mlp_kind: MlpKind,
    pub moe: MoEConfig,
    pub mla: MLAConfig,
    pub vision: VisionConfig,
    /// DeepSeek-V2: number of leading dense FFN blocks (0 = all MoE)
    pub leading_dense_blocks: usize,
    /// MoE intermediate size (separate from dense intermediate)
    pub expert_intermediate: usize,
    // Physics/world-state fields (Slice 2+)
    pub tau: f32,
    pub e: f32,
    pub age: u64,
    pub cycles: u64,
    pub h: f32,
    pub base_ms: f32,
    pub phi: f32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
    pub fn dim_kv(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }

    pub fn attn_kind(&self, layer_idx: usize) -> AttnKind {
        match &self.attn_policy {
            AttnPolicy::Global(k) => *k,
            AttnPolicy::PerLayer(v) => v.get(layer_idx).copied().unwrap_or(AttnKind::MHA),
        }
    }

    pub fn is_moe(&self) -> bool {
        self.mlp_kind == MlpKind::MoE && self.moe.enabled
    }

    pub fn is_mla(&self) -> bool {
        self.mla.enabled
    }

    pub fn is_vision(&self) -> bool {
        self.vision.enabled
    }

    pub fn is_linear(&self) -> bool {
        self.attn_kind(0) == AttnKind::Linear
    }

    /// Estimated f32 memory in GB for all weights (embed + layers + output + norms).
    pub fn estimated_f32_gb(&self) -> f64 {
        // Attention params per layer (depends on MLA vs MHA)
        let attn_params = if self.is_mla() {
            // DeepSeek-V2 MLA: q_proj + kv_a_mqa + kv_b + output + q_rope
            let q_proj = self.dim * self.mla.q_lora_rank; // attn_q.weight [dim, q_lora_rank * n_heads?]
            let kv_a = self.dim * (self.mla.kv_lora_rank + self.mla.qk_rope_head_dim); // kv_a_mqa
            let kv_b = self.mla.kv_lora_rank * self.n_heads * self.mla.v_head_dim; // kv_b
            let out_proj = self.dim * self.dim; // attn_output
            q_proj + kv_a + kv_b + out_proj
        } else {
            let dk = self.dim_kv();
            self.dim * self.dim       // wq
                + dk * self.dim       // wk
                + dk * self.dim       // wv
                + self.dim * self.dim // wo
        };

        // FFN params per layer
        let ffn_dense = self.intermediate * self.dim // w1 (gate)
            + self.dim * self.intermediate // w2 (down)
            + self.intermediate * self.dim; // w3 (up)

        let (ffn_moe, ffn_shared) = if self.is_moe() {
            let e_inter = if self.expert_intermediate > 0 { self.expert_intermediate } else { self.intermediate };
            // Routed experts: 3 3D tensors [e_inter, dim, n_experts] each
            let routed = e_inter * self.dim * self.moe.n_experts * 3;
            // Shared experts: 3 2D tensors [shared_inter, dim] each
            let shared_inter = self.moe.n_shared * e_inter; // approximation
            let shared = shared_inter * self.dim * 3;
            (routed, shared)
        } else {
            (0, 0)
        };

        let norms_per_layer = self.dim + self.dim; // attn_norm + ffn_norm

        // For non-MoE models, all layers are dense. For MoE, first `leading_dense_blocks`
        // layers are dense and the rest are MoE (with routed/shared experts).
        let (dense_layers, moe_layers) = if self.is_moe() {
            let d = self.leading_dense_blocks.min(self.n_layers);
            (d, self.n_layers.saturating_sub(d))
        } else {
            (self.n_layers, 0)
        };

        let per_dense_layer = attn_params + ffn_dense + norms_per_layer;
        let per_moe_layer = if self.is_moe() {
            attn_params + ffn_moe + ffn_shared + norms_per_layer
        } else {
            // Fallback: same as dense (handles non-MoE models misclassified by leading_dense_blocks)
            attn_params + ffn_dense + norms_per_layer
        };

        let total = self.vocab_size * self.dim       // embed
            + self.vocab_size * self.dim             // output
            + self.dim                               // final_norm
            + dense_layers * per_dense_layer
            + moe_layers * per_moe_layer;
        total as f64 * 4.0 / 1e9                    // f32 = 4 bytes
    }

    /// Tiny model for the runnable Slice-1 demo (fast, local, no weights needed).
    pub fn tiny() -> Self {
        Self {
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 512,
            intermediate: 128,
            rope_theta: 10000.0,
            max_seq: 64,
            tau: 1.0,
            e: 1.0,
            age: 0,
            cycles: 0,
            h: 0.5,
            base_ms: 1000.0,
            phi: 0.0,
            // Architecture policy
            attn_policy: AttnPolicy::Global(AttnKind::MHA),
            mlp_kind: MlpKind::Dense,
            moe: MoEConfig::default(),
            mla: MLAConfig::default(),
            vision: VisionConfig::default(),
            leading_dense_blocks: 0,
            expert_intermediate: 0,
        }
    }

    /// Dense baseline (Llama/Qwen style)
    pub fn dense_llama(dim: usize, n_layers: usize, n_heads: usize, vocab_size: usize) -> Self {
        Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads: n_heads,
            vocab_size,
            intermediate: dim * 4,
            rope_theta: 10000.0,
            max_seq: 4096,
            tau: 1.0,
            e: 1.0,
            age: 0,
            cycles: 0,
            h: 0.5,
            base_ms: 1000.0,
            phi: 0.0,
            attn_policy: AttnPolicy::Global(AttnKind::MHA),
            mlp_kind: MlpKind::Dense,
            moe: MoEConfig::default(),
            mla: MLAConfig::default(),
            vision: VisionConfig::default(),
            leading_dense_blocks: 0,
            expert_intermediate: 0,
        }
    }

    /// DeepSeek-V2 style (MoE + MLA)
    pub fn deepseek_v2(dim: usize, n_layers: usize, n_heads: usize, vocab_size: usize) -> Self {
        let moe = MoEConfig {
            enabled: true,
            n_experts: 64,
            n_shared: 2,
            top_k: 6,
            capacity_factor: 1.25,
            router_bias: true,
        };
        let mla = MLAConfig {
            enabled: true,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        };
        Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads: 1, // MLA uses single KV head
            vocab_size,
            intermediate: 10944, // DeepSeek-V2 intermediate
            rope_theta: 10000.0,
            max_seq: 163840,
            tau: 1.0,
            e: 1.0,
            age: 0,
            cycles: 0,
            h: 0.5,
            base_ms: 1000.0,
            phi: 0.0,
            attn_policy: AttnPolicy::Global(AttnKind::MLA),
            mlp_kind: MlpKind::MoE,
            moe,
            mla,
            vision: VisionConfig::default(),
            leading_dense_blocks: 1,
            expert_intermediate: 1408,
        }
    }
}
