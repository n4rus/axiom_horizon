//! Transformer model (Slice 1: dense RoPE-MHA, Llama/Qwen-style pre-norm, SwiGLU).
//! Weights stored (out, in) to match GGUF layout directly.
//! Includes full forward + backward pass for training.

use crate::config::Config;
use crate::engine;
#[allow(unused_imports)]
use crate::moe::{MoELayer, RouterOutput};
#[allow(unused_imports)]
use crate::mla::{MLAWeights, MLACache};
use crate::linear_attn::{linear_attn_forward, FeatureFn};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s};
use std::collections::HashMap;

type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;


const GRAD_CLIP: f32 = 1.0;

fn sanitize(v: f32) -> f32 {
    if v.is_nan() || v.is_infinite() { 0.0 }
    else { v.clamp(-GRAD_CLIP, GRAD_CLIP) }
}

#[inline(always)]
fn sanitize_in_place(v: &mut f32) {
    if v.is_nan() || v.is_infinite() {
        *v = 0.0;
    } else {
        *v = v.clamp(-GRAD_CLIP, GRAD_CLIP);
    }
}

fn mat(map: &TensorMap, name: &str) -> Array2<f32> {
    let (shape, data) = map.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    let r = shape[0];
    let c = shape[1];
    Array2::from_shape_vec((r, c), data.clone()).unwrap()
}
fn mat_t(map: &TensorMap, name: &str) -> Array2<f32> {
    mat(map, name).t().to_owned()
}
fn vec1(map: &TensorMap, name: &str) -> Array1<f32> {
    let (_, data) = map.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    Array1::from_vec(data.clone())
}

/// Deterministic LCG so the demo model is reproducible (no external RNG dep).
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.0 >> 33) as f32;
        (bits / (1u32 << 31) as f32) - 1.0
    }
    fn mat(&mut self, out: usize, in_: usize) -> Array2<f32> {
        let v: Vec<f32> = (0..out * in_).map(|_| self.next_f32() * 0.1).collect();
        Array2::from_shape_vec((out, in_), v).unwrap()
    }
    fn vec(&mut self, n: usize) -> Array1<f32> {
        Array1::from_vec((0..n).map(|_| 1.0).collect())
    }
}

#[derive(Clone)]
pub struct LayerWeights {
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
    pub attn_norm: Array1<f32>,
    pub ffn_norm: Array1<f32>,
    // MoE extensions
    pub moe_shared_w1: Option<Array2<f32>>,   // [n_shared * intermediate, dim]
    pub moe_shared_w2: Option<Array2<f32>>,   // [dim, n_shared * intermediate]
    pub moe_shared_w3: Option<Array2<f32>>,   // [n_shared * intermediate, dim]
    pub moe_expert_w1: Option<Vec<Array2<f32>>>, // [n_experts] of [intermediate, dim]
    pub moe_expert_w2: Option<Vec<Array2<f32>>>, // [n_experts] of [dim, intermediate]
    pub moe_expert_w3: Option<Vec<Array2<f32>>>, // [n_experts] of [intermediate, dim]
    pub moe_router_weight: Option<Array2<f32>>,  // [n_experts, dim]
    pub moe_router_bias: Option<Array1<f32>>,    // [n_experts]
    // MLA extensions
    pub mla_wq_a: Option<Array2<f32>>,        // [q_lora_rank, dim]
    pub mla_wq_b: Option<Array2<f32>>,        // [n_heads * q_head_dim, q_lora_rank]
    pub mla_wk_v_a: Option<Array2<f32>>,      // [kv_lora_rank, dim]
    pub mla_wk_b: Option<Array2<f32>>,        // [n_heads * (qk_nope + v_head), kv_lora_rank]
    pub mla_wv_b: Option<Array2<f32>>,        // [n_heads * v_head_dim, kv_lora_rank]
    #[allow(dead_code)]
    pub mla_wq_rope: Option<Array2<f32>>,     // [n_heads * qk_rope, q_lora_rank]
    pub mla_wo: Option<Array2<f32>>,          // [dim, n_heads * v_head_dim]
    // GGUF-format MLA fields (DeepSeek-V2 llama.cpp convention)
    pub mla_q: Option<Array2<f32>>,           // [dim, n_heads * qk_head_dim] = attn_q.weight
    pub mla_kv_a_mqa: Option<Array2<f32>>,    // [dim, kv_lora_rank + qk_rope_head_dim] = attn_kv_a_mqa.weight
    pub mla_kv_b: Option<Array2<f32>>,        // [kv_lora_rank, n_heads * (d_nope + v_head_dim)] = attn_kv_b.weight
    pub mla_kv_a_norm: Option<Array1<f32>>,   // [kv_lora_rank] = attn_kv_a_norm.weight
    // GGUF-format MoE fields (DeepSeek-V2 llama.cpp convention)
    pub moe_gate_exps: Option<Array3<f32>>,   // [dim, e_inter, n_experts] = ffn_gate_exps.weight
    pub moe_down_exps: Option<Array3<f32>>,   // [e_inter, dim, n_experts] = ffn_down_exps.weight  
    pub moe_up_exps: Option<Array3<f32>>,     // [dim, e_inter, n_experts] = ffn_up_exps.weight
    pub moe_gate_inp: Option<Array2<f32>>,    // [dim, n_experts] = ffn_gate_inp.weight (router)
    pub moe_shared_gate: Option<Array2<f32>>, // [dim, shared_inter] = ffn_gate_shexp.weight
    pub moe_shared_down: Option<Array2<f32>>, // [shared_inter, dim] = ffn_down_shexp.weight
    pub moe_shared_up: Option<Array2<f32>>,   // [dim, shared_inter] = ffn_up_shexp.weight
}

#[derive(Clone)]
pub struct Weights {
    pub embed: Array2<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Array1<f32>,
    pub output: Array2<f32>,
}

/// Cache of intermediate activations during forward pass for backprop.
pub struct ForwardCache {
    /// Per-layer caches
    pub layers: Vec<LayerCache>,
    /// Final norm output [T, dim]
    pub xf: Array2<f32>,
    /// Logits [T, vocab]
    pub logits: Array2<f32>,
}

/// Per-layer cache — stores intermediates needed for backprop.
pub struct LayerCache {
    pub h_attn: Array2<f32>,
    pub v: Array2<f32>,
    pub q_rope: Array2<f32>,
    pub k_rope: Array2<f32>,
    pub attn_out: Array2<f32>,
    pub h_ffn: Array2<f32>,
    pub gate: Array2<f32>,
    pub up: Array2<f32>,
    pub gate_silu: Array2<f32>,
    pub x_out: Array2<f32>,
    /// GPU tensors kept alive between forward and backward (avoids re-upload).
    #[cfg(feature = "gpu")]
    pub gpu: Option<GpuLayerCache>,
}

/// GPU-side layer cache: tensors stored on device, avoid CPU round-trips.
#[cfg(feature = "gpu")]
pub struct GpuLayerCache {
    pub h_attn: candle_core::Tensor,
    pub q_rope: candle_core::Tensor,
    pub k_rope: candle_core::Tensor,
    pub v: candle_core::Tensor,
    pub attn_out: candle_core::Tensor,
    pub h_ffn: candle_core::Tensor,
    pub gate: candle_core::Tensor,
    pub up: candle_core::Tensor,
    pub gate_silu: candle_core::Tensor,
    pub x_out: candle_core::Tensor,
}

/// Controls which parameters are frozen (no gradient computed/stored).
#[derive(Clone, Debug)]
pub struct FreezeConfig {
    /// Number of initial layers to freeze (0 = train all)
    pub freeze_layers: usize,
    /// Freeze the embedding table
    pub freeze_embed: bool,
    /// Freeze the output (lm_head) projection
    pub freeze_output: bool,
}

impl Default for FreezeConfig {
    fn default() -> Self {
        FreezeConfig { freeze_layers: 0, freeze_embed: false, freeze_output: false }
    }
}

/// Gradient accumulator for all parameters.
/// Frozen parameters have `None` gradients (not allocated).
pub struct Gradients {
    pub embed: Option<Array2<f32>>,
    pub layers: Vec<Option<LayerGrads>>,
    pub final_norm: Array1<f32>,
    pub output: Option<Array2<f32>>,
}

impl Gradients {
    /// Accumulate another gradient into this one (in-place add).
    /// Skips frozen (None) gradients.
    #[allow(dead_code)]
    pub fn accumulate(&mut self, other: &Gradients) {
        if let (Some(e), Some(oe)) = (&mut self.embed, &other.embed) {
            *e += oe;
        }
        if let (Some(o), Some(oo)) = (&mut self.output, &other.output) {
            *o += oo;
        }
        self.final_norm += &other.final_norm;
        for (li, lg) in self.layers.iter_mut().enumerate() {
            if let (Some(lg), Some(olg)) = (lg.as_mut(), other.layers[li].as_ref()) {
                lg.wq += &olg.wq;
                lg.wk += &olg.wk;
                lg.wv += &olg.wv;
                lg.wo += &olg.wo;
                lg.w1 += &olg.w1;
                lg.w2 += &olg.w2;
                lg.w3 += &olg.w3;
                lg.attn_norm += &olg.attn_norm;
                lg.ffn_norm += &olg.ffn_norm;
            }
        }
    }

    /// Zero out all non-frozen gradients (for re-use).
    #[allow(dead_code)]
    pub fn zero(&mut self) {
        if let Some(e) = &mut self.embed { e.fill(0.0); }
        if let Some(o) = &mut self.output { o.fill(0.0); }
        self.final_norm.fill(0.0);
        for lg in self.layers.iter_mut() {
            if let Some(lg) = lg.as_mut() {
                lg.wq.fill(0.0); lg.wk.fill(0.0); lg.wv.fill(0.0); lg.wo.fill(0.0);
                lg.w1.fill(0.0); lg.w2.fill(0.0); lg.w3.fill(0.0);
                lg.attn_norm.fill(0.0); lg.ffn_norm.fill(0.0);
            }
        }
    }
}

pub struct LayerGrads {
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w3: Array2<f32>,
    pub attn_norm: Array1<f32>,
    pub ffn_norm: Array1<f32>,
}

impl Weights {
    /// Create zero-initialized gradient accumulator matching this model's shape.
    /// Frozen params (per `freeze`) get `None` — no memory allocated.
    #[allow(dead_code)]
    pub fn zero_gradients(&self, cfg: &Config, freeze: &FreezeConfig) -> Gradients {
        let dk = cfg.dim_kv();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            if li < freeze.freeze_layers {
                layers.push(None);
            } else {
                layers.push(Some(LayerGrads {
                    wq: Array2::zeros((cfg.dim, cfg.dim)),
                    wk: Array2::zeros((dk, cfg.dim)),
                    wv: Array2::zeros((dk, cfg.dim)),
                    wo: Array2::zeros((cfg.dim, cfg.dim)),
                    w1: Array2::zeros((cfg.intermediate, cfg.dim)),
                    w2: Array2::zeros((cfg.dim, cfg.intermediate)),
                    w3: Array2::zeros((cfg.intermediate, cfg.dim)),
                    attn_norm: Array1::zeros(cfg.dim),
                    ffn_norm: Array1::zeros(cfg.dim),
                }));
            }
        }
        Gradients {
            embed: if freeze.freeze_embed { None } else { Some(Array2::zeros((cfg.vocab_size, cfg.dim))) },
            layers,
            final_norm: Array1::zeros(cfg.dim),
            output: if freeze.freeze_output { None } else { Some(Array2::zeros((cfg.vocab_size, cfg.dim))) },
        }
    }

    pub fn random(cfg: &Config) -> Self {
        let mut rng = Rng::new(0xC0FFEE);
        let dk = cfg.dim_kv();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(LayerWeights {
                wq: rng.mat(cfg.dim, cfg.dim),
                wk: rng.mat(dk, cfg.dim),
                wv: rng.mat(dk, cfg.dim),
                wo: rng.mat(cfg.dim, cfg.dim),
                w1: rng.mat(cfg.intermediate, cfg.dim),
                w2: rng.mat(cfg.dim, cfg.intermediate),
                w3: rng.mat(cfg.intermediate, cfg.dim),
                attn_norm: rng.vec(cfg.dim),
                ffn_norm: rng.vec(cfg.dim),
                // MoE extensions (Option::None for dense)
                moe_shared_w1: None,
                moe_shared_w2: None,
                moe_shared_w3: None,
                moe_expert_w1: None,
                moe_expert_w2: None,
                moe_expert_w3: None,
                moe_router_weight: None,
                moe_router_bias: None,
                // MLA extensions (Option::None for MHA)
                mla_wq_a: None,
                mla_wq_b: None,
                mla_wk_v_a: None,
                mla_wk_b: None,
                mla_wv_b: None,
                mla_wq_rope: None,
                mla_wo: None,
                // MoE extensions (GGUF convention)
                moe_gate_exps: None,
                moe_down_exps: None,
                moe_up_exps: None,
                moe_gate_inp: None,
                moe_shared_gate: None,
                moe_shared_down: None,
                moe_shared_up: None,
                // MLA extensions (GGUF convention)
                mla_q: None,
                mla_kv_a_mqa: None,
                mla_kv_b: None,
                mla_kv_a_norm: None,
            });
        }
        Weights {
            embed: rng.mat(cfg.vocab_size, cfg.dim),
            layers,
            final_norm: rng.vec(cfg.dim),
            output: rng.mat(cfg.vocab_size, cfg.dim),
        }
    }

    /// Forward pass -> (logits (T, vocab), final-norm hidden of last position).
    pub fn forward(&self, cfg: &Config, tokens: &[usize]) -> (Array2<f32>, Vec<f32>) {
        let t = tokens.len().min(cfg.max_seq);
        let toks: Vec<usize> = tokens.iter().take(t).cloned().collect();
        let t = toks.len();
        let hd = cfg.head_dim();

        let mut x = Array2::zeros((t, cfg.dim));
        for (i, &tok) in toks.iter().enumerate() {
            let tok = tok.min(cfg.vocab_size - 1);
            for d in 0..cfg.dim {
                x[[i, d]] = self.embed[[tok, d]];
            }
        }

        for layer in &self.layers {
            #[cfg(feature = "gpu")]
            {
                // GPU-fused inference: all operations on GPU per sub-block
                use crate::engine::{fused_attention, fused_ffn};

                let attn_proj = fused_attention(
                    &x, layer.attn_norm.as_slice().unwrap(),
                    &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                    cfg.n_heads, cfg.n_kv_heads, hd, cfg.rope_theta, engine::EPS,
                );
                x = &x + &attn_proj;

                let ff = fused_ffn(
                    &x, layer.ffn_norm.as_slice().unwrap(),
                    &layer.w1, &layer.w3, &layer.w2, engine::EPS,
                );
                x = &x + &ff;

                crate::engine::invalidate_weight_cache();
            }
            #[cfg(not(feature = "gpu"))]
            {
                // --- attention (pre-norm) ---
                let h = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);

                // Dispatch attention based on config
                let attn_out = if cfg.is_mla() {
                    // MLA path (DeepSeek-V2)
                    self.mla_forward_layer(cfg, &h.view(), layer, hd)
                } else if cfg.is_linear() {
                    // Linear/lightning attention path (MiniMax-M3)
                    self.linear_attn_forward_layer(cfg, &h.view(), layer, hd)
                } else {
                    // Standard MHA path (Llama/Qwen/Gemma)
                    self.mha_forward_layer(cfg, &h.view(), layer, hd)
                };

                let attn_proj = engine::linear(&attn_out, &layer.wo);
                x = &x + &attn_proj;

                // --- ffn (SwiGLU dense or MoE, pre-norm) ---
                let h2 = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
                let ff = if cfg.is_moe() {
                    self.moe_forward_layer(cfg, &h2.view(), layer)
                } else {
                    // Dense SwiGLU
                    let gate = engine::silu(&engine::linear(&h2, &layer.w1));
                    let up = engine::linear(&h2, &layer.w3);
                    engine::linear(&(&gate * &up), &layer.w2)
                };
                x = &x + &ff;
            }
        }

        let xf = engine::rmsnorm_rows(&x, &self.final_norm, engine::EPS);
        let logits = engine::linear(&xf, &self.output); // (T, vocab)
        let xf_last = (0..cfg.dim).map(|d| xf[[t - 1, d]]).collect();
        (logits, xf_last)
    }

    /// Forward and return only the last-token logits (for generation).
    pub fn last_logits(&self, cfg: &Config, tokens: &[usize]) -> Vec<f32> {
        let (logits, _xf) = self.forward(cfg, tokens);
        let row = logits.nrows() - 1;
        (0..logits.ncols()).map(|j| logits[[row, j]]).collect()
    }

    /// Forward pass with pre-computed embeddings instead of token IDs.
    /// `embeds`: [t, dim] — pre-computed input embeddings (e.g. vision + text).
    /// This allows multi-modal inputs where vision tokens are projected into
    /// the text model's latent space.
    pub fn forward_with_embeddings(&self, cfg: &Config, embeds: &Array2<f32>) -> (Array2<f32>, Vec<f32>) {
        let t = embeds.nrows().min(cfg.max_seq);
        let mut x = embeds.slice(ndarray::s![..t, ..]).to_owned();
        let hd = cfg.head_dim();

        for layer in &self.layers {
            #[cfg(feature = "gpu")]
            {
                use crate::engine::{fused_attention, fused_ffn};
                let attn_proj = fused_attention(
                    &x, layer.attn_norm.as_slice().unwrap(),
                    &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                    cfg.n_heads, cfg.n_kv_heads, hd, cfg.rope_theta, engine::EPS,
                );
                x = &x + &attn_proj;
                let ff = fused_ffn(
                    &x, layer.ffn_norm.as_slice().unwrap(),
                    &layer.w1, &layer.w3, &layer.w2, engine::EPS,
                );
                x = &x + &ff;
                crate::engine::invalidate_weight_cache();
            }
            #[cfg(not(feature = "gpu"))]
            {
                let h = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);
                let attn_out = if cfg.is_mla() {
                    self.mla_forward_layer(cfg, &h.view(), layer, hd)
                } else if cfg.is_linear() {
                    self.linear_attn_forward_layer(cfg, &h.view(), layer, hd)
                } else {
                    self.mha_forward_layer(cfg, &h.view(), layer, hd)
                };
                let attn_proj = engine::linear(&attn_out, &layer.wo);
                x = &x + &attn_proj;

                let h2 = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
                let ff = if cfg.is_moe() {
                    self.moe_forward_layer(cfg, &h2.view(), layer)
                } else {
                    let gate = engine::silu(&engine::linear(&h2, &layer.w1));
                    let up = engine::linear(&h2, &layer.w3);
                    engine::linear(&(&gate * &up), &layer.w2)
                };
                x = &x + &ff;
            }
        }

        let xf = engine::rmsnorm_rows(&x, &self.final_norm, engine::EPS);
        let logits = engine::linear(&xf, &self.output);
        let xf_last = (0..cfg.dim).map(|d| xf[[t - 1, d]]).collect();
        (logits, xf_last)
    }

    /// Final-norm hidden state of the last position (the context representation
    /// Kai uses to measure novelty against its assimilated prior).
    pub fn last_hidden(&self, cfg: &Config, tokens: &[usize]) -> Vec<f32> {
        let (_logits, xf) = self.forward(cfg, tokens);
        xf
    }

    /// Mean-pooled final-norm hidden state across all positions.
    /// Produces more discriminative embeddings than last_hidden for retrieval.
    /// For short texts (<1 token) returns last_hidden as fallback.
    #[allow(dead_code)]
    pub fn mean_pooled_hidden(&self, cfg: &Config, tokens: &[usize]) -> Vec<f32> {
        let t = tokens.len().min(cfg.max_seq);
        let toks: Vec<usize> = tokens.iter().take(t).cloned().collect();
        let t = toks.len();
        if t <= 1 {
            return self.last_hidden(cfg, tokens);
        }

        let hd = cfg.head_dim();
        let mut x = Array2::zeros((t, cfg.dim));
        for (i, &tok) in toks.iter().enumerate() {
            let tok = tok.min(cfg.vocab_size - 1);
            for d in 0..cfg.dim {
                x[[i, d]] = self.embed[[tok, d]];
            }
        }

        for layer in &self.layers {
            #[cfg(not(feature = "gpu"))]
            {
                let h = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);
                let mut q = engine::linear(&h, &layer.wq);
                let mut k = engine::linear(&h, &layer.wk);
                let v = engine::linear(&h, &layer.wv);
                engine::apply_rope_all(&mut q, cfg.n_heads, hd, cfg.rope_theta);
                engine::apply_rope_all(&mut k, cfg.n_kv_heads, hd, cfg.rope_theta);

                let mut attn_out = Array2::zeros((t, cfg.dim));
                let group = cfg.n_heads / cfg.n_kv_heads.max(1);
                for head in 0..cfg.n_heads {
                    let kvh = head / group;
                    for i in 0..t {
                        let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
                        for j in 0..=i {
                            let mut dot = 0.0;
                            for d in 0..hd {
                                dot += q[[i, head * hd + d]] * k[[j, kvh * hd + d]];
                            }
                            scores.push(dot / (hd as f32).sqrt());
                        }
                        let probs = engine::softmax(&scores);
                        for d in 0..hd {
                            let mut acc = 0.0;
                            for (j, &sj) in probs.iter().enumerate() {
                                acc += sj * v[[j, kvh * hd + d]];
                            }
                            attn_out[[i, head * hd + d]] = acc;
                        }
                    }
                }
                let attn_proj = engine::linear(&attn_out, &layer.wo);
                x = &x + &attn_proj;

                let h2 = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
                let gate = engine::silu(&engine::linear(&h2, &layer.w1));
                let up = engine::linear(&h2, &layer.w3);
                let ff = engine::linear(&(&gate * &up), &layer.w2);
                x = &x + &ff;
            }
        }

        let xf = engine::rmsnorm_rows(&x, &self.final_norm, engine::EPS);
        // Mean-pool across all positions
        let mut pooled = vec![0.0f32; cfg.dim];
        for i in 0..t {
            for d in 0..cfg.dim {
                pooled[d] += xf[[i, d]];
            }
        }
        for d in 0..cfg.dim {
            pooled[d] /= t as f32;
        }
        pooled
    }

    /// Load model from GGUF and compute native embedding (mean-pooled hidden state).
    /// This avoids needing a separate BERT encoder — uses the model's own representations.
    pub fn embed_with_model(gguf_path: &str, text: &str) {
        use crate::loader::{load_tensors, read_kv, build_config};
        use crate::tok::Tokenizer;

        let meta = match read_kv(gguf_path) {
            Ok(m) => m,
            Err(e) => { eprintln!("read meta failed: {e}"); return; }
        };
        let cfg = match build_config(&meta) {
            Some(c) => c,
            None => { eprintln!("could not build config"); return; }
        };
        let tok = match Tokenizer::from_gguf(&meta) {
            Some(t) => t,
            None => { eprintln!("no tokenizer in GGUF"); return; }
        };

        let (_v, map, _nt, _nk) = match load_tensors(gguf_path) {
            Ok(x) => x,
            Err(e) => { eprintln!("load tensors failed: {e}"); return; }
        };

        let w = match Weights::from_gguf(&map, &cfg) {
            Ok(w) => w,
            Err(e) => { eprintln!("from_gguf failed: {e}"); return; }
        };

        let ids = tok.encode(text);
        let emb = w.mean_pooled_hidden(&cfg, &ids);

        let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mx = emb.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mn = emb.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        println!("Kai-Fusion native embedding (mean-pooled hidden):");
        println!("  dim={}  |x|={:.4}  min={:.4}  max={:.4}", emb.len(), norm, mn, mx);
        print!("  emb[:8] = ");
        for v in emb.iter().take(8) {
            print!("{v:.4} ");
        }
        println!();
    }

    /// MoE forward for a single layer (DeepSeek-V2 style)
    fn moe_forward_layer(&self, cfg: &Config, h: &ArrayView2<f32>, layer: &LayerWeights) -> Array2<f32> {
        let batch = h.nrows();
        let dim = cfg.dim;
        let intermediate = cfg.intermediate;
        let moe = &cfg.moe;
        let _n_experts = moe.n_experts;
        let n_shared = moe.n_shared;
        let top_k = moe.top_k;

        let l = layer;
        let shared_w1 = l.moe_shared_w1.as_ref().unwrap(); // [n_shared * intermediate, dim]
        let shared_w2 = l.moe_shared_w2.as_ref().unwrap(); // [dim, n_shared * intermediate]
        let shared_w3 = l.moe_shared_w3.as_ref().unwrap(); // [n_shared * intermediate, dim]
        let expert_w1 = l.moe_expert_w1.as_ref().unwrap(); // [n_experts] of [intermediate, dim]
        let expert_w2 = l.moe_expert_w2.as_ref().unwrap(); // [n_experts] of [dim, intermediate]
        let expert_w3 = l.moe_expert_w3.as_ref().unwrap(); // [n_experts] of [intermediate, dim]
        let router_weight = l.moe_router_weight.as_ref().unwrap(); // [n_experts, dim]
        let router_bias = l.moe_router_bias.as_ref(); // [n_experts]

        let mut out = Array2::zeros((batch, dim));

        for b in 0..batch {
            let token = h.row(b);
            let mut token_out = Array1::zeros(dim);

            // Shared experts (always active)
            for s in 0..n_shared {
                let w1 = shared_w1.slice(s![s * intermediate..(s + 1) * intermediate, ..]);
                let w2 = shared_w2.slice(s![.., s * intermediate..(s + 1) * intermediate]);
                let w3 = shared_w3.slice(s![s * intermediate..(s + 1) * intermediate, ..]);

                let gate = w1.dot(&token).mapv(|x| x.max(0.0)); // SwiGLU gate
                let up = w3.dot(&token);
                let hidden = &gate * &up;
                let expert_out = w2.dot(&hidden);
                token_out += &expert_out;
            }

            // Router
            let logits = router_weight.dot(&token);
            let logits = if let Some(bias) = router_bias {
                &logits + bias
            } else {
                logits
            };

            // Top-k softmax
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits = logits.mapv(|x| (x - max_logit).exp());
            let probs = &exp_logits / exp_logits.sum();

            // Top-k indices
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_k = &indexed[..top_k];

            let sum_top = top_k.iter().map(|(_, w)| *w).sum::<f32>();

            for (e, prob) in top_k {
                let weight = prob / sum_top;
                let w1 = &expert_w1[*e];
                let w2 = &expert_w2[*e];
                let _w3 = &expert_w3[*e];

                let gate = w1.dot(&token).mapv(|x| x.max(0.0));
                let up = expert_w3[*e].dot(&token);
                let hidden = &gate * &up;
                let expert_out = w2.dot(&hidden);
                token_out += &(&expert_out * weight);
            }

            out.row_mut(b).assign(&token_out);
        }

        out
    }

/// MLA forward for a single layer (DeepSeek-V2 style).
/// Both Q and K are split into nope (without RoPE) and rope (with RoPE) components.
/// Attention score = q_nope @ k_nope^T + q_rope @ k_rope^T, so positions are encoded
/// through the rope terms and content through the nope terms.
    fn mla_forward_layer(&self, cfg: &Config, h: &ArrayView2<f32>, layer: &LayerWeights, _hd: usize) -> Array2<f32> {
        let t = h.nrows();
        let mla = &cfg.mla;
        let n_heads = cfg.n_heads;

        // Per-head dimension accounting
        // d_nope = v_head_dim - qk_rope_head_dim   (non-RoPE'd part, matched with key nope)
        // d_rope = qk_rope_head_dim                (RoPE'd part, matched with key rope)
        // Total per head = d_nope + d_rope = v_head_dim
        let (d_rope, d_nope) = {
            let r = mla.qk_rope_head_dim;
            let v = mla.v_head_dim;
            (r, v.saturating_sub(r))
        };

        let l = layer;
        let wq_a = l.mla_wq_a.as_ref().unwrap();
        let wq_b = l.mla_wq_b.as_ref().unwrap();
        let wk_v_a = l.mla_wk_v_a.as_ref().unwrap();
        let wk_b = l.mla_wk_b.as_ref().unwrap();
        let wv_b = l.mla_wv_b.as_ref().unwrap();
        let wo = l.mla_wo.as_ref().unwrap();

        let h_norm = engine::rmsnorm_rows(&h.to_owned(), &l.attn_norm, engine::EPS);

        // Q: h → q_a (compressed) → q_b (full, split into rope + nope)
        let q_a = h_norm.dot(&wq_a.t());
        // wq_b output: n_heads * (d_rope + v_head_dim) ≈ n_heads * (d_rope + d_nope + d_rope)
        // We take only the first n_heads * (d_nope + d_rope) = n_heads * v_head_dim per head
        let q_b = q_a.dot(&wq_b.t());
        let q_rope_src = q_b.slice(s![.., ..n_heads * d_rope]).to_owned();    // [T, n_heads * d_rope]
        let q_nope_src = q_b.slice(s![.., n_heads * d_rope..n_heads * (d_rope + d_nope)]).to_owned(); // [T, n_heads * d_nope]

        // K/V: h → kv_a (compressed) → kv_b (key nope + key rope), v_b (value)
        let kv_a = h_norm.dot(&wk_v_a.t());
        let kv_b = kv_a.dot(&wk_b.t()); // [T, n_heads * (d_nope + d_rope)] = [T, cfg.dim]
        let v_b = kv_a.dot(&wv_b.t());  // [T, n_heads * v_head_dim]

        // Split kv_b into k_nope (first d_nope per head) + k_rope (next d_rope per head)
        let k_nope_src = kv_b.slice(s![.., ..n_heads * d_nope]).to_owned();   // [T, n_heads * d_nope]
        let k_rope_src = kv_b.slice(s![.., n_heads * d_nope..n_heads * (d_nope + d_rope)]).to_owned(); // [T, n_heads * d_rope]

        // RoPE on Q_rope and K_rope
        let mut q_rope_reshaped = q_rope_src.to_shape((t, n_heads, d_rope)).unwrap().to_owned();
        let mut k_rope_reshaped = k_rope_src.to_shape((t, n_heads, d_rope)).unwrap().to_owned();
        for i in 0..t {
            let mut q_row = q_rope_reshaped.slice(s![i, .., ..])
                .into_shape((1, n_heads * d_rope)).unwrap().to_owned();
            engine::apply_rope_all(&mut q_row, n_heads, d_rope, cfg.rope_theta);
            q_rope_reshaped.slice_mut(s![i, .., ..])
                .assign(&q_row.into_shape((n_heads, d_rope)).unwrap());

            let mut k_row = k_rope_reshaped.slice(s![i, .., ..])
                .into_shape((1, n_heads * d_rope)).unwrap().to_owned();
            engine::apply_rope_all(&mut k_row, n_heads, d_rope, cfg.rope_theta);
            k_rope_reshaped.slice_mut(s![i, .., ..])
                .assign(&k_row.into_shape((n_heads, d_rope)).unwrap());
        }
        let q_rope_flat = q_rope_reshaped.into_shape((t, n_heads * d_rope)).unwrap();
        let k_rope_flat = k_rope_reshaped.into_shape((t, n_heads * d_rope)).unwrap();

        // Reshape nope parts: [T, n_heads, d_nope]
        let q_nope_3d = q_nope_src.to_shape((t, n_heads, d_nope)).unwrap().to_owned();
        let k_nope_3d = k_nope_src.to_shape((t, n_heads, d_nope)).unwrap().to_owned();

        // Reshape value: [T, n_heads, v_head_dim]
        let v_3d = v_b.to_shape((t, n_heads, mla.v_head_dim)).unwrap().to_owned();

        let scale = 1.0 / ((d_nope + d_rope) as f32).sqrt();
        let mut attn_out = Array3::zeros((t, n_heads, mla.v_head_dim));

        // Causal mask
        let mut mask = Array2::zeros((t, t));
        for i in 0..t {
            for j in (i + 1)..t {
                mask[[i, j]] = f32::NEG_INFINITY;
            }
        }

        for head in 0..n_heads {
            // Extract per-head components
            let q_nope_h = q_nope_3d.slice(s![.., head, ..]); // [T, d_nope]
            let q_rope_h = q_rope_flat.slice(s![.., head * d_rope..(head + 1) * d_rope]); // [T, d_rope]
            let k_nope_h = k_nope_3d.slice(s![.., head, ..]); // [T, d_nope]
            let k_rope_h = k_rope_flat.slice(s![.., head * d_rope..(head + 1) * d_rope]); // [T, d_rope]
            let v_h = v_3d.slice(s![.., head, ..]); // [T, v_head_dim]

            // Score = q_nope @ k_nope^T + q_rope @ k_rope^T  [T, T]
            let scores_nope = q_nope_h.dot(&k_nope_h.t()); // [T, T]
            let scores_rope = q_rope_h.dot(&k_rope_h.t()); // [T, T]
            let scores = (&scores_nope + &scores_rope) * scale;
            let scores_masked = &scores + &mask;

            // Softmax
            let probs = scores_masked.axis_iter(Axis(0))
                .map(|row| engine::softmax(row.as_slice().unwrap()))
                .collect::<Vec<_>>();
            let probs_arr = Array2::from_shape_vec((t, t),
                probs.into_iter().flatten().collect()).unwrap();

            // Weighted sum of V
            let out_h = probs_arr.dot(&v_h);
            attn_out.slice_mut(s![.., head, ..]).assign(&out_h);
        }

        let attn_out_flat = attn_out.to_owned().into_shape((t, n_heads * mla.v_head_dim)).unwrap();
        attn_out_flat.dot(&wo.t())
    }

    /// Standard MHA attention forward for a single layer (Llama/Qwen/Gemma style)
    fn mha_forward_layer(&self, cfg: &Config, h: &ArrayView2<f32>, layer: &LayerWeights, hd: usize) -> Array2<f32> {
        let t = h.nrows();
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let group = n_heads / n_kv_heads.max(1);

        // Convert view to owned for linear projections
        let h_owned = h.to_owned();

        // QKV projections for the entire sequence at once
        let q = engine::linear(&h_owned, &layer.wq); // [T, dim]
        let k = engine::linear(&h_owned, &layer.wk); // [T, dk]
        let v = engine::linear(&h_owned, &layer.wv); // [T, dk]

        // Reshape for multi-head: [T, n_heads, head_dim]
        let mut q = q.to_shape((t, n_heads, hd)).unwrap().to_owned();
        let mut k = k.to_shape((t, n_kv_heads, hd)).unwrap().to_owned();
        let v = v.to_shape((t, n_kv_heads, hd)).unwrap().to_owned();

        // RoPE on Q and K for the full sequence
        for i in 0..t {
            // For each position, reshape to [1, n_heads * hd] for apply_rope_all
            let mut q_row_2d = q.slice(s![i, .., ..]).into_shape((1, n_heads * hd)).unwrap().to_owned();
            engine::apply_rope_all(&mut q_row_2d, n_heads, hd, cfg.rope_theta);
            q.slice_mut(s![i, .., ..]).assign(&q_row_2d.into_shape((n_heads, hd)).unwrap());

            let mut k_row_2d = k.slice(s![i, .., ..]).into_shape((1, n_kv_heads * hd)).unwrap().to_owned();
            engine::apply_rope_all(&mut k_row_2d, n_kv_heads, hd, cfg.rope_theta);
            k.slice_mut(s![i, .., ..]).assign(&k_row_2d.into_shape((n_kv_heads, hd)).unwrap());
        }

        // Transpose to [n_heads, T, hd] for batched matmul
        let q_3d = q.permuted_axes([1, 0, 2]); // [n_heads, T, hd]
        let k_3d = k.permuted_axes([1, 0, 2]); // [n_kv_heads, T, hd]
        let v_3d = v.permuted_axes([1, 0, 2]); // [n_kv_heads, T, hd]

        // GQA: expand K,V from [n_kv_heads, T, hd] -> [n_heads, T, hd]
        let k_exp: Array3<f32> = if group > 1 {
            let k_with_axis = k_3d.insert_axis(Axis(1));
            let broad = k_with_axis.broadcast((n_kv_heads, group, t, hd)).unwrap();
            broad.to_owned().into_shape((n_heads, t, hd)).unwrap()
        } else { k_3d.to_owned() };
        let v_exp: Array3<f32> = if group > 1 {
            let v_with_axis = v_3d.insert_axis(Axis(1));
            let broad = v_with_axis.broadcast((n_kv_heads, group, t, hd)).unwrap();
            broad.to_owned().into_shape((n_heads, t, hd)).unwrap()
        } else { v_3d.to_owned() };

        // Batched causal attention: Q @ K^T / sqrt(d) + mask -> softmax -> @ V
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = Array3::zeros((n_heads, t, hd));

        // Build causal mask once
        let mut mask = Array2::zeros((t, t));
        for i in 0..t {
            for j in (i + 1)..t {
                mask[[i, j]] = f32::NEG_INFINITY;
            }
        }

        for head in 0..n_heads {
            let q_h = q_3d.slice(s![head, .., ..]); // [T, hd]
            let k_h = k_exp.slice(s![head, .., ..]); // [T, hd]
            let v_h = v_exp.slice(s![head, .., ..]); // [T, hd]

            // Scores: [T, T] = Q @ K^T * scale
            let scores = q_h.dot(&k_h.t()) * scale; // [T, T]
            let scores_masked = &scores + &mask; // broadcast mask

            // Softmax per row
            let probs = scores_masked.axis_iter(Axis(0))
                .map(|row| engine::softmax(row.as_slice().unwrap()))
                .collect::<Vec<_>>();
            let probs_arr = Array2::from_shape_vec((t, t), probs.into_iter().flatten().collect()).unwrap();

            // Weighted sum: [T, hd] = probs @ V
            let out_h = probs_arr.dot(&v_h); // [T, hd]
            attn_out.slice_mut(s![head, .., ..]).assign(&out_h);
        }

        // Transpose back to [T, n_heads, hd] and flatten
        let attn_out = attn_out.permuted_axes([1, 0, 2]); // [T, n_heads, hd]
        let attn_out_flat = attn_out.as_standard_layout().into_shape((t, n_heads * hd)).unwrap().into_owned();

        // Output projection
        engine::linear(&attn_out_flat, &layer.wo)
    }
    /// Uses the same QKV projections as MHA but replaces softmax with
    /// linear-time recurrent feature-map attention.
    fn linear_attn_forward_layer(
        &self,
        cfg: &Config,
        h: &ArrayView2<f32>,
        layer: &LayerWeights,
        _hd: usize,
    ) -> Array2<f32> {
        let t = h.nrows();
        let _dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();

        // Convert view to owned for QKV projections
        let h_owned = h.to_owned();

        // QKV projections (same as MHA)
        let q = engine::linear(&h_owned, &layer.wq); // [T, dim]
        let k = engine::linear(&h_owned, &layer.wk); // [T, dim_kv]
        let v = engine::linear(&h_owned, &layer.wv); // [T, dim_kv]

        // Apply RoPE per-token
        let mut q_out = q.clone();
        let mut k_out = k.clone();
        for i in 0..t {
            let q_row = q_out.slice(s![i..i + 1, ..]);
            let mut q_rope = q_row.to_owned();
            engine::apply_rope_all(&mut q_rope, n_heads, head_dim, cfg.rope_theta);
            q_out.slice_mut(s![i..i + 1, ..]).assign(&q_rope);

            let k_row = k_out.slice(s![i..i + 1, ..]);
            let mut k_rope = k_row.to_owned();
            engine::apply_rope_all(&mut k_rope, n_kv_heads, head_dim, cfg.rope_theta);
            k_out.slice_mut(s![i..i + 1, ..]).assign(&k_rope);
        }

        // Run full-sequence linear attention
        linear_attn_forward(
            &q_out, &k_out, &v,
            n_heads, n_kv_heads, head_dim,
            &FeatureFn::Elu1,
            crate::engine::EPS,
        )
    }

    /// Kai assimilation step: minimize prediction surprisal over `target` given
    /// `tokens` as context. Updates the readout (SGD on the output projection,
    /// the exact softmax-gradient step) and performs a Hebbian write of the
    /// target token's embedding toward the context representation. Returns the
    /// surprisal (-log p[target]) before the update.
    pub fn assimilate_step(
        &mut self,
        cfg: &Config,
        tokens: &[usize],
        target: usize,
        lr: f32,
    ) -> f32 {
        let (logits, xf) = self.forward(cfg, tokens);
        let row = logits.nrows() - 1;
        let vocab = cfg.vocab_size;
        let raw: Vec<f32> = (0..vocab).map(|j| logits[[row, j]]).collect();
        let p = engine::softmax(&raw);
        let tgt = target.min(vocab - 1);
        let surprisal = -p[tgt].max(1e-9).ln();

        // exact softmax-gradient on the readout: dL/dW[j,d] = (p_j - 1{tgt})*xf_d
        for j in 0..vocab {
            let g = (p[j] - if j == tgt { 1.0 } else { 0.0 }) * lr;
            for d in 0..cfg.dim {
                self.output[[j, d]] -= g * xf[d];
            }
        }
        // Hebbian write: bind the target token's embedding to the context state
        for d in 0..cfg.dim {
            self.embed[[tgt, d]] += lr * 0.1 * xf[d];
        }
        // GPU weight cache stale after weight write
        #[cfg(feature = "gpu")]
        crate::engine::invalidate_weight_cache();
        surprisal
    }

    /// Standard deviation of the readout weights — a proxy for the model's
    /// uncertainty/breadth (the variance term in Kai's VFE).
    pub fn readout_variance(&self) -> f32 {
        let m = self.output.len() as f32;
        let mean = self.output.iter().sum::<f32>() / m;
        let var = self.output.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / m;
        var.sqrt()
    }

    /// Inject Kai's assimilated prior (the Phase-A attractor, projected into
    /// hidden space) as a constant bias across the token-embedding table, so
    /// the decoder literally starts from the 10 architectures Kai ingested.
    pub fn seed_prior(&mut self, prior: &[f32], scale: f32) {
        let dim = prior.len().min(self.embed.ncols());
        for r in 0..self.embed.nrows() {
            for d in 0..dim {
                self.embed[[r, d]] += scale * prior[d];
            }
        }
    }

    /// Build a decoder from a real llama-style GGUF (weights stored [out, in]).
    /// Embedding is [dim, vocab] -> transposed to [vocab, dim]; lm_head is
    /// `output.weight` if present, else tied to the embedding.
    /// Also loads MoE and MLA weights if present in GGUF and config enables them.
    pub fn from_gguf(map: &TensorMap, cfg: &Config) -> Result<Weights, String> {
        let embed = mat_t(map, "token_embd.weight");
        // engine::linear expects (output, input): y = x @ W^T.
        // If GGUF stores as (input, output) (detected by nrows != expected_out), transpose.
        let fix = |w: Array2<f32>, expected_out: usize| -> Array2<f32> {
            if w.nrows() != expected_out { w.t().to_owned() } else { w }
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let wq = fix(mat(map, &format!("blk.{i}.attn_q.weight")), cfg.dim);
            let wk = fix(mat(map, &format!("blk.{i}.attn_k.weight")), cfg.dim_kv());
            let wv = fix(mat(map, &format!("blk.{i}.attn_v.weight")), cfg.dim_kv());
            let wo = fix(mat(map, &format!("blk.{i}.attn_output.weight")), cfg.dim);
            let w1 = fix(mat(map, &format!("blk.{i}.ffn_gate.weight")), cfg.intermediate);
            let w2 = fix(mat(map, &format!("blk.{i}.ffn_down.weight")), cfg.dim);
            let w3 = fix(mat(map, &format!("blk.{i}.ffn_up.weight")), cfg.intermediate);
            let attn_norm = vec1(map, &format!("blk.{i}.attn_norm.weight"));
            let ffn_norm = vec1(map, &format!("blk.{i}.ffn_norm.weight"));

            // MoE weights (optional, DeepSeek-V2 style)
            let (moe_shared_w1, moe_shared_w2, moe_shared_w3) = if cfg.is_moe() {
                let shared_w1 = map.get(&format!("blk.{i}.ffn_shared_gate.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                let shared_w2 = map.get(&format!("blk.{i}.ffn_shared_down.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                let shared_w3 = map.get(&format!("blk.{i}.ffn_shared_up.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.intermediate));
                (shared_w1, shared_w2, shared_w3)
            } else { (None, None, None) };

            let (moe_expert_w1, moe_expert_w2, moe_expert_w3, moe_router_weight, moe_router_bias) = if cfg.is_moe() {
                let n_experts = cfg.moe.n_experts;
                let mut expert_w1 = Vec::with_capacity(n_experts);
                let mut expert_w2 = Vec::with_capacity(n_experts);
                let mut expert_w3 = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    expert_w1.push(fix(mat(map, &format!("blk.{i}.ffn_expert.{e}.gate.weight")), cfg.intermediate));
                    expert_w2.push(fix(mat(map, &format!("blk.{i}.ffn_expert.{e}.down.weight")), cfg.dim));
                    expert_w3.push(fix(mat(map, &format!("blk.{i}.ffn_expert.{e}.up.weight")), cfg.intermediate));
                }
                let router_weight = fix(mat(map, &format!("blk.{i}.ffn_router.weight")), cfg.moe.n_experts);
                let router_bias = map.get(&format!("blk.{i}.ffn_router.bias"))
                    .map(|(_s, d)| Array1::from_vec(d.clone()));
                (Some(expert_w1), Some(expert_w2), Some(expert_w3), Some(router_weight), router_bias)
            } else { (None, None, None, None, None) };

            // MLA weights (optional, DeepSeek-V2 style)
            let (mla_wq_a, mla_wq_b, mla_wk_v_a, mla_wk_b, mla_wv_b, mla_wq_rope, mla_wo) = if cfg.is_mla() {
                let wq_a = map.get(&format!("blk.{i}.attn_q_a.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.mla.q_lora_rank));
                let wq_b = map.get(&format!("blk.{i}.attn_q_b.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                let wk_v_a = map.get(&format!("blk.{i}.attn_kv_a.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.mla.kv_lora_rank));
                let wk_b = map.get(&format!("blk.{i}.attn_k_b.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                let wv_b = map.get(&format!("blk.{i}.attn_v_b.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                let wq_rope = map.get(&format!("blk.{i}.attn_q_rope.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                let wo = map.get(&format!("blk.{i}.attn_output.weight"))
                    .map(|(s, d)| fix(Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap(), cfg.dim));
                (wq_a, wq_b, wk_v_a, wk_b, wv_b, wq_rope, wo)
            } else { (None, None, None, None, None, None, None) };

            layers.push(LayerWeights {
                wq, wk, wv, wo, w1, w2, w3, attn_norm, ffn_norm,
                moe_shared_w1, moe_shared_w2, moe_shared_w3,
                moe_expert_w1, moe_expert_w2, moe_expert_w3,
                moe_router_weight, moe_router_bias,
                mla_wq_a, mla_wq_b, mla_wk_v_a, mla_wk_b, mla_wv_b, mla_wq_rope, mla_wo,
                moe_gate_exps: None, moe_down_exps: None, moe_up_exps: None, moe_gate_inp: None,
                moe_shared_gate: None, moe_shared_down: None, moe_shared_up: None,
                mla_q: None, mla_kv_a_mqa: None, mla_kv_b: None, mla_kv_a_norm: None,
            });
        }
        let final_norm = vec1(map, "output_norm.weight");
        let output = if let Some((shape, d)) = map.get("output.weight") {
            let w = Array2::from_shape_vec((shape[0], shape[1]), d.clone()).unwrap();
            if w.nrows() != cfg.vocab_size { w.t().to_owned() } else { w }
        } else {
            embed.clone()
        };
        Ok(Weights { embed, layers, final_norm, output })
    }

    /// Compute attention curvature (drift from equilibrium) per layer.
    ///
    /// From the Tonal Collapse framework: the attention manifold has metric
    /// `g_{ij} = 1 - a_{ij}` where `a_{ij}` is the attention weight.
    /// High curvature = high ||∇a|| = drift from equilibrium (high VFE).
    /// Returns (per-layer curvature, total curvature).
    pub fn attention_curvature(&self, cfg: &Config, tokens: &[usize]) -> (Vec<f32>, f32) {
        let t = tokens.len().min(cfg.max_seq);
        let toks: Vec<usize> = tokens.iter().take(t).cloned().collect();
        let t = toks.len();
        let hd = cfg.head_dim();

        let mut x = Array2::zeros((t, cfg.dim));
        for (i, &tok) in toks.iter().enumerate() {
            let tok = tok.min(cfg.vocab_size - 1);
            for d in 0..cfg.dim {
                x[[i, d]] = self.embed[[tok, d]];
            }
        }

        let mut layer_curvatures = Vec::with_capacity(cfg.n_layers);

        for layer in &self.layers {
            // --- attention (pre-norm) ---
            let h = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);
            let q = engine::linear(&h, &layer.wq); // (T, dim)
            let k = engine::linear(&h, &layer.wk); // (T, dk)

            engine::apply_rope_all(&mut q.to_owned(), cfg.n_heads, hd, cfg.rope_theta);
            engine::apply_rope_all(&mut k.to_owned(), cfg.n_kv_heads, hd, cfg.rope_theta);

            let mut layer_curvature = 0.0f32;
            let group = cfg.n_heads / cfg.n_kv_heads.max(1);

            // Compute attention weights and curvature per head
            for head in 0..cfg.n_heads {
                let kvh = head / group;
                for i in 0..t {
                    for j in 0..=i {
                        let mut dot = 0.0;
                        for d in 0..hd {
                            dot += q[[i, head * hd + d]] * k[[j, kvh * hd + d]];
                        }
                        let score = dot / (hd as f32).sqrt();
                        // Curvature contribution: 1 - softmax(score) ~ 1 - a_{ij}
                        // Accumulate over context
                        let a_ij = 1.0 / (1.0 + (-score).exp()); // sigmoid approx for single score
                        layer_curvature += (1.0 - a_ij).abs();
                    }
                }
            }

            // Normalize by number of attention positions
            let n_positions = (t * (t + 1) / 2) as f32 * cfg.n_heads as f32;
            if n_positions > 0.0 {
                layer_curvature /= n_positions;
            }
            layer_curvatures.push(layer_curvature);

            // --- continue forward pass to update x for next layer ---
            let v = engine::linear(&h, &layer.wv); // (T, dk)

            let mut attn_out = Array2::zeros((t, cfg.dim));
            let group = cfg.n_heads / cfg.n_kv_heads.max(1);
            for head in 0..cfg.n_heads {
                let kvh = head / group;
                for i in 0..t {
                    let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
                    for j in 0..=i {
                        let mut dot = 0.0;
                        for d in 0..hd {
                            dot += q[[i, head * hd + d]] * k[[j, kvh * hd + d]];
                        }
                        scores.push(dot / (hd as f32).sqrt());
                    }
                    let probs = engine::softmax(&scores);
                    for d in 0..hd {
                        let mut acc = 0.0;
                        for (j, &sj) in probs.iter().enumerate() {
                            acc += sj * v[[j, kvh * hd + d]];
                        }
                        attn_out[[i, head * hd + d]] = acc;
                    }
                }
            }
            let attn_proj = engine::linear(&attn_out, &layer.wo);
            x = &x + &attn_proj;

            // ffn
            let h2 = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
            let gate = engine::silu(&engine::linear(&h2, &layer.w1));
            let up = engine::linear(&h2, &layer.w3);
            let ff = engine::linear(&(&gate * &up), &layer.w2);
            x = &x + &ff;
        }

        let total_curvature: f32 = layer_curvatures.iter().sum();
        (layer_curvatures, total_curvature)
    }

    /// Forward pass with caching for backprop. Stores all intermediate activations.
    pub fn forward_with_cache(&self, cfg: &Config, tokens: &[usize]) -> ForwardCache {
        let t = tokens.len().min(cfg.max_seq);
        let toks: Vec<usize> = tokens.iter().take(t).cloned().collect();
        let t = toks.len();
        let hd = cfg.head_dim();

        let mut x = Array2::zeros((t, cfg.dim));
        for (i, &tok) in toks.iter().enumerate() {
            let tok = tok.min(cfg.vocab_size - 1);
            for d in 0..cfg.dim {
                x[[i, d]] = self.embed[[tok, d]];
            }
        }
        let mut layer_caches = Vec::with_capacity(cfg.n_layers);

        for layer in &self.layers {
            #[cfg(feature = "gpu")]
            {
                // GPU-fused forward with intermediate extraction for backward.
                // Intermediates stay on GPU — stored in GpuLayerCache for backward.
                use crate::engine::{fused_attention_with_cache, fused_ffn_with_cache};

                let (attn_proj, h_attn_gpu, q_rope_gpu, k_rope_gpu, v_gpu, attn_out_gpu) =
                    fused_attention_with_cache(
                        &x, layer.attn_norm.as_slice().unwrap(),
                        &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                        cfg.n_heads, cfg.n_kv_heads, hd, cfg.rope_theta, engine::EPS,
                    );
                x = &x + &attn_proj;

                let (ffn_out, h_ffn_gpu, gate_gpu, gate_silu_gpu, up_gpu) = fused_ffn_with_cache(
                    &x, layer.ffn_norm.as_slice().unwrap(),
                    &layer.w1, &layer.w3, &layer.w2, engine::EPS,
                );
                let x_out_cpu = x.clone();
                x = &x + &ffn_out;

                // Download intermediates for CPU backward path fallback.
                use crate::engine::gpu;
                let h_attn = gpu::from_tensor(&h_attn_gpu);
                let q_rope = gpu::from_tensor(&q_rope_gpu);
                let k_rope = gpu::from_tensor(&k_rope_gpu);
                let v = gpu::from_tensor(&v_gpu);
                let attn_out = gpu::from_tensor(&attn_out_gpu);
                let h_ffn = gpu::from_tensor(&h_ffn_gpu);
                let gate = gpu::from_tensor(&gate_gpu);
                let gate_silu = gpu::from_tensor(&gate_silu_gpu);
                let up = gpu::from_tensor(&up_gpu);

                let gpu_cache = GpuLayerCache {
                    h_attn: h_attn_gpu,
                    q_rope: q_rope_gpu,
                    k_rope: k_rope_gpu,
                    v: v_gpu,
                    attn_out: attn_out_gpu,
                    h_ffn: h_ffn_gpu,
                    gate: gate_gpu,
                    gate_silu: gate_silu_gpu,
                    up: up_gpu,
                    x_out: gpu::to_tensor(&x_out_cpu),
                };

                layer_caches.push(LayerCache {
                    h_attn, v, q_rope, k_rope, attn_out,
                    h_ffn, gate, up, gate_silu, x_out: x_out_cpu,
                    gpu: Some(gpu_cache),
                });
                // Re-upload weights per layer: the GPU weight cache is keyed by the
                // weight *pointer*, and apply_gradients mutates weights in place
                // (same pointer) — so a cached tensor becomes STALE after a gradient
                // step. Clearing here guarantees the next layer (and next iter's
                // forward) re-uploads the live weights instead of serving the
                // pre-update tensor, which would otherwise "reset" learning.
                crate::engine::invalidate_weight_cache();
            }
            #[cfg(not(feature = "gpu"))]
            {
                let h_attn = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);
                let q = engine::linear(&h_attn, &layer.wq);
                let k = engine::linear(&h_attn, &layer.wk);
                let v = engine::linear(&h_attn, &layer.wv);
                let mut q_rope = q.clone();
                let mut k_rope = k.clone();
                engine::apply_rope_all(&mut q_rope, cfg.n_heads, hd, cfg.rope_theta);
                engine::apply_rope_all(&mut k_rope, cfg.n_kv_heads, hd, cfg.rope_theta);

                let group = cfg.n_heads / cfg.n_kv_heads.max(1);
                let mut attn_out = Array2::zeros((t, cfg.dim));
                // Parallelize over heads: each head writes a disjoint column block
                // (head*hd .. (head+1)*hd) of attn_out, so no data races.
                use rayon::prelude::*;
                let head_blocks: Vec<(usize, Vec<f32>)> = (0..cfg.n_heads)
                    .into_par_iter()
                    .map(|head| {
                        let kvh = head / group;
                        let mut block = vec![0.0f32; t * hd];
                        for i in 0..t {
                            let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
                            for j in 0..=i {
                                let mut dot = 0.0;
                                for d in 0..hd {
                                    dot += q_rope[[i, head * hd + d]] * k_rope[[j, kvh * hd + d]];
                                }
                                scores.push(dot / (hd as f32).sqrt());
                            }
                            let probs = engine::softmax(&scores);
                            for d in 0..hd {
                                let mut acc = 0.0;
                                for (j, &sj) in probs.iter().enumerate() {
                                    acc += sj * v[[j, kvh * hd + d]];
                                }
                                block[i * hd + d] = acc;
                            }
                        }
                        (head, block)
                    })
                    .collect();
                for (head, block) in head_blocks {
                    for i in 0..t {
                        for d in 0..hd {
                            attn_out[[i, head * hd + d]] = block[i * hd + d];
                        }
                    }
                }
                x = &x + &engine::linear(&attn_out, &layer.wo);

                let h_ffn = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
                let gate = engine::linear(&h_ffn, &layer.w1);
                let gate_silu = engine::silu(&gate);
                let up = engine::linear(&h_ffn, &layer.w3);
                let x_out = x.clone();
                x = &x + &engine::linear(&(&gate_silu * &up), &layer.w2);

                layer_caches.push(LayerCache {
                    h_attn, v, q_rope, k_rope, attn_out,
                    h_ffn, gate, up, gate_silu, x_out,
                });
            }
        }

        let xf = engine::rmsnorm_rows(&x, &self.final_norm, engine::EPS);
        let logits = engine::linear(&xf, &self.output);
        ForwardCache { layers: layer_caches, xf, logits }
    }

    /// Backward pass: compute gradients for all parameters.
    /// d_logits: upstream gradient of loss w.r.t. logits [T, vocab].
    /// Returns accumulated gradients.
    pub fn backward(&self, cfg: &Config, cache: &ForwardCache, d_logits: &Array2<f32>, tokens: &[usize], freeze: &FreezeConfig) -> Gradients {
        let t = cache.logits.nrows();
        let hd = cfg.head_dim();

        // Allocate gradient accumulators (skip frozen params)
        let mut d_embed: Option<Array2<f32>> = if freeze.freeze_embed { None } else { Some(Array2::zeros((cfg.vocab_size, cfg.dim))) };
        let mut d_final_norm = Array1::zeros(cfg.dim);
        let mut d_output: Option<Array2<f32>> = if freeze.freeze_output { None } else { Some(Array2::zeros(self.output.raw_dim())) };
        let mut layer_grads: Vec<Option<LayerGrads>> = Vec::with_capacity(cfg.n_layers);

        // --- Gradient through output projection and final RMSNorm ---
        // logits = xf @ output^T  =>  d_output += d_logits^T @ xf, d_xf = d_logits @ output
        let d_xf_raw = d_logits.dot(&self.output); // [T, dim]

        // d_output = d_logits^T @ xf = [vocab, T] @ [T, dim] = [vocab, dim]
        if let Some(ref mut d_out) = d_output {
            *d_out = engine::grad_weight(&d_logits, &cache.xf);
        }

        // RMSNorm backward
        // xf[i,d] = x[i,d] * inv_std[i] * fnorm[d]
        // dL/dx[i,d] = dL/dxf[i,d] * inv_std[i] * fnorm[d] + correction from mean_sq
        // dL/dfnorm[d] = sum_i dL/dxf[i,d] * x[i,d] * inv_std[i]
        let mut d_x: Array2<f32> = Array2::zeros((t, cfg.dim));
        for i in 0..t {
            let row: Vec<f32> = (0..cfg.dim).map(|d| cache.xf[[i, d]]).collect();
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cfg.dim as f32;
            let inv_std = 1.0 / (mean_sq + engine::EPS).sqrt();
            for d in 0..cfg.dim {
                d_final_norm[d] += d_xf_raw[[i, d]] * (cache.xf[[i, d]] / self.final_norm[d].max(1e-9));
            }
            // Full RMSNorm backward with gradient through norm
            for d in 0..cfg.dim {
                let m = inv_std * self.final_norm[d];
                d_x[[i, d]] = d_xf_raw[[i, d]] * m;
            }
            // Norm correction term
            let mut sum_term = 0.0;
            for d in 0..cfg.dim {
                sum_term += d_xf_raw[[i, d]] * self.final_norm[d] * (cache.xf[[i, d]] / self.final_norm[d].max(1e-9));
            }
            sum_term *= -inv_std.powi(3) / cfg.dim as f32;
            for d in 0..cfg.dim {
                d_x[[i, d]] += sum_term * row[d];
            }
        }

        // Process layers in reverse
        for li in (0..cfg.n_layers).rev() {
            let layer = &self.layers[li];
            let lc = &cache.layers[li];
            let is_frozen = li < freeze.freeze_layers;

            // Weight gradients are created by engine::grad_weight/gpu_matmul below.
            // These are accumulated here (unused gradient entries go out of scope
            // for frozen layers).
            let mut d_attn_norm = Array1::zeros(cfg.dim);
            let mut d_ffn_norm = Array1::zeros(cfg.dim);

            // --- FFN residual ---
            // x = x_out + ff, so d_ff = d_x (upstream), and d_x_for_ffn = d_x (residual)
            let d_ff = d_x.clone();
            let d_x_after_ffn = d_x.clone();

            // --- FFN output: ff = gate_silu * up @ w2^T ---
            let in_ff = &lc.gate_silu * &lc.up; // [T, intermediate]
            let d_w2 = engine::grad_weight(&d_ff, &in_ff); // d_ff.T @ in_ff = [dim, intermediate]
            let d_in_ff = engine::gpu_matmul(&d_ff, &layer.w2); // d_ff @ w2 = [T, intermediate]

            // --- gate_silu * up (element-wise multiply) ---
            // d_up = d_in_ff * gate_silu, d_gate_silu = d_in_ff * up
            let mut d_up: Array2<f32> = Array2::zeros((t, cfg.intermediate));
            let mut d_gate_silu: Array2<f32> = Array2::zeros((t, cfg.intermediate));
            for i in 0..t {
                for j in 0..cfg.intermediate {
                    d_up[[i, j]] = d_in_ff[[i, j]] * lc.gate_silu[[i, j]];
                    d_gate_silu[[i, j]] = d_in_ff[[i, j]] * lc.up[[i, j]];
                }
            }

            // --- SiLU backward ---
            // d_gate = d_gate_silu * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
            let mut d_gate: Array2<f32> = Array2::zeros((t, cfg.intermediate));
            for i in 0..t {
                for j in 0..cfg.intermediate {
                    let g = lc.gate[[i, j]];
                    let s = 1.0 / (1.0 + (-g).exp());
                    let ds = s * (1.0 - s);
                    d_gate[[i, j]] = d_gate_silu[[i, j]] * (s + g * ds);
                }
            }

            // --- w3: up = h_ffn @ w3^T ---
            let d_w3 = engine::grad_weight(&d_up, &lc.h_ffn); // d_up.T @ h_ffn = [intermediate, dim]
            let d_x_up = engine::gpu_matmul(&d_up, &layer.w3); // d_up @ w3 = [T, dim]

            // --- w1: gate = h_ffn @ w1^T ---
            let d_w1 = engine::grad_weight(&d_gate, &lc.h_ffn); // d_gate.T @ h_ffn = [intermediate, dim]
            let d_x_gate = engine::gpu_matmul(&d_gate, &layer.w1); // d_gate @ w1 = [T, dim]

            // --- FFN RMSNorm backward ---
            let mut d_x_ffn_norm: Array2<f32> = Array2::zeros((t, cfg.dim));
            let d_x_ffn_total = &d_x_up + &d_x_gate;
            for i in 0..t {
                let row: Vec<f32> = (0..cfg.dim).map(|d| lc.h_ffn[[i, d]]).collect();
                let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cfg.dim as f32;
                let inv_std = 1.0 / (mean_sq + engine::EPS).sqrt();
                for d in 0..cfg.dim {
                    d_ffn_norm[d] += d_x_ffn_total[[i, d]] * (lc.h_ffn[[i, d]] / layer.ffn_norm[d].max(1e-9));
                    d_x_ffn_norm[[i, d]] = d_x_ffn_total[[i, d]] * inv_std * layer.ffn_norm[d];
                }
                let mut sum_term = 0.0;
                for d in 0..cfg.dim {
                    sum_term += d_x_ffn_total[[i, d]] * layer.ffn_norm[d] * (lc.h_ffn[[i, d]] / layer.ffn_norm[d].max(1e-9));
                }
                sum_term *= -inv_std.powi(3) / cfg.dim as f32;
                for d in 0..cfg.dim {
                    d_x_ffn_norm[[i, d]] += sum_term * row[d];
                }
            }

            // --- FFN residual add: x = x_after_attn + attn_proj, d_x_upstream = d_x_after_ffn + d_x_ffn_norm
            let d_x_after_attn = &d_x_after_ffn + &d_x_ffn_norm;

            // --- wo: attn_proj = attn_out @ wo^T ---
            let d_attn_proj = d_x_after_attn.clone();
            let d_wo = engine::grad_weight(&d_attn_proj, &lc.attn_out); // d_attn_proj.T @ attn_out = [dim, dim]
            let d_attn_out = engine::gpu_matmul(&d_attn_proj, &layer.wo); // d_attn_proj @ wo = [T, dim]

            // --- Attention: attn_out = softmax(QK^T/sqrt(hd)) @ V ---
            // Pre-allocate for CPU path; GPU path overwrites immediately.
            // The #[allow] silences "assigned but never read" on the zero init.
            #[allow(unused_assignments)]
            let mut d_q: Array2<f32> = Array2::zeros((t, cfg.n_heads * hd));
            #[allow(unused_assignments)]
            let mut d_k: Array2<f32> = Array2::zeros((t, cfg.n_kv_heads * hd));
            #[allow(unused_assignments)]
            let mut d_v: Array2<f32> = Array2::zeros((t, cfg.dim_kv()));

            #[cfg(feature = "gpu")]
            {
                // Batched GPU attention backward using cached GPU tensors
                // (no CPU re-upload of Q, K, V from layer cache).
                if let Some(ref gpu_cache) = lc.gpu {
                    use crate::engine::gpu_attn_backward_batched_from_cache;
                    let (dq_gpu, dk_gpu, dv_gpu) = gpu_attn_backward_batched_from_cache(
                        &gpu_cache.q_rope, &gpu_cache.k_rope, &gpu_cache.v,
                        &d_attn_out,
                        cfg.n_heads, cfg.n_kv_heads, hd,
                    );
                    d_q = dq_gpu;
                    d_k = dk_gpu;
                    d_v = dv_gpu;
                } else {
                    // Fallback: re-upload from CPU arrays
                    use crate::engine::gpu_attn_backward_batched;
                    let (dq_gpu, dk_gpu, dv_gpu) = gpu_attn_backward_batched(
                        &lc.q_rope, &lc.k_rope, &lc.v, &d_attn_out,
                        cfg.n_heads, cfg.n_kv_heads, hd,
                    );
                    d_q = dq_gpu;
                    d_k = dk_gpu;
                    d_v = dv_gpu;
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                // CPU per-head nested-loop attention backward (parallel over heads)
                let group = cfg.n_heads / cfg.n_kv_heads.max(1);
                use rayon::prelude::*;
                // Each head computes its OWN partial d_q (head*hd cols, disjoint) and
                // partial d_k/d_v (kvh*hd cols). GQA means heads can share a kvh, so we
                // accumulate d_k/d_v into per-head local buffers and merge after the
                // parallel section to avoid data races.
                let partials: Vec<(usize, usize, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..cfg.n_heads)
                    .into_par_iter()
                    .map(|head| {
                        let kvh = head / group;
                        let mut pd_q = vec![0.0f32; t * hd];
                        let mut pd_k = vec![0.0f32; t * hd];
                        let mut pd_v = vec![0.0f32; t * hd];
                        for i in 0..t {
                            let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
                            for j in 0..=i {
                                let mut dot = 0.0;
                                for d in 0..hd {
                                    dot += lc.q_rope[[i, head * hd + d]] * lc.k_rope[[j, kvh * hd + d]];
                                }
                                scores.push(dot / (hd as f32).sqrt());
                            }
                            let probs = engine::softmax(&scores);

                            for j in 0..=i {
                                for d in 0..hd {
                                    pd_v[j * hd + d] += d_attn_out[[i, head * hd + d]] * probs[j];
                                }
                            }

                            let mut d_scores = vec![0.0; i + 1];
                            for j in 0..=i {
                                for d in 0..hd {
                                    d_scores[j] += d_attn_out[[i, head * hd + d]] * lc.v[[j, kvh * hd + d]];
                                }
                            }

                            let mut d_score_raw = vec![0.0; i + 1];
                            for k in 0..=i {
                                let mut s = 0.0;
                                for j in 0..=i {
                                    let delta = if k == j { 1.0 } else { 0.0 };
                                    s += d_scores[j] * probs[j] * (delta - probs[k]);
                                }
                                d_score_raw[k] = s / (hd as f32).sqrt();
                            }

                            for j in 0..=i {
                                for d in 0..hd {
                                    pd_q[i * hd + d] += d_score_raw[j] * lc.k_rope[[j, kvh * hd + d]];
                                    pd_k[j * hd + d] += d_score_raw[j] * lc.q_rope[[i, head * hd + d]];
                                }
                            }
                        }
                        (head, kvh, pd_q, pd_k, pd_v)
                    })
                    .collect();
                for (head, kvh, pd_q, pd_k, pd_v) in partials {
                    for i in 0..t {
                        for d in 0..hd {
                            d_q[[i, head * hd + d]] += pd_q[i * hd + d];
                            d_k[[i, kvh * hd + d]] += pd_k[i * hd + d];
                            d_v[[i, kvh * hd + d]] += pd_v[i * hd + d];
                        }
                    }
                }
            }

            // --- RoPE backward (inverse rotation) ---
            engine::apply_rope_all(&mut d_q, cfg.n_heads, hd, cfg.rope_theta);
            engine::apply_rope_all(&mut d_k, cfg.n_kv_heads, hd, cfg.rope_theta);

            // --- Q projection: q = h_attn @ wq^T ---
            let d_wq = engine::grad_weight(&d_q, &lc.h_attn); // d_q.T @ h_attn = [dim, dim]
            let d_h_q = engine::gpu_matmul(&d_q, &layer.wq); // d_q @ wq = [T, dim]

            // --- K projection: k = h_attn @ wk^T ---
            let d_wk = engine::grad_weight(&d_k, &lc.h_attn); // d_k.T @ h_attn = [dk, dim]
            let d_h_k = engine::gpu_matmul(&d_k, &layer.wk); // d_k @ wk = [T, dim]

            // --- V projection: v = h_attn @ wv^T ---
            let d_wv = engine::grad_weight(&d_v, &lc.h_attn); // d_v.T @ h_attn = [dk, dim]
            let d_h_v = engine::gpu_matmul(&d_v, &layer.wv); // d_v @ wv = [T, dim]

            // --- Attention RMSNorm backward ---
            let d_x_attn = &d_h_q + &d_h_k + &d_h_v;
            let mut d_x_attn_norm: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                let row: Vec<f32> = (0..cfg.dim).map(|d| lc.h_attn[[i, d]]).collect();
                let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cfg.dim as f32;
                let inv_std = 1.0 / (mean_sq + engine::EPS).sqrt();
                for d in 0..cfg.dim {
                    d_attn_norm[d] += d_x_attn[[i, d]] * (lc.h_attn[[i, d]] / layer.attn_norm[d].max(1e-9));
                    d_x_attn_norm[[i, d]] = d_x_attn[[i, d]] * inv_std * layer.attn_norm[d];
                }
                let mut sum_term = 0.0;
                for d in 0..cfg.dim {
                    sum_term += d_x_attn[[i, d]] * layer.attn_norm[d] * (lc.h_attn[[i, d]] / layer.attn_norm[d].max(1e-9));
                }
                sum_term *= -inv_std.powi(3) / cfg.dim as f32;
                for d in 0..cfg.dim {
                    d_x_attn_norm[[i, d]] += sum_term * row[d];
                }
            }

            // Residual: x = x_in + attn_proj + ff
            // d_x_in = d_x_after_attn (residual) + d_x_attn_norm (attention path)
            d_x = &d_x_after_attn + &d_x_attn_norm;

            if is_frozen {
                layer_grads.push(None);
            } else {
                layer_grads.push(Some(LayerGrads {
                    wq: d_wq,
                    wk: d_wk,
                    wv: d_wv,
                    wo: d_wo,
                    w1: d_w1,
                    w2: d_w2,
                    w3: d_w3,
                    attn_norm: d_attn_norm,
                    ffn_norm: d_ffn_norm,
                }));
            }
        }

        layer_grads.reverse();

        // Gradient through embedding lookup
        if let Some(ref mut d_emb) = d_embed {
            for (i, &tok) in tokens.iter().enumerate() {
                if i >= t { break; }
                let tok = tok.min(cfg.vocab_size - 1);
                for d in 0..cfg.dim {
                    d_emb[[tok, d]] += d_x[[i, d]];
                }
            }
        }

        // Sanitize all gradients (replace NaN/Inf with 0)
        if let Some(ref mut d_emb) = d_embed { for v in d_emb.iter_mut() { sanitize_in_place(v); } }
        if let Some(ref mut d_out) = d_output { for v in d_out.iter_mut() { sanitize_in_place(v); } }
        for v in d_final_norm.iter_mut() { sanitize_in_place(v); }
        for lg in &mut layer_grads {
            if let Some(ref mut lg) = lg {
                for v in lg.wq.iter_mut() { sanitize_in_place(v); }
                for v in lg.wk.iter_mut() { sanitize_in_place(v); }
                for v in lg.wv.iter_mut() { sanitize_in_place(v); }
                for v in lg.wo.iter_mut() { sanitize_in_place(v); }
                for v in lg.w1.iter_mut() { sanitize_in_place(v); }
                for v in lg.w2.iter_mut() { sanitize_in_place(v); }
                for v in lg.w3.iter_mut() { sanitize_in_place(v); }
                for v in lg.attn_norm.iter_mut() { sanitize_in_place(v); }
                for v in lg.ffn_norm.iter_mut() { sanitize_in_place(v); }
            }
        }

        Gradients {
            embed: d_embed,
            layers: layer_grads,
            final_norm: d_final_norm,
            output: d_output,
        }
    }

    /// Apply accumulated gradients with learning rate.
    /// Skips frozen (None) gradient entries.
    pub fn apply_gradients(&mut self, grads: &Gradients, lr: f32) {
        let sanitize_lr = |v: f32| lr * sanitize(v);
        if let Some(ref g_emb) = grads.embed {
            for d in 0..self.embed.nrows() {
                for j in 0..self.embed.ncols() {
                    self.embed[[d, j]] -= sanitize_lr(g_emb[[d, j]]);
                }
            }
        }
        if let Some(ref g_out) = grads.output {
            for d in 0..self.output.nrows() {
                for j in 0..self.output.ncols() {
                    self.output[[d, j]] -= sanitize_lr(g_out[[d, j]]);
                }
            }
        }
        for d in 0..self.final_norm.len() {
            self.final_norm[d] -= sanitize_lr(grads.final_norm[d]);
        }
        for (li, layer) in self.layers.iter_mut().enumerate() {
            if let Some(ref lg) = grads.layers[li] {
                for d in 0..layer.wq.nrows() {
                    for j in 0..layer.wq.ncols() {
                        layer.wq[[d, j]] -= sanitize_lr(lg.wq[[d, j]]);
                    }
                }
                for d in 0..layer.wk.nrows() {
                    for j in 0..layer.wk.ncols() {
                        layer.wk[[d, j]] -= sanitize_lr(lg.wk[[d, j]]);
                    }
                }
                for d in 0..layer.wv.nrows() {
                    for j in 0..layer.wv.ncols() {
                        layer.wv[[d, j]] -= sanitize_lr(lg.wv[[d, j]]);
                    }
                }
                for d in 0..layer.wo.nrows() {
                    for j in 0..layer.wo.ncols() {
                        layer.wo[[d, j]] -= sanitize_lr(lg.wo[[d, j]]);
                    }
                }
                for d in 0..layer.w1.nrows() {
                    for j in 0..layer.w1.ncols() {
                        layer.w1[[d, j]] -= sanitize_lr(lg.w1[[d, j]]);
                    }
                }
                for d in 0..layer.w2.nrows() {
                    for j in 0..layer.w2.ncols() {
                        layer.w2[[d, j]] -= sanitize_lr(lg.w2[[d, j]]);
                    }
                }
                for d in 0..layer.w3.nrows() {
                    for j in 0..layer.w3.ncols() {
                        layer.w3[[d, j]] -= sanitize_lr(lg.w3[[d, j]]);
                    }
                }
                for d in 0..layer.attn_norm.len() {
                    layer.attn_norm[d] -= sanitize_lr(lg.attn_norm[d]);
                }
                for d in 0..layer.ffn_norm.len() {
                    layer.ffn_norm[d] -= sanitize_lr(lg.ffn_norm[d]);
                }
            }
        }
        // GPU weight cache is now stale — invalidate so next forward re-uploads
        #[cfg(feature = "gpu")]
        crate::engine::invalidate_weight_cache();



}

}

/// impl Weights ends here — free functions follow

/// Run the forward pass for a single layer, storing intermediates needed for backprop.
pub fn forward_layer(lw: &LayerWeights, cfg: &Config, x: &Array2<f32>) -> LayerCache {
    let t = x.nrows();
    let hd = cfg.head_dim();
    let dim = cfg.dim;

    let h_attn = engine::rmsnorm_rows(x, &lw.attn_norm, engine::EPS);
    let q = engine::linear(&h_attn, &lw.wq);
    let k = engine::linear(&h_attn, &lw.wk);
    let v = engine::linear(&h_attn, &lw.wv);
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    engine::apply_rope_all(&mut q_rope, cfg.n_heads, hd, cfg.rope_theta);
    engine::apply_rope_all(&mut k_rope, cfg.n_kv_heads, hd, cfg.rope_theta);

    // Causal self-attention
    let group = cfg.n_heads / cfg.n_kv_heads.max(1);
    let mut attn_out = Array2::zeros((t, dim));
    for head in 0..cfg.n_heads {
        let kvh = head / group;
        for i in 0..t {
            let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
            for j in 0..=i {
                let mut dot = 0.0;
                for d in 0..hd {
                    dot += q_rope[[i, head * hd + d]] * k_rope[[j, kvh * hd + d]];
                }
                scores.push(dot / (hd as f32).sqrt());
            }
            let probs = engine::softmax(&scores);
            for d in 0..hd {
                let mut acc = 0.0;
                for (j, &sj) in probs.iter().enumerate() {
                    acc += sj * v[[j, kvh * hd + d]];
                }
                attn_out[[i, head * hd + d]] = acc;
            }
        }
    }

    // FFN with SwiGLU
    let x_after_attn = x.to_owned() + &engine::linear(&attn_out, &lw.wo);
    let h_ffn = engine::rmsnorm_rows(&x_after_attn, &lw.ffn_norm, engine::EPS);
    let gate = engine::linear(&h_ffn, &lw.w1);
    let gate_silu = engine::silu(&gate);
    let up = engine::linear(&h_ffn, &lw.w3);
    let x_out = &x_after_attn + &engine::linear(&(&gate_silu * &up), &lw.w2);

    LayerCache {
        h_attn, v, q_rope, k_rope,
        attn_out, h_ffn,
        gate, up, gate_silu, x_out,
        #[cfg(feature = "gpu")]
        gpu: None,
    }
}

// Each layer stores the RoPE-applied K and V for all positions seen so far,
// shaped [n_positions, n_kv_heads * head_dim].
#[derive(Clone)]
pub struct KVCache {
    /// Flat buffers per layer: contiguous K values, dim_kv per token per layer
    pub k: Vec<Vec<f32>>,
    /// Flat buffers per layer: contiguous V values, dim per token per layer
    pub v: Vec<Vec<f32>>,
    /// Per-token novelty scores (average g_ij = 1 - a_ij over all past positions)
    pub novelty: Vec<f32>,
    /// Per-token attention curvature (1 - max(softmax)) averaged across heads.
    /// High curvature → attention is diffuse; low → focused on one token.
    pub curvature: Vec<f32>,
    /// Per-layer novelty scores: per_layer_novelty[li][pos] = novelty at that position
    pub per_layer_novelty: Vec<Vec<f32>>,
    /// Per-layer curvature scores: per_layer_curvature[li][pos] = curvature at that position
    pub per_layer_curvature: Vec<Vec<f32>>,
    /// Number of tokens cached (shared across all layers)
    pub n_cached: usize,
    /// Time dilation factor τ: modulates subjective time. Updated per-token
    /// from VFE (lower VFE → higher τ → more careful sampling).
    pub tau: f32,
}

impl KVCache {
    /// Create an empty cache for a model with `n_layers` layers.
    pub fn new(n_layers: usize) -> Self {
        KVCache {
            k: (0..n_layers).map(|_| Vec::new()).collect(),
            v: (0..n_layers).map(|_| Vec::new()).collect(),
            novelty: Vec::new(),
            curvature: Vec::new(),
            per_layer_novelty: (0..n_layers).map(|_| Vec::new()).collect(),
            per_layer_curvature: (0..n_layers).map(|_| Vec::new()).collect(),
            n_cached: 0,
            tau: 1.0,
        }

    }
}


/// Incremental forward pass for a DeepSeek-V2 layer (MLA + MoE).
/// Uses GGUF-convention fields stored in LayerWeights.
pub fn forward_layer_deepseek2_kv(
    lw: &LayerWeights,
    cfg: &Config,
    x: &Array2<f32>,
    cache: &mut KVCache,
    li: usize,
    pos: usize,
) -> Array2<f32> {
    let _dim = cfg.dim;
    let mla = &cfg.mla;
    let kv_lora_rank = mla.kv_lora_rank;
    let qk_rope_head_dim = mla.qk_rope_head_dim;
    let v_head_dim = mla.v_head_dim;
    let qk_nope_head_dim = v_head_dim; // DeepSeek-V2 convention
    let qk_head_dim = qk_rope_head_dim + qk_nope_head_dim;
    let n_heads = cfg.n_heads;

    // --- Attention ---
    let h_attn = engine::rmsnorm_rows(x, &lw.attn_norm, engine::EPS);

    // Combined Q projection (nope + rope): [1, dim] x [dim, n_heads * qk_head_dim]
    let q_combined = engine::linear(&h_attn, &lw.mla_q.as_ref().unwrap_or(&lw.wq));
    // q_combined shape: [1, n_heads * qk_head_dim]

    // K/V latent projection: [1, dim] x [dim, kv_lora_rank + qk_rope_head_dim]
    let kv_a = engine::linear(&h_attn, &lw.mla_kv_a_mqa.as_ref().unwrap());

    // Split KV latent: first kv_lora_rank is compressed latent, last qk_rope_head_dim is k_rope
    let k_compressed = kv_a.slice(s![.., ..kv_lora_rank]).to_owned();
    let k_rope_input = kv_a.slice(s![.., kv_lora_rank..]).to_owned();

    // Apply KV alpha norm to compressed latent
    let k_normed = if let Some(ref norm) = lw.mla_kv_a_norm {
        let k_compressed_arr = k_compressed.row(0).to_owned();
        let mut k_normed = Array2::zeros((1, kv_lora_rank));
        for i in 0..kv_lora_rank {
            k_normed[[0, i]] = k_compressed_arr[i] * norm[i];
        }
        k_normed
    } else {
        k_compressed.clone()
    };

    // V latent is same compressed representation: [1, kv_lora_rank]
    let v_compressed = &k_normed;

    // K/V up projection: [1, kv_lora_rank] x [kv_lora_rank, n_heads * (qk_nope + v_head_dim)]
    let kv_b = engine::linear(v_compressed, &lw.mla_kv_b.as_ref().unwrap());
    // kv_b shape: [1, n_heads * (qk_nope_head_dim + v_head_dim)]

    // Split into k_nope and v
    let k_nope_part = kv_b.slice(s![.., ..n_heads * qk_nope_head_dim]).to_owned();
    let v = kv_b.slice(s![.., n_heads * qk_nope_head_dim..]).to_owned();

    // Apply RoPE to Q_rope and K_rope (2D arrays for apply_rope_all)
    let mut q_rope_2d = Array2::zeros((1, n_heads * qk_rope_head_dim));
    for h in 0..n_heads {
        for j in 0..qk_rope_head_dim {
            q_rope_2d[[0, h * qk_rope_head_dim + j]] = q_combined[[0, h * qk_head_dim + j]];
        }
    }
    engine::apply_rope_all(&mut q_rope_2d, pos, qk_rope_head_dim, 10000.0);

    let mut k_rope_2d = Array2::zeros((1, qk_rope_head_dim));
    for j in 0..qk_rope_head_dim {
        k_rope_2d[[0, j]] = k_rope_input[[0, j]];
    }
    engine::apply_rope_all(&mut k_rope_2d, pos, qk_rope_head_dim, 10000.0);

    // Full Q: [rope | nope] per head
    let q_nope_part = q_combined.slice(s![.., n_heads * qk_rope_head_dim..]).to_owned();
    let mut q_full = Array2::zeros((1, n_heads * qk_head_dim));
    for h in 0..n_heads {
        for j in 0..qk_rope_head_dim {
            q_full[[0, h * qk_head_dim + j]] = q_rope_2d[[0, h * qk_rope_head_dim + j]];
        }
        for j in 0..qk_nope_head_dim {
            q_full[[0, h * qk_head_dim + qk_rope_head_dim + j]] = q_nope_part[[0, h * qk_nope_head_dim + j]];
        }
    }

    // Store K/V in flat cache (matching forward_layer_kv pattern)
    let dk = n_heads * qk_nope_head_dim + qk_rope_head_dim;
    let dv = n_heads * v_head_dim;
    let mut k_entry = Vec::with_capacity(dk);
    for h in 0..n_heads {
        for j in 0..qk_nope_head_dim {
            k_entry.push(k_nope_part[[0, h * qk_nope_head_dim + j]]);
        }
    }
    for j in 0..qk_rope_head_dim {
        k_entry.push(k_rope_2d[[0, j]]);
    }
    cache.k[li].extend_from_slice(&k_entry);

    let mut v_entry = Vec::with_capacity(dv);
    for h in 0..n_heads {
        for j in 0..v_head_dim {
            v_entry.push(v[[0, h * v_head_dim + j]]);
        }
    }
    cache.v[li].extend_from_slice(&v_entry);
    cache.n_cached = cache.n_cached.max(pos + 1);

    // Attention over cached positions (flat buffer)
    let k_slice = &cache.k[li];
    let v_slice = &cache.v[li];
    let n_prev = k_slice.len() / dk;
    let mut attn_out = vec![0.0f32; n_heads * v_head_dim];
    let mut total_novelty = 0.0f32;
    let mut total_curvature = 0.0f32;

    for head in 0..n_heads {
        let scale = 1.0 / (qk_head_dim as f32).sqrt();
        let mut scores = Vec::with_capacity(n_prev);
        for j in 0..n_prev {
            let mut sum = 0.0f32;
            for d in 0..qk_head_dim {
                let qd = q_full[[0, head * qk_head_dim + d]];
                let kd = if d < qk_nope_head_dim {
                    k_slice[j * dk + head * qk_nope_head_dim + d]
                } else {
                    k_slice[j * dk + n_heads * qk_nope_head_dim + (d - qk_nope_head_dim)]
                };
                sum += qd * kd;
            }
            scores.push(sum * scale);
        }
        let valid = (pos + 1).min(scores.len());
        let mut probs = scores[..valid].to_vec();
        if !probs.is_empty() {
            engine::softmax_inplace(&mut probs);
        }
        if let Some(&p_max) = probs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            total_novelty += 1.0 - p_max;
            total_curvature += probs.iter().map(|p| p * p).sum::<f32>();
        }
        for d in 0..v_head_dim {
            let mut acc = 0.0f32;
            for (j, &p) in probs.iter().enumerate() {
                acc += p * v_slice[j * dv + head * v_head_dim + d];
            }
            attn_out[head * v_head_dim + d] = acc;
        }
    }

    // Physics-wired novelty/curvature
    let avg_novelty = total_novelty / n_heads as f32;
    let avg_curvature = 1.0 - total_curvature / n_heads as f32;
    if pos < cache.novelty.len() {
        cache.novelty[pos] = avg_novelty;
        cache.curvature[pos] = avg_curvature;
    } else {
        cache.novelty.push(avg_novelty);
        cache.curvature.push(avg_curvature);
    }
    if li < cache.per_layer_novelty.len() {
        if pos < cache.per_layer_novelty[li].len() {
            cache.per_layer_novelty[li][pos] = avg_novelty;
            cache.per_layer_curvature[li][pos] = avg_curvature;
        } else {
            cache.per_layer_novelty[li].push(avg_novelty);
            cache.per_layer_curvature[li].push(avg_curvature);
        }
    }

    // Output projection
    let attn_out_arr = Array2::from_shape_vec((1, n_heads * v_head_dim), attn_out).unwrap();
    let attn_proj = engine::linear(&attn_out_arr, &lw.wo);
    let mut out = x + &attn_proj;

    // --- MoE FFN ---
    let h_ffn = engine::rmsnorm_rows(&out, &lw.ffn_norm, engine::EPS);

    if let Some(ref gate_inp) = lw.moe_gate_inp {
        // GGUF-convention MoE with 3D expert tensors
        let n_experts = cfg.moe.n_experts;
        let top_k = cfg.moe.top_k;
        let e_inter = if cfg.expert_intermediate > 0 { cfg.expert_intermediate } else { cfg.intermediate };

        // Router: h_ffn @ gate_inp -> [1, n_experts]
        let router_logits = engine::linear(&h_ffn, gate_inp);
        let mut router_probs = Array2::zeros((1, n_experts));
        for i in 0..n_experts {
            router_probs[[0, i]] = router_logits[[0, i]];
        }
        engine::softmax_inplace(&mut router_probs.row_mut(0).as_slice_mut().unwrap());

        // Top-k routing
        let mut expert_scores: Vec<(usize, f32)> = (0..n_experts).map(|i| (i, router_probs[[0, i]])).collect();
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected: Vec<usize> = expert_scores.iter().take(top_k).map(|(i, _)| *i).collect();

        let mut moe_out = Array2::zeros((1, e_inter));
        for &e in &selected {
            let score = router_probs[[0, e]];
            let gate = lw.moe_gate_exps.as_ref().unwrap();
            let down = lw.moe_down_exps.as_ref().unwrap();
            let up = lw.moe_up_exps.as_ref().unwrap();
            // Extract expert e: [e_inter, dim] for gate, [dim, e_inter] for down
            let gate_e = gate.index_axis(Axis(0), e);
            let up_e = up.index_axis(Axis(0), e);
            let down_e = down.index_axis(Axis(0), e);
            let h_expert = engine::silu(&engine::linear(&h_ffn, &gate_e.to_owned()));
            let activated = &h_expert * &engine::linear(&h_ffn, &up_e.to_owned());
            let expert_out = engine::linear(&activated, &down_e.to_owned());
            let scaled = expert_out.mapv(|v| v * score);
            moe_out += &scaled;
        }

        // Add shared experts if present
        if let (Some(ref sg), Some(ref sd), Some(ref su)) = (&lw.moe_shared_gate, &lw.moe_shared_down, &lw.moe_shared_up) {
            let shared_gate = engine::silu(&engine::linear(&h_ffn, sg));
            let shared_up = engine::linear(&h_ffn, su);
            let shared_act = &shared_gate * &shared_up;
            let shared_out = engine::linear(&shared_act, sd);
            moe_out += &shared_out;
        }

        let ff = engine::linear(&moe_out, &lw.w2); // w2[dim, e_inter]
        out = &out + &ff;
    } else if let Some(ref expert_w1) = lw.moe_expert_w1 {
        // Paper-convention MoE
        let n_experts = cfg.moe.n_experts;
        let top_k = cfg.moe.top_k;
        let e_inter = if cfg.expert_intermediate > 0 { cfg.expert_intermediate } else { cfg.intermediate };

        let router_logits = if let Some(ref rw) = lw.moe_router_weight {
            engine::linear(&h_ffn, rw)
        } else if let Some(ref gi) = lw.moe_gate_inp {
            engine::linear(&h_ffn, gi)
        } else {
            return out;
        };

        let mut router_probs = router_logits.mapv(|v| v.exp());
        let sum: f32 = router_probs.sum();
        if sum > 0.0 { router_probs /= sum; }

        let mut expert_scores: Vec<(usize, f32)> = (0..n_experts).map(|i| (i, router_probs[[0, i]])).collect();
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected: Vec<usize> = expert_scores.iter().take(top_k).map(|(i, _)| *i).collect();

        let mut moe_out = Array2::zeros((1, e_inter));
        for &e in &selected {
            let score = router_probs[[0, e]];
            let gate = engine::silu(&engine::linear(&h_ffn, &expert_w1[e]));
            let up = engine::linear(&h_ffn, &lw.moe_expert_w3.as_ref().unwrap()[e]);
            let expert_out = engine::linear(&(&gate * &up), &lw.moe_expert_w2.as_ref().unwrap()[e]);
            moe_out += &(expert_out.mapv(|v| v * score));
        }

        if let (Some(ref sw1), Some(ref sw2), Some(ref sw3)) = (&lw.moe_shared_w1, &lw.moe_shared_w2, &lw.moe_shared_w3) {
            let shared_gate = engine::silu(&engine::linear(&h_ffn, sw1));
            let shared_up = engine::linear(&h_ffn, sw3);
            let shared_out = engine::linear(&(&shared_gate * &shared_up), sw2);
            moe_out += &shared_out;
        }

        let ff = engine::linear(&moe_out, &lw.w2);
        out = &out + &ff;
    } else {
        // Dense FFN
        let gate = engine::silu(&engine::linear(&h_ffn, &lw.w1));
        let up = engine::linear(&h_ffn, &lw.w3);
        let ff = engine::linear(&(&gate * &up), &lw.w2);
        out = &out + &ff;
    }

    out
}

/// Incremental forward pass for a single new token at absolute position `pos`.
///
/// `x` is the embedding row for exactly one token (shape `[1, dim]`). The layer's
/// K/V for this token are RoPE-applied and appended to the `li`-th entry of
/// `cache`; only the new token attends (causally) to all cached positions.
/// Returns the layer output for this token (shape `[1, dim]`). This is O(pos)
/// per layer instead of O(pos^2), giving O(T) generation overall.
pub fn forward_layer_kv(
    lw: &LayerWeights,
    cfg: &Config,
    x: &Array2<f32>,
    cache: &mut KVCache,
    li: usize,
    pos: usize,
) -> Array2<f32> {
    // Dispatch to DeepSeek-V2 MLA+MoE path if configured
    if cfg.is_mla() {
        return forward_layer_deepseek2_kv(lw, cfg, x, cache, li, pos);
    }

    let hd = cfg.head_dim();
    let dim = cfg.dim;
    let dim_kv = cfg.dim_kv();

    let h_attn = engine::rmsnorm_rows(x, &lw.attn_norm, engine::EPS);
    let q = engine::linear(&h_attn, &lw.wq);
    let k = engine::linear(&h_attn, &lw.wk);
    let v = engine::linear(&h_attn, &lw.wv);
    let mut q_rope = q;
    let mut k_rope = k;
    engine::apply_rope_at(&mut q_rope, cfg.n_heads, hd, cfg.rope_theta, pos);
    engine::apply_rope_at(&mut k_rope, cfg.n_kv_heads, hd, cfg.rope_theta, pos);

    // Append this token's K/V as flat contiguous rows (no full-copy reallocation)
    let dk = dim_kv;
    let dv = dim_kv;
    cache.k[li].extend_from_slice(&k_rope.as_slice().unwrap()[..dk]);
    cache.v[li].extend_from_slice(&v.as_slice().unwrap()[..dv]);
    cache.n_cached = cache.n_cached.max(pos + 1);

    // Attention over all cached positions (flat buffer indexed by j * dk + d)
    let k_slice = &cache.k[li];
    let v_slice = &cache.v[li];
    let n_prev = cache.k[li].len() / dk;
    let group = cfg.n_heads / cfg.n_kv_heads.max(1);
    let mut attn_out = vec![0.0f32; dim];
    let mut total_novelty = 0.0f32;
    let mut total_curvature = 0.0f32;
    for head in 0..cfg.n_heads {
        let kvh = head / group;
        let mut scores: Vec<f32> = Vec::with_capacity(n_prev);
        for j in 0..n_prev {
            let mut dot = 0.0;
            for d in 0..hd {
                dot += q_rope[[0, head * hd + d]] * k_slice[j * dk + kvh * hd + d];
            }
            scores.push(dot / (hd as f32).sqrt());
        }
        let probs = engine::softmax(&scores);
        let eps = 1e-12;
        let entropy: f32 = probs.iter().map(|a| {
            if *a > eps { -a * a.log10() } else { 0.0 }
        }).sum();
        let max_entropy = (n_prev as f32).log10();
        let norm_entropy = if max_entropy > eps { entropy / max_entropy } else { 0.0 };
        total_novelty += norm_entropy;
        // Attention curvature: 1 - max(softmax). Low when attention is sharply focused on one token,
        // high when attention is diffuse across many tokens.
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        total_curvature += 1.0 - max_prob;
        for d in 0..hd {
            let mut acc = 0.0;
            for (j, &sj) in probs.iter().enumerate() {
                acc += sj * v_slice[j * dv + kvh * hd + d];
            }
            attn_out[head * hd + d] = acc;
        }
    }
    let novelty = total_novelty / cfg.n_heads.max(1) as f32;
    let curvature = total_curvature / cfg.n_heads.max(1) as f32;
    if pos as usize >= cache.novelty.len() {
        cache.novelty.push(novelty);
        cache.curvature.push(curvature);
    }
    // Per-layer tracking: push regardless of whether we're the first layer at this position.
    // This gives per-layer novelty/curvature profiles across all positions.
    cache.per_layer_novelty[li].push(novelty);
    cache.per_layer_curvature[li].push(curvature);

    let attn_arr = Array2::from_shape_vec((1, dim), attn_out).unwrap();
    let x_after_attn = x.to_owned() + &engine::linear(&attn_arr, &lw.wo);
    let h_ffn = engine::rmsnorm_rows(&x_after_attn, &lw.ffn_norm, engine::EPS);
    let gate = engine::linear(&h_ffn, &lw.w1);
    let gate_silu = engine::silu(&gate);
    let up = engine::linear(&h_ffn, &lw.w3);
    x_after_attn + &engine::linear(&(&gate_silu * &up), &lw.w2)
}

/// Backward pass through RMSNorm.
/// Given input `x` (pre-norm, shape [t, dim]), weight `w` (shape [dim]),
/// and upstream gradient `d_out` (shape [t, dim]), computes gradient
/// w.r.t. input `x` and gradient w.r.t. weight `w`.
pub fn rmsnorm_backward(
    x: &Array2<f32>,
    w: &Array1<f32>,
    d_out: &Array2<f32>,
) -> (Array2<f32>, Array1<f32>) {
    let t = x.nrows();
    let dim = x.ncols();
    let mut d_x = Array2::zeros((t, dim));
    let mut d_w = Array1::zeros(dim);
    let eps = engine::EPS;

    for i in 0..t {
        // Compute row statistics
        let mut mean_sq = 0.0f32;
        for d in 0..dim {
            mean_sq += x[[i, d]] * x[[i, d]];
        }
        mean_sq /= dim as f32;
        let inv_std = 1.0 / (mean_sq + eps).sqrt();

        // d_w and first part of d_x
        let mut sum_term = 0.0f32;
        for d in 0..dim {
            let xi = x[[i, d]];
            let wi = w[d];
            let d_out_id = d_out[[i, d]];
            d_w[d] += d_out_id * (xi / wi.max(1e-9));
            d_x[[i, d]] = d_out_id * inv_std * wi;
            sum_term += d_out_id * wi * (xi / wi.max(1e-9));
        }

        // Second-order correction
        sum_term *= -inv_std.powi(3) / dim as f32;
        for d in 0..dim {
            d_x[[i, d]] += sum_term * x[[i, d]];
        }
    }

    d_x.mapv_inplace(sanitize);
    d_w.mapv_inplace(sanitize);
    (d_x, d_w)
}


/// Given the forward cache, upstream gradient `d_x` (gradient at layer output),
/// compute gradients for all layer parameters and return them along with the
/// gradient to propagate to the previous layer.
pub fn backward_layer(
    lw: &LayerWeights,
    cfg: &Config,
    lc: &LayerCache,
    d_x: &Array2<f32>,
) -> (LayerGrads, Array2<f32>) {
    let t = d_x.nrows();
    let hd = cfg.head_dim();

    // --- FFN residual ---
    let d_ff = d_x.clone();
    let d_x_after_attn = d_x.clone();

    // --- FFN output: ff = (gate_silu * up) @ w2^T ---
    let in_ff = &lc.gate_silu * &lc.up;
    let mut d_w2 = d_ff.t().dot(&in_ff);
    let mut d_in_ff = d_ff.dot(&lw.w2);
    d_w2.mapv_inplace(sanitize);
    d_in_ff.mapv_inplace(sanitize);

    // --- gate_silu * up (element-wise multiply) ---
    let mut d_up: Array2<f32> = Array2::zeros((t, cfg.intermediate));
    let mut d_gate_silu: Array2<f32> = Array2::zeros((t, cfg.intermediate));
    for i in 0..t {
        for j in 0..cfg.intermediate {
            d_up[[i, j]] = sanitize(d_in_ff[[i, j]] * lc.gate_silu[[i, j]]);
            d_gate_silu[[i, j]] = sanitize(d_in_ff[[i, j]] * lc.up[[i, j]]);
        }
    }

    // --- SiLU backward ---
    let mut d_gate: Array2<f32> = Array2::zeros((t, cfg.intermediate));
    for i in 0..t {
        for j in 0..cfg.intermediate {
            let g = lc.gate[[i, j]];
            let s = 1.0 / (1.0 + (-g).exp());
            let ds = s * (1.0 - s);
            d_gate[[i, j]] = sanitize(d_gate_silu[[i, j]] * (s + g * ds));
        }
    }

    // --- w3: up = h_ffn @ w3^T ---
    let mut d_w3 = d_up.t().dot(&lc.h_ffn);
    let mut d_x_up = d_up.dot(&lw.w3);
    d_w3.mapv_inplace(sanitize);
    d_x_up.mapv_inplace(sanitize);

    let mut d_w1 = d_gate.t().dot(&lc.h_ffn);
    let mut d_x_gate = d_gate.dot(&lw.w1);
    d_w1.mapv_inplace(sanitize);
    d_x_gate.mapv_inplace(sanitize);

    // --- FFN RMSNorm backward ---
    let d_x_ffn_total = &d_x_up + &d_x_gate;
    let (d_x_ffn_norm, d_ffn_norm) = rmsnorm_backward(&lc.h_ffn, &lw.ffn_norm, &d_x_ffn_total);

    // --- Attention residual ---
    let d_x_attn_upstream = &d_x_after_attn + &d_x_ffn_norm;

    // --- wo: attn_proj = attn_out @ wo^T ---
    let mut d_wo = d_x_attn_upstream.t().dot(&lc.attn_out);
    let mut d_attn_out = d_x_attn_upstream.dot(&lw.wo);
    d_wo.mapv_inplace(sanitize);
    d_attn_out.mapv_inplace(sanitize);

    // --- Attention backward (per head) ---
    let mut d_q: Array2<f32> = Array2::zeros((t, cfg.n_heads * hd));
    let mut d_k: Array2<f32> = Array2::zeros((t, cfg.n_kv_heads * hd));
    let mut d_v: Array2<f32> = Array2::zeros((t, cfg.dim_kv()));
    let group = cfg.n_heads / cfg.n_kv_heads.max(1);

    #[cfg(feature = "gpu")]
    {
        use crate::engine::gpu_head_attn_backward;
        for head in 0..cfg.n_heads {
            let kvh = head / group;
            let mut q_s = Array2::zeros((t, hd));
            let mut k_s = Array2::zeros((t, hd));
            let mut v_s = Array2::zeros((t, hd));
            let mut do_s = Array2::zeros((t, hd));
            for i in 0..t {
                for d in 0..hd {
                    q_s[[i, d]] = lc.q_rope[[i, head * hd + d]];
                    k_s[[i, d]] = lc.k_rope[[i, kvh * hd + d]];
                    v_s[[i, d]] = lc.v[[i, kvh * hd + d]];
                    do_s[[i, d]] = d_attn_out[[i, head * hd + d]];
                }
            }
            let (dq_h, dk_h, dv_h) = gpu_head_attn_backward(&q_s, &k_s, &v_s, &do_s);
            for i in 0..t {
                for d in 0..hd {
                    d_q[[i, head * hd + d]] = sanitize(dq_h[[i, d]]);
                    d_k[[i, kvh * hd + d]] += sanitize(dk_h[[i, d]]);
                    d_v[[i, kvh * hd + d]] += sanitize(dv_h[[i, d]]);
                }
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        for head in 0..cfg.n_heads {
            let kvh = head / group;
            for i in 0..t {
                let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
                for j in 0..=i {
                    let mut dot = 0.0;
                    for d in 0..hd {
                        dot += lc.q_rope[[i, head * hd + d]] * lc.k_rope[[j, kvh * hd + d]];
                    }
                    scores.push(dot / (hd as f32).sqrt());
                }
                let probs = engine::softmax(&scores);

                for j in 0..=i {
                    for d in 0..hd {
                        d_v[[j, kvh * hd + d]] += sanitize(d_attn_out[[i, head * hd + d]] * probs[j]);
                    }
                }

                let mut d_scores = vec![0.0; i + 1];
                for j in 0..=i {
                    for d in 0..hd {
                        d_scores[j] += d_attn_out[[i, head * hd + d]] * lc.v[[j, kvh * hd + d]];
                    }
                }

                let mut d_score_raw = vec![0.0; i + 1];
                for k in 0..=i {
                    let mut s = 0.0;
                    for j in 0..=i {
                        let delta = if k == j { 1.0 } else { 0.0 };
                        s += d_scores[j] * probs[j] * (delta - probs[k]);
                    }
                    d_score_raw[k] = s / (hd as f32).sqrt();
                }

                for j in 0..=i {
                    for d in 0..hd {
                        d_q[[i, head * hd + d]] += sanitize(d_score_raw[j] * lc.k_rope[[j, kvh * hd + d]]);
                        d_k[[j, kvh * hd + d]] += sanitize(d_score_raw[j] * lc.q_rope[[i, head * hd + d]]);
                    }
                }
            }
        }
    }

    // --- RoPE backward ---
    engine::apply_rope_all(&mut d_q, cfg.n_heads, hd, cfg.rope_theta);
    engine::apply_rope_all(&mut d_k, cfg.n_kv_heads, hd, cfg.rope_theta);

    // --- Q, K, V projections ---
    let mut d_wq = d_q.t().dot(&lc.h_attn);
    let mut d_h_q = d_q.dot(&lw.wq);
    d_wq.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
    d_h_q.mapv_inplace(sanitize);
    let mut d_wk = d_k.t().dot(&lc.h_attn);
    let mut d_h_k = d_k.dot(&lw.wk);
    d_wk.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
    d_h_k.mapv_inplace(sanitize);
    let mut d_wv = d_v.t().dot(&lc.h_attn);
    let mut d_h_v = d_v.dot(&lw.wv);
    d_wv.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
    d_h_v.mapv_inplace(sanitize);

    // --- Attention RMSNorm backward ---
    let d_x_attn = &d_h_q + &d_h_k + &d_h_v;
    let (d_x_attn_norm, d_attn_norm) = rmsnorm_backward(&lc.h_attn, &lw.attn_norm, &d_x_attn);

    let grads = LayerGrads {
        wq: d_wq, wk: d_wk, wv: d_wv, wo: d_wo,
        w1: d_w1, w2: d_w2, w3: d_w3,
        attn_norm: d_attn_norm, ffn_norm: d_ffn_norm,
    };

    // d_x_prev includes the upstream gradient from the residual skip connection
    // plus the gradient through the attention/FFN path
    let d_x_prev = &d_x_attn_norm + &d_x_ffn_norm + d_x;

    (grads, d_x_prev)
}

/// Compute attention gap at each position boundary, averaged across all layers and heads.
/// `x` should be the embedded token activations (shape [T, dim]).
/// gap[i] = mean attention from position i+1 to all positions ≤ i across all layers and heads.
/// High gap = position i+1 attends strongly to the prefix → boundary is NOT a good split.
/// Low gap = position i+1 mostly attends to itself and later positions → good chunk boundary.
/// Returns a vector of length T-1 (boundary between pos 0-1, 1-2, ..., T-2, T-1).
#[allow(dead_code)]
pub fn attention_gaps(layers: &[LayerWeights], cfg: &Config, x_in: &Array2<f32>) -> Vec<f32> {
    let t = x_in.nrows();
    if t < 2 {
        return vec![];
    }
    let hd = cfg.head_dim();
    let group = cfg.n_heads / cfg.n_kv_heads.max(1);

    let mut x = x_in.to_owned();
    let mut gap_sum = vec![0.0f32; t - 1];
    let mut gap_count = 0usize;

    for layer in layers {
        let h = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);
        let q = engine::linear(&h, &layer.wq);
        let k = engine::linear(&h, &layer.wk);
        let v = engine::linear(&h, &layer.wv);
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        engine::apply_rope_all(&mut q_rope, cfg.n_heads, hd, cfg.rope_theta);
        engine::apply_rope_all(&mut k_rope, cfg.n_kv_heads, hd, cfg.rope_theta);

        for head in 0..cfg.n_heads {
            let kvh = head / group;
            for i in 0..t {
                let n_candidates = i + 1;
                let mut scores = Vec::with_capacity(n_candidates);
                for j in 0..=i {
                    let mut dot = 0.0;
                    for d in 0..hd {
                        dot += q_rope[[i, head * hd + d]] * k_rope[[j, kvh * hd + d]];
                    }
                    scores.push(dot / (hd as f32).sqrt());
                }
                let probs = engine::softmax(&scores);
                let mut prefix_sum = 0.0;
                for j in 0..i {
                    prefix_sum += probs[j];
                    gap_sum[j] += prefix_sum;
                }
            }
        }

        let mut attn_out = Array2::zeros((t, cfg.dim));
        for head in 0..cfg.n_heads {
            let kvh = head / group;
            for i in 0..t {
                let mut scores = Vec::with_capacity(i + 1);
                for j in 0..=i {
                    let mut dot = 0.0;
                    for d in 0..hd {
                        dot += q_rope[[i, head * hd + d]] * k_rope[[j, kvh * hd + d]];
                    }
                    scores.push(dot / (hd as f32).sqrt());
                }
                let probs = engine::softmax(&scores);
                for d in 0..hd {
                    let mut acc = 0.0;
                    for (j, &sj) in probs.iter().enumerate() {
                        acc += sj * v[[j, kvh * hd + d]];
                    }
                    attn_out[[i, head * hd + d]] = acc;
                }
            }
        }
        let attn_proj = engine::linear(&attn_out, &layer.wo);
        x = &x + &attn_proj;

        let h_ffn = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
        let gate = engine::silu(&engine::linear(&h_ffn, &layer.w1));
        let up = engine::linear(&h_ffn, &layer.w3);
        let ff = engine::linear(&(&gate * &up), &layer.w2);
        x = &x + &ff;

        gap_count += cfg.n_heads;
    }

    let count = gap_count as f32;
    for (j, g) in gap_sum.iter_mut().enumerate() {
        let n_contributors = (t - 1 - j) as f32; // positions i > j
        if n_contributors > 0.0 {
            *g /= count * n_contributors;
        }
    }

    gap_sum
}

/// Partition a sequence at geodesic boundaries where attention gap drops below a threshold.
/// gap[i] = mean prefix-attention of position i+1. Low values mean good chunk boundaries.
/// Returns start positions of each chunk (always includes 0 and T).
/// `min_chunk` ensures minimum chunk size.
pub fn geodesic_chunks(gaps: &[f32], threshold: f32, min_chunk: usize) -> Vec<usize> {
    let t = gaps.len() + 1;
    let mut boundaries = vec![0usize];
    let mut last = 0usize;
    for i in 0..gaps.len() {
        if gaps[i] < threshold && i - last + 1 >= min_chunk {
            // boundary between i and i+1
            boundaries.push(i + 1);
            last = i + 1;
        }
    }
    if last < t {
        boundaries.push(t);
    }
    boundaries
}
#[cfg(test)]


mod tests {
    use super::*;

    fn make_cfg() -> Config {
        Config {
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 50,
            intermediate: 32,
            rope_theta: 10000.0,
            max_seq: 256,
            // BRACKET-LINE STATE VECTOR (1.3) — initialize to identity
            tau: 1.0,
            e: 1.0,
            age: 0,
            cycles: 0,
            h: 0.5,
            base_ms: 1000.0,
            phi: 0.0,
            attn_policy: crate::config::AttnPolicy::Global(crate::config::AttnKind::MHA),
            mlp_kind: crate::config::MlpKind::Dense,
            moe: crate::config::MoEConfig::default(),
            mla: crate::config::MLAConfig::default(),
            vision: crate::config::VisionConfig::default(),
            expert_intermediate: 0,
            leading_dense_blocks: 0,
        }
    }

    fn random_layer(cfg: &Config, seed: u64) -> LayerWeights {
        let mut s = seed;
        let dim = cfg.dim;
        let kv = cfg.dim_kv();
        let mk = |rows: usize, cols: usize, s: &mut u64| {
            let mut a = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a[[i, j]] = ((*s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
                }
            }
            a
        };
        let mk1 = |n: usize, s: &mut u64| {
            let mut a = Array1::zeros(n);
            for i in 0..n {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                a[i] = 1.0 + ((*s >> 33) as f32 / (u32::MAX as f32)) * 0.1;
            }
            a
        };
        LayerWeights {
            wq: mk(dim, dim, &mut s),
            wk: mk(kv, dim, &mut s),
            wv: mk(kv, dim, &mut s),
            wo: mk(dim, dim, &mut s),
            w1: mk(cfg.intermediate, dim, &mut s),
            w2: mk(dim, cfg.intermediate, &mut s),
            w3: mk(cfg.intermediate, dim, &mut s),
            attn_norm: mk1(dim, &mut s),
            ffn_norm: mk1(dim, &mut s),
            mla_wq_a: None, mla_wq_b: None,
            mla_wk_v_a: None, mla_wk_b: None,
            mla_wv_b: None, mla_wq_rope: None, mla_wo: None,
            moe_gate_exps: None, moe_down_exps: None, moe_up_exps: None, moe_gate_inp: None,
            moe_shared_gate: None, moe_shared_down: None, moe_shared_up: None,
            mla_q: None, mla_kv_a_mqa: None, mla_kv_b: None, mla_kv_a_norm: None,
            moe_shared_w1: None, moe_shared_w2: None, moe_shared_w3: None,
            moe_expert_w1: None, moe_expert_w2: None, moe_expert_w3: None,
            moe_router_weight: None, moe_router_bias: None,
        }
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        let cfg = make_cfg();
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|li| random_layer(&cfg, li as u64 + 1)).collect();
        let t = 6;
        let dim = cfg.dim;

        // Full forward over all T positions (no cache).
        let mut x_full = Array2::zeros((t, dim));
        for pos in 0..t {
            for d in 0..dim {
                x_full[[pos, d]] = (((pos * 7 + d * 3) % 11) as f32 / 11.0) - 0.5;
            }
        }
        let mut x = x_full.clone();
        for lw in &layers {
            let lc = forward_layer(lw, &cfg, &x);
            x = lc.x_out;
        }
        let full_last = x.row(t - 1).to_owned();

        // Incremental forward position-by-position using the KV cache.
        let mut cache = KVCache::new(cfg.n_layers);
        let mut x_inc = Array2::zeros((1, dim));
        for pos in 0..t {
            for d in 0..dim {
                x_inc[[0, d]] = x_full[[pos, d]];
            }
            for (li, lw) in layers.iter().enumerate() {
                x_inc = forward_layer_kv(lw, &cfg, &x_inc, &mut cache, li, pos);
            }
        }
        let inc_last = x_inc.row(0).to_owned();

        let mut max_err = 0.0f32;
        for d in 0..dim {
            max_err = max_err.max((full_last[d] - inc_last[d]).abs());
        }
        assert!(max_err < 1e-4, "KV cache diverged from full forward, max_err={max_err}");
    }

    /// Replicates `generate_streaming`'s full pipeline (embed -> KV-cached layers
    /// -> final RMSNorm -> output projection) for a tiny in-memory model, and
    /// returns the logits of the last prompt token plus the first sampled token.
    /// Validates the prefill/sampling loop math against a non-streaming forward.
    #[test]
    fn streaming_pipeline_matches_full_forward() {
        let cfg = make_cfg();
        let layers: Vec<LayerWeights> = (0..cfg.n_layers).map(|li| random_layer(&cfg, li as u64 + 1)).collect();
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;

        // Synthetic embed / output / final-norm in f32.
        let mut s = 99u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX as f32)) - 0.5
        };
        let mut embed = Array2::zeros((vocab, dim));
        for i in 0..vocab {
            for d in 0..dim {
                embed[[i, d]] = r();
            }
        }
        // output stored as (vocab, dim) so linear(xf [t,dim], output) -> [t,vocab].
        let mut output = Array2::zeros((vocab, dim));
        for j in 0..vocab {
            for i in 0..dim {
                output[[j, i]] = r();
            }
        }
        let mut fnorm = Array1::zeros(dim);
        for d in 0..dim {
            fnorm[d] = 1.0 + r() * 0.1;
        }

        let prompt: Vec<usize> = vec![3, 7, 11];
        let n_prompt = prompt.len();
        let max_new = 4usize;

        // --- Non-streaming reference: full forward over all positions, read prompt tail. ---
        let mut x_ref = Array2::zeros((n_prompt, dim));
        for (pos, &tid) in prompt.iter().enumerate() {
            for d in 0..dim {
                x_ref[[pos, d]] = embed[[tid, d]];
            }
        }
        let mut x = x_ref.clone();
        for lw in &layers {
            let lc = forward_layer(lw, &cfg, &x);
            x = lc.x_out;
        }
        let xf_ref = engine::rmsnorm_rows(&x, &fnorm, engine::EPS);
        let logits_ref = engine::linear(&xf_ref, &output);

        // --- Streaming (KV-cached) generation, mirroring main.rs::generate_streaming. ---
        let mut ids = prompt.clone();
        let mut cache = KVCache::new(cfg.n_layers);
        for pos in 0..(n_prompt + max_new - 1) {
            let tid = ids[pos];
            let mut x_tok = Array2::zeros((1, dim));
            for d in 0..dim {
                x_tok[[0, d]] = embed[[tid, d]];
            }
            for (li, lw) in layers.iter().enumerate() {
                x_tok = forward_layer_kv(lw, &cfg, &x_tok, &mut cache, li, pos);
            }
            if pos < n_prompt - 1 {
                continue;
            }
            let xf = engine::rmsnorm_rows(&x_tok, &fnorm, engine::EPS);
            let logits = engine::linear(&xf, &output);
            if pos == n_prompt - 1 {
                // Compare the prompt-tail logits from both paths.
                let mut max_err = 0.0f32;
                for j in 0..vocab {
                    max_err = max_err.max((logits[[0, j]] - logits_ref[[n_prompt - 1, j]]).abs());
                }
                assert!(max_err < 1e-4, "streaming prompt-tail logits diverged, max_err={max_err}");
            }
            // Greedily sample to advance the loop (argmax).
            let mut best = 0usize;
            let mut bv = logits[[0, 0]];
            for j in 1..vocab {
                if logits[[0, j]] > bv {
                    bv = logits[[0, j]];
                    best = j;
                }
            }
            if pos >= n_prompt - 1 {
                ids.push(best);
            }
        }

        // The loop must have produced exactly `max_new` new tokens appended.
        assert_eq!(ids.len(), n_prompt + max_new);
    }

    #[test]
    fn moe_forward_layer_produces_valid_output() {
        let mut cfg = make_cfg();
        cfg.mlp_kind = crate::config::MlpKind::MoE;
        cfg.moe.enabled = true;
        cfg.moe.n_experts = 4;
        cfg.moe.n_shared = 2;
        cfg.moe.top_k = 2;

        let dim = cfg.dim;
        let intermediate = cfg.intermediate;
        let n_experts = cfg.moe.n_experts;
        let n_shared = cfg.moe.n_shared;

        let mut s = 42u64;
        let mk = |rows: usize, cols: usize, s: &mut u64| {
            let mut a = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a[[i, j]] = ((*s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
                }
            }
            a
        };
        let mk1 = |n: usize, s: &mut u64| {
            let mut a = Array1::zeros(n);
            for i in 0..n {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                a[i] = 1.0 + ((*s >> 33) as f32 / (u32::MAX as f32)) * 0.1;
            }
            a
        };

        let layer = LayerWeights {
            wq: Array2::zeros((dim, dim)),
            wk: Array2::zeros((cfg.dim_kv(), dim)),
            wv: Array2::zeros((cfg.dim_kv(), dim)),
            wo: Array2::zeros((dim, dim)),
            w1: Array2::zeros((intermediate, dim)),
            w2: Array2::zeros((dim, intermediate)),
            w3: Array2::zeros((intermediate, dim)),
            attn_norm: mk1(dim, &mut s),
            ffn_norm: mk1(dim, &mut s),
            mla_wq_a: None, mla_wq_b: None,
            mla_wk_v_a: None, mla_wk_b: None,
            mla_wv_b: None, mla_wq_rope: None, mla_wo: None,
            moe_gate_exps: None, moe_down_exps: None, moe_up_exps: None, moe_gate_inp: None,
            moe_shared_gate: None, moe_shared_down: None, moe_shared_up: None,
            mla_q: None, mla_kv_a_mqa: None, mla_kv_b: None, mla_kv_a_norm: None,
            // MoE weights
            moe_shared_w1: Some(mk(n_shared * intermediate, dim, &mut s)),
            moe_shared_w2: Some(mk(dim, n_shared * intermediate, &mut s)),
            moe_shared_w3: Some(mk(n_shared * intermediate, dim, &mut s)),
            moe_expert_w1: Some(vec![mk(intermediate, dim, &mut s); n_experts]),
            moe_expert_w2: Some(vec![mk(dim, intermediate, &mut s); n_experts]),
            moe_expert_w3: Some(vec![mk(intermediate, dim, &mut s); n_experts]),
            moe_router_weight: Some(mk(n_experts, dim, &mut s)),
            moe_router_bias: None,
        };

        let weights = Weights {
            embed: Array2::zeros((cfg.vocab_size, dim)),
            layers: vec![layer],
            final_norm: mk1(dim, &mut s),
            output: Array2::zeros((dim, cfg.vocab_size)),
        };

        // Use a random input vector so some gate activations are positive
        let mut hs = 9999u64;
        let h = mk(1, dim, &mut hs);
        // h values are in [-0.5, 0.5], should give mixed positive/negative gates
        let out = weights.moe_forward_layer(&cfg, &h.view(), &weights.layers[0]);

        assert_eq!(out.shape(), &[1, dim], "MoE output shape mismatch");
        for &v in out.iter() {
            assert!(v.is_finite(), "MoE output contains non-finite value: {v}");
        }
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 1e-6, "MoE output is all zeros or near-zero (max_abs={max_abs})");
    }

    #[test]
    fn mla_forward_layer_produces_valid_output() {
        let mut cfg = make_cfg();
        cfg.mla.enabled = true;
        cfg.mla.q_lora_rank = 4;
        cfg.mla.kv_lora_rank = 4;
        cfg.mla.qk_rope_head_dim = 2;
        cfg.mla.v_head_dim = 2;

        let dim = cfg.dim;
        let n_heads = cfg.n_heads;
        let q_lora = cfg.mla.q_lora_rank;
        let kv_lora = cfg.mla.kv_lora_rank;
        let qk_rope = cfg.mla.qk_rope_head_dim;
        let v_head = cfg.mla.v_head_dim;
        let q_head = qk_rope + v_head;

        let mut s = 42u64;
        let mk = |rows: usize, cols: usize, s: &mut u64| {
            let mut a = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    a[[i, j]] = ((*s >> 33) as f32 / (u32::MAX as f32)) - 0.5;
                }
            }
            a
        };
        let mk1 = |n: usize, s: &mut u64| {
            let mut a = Array1::zeros(n);
            for i in 0..n {
                *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                a[i] = 1.0 + ((*s >> 33) as f32 / (u32::MAX as f32)) * 0.1;
            }
            a
        };

        let layer = LayerWeights {
            wq: Array2::zeros((dim, dim)),
            wk: Array2::zeros((cfg.dim_kv(), dim)),
            wv: Array2::zeros((cfg.dim_kv(), dim)),
            wo: Array2::zeros((dim, dim)),
            w1: Array2::zeros((cfg.intermediate, dim)),
            w2: Array2::zeros((dim, cfg.intermediate)),
            w3: Array2::zeros((cfg.intermediate, dim)),
            attn_norm: mk1(dim, &mut s),
            ffn_norm: mk1(dim, &mut s),
            mla_wq_a: Some(mk(q_lora, dim, &mut s)),
            mla_wq_b: Some(mk(n_heads * q_head, q_lora, &mut s)),
            mla_wk_v_a: Some(mk(kv_lora, dim, &mut s)),
            mla_wk_b: Some(mk(n_heads * (v_head + v_head), kv_lora, &mut s)),
            mla_wv_b: Some(mk(n_heads * v_head, kv_lora, &mut s)),
            mla_wq_rope: Some(mk(n_heads * qk_rope, q_lora, &mut s)),
            mla_wo: Some(mk(dim, n_heads * v_head, &mut s)),
            moe_shared_w1: None,
            moe_shared_w2: None,
            moe_shared_w3: None,
            moe_expert_w1: None,
            moe_expert_w2: None,
            moe_expert_w3: None,
            moe_router_weight: None,
            moe_router_bias: None,
            moe_gate_exps: None,
            moe_down_exps: None,
            moe_up_exps: None,
            moe_gate_inp: None,
            moe_shared_gate: None,
            moe_shared_down: None,
            moe_shared_up: None,
            mla_q: None,
            mla_kv_a_mqa: None,
            mla_kv_b: None,
            mla_kv_a_norm: None,
        };

        let weights = Weights {
            embed: Array2::zeros((cfg.vocab_size, dim)),
            layers: vec![layer],
            final_norm: mk1(dim, &mut s),
            output: Array2::zeros((dim, cfg.vocab_size)),
        };

        let mut hs = 9999u64;
        let h = mk(1, dim, &mut hs);
        let hd = cfg.dim / cfg.n_heads;
        let out = weights.mla_forward_layer(&cfg, &h.view(), &weights.layers[0], hd);

        assert_eq!(out.shape(), &[1, dim], "MLA output shape mismatch");
        for &v in out.iter() {
            assert!(v.is_finite(), "MLA output contains non-finite value: {v}");
        }
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 1e-6, "MLA output is all zeros or near-zero (max_abs={max_abs})");
    }
}
