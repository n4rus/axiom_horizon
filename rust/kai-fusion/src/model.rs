//! Transformer model (Slice 1: dense RoPE-MHA, Llama/Qwen-style pre-norm, SwiGLU).
//! Weights stored (out, in) to match GGUF layout directly.
//! Includes full forward + backward pass for training.

use crate::config::Config;
use crate::engine;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn sanitize(v: f32) -> f32 {
    if v.is_nan() || v.is_infinite() { 0.0 } else { v }
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
}

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
}

/// Gradient accumulator for all parameters
pub struct Gradients {
    pub embed: Array2<f32>,
    pub layers: Vec<LayerGrads>,
    pub final_norm: Array1<f32>,
    pub output: Array2<f32>,
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
            // --- attention (pre-norm) ---
            let h = engine::rmsnorm_rows(&x, &layer.attn_norm, engine::EPS);
            let mut q = engine::linear(&h, &layer.wq); // (T, dim)
            let mut k = engine::linear(&h, &layer.wk); // (T, dk)
            let v = engine::linear(&h, &layer.wv); // (T, dk)
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
            let attn_proj = engine::linear(&attn_out, &layer.wo); // (T, dim)
            x = &x + &attn_proj;

            // --- ffn (SwiGLU, pre-norm) ---
            let h2 = engine::rmsnorm_rows(&x, &layer.ffn_norm, engine::EPS);
            let gate = engine::silu(&engine::linear(&h2, &layer.w1));
            let up = engine::linear(&h2, &layer.w3);
            let ff = engine::linear(&(&gate * &up), &layer.w2);
            x = &x + &ff;
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

    /// Final-norm hidden state of the last position (the context representation
    /// Kai uses to measure novelty against its assimilated prior).
    pub fn last_hidden(&self, cfg: &Config, tokens: &[usize]) -> Vec<f32> {
        let (_logits, xf) = self.forward(cfg, tokens);
        xf
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
            layers.push(LayerWeights { wq, wk, wv, wo, w1, w2, w3, attn_norm, ffn_norm });
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

        let xf = engine::rmsnorm_rows(&x, &self.final_norm, engine::EPS);
        let logits = engine::linear(&xf, &self.output);
        ForwardCache { layers: layer_caches, xf, logits }
    }

    /// Backward pass: compute gradients for all parameters.
    /// d_logits: upstream gradient of loss w.r.t. logits [T, vocab].
    /// Returns accumulated gradients.
    pub fn backward(&self, cfg: &Config, cache: &ForwardCache, d_logits: &Array2<f32>, tokens: &[usize]) -> Gradients {
        let t = cache.logits.nrows();
        let hd = cfg.head_dim();
        let dk = cfg.dim_kv();

        // Allocate gradient accumulators
        let mut d_embed: Array2<f32> = Array2::zeros((cfg.vocab_size, cfg.dim));
        let mut d_final_norm = Array1::zeros(cfg.dim);
        let mut d_output: Array2<f32> = Array2::zeros(self.output.raw_dim());
        let mut layer_grads = Vec::with_capacity(cfg.n_layers);

        // --- Gradient through output projection and final RMSNorm ---
        // logits = xf @ output^T  =>  d_output += d_logits^T @ xf, d_xf = d_logits @ output
        let d_xf_raw = d_logits.dot(&self.output); // [T, dim]

        // d_output = d_logits^T @ xf = [vocab, T] @ [T, dim] = [vocab, dim]
        for j in 0..cfg.vocab_size {
            for d in 0..cfg.dim {
                let mut s = 0.0;
                for i in 0..t {
                    s += d_logits[[i, j]] * cache.xf[[i, d]];
                }
                d_output[[j, d]] = s;
            }
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
            let mut d_wq: Array2<f32> = Array2::zeros(layer.wq.raw_dim());
            let mut d_wk: Array2<f32> = Array2::zeros(layer.wk.raw_dim());
            let mut d_wv: Array2<f32> = Array2::zeros(layer.wv.raw_dim());
            let mut d_wo: Array2<f32> = Array2::zeros(layer.wo.raw_dim());
            let mut d_w1: Array2<f32> = Array2::zeros(layer.w1.raw_dim());
            let mut d_w2: Array2<f32> = Array2::zeros(layer.w2.raw_dim());
            let mut d_w3: Array2<f32> = Array2::zeros(layer.w3.raw_dim());
            let mut d_attn_norm = Array1::zeros(cfg.dim);
            let mut d_ffn_norm = Array1::zeros(cfg.dim);

            // --- FFN residual ---
            // x = x_out + ff, so d_ff = d_x (upstream), and d_x_for_ffn = d_x (residual)
            let d_ff = d_x.clone();
            let d_x_after_ffn = d_x.clone();

            // --- FFN output: ff = gate_silu * up @ w2^T ---
            // d_w2 += (gate_silu * up)^T @ d_ff
            // d_in_ff = d_ff @ w2
            let in_ff = &lc.gate_silu * &lc.up; // [T, intermediate]
            for j in 0..cfg.intermediate {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_ff[[i, d]] * in_ff[[i, j]];
                    }
                    d_w2[[d, j]] = s;
                }
            }
            let mut d_in_ff: Array2<f32> = Array2::zeros((t, cfg.intermediate));
            for i in 0..t {
                for j in 0..cfg.intermediate {
                    let mut s = 0.0;
                    for d in 0..cfg.dim {
                        s += d_ff[[i, d]] * layer.w2[[d, j]];
                    }
                    d_in_ff[[i, j]] = s;
                }
            }

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
            // d_w3 += d_up^T @ h_ffn, d_x_up = d_up @ w3
            for j in 0..cfg.intermediate {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_up[[i, j]] * lc.h_ffn[[i, d]];
                    }
                    d_w3[[j, d]] = s;
                }
            }
            let mut d_x_up: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for j in 0..cfg.intermediate {
                        s += d_up[[i, j]] * layer.w3[[j, d]];
                    }
                    d_x_up[[i, d]] = s;
                }
            }

            // --- w1: gate = h_ffn @ w1^T ---
            for j in 0..cfg.intermediate {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_gate[[i, j]] * lc.h_ffn[[i, d]];
                    }
                    d_w1[[j, d]] = s;
                }
            }
            let mut d_x_gate: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for j in 0..cfg.intermediate {
                        s += d_gate[[i, j]] * layer.w1[[j, d]];
                    }
                    d_x_gate[[i, d]] = s;
                }
            }

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
            // d_wo += d_attn_proj^T @ attn_out, d_attn_out = d_attn_proj @ wo
            let d_attn_proj = d_x_after_attn.clone();
            for j in 0..cfg.dim {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_attn_proj[[i, j]] * lc.attn_out[[i, d]];
                    }
                    d_wo[[j, d]] = s;
                }
            }
            let mut d_attn_out: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for j in 0..cfg.dim {
                        s += d_attn_proj[[i, j]] * layer.wo[[j, d]];
                    }
                    d_attn_out[[i, d]] = s;
                }
            }

            // --- Attention: attn_out = softmax(QK^T/sqrt(hd)) @ V ---
            // For each position i, head h, dimension d:
            // attn_out[i, h*hd+d] = sum_j softmax(Q_i @ K_j / sqrt(hd))_j * V[j, kvh*hd+d]
            // This requires computing through the softmax and V for each head position

            // Per-head backward through attention
            let group = cfg.n_heads / cfg.n_kv_heads.max(1);
            let mut d_q: Array2<f32> = Array2::zeros((t, cfg.n_heads * hd));
            let mut d_k: Array2<f32> = Array2::zeros((t, cfg.n_kv_heads * hd));
    let mut d_v: Array2<f32> = Array2::zeros((t, cfg.dim_kv()));

            for head in 0..cfg.n_heads {
                let kvh = head / group;
                for i in 0..t {
                    // For each position i, compute scores and probs
                    let mut scores: Vec<f32> = Vec::with_capacity(i + 1);
                    for j in 0..=i {
                        let mut dot = 0.0;
                        for d in 0..hd {
                            dot += lc.q_rope[[i, head * hd + d]] * lc.k_rope[[j, kvh * hd + d]];
                        }
                        scores.push(dot / (hd as f32).sqrt());
                    }
                    let probs = engine::softmax(&scores);

                    // dL/dV[j, kvh*hd+d] += sum_i d_attn_out[i, h*hd+d] * probs[j] (for j <= i)
                    for j in 0..=i {
                        for d in 0..hd {
                            d_v[[j, kvh * hd + d]] += d_attn_out[[i, head * hd + d]] * probs[j];
                        }
                    }

                    // dL/dscores[j] = sum_d d_attn_out[i, h*hd+d] * V[j, kvh*hd+d]
                    let mut d_scores = vec![0.0; i + 1];
                    for j in 0..=i {
                        for d in 0..hd {
                            d_scores[j] += d_attn_out[[i, head * hd + d]] * lc.v[[j, kvh * hd + d]];
                        }
                    }

                    // Softmax backward: dL/dscore_k = sum_j dL/dp_j * p_j * (delta_{kj} - p_k)
                    // where dL/dp_j is the d_scores we just computed (before softmax, for the attn_out computation)
                    // Actually dL/ds is the gradient w.r.t. the softmax input (scores)
                    let mut d_score_raw = vec![0.0; i + 1];
                    for k in 0..=i {
                        let mut s = 0.0;
                        for j in 0..=i {
                            let delta = if k == j { 1.0 } else { 0.0 };
                            s += d_scores[j] * probs[j] * (delta - probs[k]);
                        }
                        d_score_raw[k] = s / (hd as f32).sqrt();
                    }

                    // dL/dQ[i, h*hd+d] += sum_j d_score_raw[j] * K[j, kvh*hd+d]
                    // dL/dK[j, kvh*hd+d] += d_score_raw[j] * Q[i, h*hd+d]
                    for j in 0..=i {
                        for d in 0..hd {
                            d_q[[i, head * hd + d]] += d_score_raw[j] * lc.k_rope[[j, kvh * hd + d]];
                            d_k[[j, kvh * hd + d]] += d_score_raw[j] * lc.q_rope[[i, head * hd + d]];
                        }
                    }
                }
            }

            // --- RoPE backward (inverse rotation) ---
            engine::apply_rope_all(&mut d_q, cfg.n_heads, hd, cfg.rope_theta);
            engine::apply_rope_all(&mut d_k, cfg.n_kv_heads, hd, cfg.rope_theta);

            // --- Q projection: q = h_attn @ wq^T ---
            for j in 0..cfg.dim {
                for d in 0..cfg.n_heads * hd {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_q[[i, j]] * lc.h_attn[[i, d]];
                    }
                    d_wq[[j, d]] = s;
                }
            }
            let mut d_h_q: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                for d in 0..cfg.dim {
                    if d < cfg.n_heads * hd {
                        for j in 0..cfg.dim {
                            d_h_q[[i, d]] += d_q[[i, j]] * layer.wq[[j, d]];
                        }
                    }
                }
            }

            // --- K projection: k = h_attn @ wk^T ---
            for j in 0..dk {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_k[[i, j]] * lc.h_attn[[i, d]];
                    }
                    d_wk[[j, d]] = s;
                }
            }
            let mut d_h_k: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                for d in 0..cfg.dim {
                    if d < dk {
                        for j in 0..dk {
                            d_h_k[[i, d]] += d_k[[i, j]] * layer.wk[[j, d]];
                        }
                    }
                }
            }

            // --- V projection: v = h_attn @ wv^T ---
            for j in 0..dk {
                for d in 0..cfg.dim {
                    let mut s = 0.0;
                    for i in 0..t {
                        s += d_v[[i, j]] * lc.h_attn[[i, d]];
                    }
                    d_wv[[j, d]] = s;
                }
            }
            let mut d_h_v: Array2<f32> = Array2::zeros((t, cfg.dim));
            for i in 0..t {
                for d in 0..cfg.dim {
                    if d < dk {
                        for j in 0..dk {
                            d_h_v[[i, d]] += d_v[[i, j]] * layer.wv[[j, d]];
                        }
                    }
                }
            }

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

            layer_grads.push(LayerGrads {
                wq: d_wq,
                wk: d_wk,
                wv: d_wv,
                wo: d_wo,
                w1: d_w1,
                w2: d_w2,
                w3: d_w3,
                attn_norm: d_attn_norm,
                ffn_norm: d_ffn_norm,
            });
        }

        layer_grads.reverse();

        // Gradient through embedding lookup
        for (i, &tok) in tokens.iter().enumerate() {
            if i >= t { break; }
            let tok = tok.min(cfg.vocab_size - 1);
            for d in 0..cfg.dim {
                d_embed[[tok, d]] += d_x[[i, d]];
            }
        }

        // Sanitize all gradients
        for v in d_embed.iter_mut() { *v = sanitize(*v); }
        for v in d_output.iter_mut() { *v = sanitize(*v); }
        for v in d_final_norm.iter_mut() { *v = sanitize(*v); }
        for lg in &mut layer_grads {
            for v in lg.wq.iter_mut() { *v = sanitize(*v); }
            for v in lg.wk.iter_mut() { *v = sanitize(*v); }
            for v in lg.wv.iter_mut() { *v = sanitize(*v); }
            for v in lg.wo.iter_mut() { *v = sanitize(*v); }
            for v in lg.w1.iter_mut() { *v = sanitize(*v); }
            for v in lg.w2.iter_mut() { *v = sanitize(*v); }
            for v in lg.w3.iter_mut() { *v = sanitize(*v); }
            for v in lg.attn_norm.iter_mut() { *v = sanitize(*v); }
            for v in lg.ffn_norm.iter_mut() { *v = sanitize(*v); }
        }

        Gradients {
            embed: d_embed,
            layers: layer_grads,
            final_norm: d_final_norm,
            output: d_output,
        }
    }

    /// Apply accumulated gradients with learning rate.
    pub fn apply_gradients(&mut self, grads: &Gradients, lr: f32) {
        let sanitize_lr = |v: f32| lr * sanitize(v);
        for d in 0..self.embed.nrows() {
            for j in 0..self.embed.ncols() {
                self.embed[[d, j]] -= sanitize_lr(grads.embed[[d, j]]);
            }
        }
        for d in 0..self.output.nrows() {
            for j in 0..self.output.ncols() {
                self.output[[d, j]] -= sanitize_lr(grads.output[[d, j]]);
            }
        }
        for d in 0..self.final_norm.len() {
            self.final_norm[d] -= sanitize_lr(grads.final_norm[d]);
        }
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let lg = &grads.layers[li];
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
}

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
    }
}

/// Per-layer key/value cache used during incremental (KV-cached) generation.
/// Each layer stores the RoPE-applied K and V for all positions seen so far,
/// shaped [n_positions, n_kv_heads * head_dim].
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
}
