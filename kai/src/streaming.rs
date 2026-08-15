//! Streaming generation — batched prefill + per-token layer-by-layer

use crate::config::Config;
use crate::gguf::{self, TensorInfo, Weights, LayerWeights, Tokenizer};
use crate::vfe::{compute_vfe, sample_top_p, argmax, VFEState};
use crate::attractor::Attractor;
use crate::config::PhysicsParams;
use ndarray::{Array1, Array2};

/// Key-Value cache with physics metrics.
/// Pre-allocated with max_seq_len (n_prompt + max_new) to avoid O(n²) copies.
/// Each layer's array is indexed by absolute position (0 .. max_seq_len).
pub struct KVCache {
    pub keys: Vec<Array2<f32>>,
    pub values: Vec<Array2<f32>>,
    /// Maximum sequence length (pre-allocated).
    pub max_len: usize,
    pub novelty: Vec<f32>,
    pub curvature: Vec<f32>,
    pub per_layer_novelty: Vec<Vec<f32>>,
    pub tau: f32,
}

impl KVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let ncols = n_kv_heads * head_dim;
        Self {
            keys: vec![Array2::zeros((max_seq_len, ncols)); n_layers],
            values: vec![Array2::zeros((max_seq_len, ncols)); n_layers],
            max_len: max_seq_len,
            novelty: vec![0.0; n_layers],
            curvature: vec![0.0; n_layers],
            per_layer_novelty: vec![vec![]; n_layers],
            tau: 1.0,
        }
    }

    /// Write K,V at absolute position `pos`. O(1) slice assign.
    fn set(&mut self, li: usize, pos: usize, k: Array2<f32>, v: Array2<f32>) {
        self.keys[li].slice_mut(ndarray::s![pos, ..]).assign(&k.row(0));
        self.values[li].slice_mut(ndarray::s![pos, ..]).assign(&v.row(0));
    }
}

const EPS: f32 = 1e-5;

/// Generate tokens using streaming inference
/// Simple wrapper — constructs PhysicsParams from temperature and delegates.
pub fn generate(
    model_path: &str,
    prompt: &str,
    cfg: &Config,
    max_new: usize,
    temperature: f32,
    no_engram: bool,
) -> Result<usize, String> {
    let phys = PhysicsParams {
        base_temperature: temperature,
        top_p: 0.9,
        ..Default::default()
    };
    generate_with_phys(model_path, prompt, cfg, max_new, &phys, no_engram)
}

/// Core generation with full PhysicsParams control (used by Darwin evolution)
pub fn generate_with_phys(
    model_path: &str,
    prompt: &str,
    cfg: &Config,
    max_new: usize,
    phys: &PhysicsParams,
    no_engram: bool,
) -> Result<usize, String> {
    let meta = gguf::read_kv(model_path)?;
    let tokenizer = Tokenizer::from_gguf(&meta)
        .ok_or_else(|| "Failed to build tokenizer from GGUF metadata".to_string())?;
    let (mmap, tensors) = gguf::read_tensors(model_path)?;
    let weights = load_weights(&tensors, &mmap, cfg)?;
    let vocab = cfg.vocab_size.max(weights.embed.shape()[0]);
    let mut engram = if !no_engram {
        Some(Attractor::new(".").map_err(|e| format!("Attractor init: {}", e))?)
    } else {
        None
    };
    let mut vfe = VFEState::default();
    vfe.tau = 1.0;

    // Tokenize
    let mut ids = tokenizer.encode(prompt);
    let n_prompt = ids.len();
    if n_prompt == 0 {
        return Err("Empty prompt".to_string());
    }

    // ── Phase 1: PREFILL (batched by layer) ─────────────────────────
    println!("[prefill] {} tokens × {} layers", n_prompt, cfg.n_layers);

    let head_dim = cfg.dim / cfg.n_heads;
    let max_seq_len = n_prompt + max_new + 1; // +1 for safety
    let mut cache = KVCache::new(cfg.n_layers, cfg.n_kv_heads, head_dim, max_seq_len);

    // Initialize hidden states from embeddings
    let mut hidden_states: Vec<Array2<f32>> = ids.iter().map(|&id| {
        let mut x = Array2::zeros((1, cfg.dim));
        let tid = id.min(vocab - 1);
        x.row_mut(0).assign(&weights.embed.row(tid));
        x
    }).collect();

    // ── Phase 1b: PREFILL — BATCHED by stacking all prompt tokens ─────
    // Instead of (n_prompt × layers × projections), we stack H and do
    // a single H @ W per layer, giving ~6× better cache utilization.
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];

        // Stack all prompt hidden states into one matrix (n_prompt × dim)
        let mut h_stack = Array2::zeros((n_prompt, cfg.dim));
        for (pos, hs) in hidden_states.iter().enumerate() {
            h_stack.row_mut(pos).assign(&hs.row(0));
        }

        // Pre-attention norm (batched)
        let h_norm = rmsnorm_rows(&h_stack, &layer.attn_norm, EPS);

        // Q, K, V projections — ONE matmul per weight instead of n_prompt
        let q_all = linear(&h_norm, &layer.wq)?;   // (n_prompt, dim)
        let k_all = linear(&h_norm, &layer.wk)?;   // (n_prompt, dim_kv)
        let v_all = linear(&h_norm, &layer.wv)?;   // (n_prompt, dim_kv)

        // Per-position: RoPE, cache write, attention
        for pos in 0..n_prompt {
            // Slice single-position Q, K, V
            let mut q = q_all.slice(ndarray::s![pos..pos + 1, ..]).to_owned();
            let mut k = k_all.slice(ndarray::s![pos..pos + 1, ..]).to_owned();
            let v = v_all.slice(ndarray::s![pos..pos + 1, ..]).to_owned();

            // Apply RoPE
            apply_rope(&mut q, &mut k, pos, cfg);

            // Attention with KV cache
            let (attn_out, novelty, curvature) = attention_with_cache(cfg, &mut cache, li, pos, &q, &k, &v)?;

            // Novelty/curvature tracking
            cache.novelty[li] = novelty;
            cache.curvature[li] = curvature;
            cache.per_layer_novelty[li].push(novelty);

            // Output projection + residual
            let attn_proj = linear(&attn_out, &layer.wo)?;
            let residual = &hidden_states[pos] + &attn_proj;

            // FFN (per-position for now — could also batch)
            let h_ffn = rmsnorm_rows(&residual, &layer.ffn_norm, EPS);
            let gate = linear(&h_ffn, &layer.w1)?;
            let up = linear(&h_ffn, &layer.w3)?;
            let silu = gate.mapv(|v| v * (1.0 / (1.0 + (-v).exp())));
            let down = linear(&(silu * up), &layer.w2)?;

            hidden_states[pos] = residual + down;
        }
    }

    // Initial attractor query (if enabled)
    let mut query_emb = Array1::zeros(cfg.dim);
    if let Some(ref mut engram) = engram {
        // Set attractor dimension to match model
        engram.set_dim(cfg.dim);
        if !hidden_states.is_empty() {
            let last = &hidden_states[n_prompt - 1];
            let xf = rmsnorm_rows(last, &weights.final_norm, EPS);
            query_emb.assign(&xf.row(0));
            let memories = engram.query(query_emb.as_slice().unwrap(), 3);
            if !memories.is_empty() {
                println!("  attractor: {} memories", memories.len());
            }
        }
    }

    // ── Phase 2: GENERATION (per-token, per-layer) ──────────────────
    let mut x = hidden_states.pop().unwrap_or_else(|| {
        let mut x = Array2::zeros((1, cfg.dim));
        let tid = ids.last().copied().unwrap_or(0).min(cfg.vocab_size - 1);
        x.row_mut(0).assign(&weights.embed.row(tid));
        x
    });

    println!("[gen] max_new={}", max_new);
    let mut generated = String::new();

    for gen_pos in 0..max_new {
        let abs_pos = n_prompt + gen_pos;

        for li in 0..cfg.n_layers {
            x = forward_layer(cfg, &mut cache, li, abs_pos, &x, &weights.layers[li])?;
        }

        // Final norm + logits
        let xf = rmsnorm_rows(&x, &weights.final_norm, EPS);
        let logits = xf.dot(&weights.output);

        // Physics-wired sampling
        let logits_row = logits.row(0);
        let logits_vec: Vec<f32> = logits_row.iter().copied().collect();
        let probs = softmax_1d(&logits_vec);

        // VFE adaptive sampling with attractor-based prior
        let target = argmax(&probs);
        let logits_view = logits.row(0);

        // Build prior from attractor memory (or uniform if disabled)
        let prior = if let Some(ref engram) = engram {
            let emb_slice = xf.row(0);
            let prior_vec = engram.build_token_prior(emb_slice.as_slice().unwrap(), vocab, 5, 0.3);
            Array1::from_vec(prior_vec)
        } else {
            Array1::ones(vocab) / vocab as f32
        };

        let temp_eff = compute_vfe(&logits_view, target, &prior.view(), &mut vfe, &phys);

        let next = if temp_eff > 0.0 {
            sample_top_p(&logits_vec, temp_eff, 0.9)
        } else {
            argmax(&probs)
        };

        // Output token
        let token_str = simple_decode(&tokenizer, next);
        print!("{}", token_str);
        use std::io::Write;
        std::io::stdout().flush().ok();
        generated.push_str(&token_str);

        // Store in attractor memory (push embedding + generated tokens)
        if let Some(ref mut engram) = engram {
            let emb = xf.row(0).to_owned();
            let gen_tokens: Vec<usize> = vec![next];
            engram.push(
                &format!("gen_{}", ids.len()),
                &token_str,
                &gen_tokens,
                &emb.view(),
            )
            .map_err(|e| format!("Attractor push: {}", e))?;
        }

        // Update for next iteration
        ids.push(next);
        if next == tokenizer.eos { break; }

        // Prepare next input
        x = Array2::zeros((1, cfg.dim));
        let tid = next.min(cfg.vocab_size - 1);
        x.row_mut(0).assign(&weights.embed.row(tid));
    }

    println!();

    // Return number of generated tokens (excluding prompt)
    let gen_count = if max_new <= ids.len().saturating_sub(n_prompt) {
        max_new
    } else {
        ids.len() - n_prompt
    };
    Ok(gen_count)
}

fn forward_layer(
    cfg: &Config,
    cache: &mut KVCache,
    li: usize,
    pos: usize,
    x: &Array2<f32>,
    layer: &LayerWeights,
) -> Result<Array2<f32>, String> {
    // Pre-attention norm
    let h = rmsnorm_rows(x, &layer.attn_norm, EPS);

    // Q, K, V projections
    let mut q = linear(&h, &layer.wq)?;
    let mut k = linear(&h, &layer.wk)?;
    let v = linear(&h, &layer.wv)?;

    // Apply RoPE to Q and K
    apply_rope(&mut q, &mut k, pos, cfg);

    // Attention with KV cache
    let (attn_out, novelty, curvature) = attention_with_cache(cfg, cache, li, pos, &q, &k, &v)?;

    // Novelty/curvature tracking
    cache.novelty[li] = novelty;
    cache.curvature[li] = curvature;
    cache.per_layer_novelty[li].push(novelty);

    // Output projection + residual
    let attn_proj = linear(&attn_out, &layer.wo)?;
    let h = x + attn_proj;

    // FFN
    let h_ffn = rmsnorm_rows(&h, &layer.ffn_norm, EPS);
    let gate = linear(&h_ffn, &layer.w1)?;  // SiLU gate
    let up = linear(&h_ffn, &layer.w3)?;
    // SiLU(x) = x * sigmoid(x) = x * (1/(1+e^-x))
    let silu = gate.mapv(|v| v * (1.0 / (1.0 + (-v).exp())));
    let down = linear(&(silu * up), &layer.w2)?;

    // Residual
    Ok(h + down)
}

fn attention_with_cache(
    cfg: &Config,
    cache: &mut KVCache,
    li: usize,
    pos: usize,
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
) -> Result<(Array2<f32>, f32, f32), String> {
    let n_kv = cfg.n_kv_heads;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.dim / n_heads;
    let _n_repeat = n_heads / n_kv;

    // Store K, V in cache — O(1) pre-allocated write at absolute position
    let k_flat = k.clone().into_shape((1, n_kv * head_dim))
        .map_err(|e| format!("k flat: {}", e))?;
    let v_flat = v.clone().into_shape((1, n_kv * head_dim))
        .map_err(|e| format!("v flat: {}", e))?;
    cache.set(li, pos, k_flat, v_flat);

    // Reshape Q to [1, n_heads, head_dim]
    let q_r = q.clone().into_shape((1, n_heads, head_dim))
        .map_err(|e| format!("q reshape: {}", e))?;

    // Reconstruct full K, V with GQA repeat
    let seq_len = pos + 1; // number of cached positions for this layer
    // Slice pre-allocated array to actual seq_len before reshape
    let k_stored = cache.keys[li].slice(ndarray::s![..seq_len, ..]).to_owned();
    let v_stored = cache.values[li].slice(ndarray::s![..seq_len, ..]).to_owned();

    // K: [seq_len, n_kv*head_dim] -> [seq_len, n_kv, head_dim] -> repeat to [seq_len, n_heads, head_dim]
    let k_3d = k_stored.into_shape((seq_len, n_kv, head_dim))
        .map_err(|e| format!("k_3d reshape: {}", e))?;
    let v_3d = v_stored.into_shape((seq_len, n_kv, head_dim))
        .map_err(|e| format!("v_3d reshape: {}", e))?;

    // Build full K, V by repeating along head dim
    let mut k_full = Array3D::zeros((seq_len, n_heads, head_dim));
    let mut v_full = Array3D::zeros((seq_len, n_heads, head_dim));
    for h in 0..n_heads {
        let kv_h = h % n_kv;
        k_full.slice_mut(ndarray::s![.., h, ..]).assign(&k_3d.slice(ndarray::s![.., kv_h, ..]));
        v_full.slice_mut(ndarray::s![.., h, ..]).assign(&v_3d.slice(ndarray::s![.., kv_h, ..]));
    }

    // Attention scores: [1, n_heads, head_dim] @ [head_dim, seq_len] -> [1, n_heads, seq_len]
    // We need to do per-head or reshape to 2D
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape q to [n_heads, head_dim] and k_full to [seq_len, n_heads, head_dim]
    // Then for each head: scores[h] = q[h] @ k[:,h,:]^T = [1, seq_len]
    let mut scores = Array2::zeros((n_heads, seq_len));
    for h in 0..n_heads {
        let q_h = q_r.slice(ndarray::s![0, h, ..]);
        let k_h = k_full.slice(ndarray::s![.., h, ..]);
        for s in 0..seq_len {
            let k_s = k_h.slice(ndarray::s![s, ..]);
            // Use fold for dot product to avoid ambiguity
            let dot: f32 = q_h.iter().zip(k_s.iter()).map(|(a, b)| a * b).sum();
            scores[[h, s]] = dot * scale;
        }
    }

    // Softmax over seq_len dimension for each head
    let mut attn_weights = Array2::zeros((n_heads, seq_len));
    for h in 0..n_heads {
        let row = scores.row(h);
        let max_s = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_row: Array1<f32> = row.mapv(|x| (x - max_s).exp());
        let sum_exp = exp_row.sum();
        attn_weights.row_mut(h).assign(&(exp_row / sum_exp));
    }

    // Novelty = normalized entropy, Curvature = 1 - max attention
    let entropy = attn_weights.mapv(|x| if x > 1e-10 { -x * x.ln() } else { 0.0 }).sum() / n_heads as f32;
    let max_attn = attn_weights.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let novelty = entropy / (n_heads as f32).ln();
    let curvature = 1.0 - max_attn;

    // Attention output: attn_weights [n_heads, seq_len] @ v_full [seq_len, n_heads, head_dim]
    // Result: [n_heads, head_dim]
    let mut attn_out = Array2::zeros((n_heads, head_dim));
    for h in 0..n_heads {
        let mut sum = Array1::zeros(head_dim);
        for s in 0..seq_len {
            let w = attn_weights[[h, s]];
            let v_h = v_full.slice(ndarray::s![s, h, ..]);
            sum = sum + v_h.to_owned() * w;
        }
        attn_out.row_mut(h).assign(&sum);
    }

    // Flatten to [1, dim]
    let out = attn_out.into_shape((1, cfg.dim))
        .map_err(|e| format!("attn out reshape: {}", e))?;

    Ok((out, novelty, curvature))
}

type Array3D = ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>;

/// Linear transformation: y = x @ W^T   (1D x: [dim_in], W: [dim_out, dim_in])
fn linear(x: &Array2<f32>, w: &Array2<f32>) -> Result<Array2<f32>, String> {
    // x: [batch, dim_in], w: [dim_in, dim_out] (GGUF storage convention)
    // Result: [batch, dim_out]
    Ok(x.dot(w))
}

/// Apply Rotary Position Embeddings to Q and K in-place.
/// q: [batch, n_heads * head_dim], k: [batch, n_kv_heads * head_dim]
/// Only the first rope_dim dimensions of each head are rotated.
fn apply_rope(q: &mut Array2<f32>, k: &mut Array2<f32>, pos: usize, cfg: &Config) {
    let head_dim = cfg.dim / cfg.n_heads;
    let rope_dim = cfg.rope_dim.min(head_dim);
    let theta = cfg.rope_theta;

    // Precompute frequency sin/cos for this position
    let mut freq = vec![0.0f32; rope_dim];
    for i in 0..rope_dim / 2 {
        let theta_i = theta.powf(-2.0 * i as f32 / rope_dim as f32);
        let angle = pos as f32 * theta_i;
        let (sin, cos) = angle.sin_cos();
        freq[2 * i] = cos;
        freq[2 * i + 1] = sin;
    }

    // Rotate a single matrix given number of heads
    let rotate = |mat: &mut Array2<f32>, n_heads: usize| {
        let batch = mat.shape()[0];
        for b in 0..batch {
            for h in 0..n_heads {
                let offset = h * head_dim;
                for i in 0..rope_dim / 2 {
                    let i2 = 2 * i;
                    let idx0 = offset + i2;
                    let idx1 = offset + i2 + 1;
                    let x0 = mat[[b, idx0]];
                    let x1 = mat[[b, idx1]];
                    mat[[b, idx0]] = x0 * freq[i2] - x1 * freq[i2 + 1];
                    mat[[b, idx1]] = x0 * freq[i2 + 1] + x1 * freq[i2];
                }
            }
        }
    };

    rotate(q, cfg.n_heads);
    rotate(k, cfg.n_kv_heads);
}

fn rmsnorm_rows(x: &Array2<f32>, w: &Array1<f32>, eps: f32) -> Array2<f32> {
    let mut y = x.clone();
    for mut row in y.rows_mut() {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let rms = (mean_sq + eps).sqrt();
        for j in 0..row.len() {
            row[j] = row[j] / rms * w[j];
        }
    }
    y
}

fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|v| v / sum).collect()
}

fn simple_decode(tokenizer: &Tokenizer, id: usize) -> String {
    // Reverse lookup: find the token string for a given id
    for (s, &i) in &tokenizer.vocab {
        if i == id {
            // Replace the model-specific space prefix with an actual space
            return s.replace(&tokenizer.space_prefix, " ")
                     .replace("</s>", "")
                     .replace("<unk>", "?");
        }
    }
    format!("[{}]", id)
}

fn load_weights(
    tensors: &std::collections::HashMap<String, TensorInfo>,
    mmap: &memmap2::Mmap,
    cfg: &Config,
) -> Result<Weights, String> {
    let dequant = |info: &TensorInfo| -> Result<Array2<f32>, String> {
        let data = &mmap[info.file_offset as usize..];
        let n = info.shape.iter().product::<usize>();
        let mut out = Vec::with_capacity(n);

        match info.ggml_type {
            0 => { // F32
                let ptr = data.as_ptr() as *const f32;
                for i in 0..n { out.push(unsafe { *ptr.add(i) }); }
            }
            1 => { // F16
                let ptr = data.as_ptr() as *const u16;
                for i in 0..n { out.push(f16_to_f32(unsafe { *ptr.add(i) })); }
            }
            2 => { // Q4_0
                let blocks = n / 32;
                let ptr = data.as_ptr() as *const u8;
                for i in 0..blocks {
                    let d = unsafe { &*ptr.add(i * 18).cast::<[u8; 18]>() };
                    let scale = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
                    let mut qs = [0i8; 32];
                    for j in 0..16 {
                        qs[j*2] = (d[2+j] & 0x0F) as i8 - 8;
                        qs[j*2+1] = (d[2+j] >> 4) as i8 - 8;
                    }
                    for q in qs { out.push(q as f32 * scale); }
                }
            }
            3 => { // Q4_1
                let blocks = n / 32;
                let ptr = data.as_ptr() as *const u8;
                for i in 0..blocks {
                    let d = unsafe { &*ptr.add(i * 20).cast::<[u8; 20]>() };
                    let scale = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
                    let min = f16_to_f32(u16::from_le_bytes([d[2], d[3]]));
                    let mut qs = [0i8; 32];
                    for j in 0..16 {
                        qs[j*2] = (d[4+j] & 0x0F) as i8 - 8;
                        qs[j*2+1] = (d[4+j] >> 4) as i8 - 8;
                    }
                    for q in qs { out.push(min + q as f32 * scale); }
                }
            }
            6 => { // Q5_0
                let blocks = n / 32;
                let ptr = data.as_ptr() as *const u8;
                for i in 0..blocks {
                    let d = unsafe { &*ptr.add(i * 22).cast::<[u8; 22]>() };
                    let scale = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
                    let mut qs = [0i8; 32];
                    for j in 0..16 {
                        qs[j*2] = (d[2+j] & 0x0F) as i8 - 16;
                        qs[j*2+1] = (d[2+j] >> 4) as i8 - 16;
                    }
                    let high = d[18];
                    for j in 0..16 {
                        let bit = (high >> j) & 1;
                        qs[j*2] |= (bit << 4) as i8;
                        qs[j*2+1] |= (bit << 5) as i8;
                    }
                    for q in qs { out.push(q as f32 * scale); }
                }
            }
            7 => { // Q5_1
                let blocks = n / 32;
                let ptr = data.as_ptr() as *const u8;
                for i in 0..blocks {
                    let d = unsafe { &*ptr.add(i * 24).cast::<[u8; 24]>() };
                    let scale = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
                    let min = f16_to_f32(u16::from_le_bytes([d[2], d[3]]));
                    let mut qs = [0i8; 32];
                    for j in 0..16 {
                        qs[j*2] = (d[4+j] & 0x0F) as i8 - 16;
                        qs[j*2+1] = (d[4+j] >> 4) as i8 - 16;
                    }
                    let high = d[20];
                    for j in 0..16 {
                        let bit = (high >> j) & 1;
                        qs[j*2] |= (bit << 4) as i8;
                        qs[j*2+1] |= (bit << 5) as i8;
                    }
                    for q in qs { out.push(min + q as f32 * scale); }
                }
            }
            8 => { // Q8_0
                let blocks = n / 32;
                let ptr = data.as_ptr() as *const u8;
                for i in 0..blocks {
                    let d = unsafe { &*ptr.add(i * 34).cast::<[u8; 34]>() };
                    let scale = f16_to_f32(u16::from_le_bytes([d[0], d[1]]));
                    for j in 0..32 {
                        out.push(d[2+j] as i8 as f32 * scale);
                    }
                }
            }
             12 => { // Q4_K: 144 bytes per 256 elements
                const QK_K: usize = 256;
                const BLOCK_SIZE: usize = 144; // d[2] + dmin[2] + scales[12] + qs[128]
                let blocks = n / QK_K;
                let ptr = data.as_ptr() as *const u8;
                let get_scale_min = |j: usize, sc: &[u8]| -> (u8, u8) {
                    if j < 4 {
                        (sc[j] & 63, sc[j + 4] & 63)
                    } else {
                        let d = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                        let m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
                        (d, m)
                    }
                };
                for i in 0..blocks {
                    let block = unsafe { &*ptr.add(i * BLOCK_SIZE).cast::<[u8; BLOCK_SIZE]>() };
                    let d_val = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    let min   = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                    let scales = &block[4..16];
                    let qs = &block[16..144];
                    let mut is = 0;
                    let mut qs_off = 0;
                    for _j in (0..QK_K).step_by(64) {
                        let (sc0, m0) = get_scale_min(is + 0, scales);
                        let d1 = d_val * sc0 as f32;
                        let m1 = min * m0 as f32;
                        let (sc1, m1v) = get_scale_min(is + 1, scales);
                        let d2 = d_val * sc1 as f32;
                        let m2 = min * m1v as f32;
                        // First sub-block: low nibbles of qs[qs_off..qs_off+32]
                        for l in 0..32 {
                            let q = (qs[qs_off + l] & 0x0F) as f32;
                            out.push(d1 * q - m1);
                        }
                        // Second sub-block: high nibbles of qs[qs_off..qs_off+32]
                        for l in 0..32 {
                            let q = (qs[qs_off + l] >> 4) as f32;
                            out.push(d2 * q - m2);
                        }
                        qs_off += 32;
                        is += 2;
                    }
                }
            }
             14 => { // Q6_K: 210 bytes per 256 elements
                const QK_K: usize = 256;
                const BLOCK_SIZE: usize = 210; // ql[128] + qh[64] + scales[16] + d[2]
                let blocks = n / QK_K;
                let ptr = data.as_ptr() as *const u8;
                for i in 0..blocks {
                    let block = unsafe { &*ptr.add(i * BLOCK_SIZE).cast::<[u8; BLOCK_SIZE]>() };
                    let ql = &block[0..128];
                    let qh = &block[128..192];
                    // Scales are int8_t (signed) — cast from u8 to i8 to f32
                    let sc = &block[192..208];
                    let d_val = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                    
                    // Process in two 128-element halves
                    for half in 0..2 {
                        let ql_off = half * 64;
                        let qh_off = half * 32;
                        let sc_off = half * 8;
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[ql_off + l] & 0x0F) as i32 | (((qh[qh_off + l] as i32 >> 0) & 3) << 4)) - 32;
                            let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32 | (((qh[qh_off + l] as i32 >> 2) & 3) << 4)) - 32;
                            let q3 = ((ql[ql_off + l] as i32 >> 4) | (((qh[qh_off + l] as i32 >> 4) & 3) << 4)) - 32;
                            let q4 = ((ql[ql_off + l + 32] as i32 >> 4) | (((qh[qh_off + l] as i32 >> 6) & 3) << 4)) - 32;
                            // Scales are signed int8_t in the block
                            let s0 = (sc[sc_off + is + 0] as i8) as f32;
                            let s1 = (sc[sc_off + is + 2] as i8) as f32;
                            let s2 = (sc[sc_off + is + 4] as i8) as f32;
                            let s3 = (sc[sc_off + is + 6] as i8) as f32;
                            out.push(d_val * s0 * q1 as f32);
                            out.push(d_val * s1 * q2 as f32);
                            out.push(d_val * s2 * q3 as f32);
                            out.push(d_val * s3 * q4 as f32);
                        }
                    }
                }
            }
            _ => return Err(format!("Unsupported GGML type: {}", info.ggml_type)),
        }

        if info.shape.len() == 2 {
            Ok(Array2::from_shape_vec((info.shape[0], info.shape[1]), out)
                .map_err(|e| format!("shape {}x{}: {}", info.shape[0], info.shape[1], e))?)
        } else if info.shape.len() == 1 {
            Ok(Array2::from_shape_vec((1, info.shape[0]), out)
                .map_err(|e| format!("shape 1x{}: {}", info.shape[0], e))?)
        } else {
            Err(format!("Unsupported tensor shape: {:?}", info.shape))
        }
    };

    // Load embeddings
    let embed_info = tensors.get("token_embd.weight")
        .or_else(|| tensors.get("model.embed_tokens.weight"))
        .or_else(|| tensors.get("transformer.wte.weight"))
        .ok_or("No embedding weight found")?;
    let embed_arr = dequant(embed_info)?;
    // GGUF stores as [dim, vocab]; we need [vocab, dim] for row lookup.
    // Keep a copy in the original layout for use as the output projection.
    let embed_out = embed_arr.clone();
    let embed = if embed_arr.shape()[0] == cfg.dim {
        // [dim, vocab] — transpose for row lookup
        embed_arr.reversed_axes()
    } else {
        embed_arr
    };
    // Prepare output: defaults to original (un-transposed) embed for tied weights
    let mut output = embed_out;

    // If a separate output tensor exists, use it instead
    if let Some(info) = tensors.get("output.weight")
        .or_else(|| tensors.get("lm_head.weight"))
        .or_else(|| tensors.get("transformer.wte.weight")) {
        output = dequant(info)?;
        // Ensure output is [dim, vocab] for x.dot(&output)
        if output.shape()[0] != cfg.dim {
            output = output.reversed_axes();
        }
    }
    // output stays as [dim, vocab]; used as x.dot(&output)

    // Load final norm
    let final_norm_info = tensors.get("output_norm.weight")
        .or_else(|| tensors.get("model.norm.weight"))
        .or_else(|| tensors.get("transformer.ln_f.weight"))
        .ok_or("No final norm weight found")?;
    let final_norm_arr = dequant(final_norm_info)?;
    let final_norm = final_norm_arr.into_shape(final_norm_info.shape[0])
        .map_err(|e| format!("final_norm shape: {}", e))?;

    // Load layers
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for li in 0..cfg.n_layers {
        let prefix = format!("blk.{li}.");

        let get = |name: &str| -> Result<Array2<f32>, String> {
            tensors.get(&format!("{prefix}{name}"))
                .ok_or_else(|| format!("Missing {prefix}{name}"))
                .and_then(|info| dequant(info))
        };

        let attn_norm_arr = get("attn_norm.weight")?;
        let attn_norm = attn_norm_arr.into_shape(cfg.dim)
            .map_err(|e| format!("attn_norm shape: {}", e))?;
        let ffn_norm_arr = get("ffn_norm.weight")?;
        let ffn_norm = ffn_norm_arr.into_shape(cfg.dim)
            .map_err(|e| format!("ffn_norm shape: {}", e))?;

        layers.push(LayerWeights {
            attn_norm,
            ffn_norm,
            wq: get("attn_q.weight")?,
            wk: get("attn_k.weight")?,
            wv: get("attn_v.weight")?,
            wo: get("attn_output.weight")?,
            w1: get("ffn_gate.weight")?,
            w2: get("ffn_down.weight")?,
            w3: get("ffn_up.weight")?,
        });
    }

    Ok(Weights { embed, output, final_norm, layers })
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign != 0 { -0.0 } else { 0.0 }; }
        let mut m = mant as i32;
        let mut e = -14;
        while (m & 0x400) == 0 { m <<= 1; e -= 1; }
        m &= !0x400;
        return f32::from_bits((sign << 31) | ((e + 127) as u32) << 23 | (m as u32) << 13);
    } else if exp == 31 {
        return if mant == 0 { f32::INFINITY * if sign != 0 { -1.0 } else { 1.0 } } else { f32::NAN };
    }
    f32::from_bits((sign << 31) | ((exp - 15 + 127) as u32) << 23 | (mant << 13))
}
