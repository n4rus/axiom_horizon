use crate::config::Config;
use crate::engine;
use crate::gguf::{self, GgufBuffer};
use crate::model;
use crate::tok::Tokenizer;
use ndarray::{Array1, Array2, s};

struct PairCache {
    x_in: Array2<f32>,
}

struct WeightCache {
    embed: Array2<f32>,
    output: Array2<f32>,
    final_norm: Array1<f32>,
    layers: Vec<model::LayerWeights>,
    needs_output_transpose: bool,
}

/// Returns true if the model is small enough to cache all weights in f32.
fn should_cache(cfg: &Config) -> bool {
    cfg.estimated_f32_gb() < 4.0
}

fn load_all_weights(buf: &GgufBuffer, cfg: &Config) -> Result<WeightCache, String> {
    let embed_raw = buf.dequant_arr2("token_embd.weight")?;
    let embed = embed_raw.t().to_owned();
    let (output, needs_output_transpose) = if buf.tensor("output.weight").is_some() {
        let w = buf.dequant_arr2("output.weight")?;
        if w.nrows() != cfg.vocab_size {
            (w.t().to_owned(), true)
        } else {
            (w, false)
        }
    } else {
        (embed.clone(), false)
    };
    let final_norm = {
        let tinfo = buf.tensor("output_norm.weight").ok_or("missing output_norm.weight")?;
        let d = gguf::read_tensor(&buf.bytes, tinfo, buf.data_start)?;
        Array1::from_shape_vec(tinfo.shape[0], d).map_err(|e| e.to_string())?
    };
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for li in 0..cfg.n_layers {
        layers.push(load_layer(buf, cfg, li)?);
    }
    Ok(WeightCache { embed, output, final_norm, layers, needs_output_transpose })
}

/// Train using chunked streaming with full backward pass and optional distillation.
/// `chunks`: slice of (start, end) token-range pairs for independent training.
/// Each chunk is trained independently (no causal masking across chunk boundaries).
/// `distillation_lambda`: weight for KL(student || teacher) loss (0.0 = no distillation).
/// `teacher_path`: optional path to teacher model (defaults to same path for self-distillation).
pub fn train_chunked(
    path: &str,
    text: &str,
    iters: usize,
    lr: f32,
    output_path: &str,
    chunks: &[(usize, usize)],
    distillation_lambda: f32,
    teacher_path: Option<&str>,
) -> Result<(), String> {
    let meta = gguf::read_kv(path).map_err(|e| format!("meta: {e}"))?;
    let cfg = gguf::build_config(&meta).ok_or("could not build Config")?;
    let tok = Tokenizer::from_gguf(&meta).ok_or("no tokenizer")?;
    let ids = tok.encode(text);
    if ids.len() < 2 {
        return Err("need >=2 tokens".into());
    }
    if chunks.is_empty() {
        return Err("no chunks provided".into());
    }

    let mut buf = GgufBuffer::open(path)?;
    let dim = cfg.dim;

    let teacher_path = teacher_path.unwrap_or(path);

    let n_chunks = chunks.len();
    println!(
        "Chunked training (full backward): dim={} layers={} vocab={} text_tokens={} chunks={} lr={} iters={} distill_lambda={}",
        dim, cfg.n_layers, cfg.vocab_size, ids.len(), n_chunks, lr, iters, distillation_lambda
    );

    // --- TEACHER FORWARD PASS: run full sequence through teacher to get logits for all positions ---
    let teacher_logits = {
        let meta = gguf::read_kv(teacher_path).map_err(|e| format!("teacher meta: {e}"))?;
        let teacher_cfg = gguf::build_config(&meta).ok_or("could not build teacher Config")?;
        let teacher_tok = Tokenizer::from_gguf(&meta).ok_or("no teacher tokenizer")?;
        let ids = teacher_tok.encode(text);
        if ids.len() < 2 {
            return Err("teacher needs >=2 tokens".into());
        }

        let buf = GgufBuffer::open(teacher_path)?;
        let dim = teacher_cfg.dim;
        let t = ids.len().min(teacher_cfg.max_seq);

        let embed_raw = buf.dequant_arr2("token_embd.weight")?;
        let embed = embed_raw.t().to_owned();
        let output = if buf.tensor("output.weight").is_some() {
            let w = buf.dequant_arr2("output.weight")?;
            if w.nrows() != teacher_cfg.vocab_size { w.t().to_owned() } else { w }
        } else {
            embed.clone()
        };
        let final_norm = {
            let tinfo = buf.tensor("output_norm.weight").ok_or("missing output_norm.weight")?;
            let d = gguf::read_tensor(&buf.bytes, tinfo, buf.data_start)?;
            Array1::from_shape_vec(tinfo.shape[0], d).map_err(|e| e.to_string())?
        };

        let mut x = Array2::zeros((t, dim));
        for (pos, &tok_id) in ids.iter().enumerate().take(t) {
            let tok_id = tok_id.min(teacher_cfg.vocab_size - 1);
            for d in 0..dim { x[[pos, d]] = embed[[tok_id, d]]; }
        }

        for li in 0..teacher_cfg.n_layers {
            let lw = load_layer(&buf, &teacher_cfg, li)?;
            let lc = model::forward_layer(&lw, &teacher_cfg, &x);
            x = lc.x_out;
        }

        let xf = engine::rmsnorm_rows(&x, &final_norm, engine::EPS);
        engine::linear(&xf, &output)
    };
    // teacher_logits shape: [T, vocab_size]

    if should_cache(&cfg) {
        let mut cache = load_all_weights(&buf, &cfg)?;
        let mut d_output = Array2::zeros(cache.output.raw_dim());

        for it in 0..iters {
            let mut total_loss = 0.0f32;
            let mut n = 0usize;
            let mut total_ce = 0.0f32;
            let mut total_kl = 0.0f32;

            for &(chunk_start, chunk_end) in chunks {
                for i in chunk_start..chunk_end - 1 {
                    let ctx: Vec<usize> = ids[chunk_start..=i].to_vec();
                    let target = ids[i + 1];
                    let t = ctx.len();

                    let mut x = Array2::zeros((t, dim));
                    for (pos, &tok_id) in ctx.iter().enumerate() {
                        let tok_id = tok_id.min(cfg.vocab_size - 1);
                        for d in 0..dim { x[[pos, d]] = cache.embed[[tok_id, d]]; }
                    }

                    let mut pair_caches: Vec<PairCache> = Vec::with_capacity(cfg.n_layers);
                    for li in 0..cfg.n_layers {
                        let lc = model::forward_layer(&cache.layers[li], &cfg, &x);
                        pair_caches.push(PairCache { x_in: x.clone() });
                        x = lc.x_out;
                    }

                    let xf = engine::rmsnorm_rows(&x, &cache.final_norm, engine::EPS);
                    let logits = engine::linear(&xf, &cache.output);

                    let raw: Vec<f32> = (0..cfg.vocab_size).map(|j| logits[[t - 1, j]]).collect();
                    let p = engine::softmax(&raw);
                    let ce_loss = -p[target].max(1e-9).ln();
                    let global_target_pos = chunk_start + i + 1;
                    let kl_loss = if distillation_lambda > 0.0 && global_target_pos < teacher_logits.nrows() {
                        let teacher_row = teacher_logits.slice(s![global_target_pos, ..]).to_vec();
                        engine::kl_divergence(&raw, &teacher_row)
                    } else { 0.0 };

                    total_loss += ce_loss + distillation_lambda * kl_loss;
                    total_ce += ce_loss;
                    total_kl += kl_loss;
                    n += 1;

                    let mut d_logits = Array2::zeros((t, cfg.vocab_size));
                    for j in 0..cfg.vocab_size { d_logits[[t - 1, j]] = p[j] - if j == target { 1.0 } else { 0.0 }; }
                    if distillation_lambda > 0.0 && global_target_pos < teacher_logits.nrows() {
                        let teacher_row = teacher_logits.slice(s![global_target_pos, ..]).to_vec();
                        let sum_t: f32 = teacher_row.iter().map(|&x| (x).exp()).sum();
                        for j in 0..cfg.vocab_size {
                            let q = (teacher_row[j]).exp() / sum_t;
                            let pj = p[j];
                            if pj > 1e-9 && q > 1e-9 { d_logits[[t - 1, j]] += distillation_lambda * (pj - q); }
                        }
                    }

                    d_output.fill(0.0);
                    let d_out_mat = d_logits.t().dot(&xf);
                    ndarray::Zip::from(&mut d_output).and(&d_out_mat).for_each(|out, &v| { *out = if v.is_finite() { v } else { 0.0 }; });
                    let d_xf_raw = d_logits.dot(&cache.output);

                    let mut d_x = Array2::zeros((t, dim));
                    for pos in 0..t {
                        let row: Vec<f32> = (0..dim).map(|d| xf[[pos, d]]).collect();
                        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                        let inv_std: f32 = 1.0 / (mean_sq + engine::EPS).sqrt();
                        for d in 0..dim { d_x[[pos, d]] = d_xf_raw[[pos, d]] * inv_std * cache.final_norm[d]; }
                        let mut sum_term = 0.0;
                        for d in 0..dim { sum_term += d_xf_raw[[pos, d]] * cache.final_norm[d] * row[d]; }
                        sum_term *= -inv_std.powi(3) / dim as f32;
                        for d in 0..dim { d_x[[pos, d]] += sum_term * row[d]; }
                    }

                    for li in (0..cfg.n_layers).rev() {
                        let pair = &pair_caches[li];
                        let lc = model::forward_layer(&cache.layers[li], &cfg, &pair.x_in);
                        let (grads, d_x_prev) = model::backward_layer(&cache.layers[li], &cfg, &lc, &d_x);
                        lr_apply_layer(&mut cache.layers[li], &grads, lr);
                        d_x = d_x_prev;
                    }

                    ndarray::Zip::from(&mut cache.output).and(&d_output).for_each(|out, &g: &f32| {
                        if g.is_finite() { *out -= lr * g.max(-1.0).min(1.0); }
                    });
                }
            }

            let avg_loss = total_loss / n as f32;
            let avg_ce = total_ce / n as f32;
            let avg_kl = total_kl / n as f32;
            println!("  iter {it:>3}:  loss={avg_loss:.4}  CE={avg_ce:.4}  KL={avg_kl:.4}");
        }

        save_all_weights(&mut buf, &cache)?;
    } else {
        println!("Model is too large for f32 cache. Using streaming dequant per step (slower, lower memory).");

        for it in 0..iters {
            let mut total_loss = 0.0f32;
            let mut n = 0usize;
            let mut total_ce = 0.0f32;
            let mut total_kl = 0.0f32;

            for &(chunk_start, chunk_end) in chunks {
                for i in chunk_start..chunk_end - 1 {
                    let ctx: Vec<usize> = ids[chunk_start..=i].to_vec();
                    let target = ids[i + 1];
                    let t = ctx.len();

                    // Embedding
                    let embed_raw = buf.dequant_arr2("token_embd.weight")?;
                    let embed = embed_raw.t().to_owned();
                    let mut x = Array2::zeros((t, dim));
                    for (pos, &tok_id) in ctx.iter().enumerate() {
                        let tok_id = tok_id.min(cfg.vocab_size - 1);
                        for d in 0..dim { x[[pos, d]] = embed[[tok_id, d]]; }
                    }

                    // Forward pass — load one layer at a time
                    let mut pair_caches: Vec<PairCache> = Vec::with_capacity(cfg.n_layers);
                    for li in 0..cfg.n_layers {
                        let lw = load_layer(&buf, &cfg, li)?;
                        let lc = model::forward_layer(&lw, &cfg, &x);
                        pair_caches.push(PairCache { x_in: x.clone() });
                        x = lc.x_out;
                    }

                    // Final norm + output projection
                    let fnorm = {
                        let tinfo = buf.tensor("output_norm.weight").ok_or("missing output_norm.weight")?;
                        let d = gguf::read_tensor(&buf.bytes, tinfo, buf.data_start)?;
                        Array1::from_shape_vec(tinfo.shape[0], d).map_err(|e| e.to_string())?
                    };
                    let out = if buf.tensor("output.weight").is_some() {
                        let w = buf.dequant_arr2("output.weight")?;
                        if w.nrows() != cfg.vocab_size { w.t().to_owned() } else { w }
                    } else {
                        embed.clone()
                    };

                    let xf = engine::rmsnorm_rows(&x, &fnorm, engine::EPS);
                    let logits = engine::linear(&xf, &out);

                    let raw: Vec<f32> = (0..cfg.vocab_size).map(|j| logits[[t - 1, j]]).collect();
                    let p = engine::softmax(&raw);
                    let ce_loss = -p[target].max(1e-9).ln();
                    let global_target_pos = chunk_start + i + 1;
                    let kl_loss = if distillation_lambda > 0.0 && global_target_pos < teacher_logits.nrows() {
                        let teacher_row = teacher_logits.slice(s![global_target_pos, ..]).to_vec();
                        engine::kl_divergence(&raw, &teacher_row)
                    } else { 0.0 };

                    total_loss += ce_loss + distillation_lambda * kl_loss;
                    total_ce += ce_loss;
                    total_kl += kl_loss;
                    n += 1;

                    // d_logits
                    let mut d_logits = Array2::zeros((t, cfg.vocab_size));
                    for j in 0..cfg.vocab_size { d_logits[[t - 1, j]] = p[j] - if j == target { 1.0 } else { 0.0 }; }
                    if distillation_lambda > 0.0 && global_target_pos < teacher_logits.nrows() {
                        let teacher_row = teacher_logits.slice(s![global_target_pos, ..]).to_vec();
                        let sum_t: f32 = teacher_row.iter().map(|&x| (x).exp()).sum();
                        for j in 0..cfg.vocab_size {
                            let q = (teacher_row[j]).exp() / sum_t;
                            let pj = p[j];
                            if pj > 1e-9 && q > 1e-9 { d_logits[[t - 1, j]] += distillation_lambda * (pj - q); }
                        }
                    }

                    let d_out = d_logits.t().dot(&xf);
                    let d_xf_raw = d_logits.dot(&out);

                    // RMSNorm backward
                    let (mut d_x, _) = model::rmsnorm_backward(&xf, &fnorm, &d_xf_raw);

                    // Backward + update per layer
                    for li in (0..cfg.n_layers).rev() {
                        let pair = &pair_caches[li];
                        let mut lw_mut = load_layer(&buf, &cfg, li)?;
                        let lc = model::forward_layer(&lw_mut, &cfg, &pair.x_in);
                        let (grads, d_x_prev) = model::backward_layer(&lw_mut, &cfg, &lc, &d_x);
                        lr_apply_layer(&mut lw_mut, &grads, lr);
                        save_layer(&mut buf, li, &lw_mut)?;
                        d_x = d_x_prev;
                    }

                    // Update output weight
                    let out_name = if buf.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
                    let (mut out_w, needs_transpose) = if buf.tensor("output.weight").is_some() {
                        let w = buf.dequant_arr2("output.weight")?;
                        if w.nrows() != cfg.vocab_size { (w.t().to_owned(), true) } else { (w, false) }
                    } else { (buf.dequant_arr2("token_embd.weight")?, false) };
                    for j in 0..cfg.vocab_size {
                        for d in 0..dim {
                            let g = d_out[[j, d]];
                            if g.is_finite() { out_w[[j, d]] -= lr * g.max(-1.0).min(1.0); }
                        }
                    }
                    let tinfo = buf.tensor(out_name).ok_or("missing output/embedding")?;
                    if [0, 1, 2, 8, 12].contains(&tinfo.ggml_type) {
                        let final_data = if needs_transpose { out_w.t().to_owned() } else { out_w };
                        buf.overwrite(out_name, final_data.as_slice().unwrap())?;
                    }
                }
            }

            let avg_loss = total_loss / n as f32;
            let avg_ce = total_ce / n as f32;
            let avg_kl = total_kl / n as f32;
            println!("  iter {it:>3}:  loss={avg_loss:.4}  CE={avg_ce:.4}  KL={avg_kl:.4}");
        }
    }

    buf.save(output_path)?;
    println!("saved trained model -> {output_path} ({:.2} MB)", buf.bytes.len() as f64 / 1e6);
    Ok(())
}

/// Compute attention gaps using vectorized batched matmul (no nested loops).
pub fn chunked_attention_gaps(buf: &GgufBuffer, cfg: &Config, x_in: &Array2<f32>) -> Result<Vec<f32>, String> {
    let t = x_in.nrows();
    if t < 2 {
        return Ok(vec![]);
    }
    let hd = cfg.head_dim();
    let group = cfg.n_heads / cfg.n_kv_heads.max(1);

    let mut x = x_in.to_owned();
    let mut gap_sum = vec![0.0f32; t - 1];
    let mut gap_count = 0usize;

    for li in 0..cfg.n_layers {
        let lw = load_layer(buf, cfg, li)?;

        let h = engine::rmsnorm_rows(&x, &lw.attn_norm, engine::EPS);
        let q = engine::linear(&h, &lw.wq);
        let k = engine::linear(&h, &lw.wk);
        let v = engine::linear(&h, &lw.wv);
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        engine::apply_rope_all(&mut q_rope, cfg.n_heads, hd, cfg.rope_theta);
        engine::apply_rope_all(&mut k_rope, cfg.n_kv_heads, hd, cfg.rope_theta);

        // Reshape for batched matmul: [T, n_heads, hd] -> [n_heads, T, hd]
        let q_3d = q_rope
            .into_shape((t, cfg.n_heads, hd)).map_err(|e| e.to_string())?
            .permuted_axes([1, 0, 2]);
        let k_3d = k_rope
            .into_shape((t, cfg.n_kv_heads, hd)).map_err(|e| e.to_string())?
            .permuted_axes([1, 0, 2]);
        let v_3d = v
            .into_shape((t, cfg.n_kv_heads, hd)).map_err(|e| e.to_string())?
            .permuted_axes([1, 0, 2]);

        // For each head, compute causal attention scores and prefix gaps
        for head in 0..cfg.n_heads {
            let kvh = head / group;
            let q_head = q_3d.slice(s![head, .., ..]); // [T, hd]
            let k_head = k_3d.slice(s![kvh, .., ..]); // [T, hd]

            // Scores: [T, T] = q_head @ k_head.T
            let scores = q_head.dot(&k_head.t()) * (1.0 / (hd as f32).sqrt());

            // Causal mask + softmax + prefix gaps (vectorized)
            for i in 0..t {
                let mut probs = scores.slice(s![i, 0..=i]).to_vec();
                engine::softmax_inplace(&mut probs);
                let mut prefix = 0.0;
                for j in 0..i {
                    prefix += probs[j];
                    gap_sum[j] += prefix;
                }
            }
        }

        // Attention output for forward pass
        let mut attn_out = Array2::zeros((t, cfg.dim));
        for head in 0..cfg.n_heads {
            let kvh = head / group;
            let q_head = q_3d.slice(s![head, .., ..]);
            let k_head = k_3d.slice(s![kvh, .., ..]);
            let v_head = v_3d.slice(s![kvh, .., ..]);

            let scores = q_head.dot(&k_head.t()) * (1.0 / (hd as f32).sqrt());
            let mut attn_head = Array2::zeros((t, hd));
            for i in 0..t {
                let mut probs = scores.slice(s![i, 0..=i]).to_vec();
                engine::softmax_inplace(&mut probs);
                for d in 0..hd {
                    let mut acc = 0.0;
                    for (j, &pj) in probs.iter().enumerate() {
                        acc += pj * v_head[[j, d]];
                    }
                    attn_head[[i, d]] = acc;
                }
            }
            for i in 0..t {
                for d in 0..hd {
                    attn_out[[i, head * hd + d]] = attn_head[[i, d]];
                }
            }
        }

        let attn_proj = engine::linear(&attn_out, &lw.wo);
        x = &x + &attn_proj;

        let h_ffn = engine::rmsnorm_rows(&x, &lw.ffn_norm, engine::EPS);
        let gate = engine::silu(&engine::linear(&h_ffn, &lw.w1));
        let up = engine::linear(&h_ffn, &lw.w3);
        let ff = engine::linear(&(&gate * &up), &lw.w2);
        x = &x + &ff;

        gap_count += cfg.n_heads;
    }

    let count = gap_count as f32;
    for (j, g) in gap_sum.iter_mut().enumerate() {
        let n_contributors = (t - 1 - j) as f32;
        if n_contributors > 0.0 {
            *g /= count * n_contributors;
        }
    }

    Ok(gap_sum)
}

fn save_all_weights(buf: &mut GgufBuffer, cache: &WeightCache) -> Result<(), String> {
    for li in 0..cache.layers.len() {
        save_layer(buf, li, &cache.layers[li])?;
    }
    let out_name = if buf.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let t = buf.tensor(out_name).ok_or("missing output/embedding")?;
    if ![0, 1, 2, 8, 12].contains(&t.ggml_type) {
        // skip unsupported quantization
    } else {
        let final_data = if cache.needs_output_transpose { cache.output.t().to_owned() } else { cache.output.clone() };
        buf.overwrite(out_name, final_data.as_slice().unwrap())?;
    }
    Ok(())
}

fn lr_apply_layer(lw: &mut model::LayerWeights, lg: &model::LayerGrads, lr: f32) {
    let sanitize = |v: f32| if v.is_finite() { v.max(-1.0).min(1.0) } else { 0.0 };
    let apply_2d = |w: &mut Array2<f32>, g: &Array2<f32>, lr: f32| {
        assert_eq!(w.len(), g.len(), "weight/grad len mismatch");
        // Use indexed access for safety (works on non-contiguous arrays)
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                w[[i, j]] -= lr * sanitize(g[[i, j]]);
            }
        }
    };
    apply_2d(&mut lw.wq, &lg.wq, lr);
    apply_2d(&mut lw.wk, &lg.wk, lr);
    apply_2d(&mut lw.wv, &lg.wv, lr);
    apply_2d(&mut lw.wo, &lg.wo, lr);
    apply_2d(&mut lw.w1, &lg.w1, lr);
    apply_2d(&mut lw.w2, &lg.w2, lr);
    apply_2d(&mut lw.w3, &lg.w3, lr);
    for d in 0..lw.attn_norm.len() {
        lw.attn_norm[d] -= lr * sanitize(lg.attn_norm[d]);
    }
    for d in 0..lw.ffn_norm.len() {
        lw.ffn_norm[d] -= lr * sanitize(lg.ffn_norm[d]);
    }
}

pub fn load_layer(buf: &GgufBuffer, cfg: &Config, li: usize) -> Result<model::LayerWeights, String> {
    // engine::linear expects (output, input): y = x @ W^T.
    // If GGUF stores as (input, output) (detected by nrows != expected_out), transpose.
    let fix = |w: Array2<f32>, expected_out: usize| -> Array2<f32> {
        if w.nrows() != expected_out {
            w.t().to_owned() // was (input, output) → transpose to (output, input)
        } else {
            w
        }
    };
    
    // Try to load MLA tensors (supports both paper and GGUF naming conventions)
    let (mla_wq_a, mla_wq_b, mla_wk_v_a, mla_wk_b, mla_wv_b, mla_wq_rope, mla_wo,
         mla_q, mla_kv_a_mqa, mla_kv_b, mla_kv_a_norm) = if cfg.is_mla() {
        // Try GGUF convention first (DeepSeek-V2 llama.cpp format)
        // attn_q.weight combines qk_nope + qk_rope into one weight
        // attn_kv_a_mqa.weight combines v_latent + k_rope projections
        // attn_kv_b.weight maps v_latent to k_nope + v
        let mla_q_gguf = match buf.dequant_arr2(&format!("blk.{li}.attn_q.weight")) {
            Ok(w) => Some(fix(w, cfg.dim)),
            Err(_) => None,
        };
        let mla_kv_a = match buf.dequant_arr2(&format!("blk.{li}.attn_kv_a_mqa.weight")) {
            Ok(w) => Some(fix(w, cfg.dim)),
            Err(_) => None,
        };
        let mla_kvb = match buf.dequant_arr2(&format!("blk.{li}.attn_kv_b.weight")) {
            Ok(w) => Some(if w.nrows() != cfg.mla.kv_lora_rank { w.t().to_owned() } else { w }),
            Err(_) => None,
        };
        let mla_kv_an = {
            let t = buf.tensor(&format!("blk.{li}.attn_kv_a_norm.weight"));
            t.and_then(|tt| {
                let d = gguf::read_tensor(&buf.bytes, tt, buf.data_start).ok()?;
                Some(Array1::from_shape_vec(tt.shape[0], d).ok()?)
            })
        };
        // Try paper convention tensors as fallback
        let wq_a = match buf.dequant_arr2(&format!("blk.{li}.attn_q_a.weight")) {
            Ok(w) => Some(fix(w, cfg.mla.q_lora_rank)),
            Err(_) => None,
        };
        let wq_b = match buf.dequant_arr2(&format!("blk.{li}.attn_q_b.weight")) {
            Ok(w) => Some(fix(w, cfg.dim)),
            Err(_) => None,
        };
        let kv_a = mla_kv_a.clone().or_else(|| {
            match buf.dequant_arr2(&format!("blk.{li}.attn_kv_a.weight")) {
                Ok(w) => Some(fix(w, cfg.mla.kv_lora_rank)),
                Err(_) => None,
            }
        });
        let kv_b = mla_kvb.clone().or_else(|| {
            match buf.dequant_arr2(&format!("blk.{li}.attn_kv_b.weight")) {
                Ok(w) => Some(fix(w, cfg.dim)),
                Err(_) => None,
            }
        });
        let v_b = match buf.dequant_arr2(&format!("blk.{li}.attn_v_b.weight")) {
            Ok(w) => Some(fix(w, cfg.dim)),
            Err(_) => None,
        };
        let q_rope = match buf.dequant_arr2(&format!("blk.{li}.attn_q_rope.weight")) {
            Ok(w) => Some(fix(w, cfg.dim)),
            Err(_) => None,
        };
        let wo = match buf.dequant_arr2(&format!("blk.{li}.attn_output.weight")) {
            Ok(w) => Some(fix(w, cfg.dim)),
            Err(_) => None,
        };

        (wq_a, wq_b, kv_a, kv_b, v_b, q_rope, wo,
         mla_q_gguf, mla_kv_a, mla_kvb, mla_kv_an)
    } else {
        (None, None, None, None, None, None, None,
         None, None, None, None)
    };

    // Load standard MHA tensors (may be placeholder for MLA models)
    let (wq, wk, wv) = if mla_wq_a.is_none() {
        // Standard MHA
        (
            fix(buf.dequant_arr2(&format!("blk.{li}.attn_q.weight"))?, cfg.dim),
            fix(buf.dequant_arr2(&format!("blk.{li}.attn_k.weight"))?, cfg.dim_kv()),
            fix(buf.dequant_arr2(&format!("blk.{li}.attn_v.weight"))?, cfg.dim_kv())
        )
    } else {
        // MLA model - use placeholder tensors (will be unused by forward_layer_kv dispatch)
        (
            Array2::zeros((cfg.dim, cfg.dim)),
            Array2::zeros((cfg.dim_kv(), cfg.dim)),
            Array2::zeros((cfg.dim_kv(), cfg.dim))
        )
    };

    // Load MoE weights (supports both naming conventions)
    let (moe_shared_w1, moe_shared_w2, moe_shared_w3,
         moe_expert_w1, moe_expert_w2, moe_expert_w3,
         moe_router_weight, moe_router_bias,
         moe_gate_exps, moe_down_exps, moe_up_exps, moe_gate_inp,
         moe_shared_gate, moe_shared_down, moe_shared_up) = if cfg.is_moe() {
        let n_experts = cfg.moe.n_experts;
        let e_inter = if cfg.expert_intermediate > 0 { cfg.expert_intermediate } else { cfg.intermediate };
        
        // Try GGUF convention: ffn_gate_exps.weight (3D) for routed experts
        let has_gguf_moe = buf.tensor(&format!("blk.{li}.ffn_gate_exps.weight")).is_some();
        
        if has_gguf_moe {
            // GGUF convention: load 3D tensors
            let ge = Some(buf.dequant_arr3(&format!("blk.{li}.ffn_gate_exps.weight"))?);
            let de = Some(buf.dequant_arr3(&format!("blk.{li}.ffn_down_exps.weight"))?);
            let ue = Some(buf.dequant_arr3(&format!("blk.{li}.ffn_up_exps.weight"))?);
            let inp = Some(fix(buf.dequant_arr2(&format!("blk.{li}.ffn_gate_inp.weight"))?, n_experts));
            
            // Shared experts (may not exist for dense layer 0)
            let sg = match buf.dequant_arr2(&format!("blk.{li}.ffn_gate_shexp.weight")) {
                Ok(w) => Some(fix(w, cfg.dim)),
                Err(_) => None,
            };
            let sd = match buf.dequant_arr2(&format!("blk.{li}.ffn_down_shexp.weight")) {
                Ok(w) => Some(fix(w, cfg.dim)),
                Err(_) => None,
            };
            let su = match buf.dequant_arr2(&format!("blk.{li}.ffn_up_shexp.weight")) {
                Ok(w) => Some(fix(w, cfg.dim)),
                Err(_) => None,
            };
            
            (None, None, None, None, None, None, None, None,
             ge, de, ue, inp, sg, sd, su)
        } else {
            // Paper convention: individual expert tensors
            let n_shared = cfg.moe.n_shared;
            let shared_w1 = match buf.dequant_arr2(&format!("blk.{li}.ffn_shared_gate.weight")) {
                Ok(w) => Some(fix(w, n_shared * e_inter)),
                Err(_) => None,
            };
            let shared_w2 = match buf.dequant_arr2(&format!("blk.{li}.ffn_shared_down.weight")) {
                Ok(w) => Some(fix(w, cfg.dim)),
                Err(_) => None,
            };
            let shared_w3 = match buf.dequant_arr2(&format!("blk.{li}.ffn_shared_up.weight")) {
                Ok(w) => Some(fix(w, n_shared * e_inter)),
                Err(_) => None,
            };
            
            let mut expert_w1 = Vec::with_capacity(n_experts);
            let mut expert_w2 = Vec::with_capacity(n_experts);
            let mut expert_w3 = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                match buf.dequant_arr2(&format!("blk.{li}.ffn_expert.{e}.gate.weight")) {
                    Ok(w) => expert_w1.push(fix(w, e_inter)),
                    Err(_) => expert_w1.push(Array2::zeros((e_inter, cfg.dim))),
                }
                match buf.dequant_arr2(&format!("blk.{li}.ffn_expert.{e}.down.weight")) {
                    Ok(w) => expert_w2.push(fix(w, cfg.dim)),
                    Err(_) => expert_w2.push(Array2::zeros((cfg.dim, e_inter))),
                }
                match buf.dequant_arr2(&format!("blk.{li}.ffn_expert.{e}.up.weight")) {
                    Ok(w) => expert_w3.push(fix(w, e_inter)),
                    Err(_) => expert_w3.push(Array2::zeros((e_inter, cfg.dim))),
                }
            }
            
            let router_weight = match buf.dequant_arr2(&format!("blk.{li}.ffn_router.weight")) {
                Ok(w) => Some(fix(w, n_experts)),
                Err(_) => None,
            };
            let router_bias = match buf.dequant_arr1(&format!("blk.{li}.ffn_router.bias")) {
                Ok(b) => Some(b),
                Err(_) => None,
            };
            
            (shared_w1, shared_w2, shared_w3,
             Some(expert_w1), Some(expert_w2), Some(expert_w3),
             router_weight, router_bias,
             None, None, None, None, None, None, None)
        }
    } else {
        (None, None, None, None, None, None, None, None,
         None, None, None, None, None, None, None)
    };

    // For MoE layers (GGUF format: has ffn_gate_exps but not ffn_gate.weight),
    // use placeholder w1/w2/w3 to avoid failed loads.
    let (w1, w2, w3) = if moe_gate_exps.is_some() || moe_expert_w1.is_some() {
        // MoE layer — use zero arrays for dense FFN fields (unused)
        (Array2::zeros((cfg.intermediate, cfg.dim)),
         Array2::zeros((cfg.dim, cfg.intermediate)),
         Array2::zeros((cfg.intermediate, cfg.dim)))
    } else {
        // Dense FFN layer
        (
            fix(buf.dequant_arr2(&format!("blk.{li}.ffn_gate.weight"))?, cfg.intermediate),
            fix(buf.dequant_arr2(&format!("blk.{li}.ffn_down.weight"))?, cfg.dim),
            fix(buf.dequant_arr2(&format!("blk.{li}.ffn_up.weight"))?, cfg.intermediate),
        )
    };

    let wo = fix(buf.dequant_arr2(&format!("blk.{li}.attn_output.weight"))?, cfg.dim);
    
    Ok(model::LayerWeights {
        wq,
        wk,
        wv,
        wo,
        w1,
        w2,
        w3,
        attn_norm: {
            let t = buf.tensor(&format!("blk.{li}.attn_norm.weight")).ok_or("missing attn_norm")?;
            let d = gguf::read_tensor(&buf.bytes, t, buf.data_start)?;
            Array1::from_shape_vec(t.shape[0], d).map_err(|e| e.to_string())?
        },
        ffn_norm: {
            let t = buf.tensor(&format!("blk.{li}.ffn_norm.weight")).ok_or("missing ffn_norm")?;
            let d = gguf::read_tensor(&buf.bytes, t, buf.data_start)?;
            Array1::from_shape_vec(t.shape[0], d).map_err(|e| e.to_string())?
        },
        // MoE extensions (paper convention)
        moe_shared_w1,
        moe_shared_w2,
        moe_shared_w3,
        moe_expert_w1,
        moe_expert_w2,
        moe_expert_w3,
        moe_router_weight,
        moe_router_bias,
        // MoE extensions (GGUF convention: 3D expert tensors + shared experts)
        moe_gate_exps,
        moe_down_exps,
        moe_up_exps,
        moe_gate_inp,
        moe_shared_gate,
        moe_shared_down,
        moe_shared_up,
        // MLA extensions (paper convention)
        mla_wq_a,
        mla_wq_b,
        mla_wk_v_a,
        mla_wk_b,
        mla_wv_b,
        mla_wq_rope,
        mla_wo,
        // MLA extensions (GGUF convention: combined Q + kv_a_mqa + kv_b + kv_a_norm)
        mla_q,
        mla_kv_a_mqa,
        mla_kv_b,
        mla_kv_a_norm,
    })
}

fn save_layer(buf: &mut GgufBuffer, li: usize, lw: &model::LayerWeights) -> Result<(), String> {
    let save = |buf: &mut GgufBuffer, name: &str, arr: &Array2<f32>| -> Result<(), String> {
        let t = buf.tensor(name).ok_or_else(|| format!("missing {name}"))?;
        if ![0, 1, 2, 8, 12, 14].contains(&t.ggml_type) {
            return Ok(()); // skip saving this tensor
        }
        if t.shape[0] != arr.nrows() || t.shape[1] != arr.ncols() {
            let back = arr.t().to_owned();
            let flat: Vec<f32> = back.iter().copied().collect();
            buf.overwrite(name, &flat)
        } else {
            let flat: Vec<f32> = arr.iter().copied().collect();
            buf.overwrite(name, &flat)
        }
    };
    save(buf, &format!("blk.{li}.attn_q.weight"), &lw.wq)?;
    save(buf, &format!("blk.{li}.attn_k.weight"), &lw.wk)?;
    save(buf, &format!("blk.{li}.attn_v.weight"), &lw.wv)?;
    save(buf, &format!("blk.{li}.attn_output.weight"), &lw.wo)?;
    save(buf, &format!("blk.{li}.ffn_gate.weight"), &lw.w1)?;
    save(buf, &format!("blk.{li}.ffn_down.weight"), &lw.w2)?;
    save(buf, &format!("blk.{li}.ffn_up.weight"), &lw.w3)?;

    let an_flat: Vec<f32> = lw.attn_norm.iter().copied().collect();
    buf.overwrite(&format!("blk.{li}.attn_norm.weight"), &an_flat)?;
    let fn_flat: Vec<f32> = lw.ffn_norm.iter().copied().collect();
    buf.overwrite(&format!("blk.{li}.ffn_norm.weight"), &fn_flat)?;

    Ok(())
}
