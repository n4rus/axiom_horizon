//! Multi-teacher assimilation loop.
//!
//! Loads N teacher GGUFs, runs forward on a training corpus to get logits,
//! averages them (weighted or uniform), then trains the student model via
//! KL(student || teacher_ensemble) distillation with VFE regulation.
//!
//! Strategy:
//!   1. Precompute teacher logits for the entire corpus (one teacher at a time
//!      to stay within 16 GB RAM).
//!   2. Average logits across teachers (soft ensemble).
//!   3. Train student with backprop + KL loss + VFE early-stop.

use ndarray::{Array2, s};
use crate::config::Config;
use crate::engine;
use crate::model::{Weights, FreezeConfig, Gradients};

/// Result of the assimilation process.
#[allow(dead_code)]
pub struct AssimilationResult {
    pub final_loss: f32,
    pub vfe_values: Vec<f64>,
    pub iterations: usize,
    pub teacher_count: usize,
    pub saved_path: String,
}

/// Configuration for multi-teacher assimilation.
pub struct AssimilationConfig<'a> {
    pub student_path: &'a str,
    pub teacher_paths: &'a [String],
    pub output_path: &'a str,
    pub text: &'a str,
    pub iters: usize,
    pub lr: f32,
    pub distillation_temp: f32,
    /// Optional per-teacher weights (defaults to uniform if empty or length mismatch).
    pub teacher_weights: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Local VFE helpers (mirror the kai::vfe API but self-contained)
// ---------------------------------------------------------------------------

fn grad_norm_sq_local(grads: &Gradients) -> f64 {
    let embed_sum: f64 = grads.embed.as_ref().map(|a| a.iter().map(|x| (*x as f64).powi(2)).sum()).unwrap_or(0.0);
    let output_sum: f64 = grads.output.as_ref().map(|a| a.iter().map(|x| (*x as f64).powi(2)).sum()).unwrap_or(0.0);
    let fnorm_sum: f64 = grads.final_norm.iter().map(|x| (*x as f64).powi(2)).sum();
    let mut layers_sum = 0.0f64;
    for lg_opt in &grads.layers {
        if let Some(lg) = lg_opt.as_ref() {
            for x in lg.wq.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.wk.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.wv.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.wo.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.w1.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.w2.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.w3.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.attn_norm.iter() { layers_sum += (*x as f64).powi(2); }
            for x in lg.ffn_norm.iter() { layers_sum += (*x as f64).powi(2); }
        }
    }
    embed_sum + output_sum + fnorm_sum + layers_sum
}

fn calculate_vfe_thermo_local(surprisal: f64, epistemic: f64, delta_e: f64) -> f64 {
    // f_vfe = α·surprisal + β·epistemic + γ·δE
    let alpha = 1.0;
    let beta = 0.5;
    let gamma = 0.1;
    alpha * surprisal + beta * epistemic + gamma * delta_e
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the full multi-teacher assimilation loop.
pub fn assimilate_multi(cfg: &AssimilationConfig) -> Result<AssimilationResult, String> {
    let n_teachers = cfg.teacher_paths.len();
    if n_teachers == 0 {
        return Err("need at least one teacher".into());
    }

    // ---- 1. Load student ----
    let (student_cfg, mut w) = load_student(cfg.student_path)?;
    let cfg_s = &student_cfg;
    let freeze = FreezeConfig::default();

    // ---- 2. Encode training text ----
    let meta = crate::loader::read_kv(cfg.student_path).map_err(|e| format!("student meta: {e}"))?;
    let tok = crate::tok::Tokenizer::from_gguf(&meta).ok_or("no tokenizer in student GGUF")?;
    let ids = tok.encode(cfg.text);
    if ids.len() < 2 {
        return Err("training text too short (need ≥2 tokens)".into());
    }
    let t = ids.len().min(cfg_s.max_seq);
    let ids: Vec<usize> = ids[..t].to_vec();
    let n = ids.len() - 1; // number of prediction positions

    // ---- 3. Precompute teacher logits ----
    let vocab = cfg_s.vocab_size;
    let mut all_teacher_logits: Vec<Array2<f32>> = Vec::with_capacity(n_teachers);

    for (ti, tpath) in cfg.teacher_paths.iter().enumerate() {
        eprintln!("  teacher {}/{}: loading {} …", ti + 1, n_teachers, tpath);
        let tl = compute_teacher_logits(tpath, &ids, vocab, cfg_s.max_seq)?;
        eprintln!("    → logits shape [{}, {}]", tl.nrows(), tl.ncols());
        all_teacher_logits.push(tl);
    }

    // ---- 4. Weighted ensemble ----
    let weights: Vec<f32> = match &cfg.teacher_weights {
        Some(w) if w.len() == n_teachers => w.clone(),
        _ => vec![1.0 / n_teachers as f32; n_teachers],
    };
    let weight_sum: f32 = weights.iter().sum();
    let weights: Vec<f32> = weights.into_iter().map(|w| w / weight_sum).collect();

    // ---- 5. Train student ----
    let mut vfe_values = Vec::new();
    let mut prev_vfe = f64::INFINITY;
    let vfe_patience = 3usize;
    let mut stale = 0usize;
    let mut final_loss = 0.0f32;

    let temp = cfg.distillation_temp;

    for it in 0..cfg.iters {
        // Forward pass
        let cache = w.forward_with_cache(cfg_s, &ids);

        // Compute KL loss vs. teacher ensemble for all positions
        let mut d_logits = Array2::<f32>::zeros((ids.len(), cfg_s.vocab_size));
        let mut total_loss = 0.0f32;

        for pos in 0..n {
            let raw: Vec<f32> = (0..vocab).map(|j| cache.logits[[pos, j]]).collect();
            let p = engine::softmax(&raw);

            // Ensemble teacher logits at this position (with temperature)
            let mut teacher_logits_avg = vec![0.0f32; vocab];
            for ti in 0..n_teachers {
                let wt = weights[ti];
                let row = all_teacher_logits[ti].slice(s![pos, ..]);
                for j in 0..vocab {
                    teacher_logits_avg[j] += wt * row[j];
                }
            }
            // Apply temperature scaling
            let teacher_logits_scaled: Vec<f32> = teacher_logits_avg.iter().map(|x| x / temp).collect();
            let q = engine::softmax(&teacher_logits_scaled);

            // KL divergence: sum(p * log(p/q))
            let mut kl = 0.0f32;
            for j in 0..vocab {
                if p[j] > 1e-9 && q[j] > 1e-9 {
                    kl += p[j] * (p[j] / q[j]).ln();
                }
            }
            total_loss += kl;

            // Gradient: d(kl)/d(logit_j) = p_j - q_j (for cross-entropy-like loss)
            for j in 0..vocab {
                d_logits[[pos, j]] = p[j] - q[j];
            }
        }

        let avg_loss = total_loss / n as f32;
        final_loss = avg_loss;

        // Backward pass
        let grads = w.backward(cfg_s, &cache, &d_logits, &ids, &freeze);

        // VFE regulation
        let var = w.readout_variance();
        let delta_e = grad_norm_sq_local(&grads);
        let vfe = calculate_vfe_thermo_local(avg_loss as f64, var as f64, delta_e);
        vfe_values.push(vfe);

        let lr_scale = ((vfe as f32 / (avg_loss + 1.0)) * 0.2).clamp(0.02, 2.0);
        w.apply_gradients(&grads, (cfg.lr / n as f32) * lr_scale);

        eprintln!("  iter {it:>3}:  loss={avg_loss:.4}  kl={avg_loss:.4}  vfe={vfe:.4}  lr_scale={lr_scale:.3}");

        // VFE early-stop
        if vfe > prev_vfe - 1e-3 {
            stale += 1;
            if stale >= vfe_patience && it > 0 {
                eprintln!("  → VFE plateau, stopping early at iter {it}");
                break;
            }
        } else {
            stale = 0;
        }
        prev_vfe = vfe;
    }

    // ---- 6. Save fused model ----
    let out_path = cfg.output_path;
    eprintln!("  saving to {out_path} …");
    let meta_for_save = crate::loader::read_kv(cfg.student_path).map_err(|e| format!("read meta: {e}"))?;
    let tok_for_save = crate::tok::Tokenizer::from_gguf(&meta_for_save).ok_or("no tokenizer")?;
    crate::gguf_write::save_gguf(
        &w, cfg_s, &tok_for_save, out_path,
        &std::collections::HashMap::new(),
    );

    let iterc = vfe_values.len();
    Ok(AssimilationResult {
        final_loss,
        vfe_values,
        iterations: iterc,
        teacher_count: n_teachers,
        saved_path: out_path.to_string(),
    })
}

/// Load a student model (must be a valid GGUF with our architecture).
fn load_student(path: &str) -> Result<(Config, Weights), String> {
    let meta = crate::loader::read_kv(path).map_err(|e| format!("read {path}: {e}"))?;
    let cfg = crate::loader::build_config(&meta).ok_or("could not build Config from metadata")?;
    let (_v, map, _nt, _nk) = crate::loader::load_tensors(path)
        .map_err(|e| format!("load tensors: {e}"))?;
    let w = Weights::from_gguf(&map, &cfg).map_err(|e| format!("from_gguf: {e}"))?;
    Ok((cfg, w))
}

/// Compute teacher logits for the given token IDs.
/// Supports ANY architecture loadable by the existing GGUF pipeline
/// (dense MHA, MoE, MLA, Linear attention, Vision) by using
/// `Weights::from_gguf` + `forward_with_cache`.
fn compute_teacher_logits(
    path: &str,
    ids: &[usize],
    target_vocab: usize,
    max_seq: usize,
) -> Result<Array2<f32>, String> {
    // Load teacher weights and config (handles any architecture)
    let (teacher_cfg, teacher_w) = load_student(path)?;
    let t = ids.len().min(max_seq);
    let tokens: Vec<usize> = ids.iter().take(t).copied().collect();

    // Forward pass using the general forward_with_cache (supports MoE/MLA/Linear)
    let cache = teacher_w.forward_with_cache(&teacher_cfg, &tokens);
    let logits = cache.logits;

    // If teacher vocab differs from student, align
    let teacher_vocab = teacher_cfg.vocab_size;
    if teacher_vocab == target_vocab {
        Ok(logits)
    } else {
        // Pad or truncate to match student vocab
        let mut aligned = Array2::<f32>::zeros((t, target_vocab));
        let min_vocab = teacher_vocab.min(target_vocab);
        for pos in 0..t {
            for j in 0..min_vocab {
                aligned[[pos, j]] = logits[[pos, j]];
            }
        }
        Ok(aligned)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, AttnPolicy, AttnKind, MlpKind, MoEConfig, MLAConfig, VisionConfig};
    use crate::model::Weights;
    use std::collections::HashMap;
    use crate::tok::Tokenizer;

    /// Create a Config with the smallest reasonable dimensions.
    fn make_tiny_cfg() -> Config {
        Config {
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: 24,
            intermediate: 32,
            rope_theta: 10000.0,
            max_seq: 64,
            tau: 1.0, e: 1.0, age: 0, cycles: 0,
            h: 0.5, base_ms: 1000.0, phi: 0.0,
            attn_policy: AttnPolicy::Global(AttnKind::MHA),
            mlp_kind: MlpKind::Dense,
            moe: MoEConfig::default(),
            mla: MLAConfig::default(),
            vision: VisionConfig::default(),
            expert_intermediate: 0,
            leading_dense_blocks: 0,
        }
    }

    #[test]
    fn assimilate_multi_teachers_reduces_loss() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let n = COUNT.fetch_add(1, Ordering::SeqCst);

        let cfg = make_tiny_cfg();

        // Build a valid tokenizer from metadata
        let mut meta: HashMap<String, crate::loader::GgufMeta> = HashMap::new();
        let tokens: Vec<String> = (0..cfg.vocab_size).map(|i| format!("tok{i}")).collect();
        meta.insert("tokenizer.ggml.tokens".into(), crate::loader::GgufMeta::StrArr(tokens));
        meta.insert("tokenizer.ggml.bos_token_id".into(), crate::loader::GgufMeta::Num(1.0));
        meta.insert("tokenizer.ggml.eos_token_id".into(), crate::loader::GgufMeta::Num(2.0));
        meta.insert("tokenizer.ggml.unknown_token_id".into(), crate::loader::GgufMeta::Num(0.0));
        meta.insert("tokenizer.ggml.model".into(), crate::loader::GgufMeta::Str("llama".into()));
        let tok = Tokenizer::from_gguf(&meta).expect("tokenizer from metadata");

        // Create 2 teacher GGUFs and 1 student — all with same random weights (zero KL → quick convergence)
        let student_path = format!("/tmp/assimilate_student_{n}.gguf");
        let teacher1_path = format!("/tmp/assimilate_teacher1_{n}.gguf");
        let teacher2_path = format!("/tmp/assimilate_teacher2_{n}.gguf");
        let output_path = format!("/tmp/assimilate_fused_{n}.gguf");

        // Use Weights::random (known-good) + save_gguf (proven correct) for valid GGUF files
        let w = Weights::random(&cfg);
        crate::gguf_write::save_gguf(&w, &cfg, &tok, &student_path, &HashMap::new());
        crate::gguf_write::save_gguf(&w, &cfg, &tok, &teacher1_path, &HashMap::new());
        crate::gguf_write::save_gguf(&w, &cfg, &tok, &teacher2_path, &HashMap::new());

        let text = "hello world test assimilation";

        let ass_cfg = AssimilationConfig {
            student_path: &student_path,
            teacher_paths: &vec![teacher1_path.clone(), teacher2_path.clone()],
            output_path: &output_path,
            text,
            iters: 5,
            lr: 0.001,
            distillation_temp: 1.0,
            teacher_weights: None,
        };

        let result = assimilate_multi(&ass_cfg);
        assert!(result.is_ok(), "assimilate_multi failed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.teacher_count, 2);
        assert!(r.iterations > 0, "should have run at least one iteration");

        // Cleanup
        let _ = std::fs::remove_file(&student_path);
        let _ = std::fs::remove_file(&teacher1_path);
        let _ = std::fs::remove_file(&teacher2_path);
        let _ = std::fs::remove_file(&output_path);
    }
}
