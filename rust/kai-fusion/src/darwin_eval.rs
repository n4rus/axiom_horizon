//! darwin_eval.rs — batch physics-parameter evaluation for `darwin evolve`.
//!
//! The stock path (`generate_streaming`) reloads every layer from disk for
//! EVERY generated token of EVERY candidate: N candidates × T tokens × L layers
//! GGUF dequantizations. For tinyllama that is 36×20×22 = 15,840 disk loads —
//! minutes of CPU while the physics metrics measure the same fixed model.
//!
//! This decoder:
//!   1. Loads ALL layer weights once (memory resident).
//!   2. Prefills the benchmark prompt once → shared prompt KV cache.
//!   3. Per candidate: clone the prompt cache, decode tokens layer-major so
//!      each resident layer is applied once per token (no disk re-loads).
//!   4. Runs the same adaptive-physics loop (novelty/curvature/tau/VFE +
//!      L2 multi-prior + L3 curvature wiring) as `generate_streaming`.
//!
//! Layer loads drop from 15,840 to 22.

use ndarray::{Array1, Array2};

use crate::gguf;
use crate::model;
use crate::config::Config;
use crate::chunked;
use crate::tok;
use crate::engine;
use crate::vfe;
use crate::attractor;

/// Physics metrics for one candidate (mirrors main.rs `PhysicsMetrics`).
#[derive(Default, Clone)]
pub struct CandMetrics {
    pub avg_vfe: f32,
    pub avg_novelty: f32,
    pub avg_curvature: f32,
    pub final_tau: f32,
    pub total_tokens: usize,
    pub tokens_per_sec: f32,
    pub final_hidden: Vec<f32>,
    /// Entropy of the per-token softmax over sampled steps (surprisal averaged
    /// as -Σ p·ln p). High = the model is exploring, not stuck at argmax.
    pub avg_entropy: f32,
    /// Fraction of distinct tokens among all generated tokens (0..1).
    pub token_diversity: f32,
    /// Longest run of identical repeated tokens (coherence-vs-repetition cost).
    pub max_repeat_run: usize,
    /// Average stepwise hidden-state drift ∥h_{t+1}-h_t∥, normalized.
    /// Low drift = coherent flow; high drift = mode-hopping (novelty but noisy).
    pub avg_drift: f32,
    /// Average effective sampling temperature actually applied (base temp after
    /// adaptive novelty/τ/advance modulation). Shows how much the physics layer
    /// deviates from the static config.
    pub avg_eff_temp: f32,
}

/// Composite fitness for the physics benchmark: maximize BOTH
/// novelty (exploration) and coherence (stability), i.e. find the
/// temperature/τ frontier where the model is neither pressured to sample
/// identically (flat-distribution collapse) nor incoherently hopping.
/// score ∈ [0,1]; higher = better novelty-preserving-coherence.
pub fn benchmark_fitness(m: &CandMetrics) -> f32 {
    // Coherence side: -log-surprisal (low VFE), low drift, bounded repetition.
    let coh_vfe = (1.0 - m.avg_vfe.clamp(0.0, 1.0)) * 0.4;
    let coh_drift = (1.0 - m.avg_drift.clamp(0.0, 1.0)) * 0.3;
    // Repetition penalty: perfect (log-ish) run length in [0,1] terms.
    let rep_penalty = if m.total_tokens == 0 {
        0.0
    } else {
        let rep_x = (m.max_repeat_run as f32) / (m.total_tokens as f32 + 1.0);
        (1.0 - rep_x).clamp(0.0, 1.0)
    };
    let coherence = coh_vfe + coh_drift * 0.0 + rep_penalty * 0.3;

    // Novelty side: curvature (attention diffusion) + token diversity
    // (sampling not collapsing to a single token) + entropy mass.
    let div = if m.total_tokens == 0 { 0.0 } else { m.token_diversity * 0.5 + (m.avg_entropy.clamp(0.0, 12.0) / 12.0) * 0.5 };
    let novelty = (m.avg_curvature.clamp(0.0, 1.0) * 0.5 + div * 0.5).clamp(0.0, 1.0);

    // Multi-objective: geometric mean so both must be nonzero to score high.
    (coherence * novelty).sqrt()
}

/// Batch evaluator: one model load, one prompt prefill, many candidate decodes.
pub struct DarwinDecoder {
    cfg: Config,
    weights: Vec<model::LayerWeights>,
    embed: Array2<f32>,
    output: Array2<f32>,
    fnorm: Array1<f32>,
    prompt_ids: Vec<usize>,
    base_cache: model::KVCache,
    prompt_hidden: Array2<f32>, // final prefill hidden state (1 x dim)
    rng: SeededRng,
}

impl DarwinDecoder {
    /// Number of prompt tokens already prefilled (shared across candidates).
    pub fn prompt_token_count(&self) -> usize {
        self.prompt_ids.len()
    }

    /// Open model, load all layers resident, prefill the prompt once.
    pub fn open(gguf_path: &str, prompt: &str, max_prompt: usize) -> Result<DarwinDecoder, String> {
        let meta = gguf::read_kv(gguf_path)?;
        let cfg = gguf::build_config(&meta).ok_or_else(|| "not a decoder arch".to_string())?;
        let tok = tok::Tokenizer::from_gguf(&meta).ok_or_else(|| "no tokenizer in GGUF".to_string())?;
        let buf = gguf::GgufBuffer::open(gguf_path)?;
        let dim = cfg.dim;

        let embed_raw = buf.dequant_arr2("token_embd.weight")?;
        let embed = embed_raw.t().to_owned(); // [vocab, dim]
        let output = if buf.tensor("output.weight").is_some() {
            let w = buf.dequant_arr2("output.weight")?;
            if w.nrows() != cfg.vocab_size { w.t().to_owned() } else { w }
        } else {
            embed.clone()
        };

        let fnorm = {
            let tinfo = buf.tensor("output_norm.weight")
                .ok_or_else(|| "missing output_norm.weight".to_string())?;
            let d = gguf::read_tensor(&buf.bytes, tinfo, buf.data_start)?;
            Array1::from_shape_vec(tinfo.shape[0], d)
                .map_err(|e| format!("final_norm shape: {e}"))?
        };

        // Load ALL layers resident once.
        let mut weights = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            weights.push(chunked::load_layer(&buf, &cfg, li)?);
        }

        // Prefill the prompt (Phase-1 style: layer-major).
        let mut ids = tok.encode(prompt);
        if ids.is_empty() { ids = vec![0]; }
        if ids.len() > max_prompt { ids.truncate(max_prompt); }
        let n_p = ids.len();
        let mut hidden: Vec<Array2<f32>> = (0..n_p).map(|pi| {
            let tid = ids[pi].min(cfg.vocab_size - 1);
            let mut x = Array2::zeros((1, dim));
            for d in 0..dim { x[[0, d]] = embed[[tid, d]]; }
            x
        }).collect();
        let mut cache = model::KVCache::new(cfg.n_layers);
        for li in 0..cfg.n_layers {
            for pos in 0..n_p {
                hidden[pos] = model::forward_layer_kv(&weights[li], &cfg, &hidden[pos], &mut cache, li, pos);
            }
        }
        let prompt_hidden = hidden.pop().unwrap_or_else(|| {
            let tid = ids.last().copied().unwrap_or(0).min(cfg.vocab_size - 1);
            let mut x0 = Array2::zeros((1, dim));
            for d in 0..dim { x0[[0, d]] = embed[[tid, d]]; }
            x0
        });

        Ok(DarwinDecoder {
            cfg, weights, embed, output, fnorm, prompt_ids: ids,
            base_cache: cache, prompt_hidden,
            rng: SeededRng::new(std::env::var("KAI_DARWIN_SEED")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0x5EED_DA1D)),
        })
    }

    /// Decode one candidate with the given physics params. All weights are
    /// already resident; the prompt KV cache is cloned, so this is the cheap
    /// part (~one pass over the resident layers per token).
    pub fn evaluate(
        &mut self,
        base_temp: f32,
        top_p: f32,
        vfe_tau_rate: f32,
        tau_min: f32,
        tau_max: f32,
        max_new: usize,
    ) -> CandMetrics {
        // Snapshot the RNG so every candidate decode starts from the same
        // deterministic stream regardless of what previous candidates consumed.
        let rng_saved = self.rng.0;
        let cfg = &self.cfg;
        let dim = cfg.dim;
        let n_p = self.prompt_ids.len();
        let mut cache = self.base_cache.clone();
        let mut x = self.prompt_hidden.clone();

        let mut m = CandMetrics::default();
        let mut last_embedding: Vec<f32> = Vec::new();

        // Domain priors from prompt embeddings (matches main.rs wiring).
        let mut pool: Vec<Vec<f32>> = Vec::new();
        for &tid0 in self.prompt_ids.iter().take(32).skip(1) {
            let t = tid0.min(cfg.vocab_size - 1);
            pool.push((0..dim).map(|d| self.embed[[t, d]]).collect());
        }
        let domain_priors = attractor::make_domain_priors(&pool, 3, 2, 0xA11CEu64);
        let mut cloud_ring: std::collections::VecDeque<Vec<f32>> = std::collections::VecDeque::new();

        let start = std::time::Instant::now();
        // Novelty/coherence trackers: entropy, diversity, repeat-run, drift.
        let mut entropy_acc = 0.0f32;
        let mut appear: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut cur_run = 0usize;
        let mut best_run = 0usize;
        let mut last_next: usize = 0;
        let mut prev_embed: Option<Vec<f32>> = None;
        let mut drift_acc = 0.0f32;
        // Adaptive-temperature state: the L2/L3 advance of the previous token is
        // what temperatures the CURRENT sample (mirrors generate_streaming).
        let mut physics_adv: Option<vfe::PhysicsAdvance> = None;
        let mut eff_temp_acc = 0.0f32;
        let mut eff_temp_n = 0usize;
        for gen_pos in 0..max_new {
            let abs_pos = n_p + gen_pos;
            for li in 0..cfg.n_layers {
                x = model::forward_layer_kv(&self.weights[li], cfg, &x, &mut cache, li, abs_pos);
            }
            let xf = engine::rmsnorm_rows(&x, &self.fnorm, engine::EPS);
            last_embedding = xf.row(0).to_vec();

            let current_novelty = *cache.novelty.last().unwrap_or(&0.0);
            let current_curvature = *cache.curvature.last().unwrap_or(&0.0);

            // Adaptive temperature: prefer the previous token's advance_physics
            // temperature (L2 multi-prior + L3 curvature route), coupled with τ:
            // subjective-time dilation means a more careful (lower-temp) sample
            // when τ grows. First token falls back to the classic novelty/τ
            // formula (matches main.rs).
            let temp_eff = if let Some(adv) = &physics_adv {
                (adv.temperature / cache.tau.clamp(0.2, 10.0)).max(0.01)
            } else if base_temp > 0.0 {
                let boost = (current_novelty + current_curvature) * 0.5 - 0.5;
                (base_temp * (1.0 + 0.5 * boost) / cache.tau.max(0.1)).max(0.01)
            } else {
                0.0
            };
            eff_temp_acc += temp_eff;
            eff_temp_n += 1;

            let logits = engine::linear(&xf, &self.output);
            let raw: Vec<f32> = (0..cfg.vocab_size).map(|j| logits[[0, j]]).collect();
            let next = if temp_eff > 0.0 {
                sample_top_p_rng(&raw, temp_eff, top_p, &mut self.rng)
            } else {
                argmax_local(&raw)
            };
            let probs = engine::softmax(&raw);
            let confidence = probs[next.min(probs.len() - 1)].max(1e-10);
            let vfe = (-confidence.ln()) as f32;

            // ---- benchmark trackers (novelty-vs-coherence frontier) ----
            let ent: f32 = probs.iter().filter(|&&p| p > 1e-12)
                .map(|&p| -p * p.ln()).sum::<f32>();
            entropy_acc += ent;
            appear.insert(next);
            cur_run = if gen_pos > 0 && next == last_next { cur_run + 1 } else { 1 };
            if cur_run > best_run { best_run = cur_run; }
            if let Some(prev) = &prev_embed {
                let mut dot = 0.0f32; let mut np = 0.0f32; let mut nc = 0.0f32;
                for (a, b) in prev.iter().zip(last_embedding.iter()) {
                    dot += a * b; np += a * a; nc += b * b;
                }
                let cos = if np > 0.0 && nc > 0.0 { dot / (np * nc).sqrt() } else { 0.0 };
                drift_acc += (1.0 - cos.clamp(0.0, 1.0)) as f32;
            }
            prev_embed = Some(last_embedding.clone());
            last_next = next.min(cfg.vocab_size - 1);

            // L2+L3 layered physics advance (same wiring as generate_streaming).
            if !last_embedding.is_empty() {
                cloud_ring.push_back(last_embedding.clone());
                while cloud_ring.len() > 8 { cloud_ring.pop_front(); }
            }
            if !cloud_ring.is_empty() {
                let cloud: Vec<Vec<f32>> = cloud_ring.iter().cloned().collect();
                let mut actual = vec![0.0f32; probs.len()];
                actual[next.min(probs.len() - 1)] = 1.0;
                let top_src = vfe::Estimate { value: probs.iter().fold(0.0f32, |a, b| a + b * b), variance: 0.5 };
                let tail: Vec<f32> = probs.iter().take(probs.len().min(16)).cloned().collect();
                let tv = if tail.is_empty() { 0.0 } else { tail.iter().sum::<f32>() / tail.len() as f32 };
                let tail_src = vfe::Estimate { value: tv, variance: 0.75 };
let adv = vfe::advance_physics(
                    temp_eff, cache.tau, vfe, &probs, &actual, &last_embedding,
                    current_curvature.max(0.5), &cloud, &domain_priors, &[top_src, tail_src],
                    vfe_tau_rate,
                );
                // The advance's temperature temps the NEXT token; its tau and
                // curved VFE enter the running statistics here.
                let adv_tau = adv.tau.clamp(tau_min, tau_max);
                cache.tau = adv_tau;
                m.avg_vfe = (m.avg_vfe * m.total_tokens as f32 + adv.vfe_curved)
                    / (m.total_tokens + 1) as f32;
                physics_adv = Some(adv);
            } else {
                m.avg_vfe = (m.avg_vfe * m.total_tokens as f32 + vfe)
                    / (m.total_tokens + 1) as f32;
            }
            m.avg_novelty = (m.avg_novelty * m.total_tokens as f32 + current_novelty)
                / (m.total_tokens + 1) as f32;
            m.avg_curvature = (m.avg_curvature * m.total_tokens as f32 + current_curvature)
                / (m.total_tokens + 1) as f32;
            m.final_tau = cache.tau;
            m.total_tokens += 1;

            // Next-token input.
            let tid_next = next.min(cfg.vocab_size - 1);
            let mut x_next = Array2::zeros((1, dim));
            for d in 0..dim { x_next[[0, d]] = self.embed[[tid_next, d]]; }
            x = x_next;
        }
        let elapsed = start.elapsed().as_secs_f32();
        m.tokens_per_sec = if elapsed > 0.0 { m.total_tokens as f32 / elapsed } else { 0.0 };
        m.final_hidden = last_embedding;
        m.avg_entropy = if m.total_tokens > 0 { entropy_acc / m.total_tokens as f32 } else { 0.0 };
        m.token_diversity = if m.total_tokens > 0 {
            appear.len() as f32 / m.total_tokens as f32
        } else { 0.0 };
        m.max_repeat_run = best_run;
        m.avg_drift = if m.total_tokens > 1 { drift_acc / (m.total_tokens - 1) as f32 } else { 0.0 };
        m.avg_eff_temp = if eff_temp_n > 0 { eff_temp_acc / eff_temp_n as f32 } else { 0.0 };
        self.rng.0 = rng_saved; // replay-safe: same stream every call
        m
    }
}

/// Seeded, reproducible xorshift64* PRNG (identical across platforms).
struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        SeededRng(seed.max(1))
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Local top-p sampler (mirrors main.rs's private `sample_top_p`).
/// Uses a caller-supplied seeded RNG so sampling is genuinely stochastic
/// (sensitive to temperature/top_p) yet reproducible per candidate:
/// softmax over scaled logits → sort desc → cumulative-probability top-p
/// cutoff → renormalize the surviving mass → seeded cumulative walk.
fn sample_top_p_rng(logits: &[f32], temperature: f32, top_p: f32, rng: &mut SeededRng) -> usize {
    if temperature <= 0.0 {
        return argmax_local(logits);
    }
    if logits.is_empty() {
        return 0;
    }
    // scaled softmax probabilities (temperature applied before the exp).
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut scaled: Vec<f32> = logits.iter().map(|&l| ((l - max_l) / temperature).exp()).collect();
    let total: f32 = scaled.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return argmax_local(logits);
    }
    for p in scaled.iter_mut() { *p /= total; }

    // Sort descending by probability.
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Top-p cutoff on cumulative probability.
    let mut cutoff = order.len();
    {
        let mut cum = 0.0f32;
        for (k, &idx) in order.iter().enumerate() {
            cum += scaled[idx];
            if cum >= top_p { cutoff = k + 1; break; }
        }
    }
    if cutoff == 0 { cutoff = 1; }

    // Renormalize the surviving set (in descending order).
    let mut mass: f32 = order[..cutoff].iter().map(|&idx| scaled[idx]).sum();
    if mass <= 0.0 { mass = 1.0; }

    // Seeded cumulative walk over the surviving tokens.
    let r = rng.next_f32() * mass;
    let mut acc = 0.0f32;
    for &idx in order[..cutoff].iter() {
        acc += scaled[idx];
        if acc >= r {
            return idx;
        }
    }
    order[cutoff - 1]
}

/// Argmax (greedy).
fn argmax_local(v: &[f32]) -> usize {
    v.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::model::{LayerWeights, Weights};
    use crate::gguf_write;
    use crate::tok::Tokenizer;

    /// Deterministic pseudo-random fill in [-1, 1].
    fn seeded(seed: usize) -> f32 {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1013904223);
        x ^= x >> 16;
        x = x.wrapping_mul(2246822519);
        x ^= x >> 13;
        (x as u32 % 2000) as f32 / 1000.0 - 1.0
    }

    fn tiny_weights(cfg: &Config) -> Weights {
        let d = cfg.dim;
        let v = cfg.vocab_size;
        let ik = cfg.intermediate;
        let mk = |rows: usize, cols: usize, base: usize| {
            ndarray::Array2::from_shape_fn((rows, cols), |(r, c)| seeded(base + r * cols + c) * 0.2)
        };
        let mut layers = Vec::new();
        for li in 0..cfg.n_layers {
            let b = 10_000 + li * 100_000;
            layers.push(LayerWeights {
                wq: mk(d, d, b),
                wk: mk(d, d, b + 1),
                wv: mk(d, d, b + 2),
                wo: mk(d, d, b + 3),
                w1: mk(ik, d, b + 4),
                w2: mk(d, ik, b + 5),
                w3: mk(ik, d, b + 6),
                attn_norm: ndarray::Array1::from_shape_fn(d, |i| seeded(b + i) * 0.5 + 1.0),
                ffn_norm: ndarray::Array1::from_shape_fn(d, |i| seeded(b + 900 + i) * 0.5 + 1.0),
                // MoE / MLA extensions unused for the tiny dense test model.
                moe_shared_w1: None, moe_shared_w2: None, moe_shared_w3: None,
                moe_expert_w1: None, moe_expert_w2: None, moe_expert_w3: None,
                moe_router_weight: None, moe_router_bias: None,
                mla_wq_a: None, mla_wq_b: None, mla_wk_v_a: None, mla_wk_b: None,
                mla_wv_b: None, mla_wq_rope: None, mla_wo: None,
                mla_q: None, mla_kv_a_mqa: None, mla_kv_b: None, mla_kv_a_norm: None,
                moe_gate_exps: None, moe_down_exps: None, moe_up_exps: None,
                moe_gate_inp: None, moe_shared_gate: None, moe_shared_down: None,
                moe_shared_up: None,
            });
        }
        Weights {
            // Writer declares embed as [dim, vocab]; reader dequant_arr2 + t() → [vocab, dim].
            embed: mk(d, v, 0),
            layers,
            final_norm: ndarray::Array1::from_shape_fn(d, |i| seeded(i + 7) * 0.5 + 1.0),
            // Writer declares output as [vocab, dim]; reader keeps nrows==vocab.
            output: mk(v, d, 700_000),
        }
    }

    /// Meta used to build a Tokenizer via from_gguf.
    fn tokenizer_meta(cfg: &Config) -> std::collections::HashMap<String, crate::loader::GgufMeta> {
        use crate::loader::GgufMeta;
        let mut m = std::collections::HashMap::new();
        let tokens: Vec<String> = (0..cfg.vocab_size).map(|i| format!("tok{i}")).collect();
        m.insert("tokenizer.ggml.tokens".into(), GgufMeta::StrArr(tokens));
        m.insert("tokenizer.ggml.bos_token_id".into(), GgufMeta::Num(1.0));
        m.insert("tokenizer.ggml.eos_token_id".into(), GgufMeta::Num(2.0));
        m.insert("tokenizer.ggml.unknown_token_id".into(), GgufMeta::Num(0.0));
        m
    }

    #[test]
    fn darwin_sampler_temp_sensitive() {
        // Bimodal logits: sharp peak + long tail. At low temp the peak dominates;
        // at high temp the tail gets picked with meaningful probability.
        let mut logits = vec![0.0f32; 128];
        logits[0] = 6.0; // strong peak
        for i in 1..64 { logits[i] = 1.0 - i as f32 * 0.1; } // mild tail
        let mut rng = SeededRng::new(42);
        let mut greedy_count = 0;
        for _ in 0..400 {
            if sample_top_p_rng(&logits, 0.05, 0.95, &mut rng) == 0 { greedy_count += 1; }
        }
        // At temp 0.05 the peak should win virtually always.
        if greedy_count != 400 {
            panic!("near-greedy temp must pick the peak (got {greedy_count}/400)");
        }
        let mut rng2 = SeededRng::new(42);
        let mut peak = 0;
        for _ in 0..400 {
            if sample_top_p_rng(&logits, 5.0, 0.95, &mut rng2) == 0 { peak += 1; }
        }
        // At temp 5.0 the distribution approaches uniform within the 0.95 mass,
        // so the single peak token must NOT always win.
        assert!(peak < 400, "hot temp must not be greedy ({peak}/400 picked peak)");
    }

    #[test]
    fn darwin_batch_decoder_end_to_end() {
        let cfg = Config::tiny();
        assert_eq!(cfg.dim, 64, "Config::tiny dim assumption");

        // Synthesize a tiny llama GGUF.
        let tok = Tokenizer::from_gguf(&tokenizer_meta(&cfg)).unwrap();
        let w = tiny_weights(&cfg);
        let out = std::env::temp_dir().join("kai_darwin_eval_test.gguf");
        let out_str = out.to_string_lossy().to_string();
        gguf_write::save_gguf(&w, &cfg, &tok, &out_str, &std::collections::HashMap::new());

        let prompt = "tok1 tok7 tok3";
        let mut dec = DarwinDecoder::open(&out_str, prompt, 8)
            .expect("batch decoder opens the synthetic GGUF");
        assert!(!dec.prompt_ids.is_empty(), "prompt must tokenize");

        // Batch 1: cold, greedy-ish, slow tau.
        let m1 = dec.evaluate(0.2, 0.9, 0.05, 0.5, 2.0, 12);
        assert!(m1.total_tokens > 0, "must generate tokens");
        assert!(m1.tokens_per_sec > 0.0, "must measure speed");
        assert!(m1.avg_vfe.is_finite() && m1.avg_vfe > 0.0, "vfe must be positive");
        assert!(m1.avg_novelty >= 0.0, "novelty is a g_ij average");
        assert!(m1.avg_curvature >= 0.0 && m1.avg_curvature <= 1.0);
        assert!(m1.final_tau >= 0.5 && m1.final_tau <= 2.0);

        // Batch 2: same model, hot temp + fast tau → different trajectory.
        let m2 = dec.evaluate(1.2, 0.8, 0.3, 0.2, 3.0, 12);
        assert!(m2.total_tokens > 0);
        // Higher VFE_rate should make tau respond more strongly (candidates differ).
        assert_ne!(m2.final_hidden, m1.final_hidden, "physics params must alter decode");

        // Regression: identical temp/top_p but different vfe_tau_rate must yield
        // a different effective sampling temperature — τ must genuinely couple
        // into decode (was a flat-passthrough bug before the τ-coupling fix).
        let ms = dec.evaluate(0.8, 0.9, 0.05, 0.5, 2.0, 16);
        let mf = dec.evaluate(0.8, 0.9, 0.5, 0.5, 2.0, 16);
        assert!(
            (ms.avg_eff_temp - mf.avg_eff_temp).abs() > 1e-4,
            "tau_rate must modulate eff temp (slow={:.4} fast={:.4})",
            ms.avg_eff_temp, mf.avg_eff_temp
        );

        // Determinism: replay batch 1 must be identical.
        let m1b = dec.evaluate(0.2, 0.9, 0.05, 0.5, 2.0, 12);
        assert_eq!(m1b.final_hidden, m1.final_hidden, "evaluate must be deterministic");

        let _ = std::fs::remove_file(&out);
    }
}
