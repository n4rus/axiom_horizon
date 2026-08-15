//! Weight-space fusion — merge multiple GGUF models into one.
//!
//! Implements three merging algorithms:
//! - **Soup**: Simple weighted averaging (`λ * A + (1-λ) * B`)
//! - **TIES**: Trim + Elect Sign + Merge (handles sign conflicts)
#![allow(dead_code)]
//! - **DARE**: Drop And REscale (sparse delta merging)
//!
//! From Kai_FUSION_ARCHITECTURE.md §7-"The actual path" step 3:
//! "Weight-space merging (model soup, TIES, DARE) across quantized models."

use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::Config;
use crate::loader;
use crate::model::{LayerWeights, Weights};

/// Available merge methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeMethod {
    /// Simple weighted average: `λ * A + (1-λ) * B`
    Soup,
    /// TIES: Trim bottom-k% deltas, Elect Sign majority, Merge survivors
    Ties,
    /// DARE: Drop p% of delta params, rescale survivors
    Dare,
}

/// Configuration for merging
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Merge method
    pub method: MergeMethod,
    /// Lambda for Soup (0.0 = all B, 1.0 = all A)
    pub lambda: f32,
    /// Trim fraction for TIES (default 0.2 = remove bottom 20% by magnitude)
    pub trim_fraction: f32,
    /// Drop probability for DARE (default 0.9 = drop 90% of deltas)
    pub drop_prob: f32,
    /// Whether to verify model compatibility
    pub verify: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            method: MergeMethod::Soup,
            lambda: 0.5,
            trim_fraction: 0.2,
            drop_prob: 0.9,
            verify: true,
        }
    }
}

/// Error during merge
#[derive(Debug)]
pub enum MergeError {
    #[allow(dead_code)]
    MismatchedConfig(String),
    LoadError(String),
    SaveError(String),
    NoModels,
    IncompatibleModels(usize, String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::MismatchedConfig(msg) => write!(f, "Config mismatch: {msg}"),
            MergeError::LoadError(msg) => write!(f, "Load error: {msg}"),
            MergeError::SaveError(msg) => write!(f, "Save error: {msg}"),
            MergeError::NoModels => write!(f, "No models provided"),
            MergeError::IncompatibleModels(n, msg) => write!(f, "Model {n} incompatible: {msg}"),
        }
    }
}

impl std::error::Error for MergeError {}

/// Load a model and its config from a GGUF path
fn load_model(path: &str) -> Result<(Weights, Config, HashMap<String, u32>, Vec<String>), MergeError> {
    let meta = loader::read_kv(path).map_err(|e| MergeError::LoadError(e.to_string()))?;
    let cfg = loader::build_config(&meta)
        .ok_or_else(|| MergeError::LoadError("could not build Config (not a decoder arch?)".to_string()))?;
    let tok = crate::tok::Tokenizer::from_gguf(&meta)
        .ok_or_else(|| MergeError::LoadError("no tokenizer in GGUF".to_string()))?;
    let (_ver, info, _data_start) = loader::parse(path)
        .map_err(|e| MergeError::LoadError(e.to_string()))?;
    let mut tensor_types = HashMap::new();
    for t in &info {
        tensor_types.insert(t.name.clone(), t.ggml_type);
    }
    let (_v, map, _nt, _nk) = loader::load_tensors(path)
        .map_err(|e| MergeError::LoadError(e.to_string()))?;
    let model = Weights::from_gguf(&map, &cfg)
        .map_err(|e| MergeError::LoadError(e.to_string()))?;
    Ok((model, cfg, tensor_types, tok.vocab.clone()))
}

/// Verify that two models have compatible architectures for merging
fn verify_compatible(cfgs: &[&Config], vocabs: &[&Vec<String>]) -> Result<(), MergeError> {
    if cfgs.len() < 2 {
        return Err(MergeError::NoModels);
    }
    let ref_cfg = cfgs[0];
    let ref_vocab = vocabs[0];

    for (i, cfg) in cfgs.iter().enumerate().skip(1) {
        if cfg.dim != ref_cfg.dim {
            return Err(MergeError::IncompatibleModels(i,
                format!("dim mismatch: {} vs {}", cfg.dim, ref_cfg.dim)));
        }
        if cfg.n_layers != ref_cfg.n_layers {
            return Err(MergeError::IncompatibleModels(i,
                format!("layers mismatch: {} vs {}", cfg.n_layers, ref_cfg.n_layers)));
        }
        if cfg.vocab_size != ref_cfg.vocab_size {
            return Err(MergeError::IncompatibleModels(i,
                format!("vocab mismatch: {} vs {}", cfg.vocab_size, ref_cfg.vocab_size)));
        }
        if cfg.n_heads != ref_cfg.n_heads {
            return Err(MergeError::IncompatibleModels(i,
                format!("heads mismatch: {} vs {}", cfg.n_heads, ref_cfg.n_heads)));
        }
    }

    // Check vocab content compatibility (same tokens in same positions)
    for (i, vocab) in vocabs.iter().enumerate().skip(1) {
        if vocab.len() != ref_vocab.len() {
            return Err(MergeError::IncompatibleModels(i,
                format!("vocab length mismatch: {} vs {}", vocab.len(), ref_vocab.len())));
        }
        // Check first 100 tokens match (should be special tokens + common words)
        let n_check = 100.min(vocab.len());
        for j in 0..n_check {
            if vocab[j] != ref_vocab[j] {
                return Err(MergeError::IncompatibleModels(i,
                    format!("vocab token {j} mismatch: '{}' vs '{}'", vocab[j], ref_vocab[j])));
            }
        }
    }

    Ok(())
}

/// Apply delta to an Array2 (element-wise addition)
fn add_delta(w: &mut Array2<f32>, delta: &Array2<f32>) {
    ndarray::Zip::from(w).and(delta).for_each(|w_elem, &d| {
        *w_elem += d;
    });
}

/// Apply delta to an Array1
fn add_delta_1d(w: &mut ndarray::Array1<f32>, delta: &ndarray::Array1<f32>) {
    ndarray::Zip::from(w).and(delta).for_each(|w_elem, &d| {
        *w_elem += d;
    });
}

// ---------------------------------------------------------------------------
// Model Soup — simple weighted averaging
// ---------------------------------------------------------------------------

fn soup_weights(weights: &mut [Weights], lambda: f32) -> Weights {
    let n = weights.len();
    if n == 0 {
        panic!("no weights to merge");
    }
    // Take model 0 as base
    let mut base = weights[0].clone();

    if n == 1 {
        return base;
    }

    // λ for each model: model 0 gets λ, model 1 gets (1-λ), equal for >2
    let w0_weight = lambda;
    let w1_weight = 1.0 - lambda;

    // Merge embed
    ndarray::Zip::from(&mut base.embed).and(&weights[0].embed).and(&weights[1].embed)
        .for_each(|b, &a, &c| {
            *b = a * w0_weight + c * w1_weight;
        });

    // Merge output
    ndarray::Zip::from(&mut base.output).and(&weights[0].output).and(&weights[1].output)
        .for_each(|b, &a, &c| {
            *b = a * w0_weight + c * w1_weight;
        });

    // Merge final_norm
    ndarray::Zip::from(&mut base.final_norm).and(&weights[0].final_norm).and(&weights[1].final_norm)
        .for_each(|b, &a, &c| {
            *b = a * w0_weight + c * w1_weight;
        });

    // Merge per-layer weights
    for (li, layer) in base.layers.iter_mut().enumerate() {
        let w0 = &weights[0].layers[li];
        let w1 = &weights[1].layers[li];

        fn merge_2d(l: &mut Array2<f32>, a: &Array2<f32>, b: &Array2<f32>, wa: f32, wb: f32) {
            ndarray::Zip::from(l).and(a).and(b).for_each(|l_e, &a_e, &b_e| {
                *l_e = a_e * wa + b_e * wb;
            });
        }
        fn merge_1d(l: &mut ndarray::Array1<f32>, a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>, wa: f32, wb: f32) {
            ndarray::Zip::from(l).and(a).and(b).for_each(|l_e, &a_e, &b_e| {
                *l_e = a_e * wa + b_e * wb;
            });
        }

        merge_2d(&mut layer.wq, &w0.wq, &w1.wq, w0_weight, w1_weight);
        merge_2d(&mut layer.wk, &w0.wk, &w1.wk, w0_weight, w1_weight);
        merge_2d(&mut layer.wv, &w0.wv, &w1.wv, w0_weight, w1_weight);
        merge_2d(&mut layer.wo, &w0.wo, &w1.wo, w0_weight, w1_weight);
        merge_2d(&mut layer.w1, &w0.w1, &w1.w1, w0_weight, w1_weight);
        merge_2d(&mut layer.w2, &w0.w2, &w1.w2, w0_weight, w1_weight);
        merge_2d(&mut layer.w3, &w0.w3, &w1.w3, w0_weight, w1_weight);
        merge_1d(&mut layer.attn_norm, &w0.attn_norm, &w1.attn_norm, w0_weight, w1_weight);
        merge_1d(&mut layer.ffn_norm, &w0.ffn_norm, &w1.ffn_norm, w0_weight, w1_weight);
    }

    base
}

// ---------------------------------------------------------------------------
// TIES-Merging (Trim + Elect Sign + Merge)
// ---------------------------------------------------------------------------

/// TIES: Trim bottom-k% of deltas by magnitude, then Elect Sign by majority.
fn ties_weights(weights: &mut [Weights], trim_fraction: f32) -> Weights {
    let n = weights.len();
    if n < 2 {
        return weights[0].clone();
    }

    let base = weights[0].clone();

    // Compute task vectors (deltas) from base to each other model
    let mut deltas: Vec<Weights> = Vec::new();
    for w in weights.iter().skip(1) {
        let delta = compute_delta(&base, w);
        deltas.push(delta);
    }

    // Trim: zero out bottom-k% of each delta by magnitude
    for delta in &mut deltas {
        trim_deltas(delta, trim_fraction);
    }

    // Elect Sign: for each parameter, take majority sign across deltas
    // Merge: average of surviving (non-zero, sign-matched) deltas
    let merged_delta = merge_signed_deltas(&deltas);

    // Add merged delta back to base
    let mut result = base.clone();
    add_delta_weights(&mut result, &merged_delta);

    result
}

fn compute_delta(base: &Weights, other: &Weights) -> Weights {
    fn delta_2d(base: &Array2<f32>, other: &Array2<f32>) -> Array2<f32> {
        other - base
    }
    fn delta_1d(base: &ndarray::Array1<f32>, other: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
        other - base
    }

    let layers: Vec<LayerWeights> = base.layers.iter().zip(other.layers.iter()).map(|(bl, ol)| {
        LayerWeights {
            wq: delta_2d(&bl.wq, &ol.wq),
            wk: delta_2d(&bl.wk, &ol.wk),
            wv: delta_2d(&bl.wv, &ol.wv),
            wo: delta_2d(&bl.wo, &ol.wo),
            w1: delta_2d(&bl.w1, &ol.w1),
            w2: delta_2d(&bl.w2, &ol.w2),
            w3: delta_2d(&bl.w3, &ol.w3),
            attn_norm: delta_1d(&bl.attn_norm, &ol.attn_norm),
            ffn_norm: delta_1d(&bl.ffn_norm, &ol.ffn_norm),
            // MoE extensions
            moe_shared_w1: None,
            moe_shared_w2: None,
            moe_shared_w3: None,
            moe_expert_w1: None,
            moe_expert_w2: None,
            moe_expert_w3: None,
            moe_router_weight: None,
            moe_router_bias: None,
            // MLA extensions
            mla_wq_a: None,
            mla_wq_b: None,
            mla_wk_v_a: None,
            mla_wk_b: None,
            mla_wv_b: None,
            mla_wq_rope: None,
            mla_wo: None,
            // MoE extensions (GGUF)
            moe_gate_exps: None,
            moe_down_exps: None,
            moe_up_exps: None,
            moe_gate_inp: None,
            moe_shared_gate: None,
            moe_shared_down: None,
            moe_shared_up: None,
            // MLA extensions (GGUF)
            mla_q: None,
            mla_kv_a_mqa: None,
            mla_kv_b: None,
            mla_kv_a_norm: None,
        }
    }).collect();

    Weights {
        embed: delta_2d(&base.embed, &other.embed),
        layers,
        final_norm: delta_1d(&base.final_norm, &other.final_norm),
        output: delta_2d(&base.output, &other.output),
    }
}

fn trim_deltas(delta: &mut Weights, fraction: f32) {
    fn trim_2d(w: &mut Array2<f32>, frac: f32) {
        // Collect all absolute values
        let mut abs_vals: Vec<f32> = w.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Find threshold at the given percentile
        let idx = ((abs_vals.len() as f32) * frac) as usize;
        let threshold = if idx < abs_vals.len() { abs_vals[idx] } else { f32::MAX };
        // Zero out values below threshold
        w.mapv_inplace(|x| if x.abs() < threshold { 0.0 } else { x });
    }
    fn trim_1d(w: &mut ndarray::Array1<f32>, frac: f32) {
        let mut abs_vals: Vec<f32> = w.iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((abs_vals.len() as f32) * frac) as usize;
        let threshold = if idx < abs_vals.len() { abs_vals[idx] } else { f32::MAX };
        w.mapv_inplace(|x| if x.abs() < threshold { 0.0 } else { x });
    }

    trim_2d(&mut delta.embed, fraction);
    trim_2d(&mut delta.output, fraction);
    trim_1d(&mut delta.final_norm, fraction);
    for layer in &mut delta.layers {
        trim_2d(&mut layer.wq, fraction);
        trim_2d(&mut layer.wk, fraction);
        trim_2d(&mut layer.wv, fraction);
        trim_2d(&mut layer.wo, fraction);
        trim_2d(&mut layer.w1, fraction);
        trim_2d(&mut layer.w2, fraction);
        trim_2d(&mut layer.w3, fraction);
        trim_1d(&mut layer.attn_norm, fraction);
        trim_1d(&mut layer.ffn_norm, fraction);
    }
}

/// Merge multiple deltas with sign voting: for each param, take majority sign,
/// average only sign-matched survivors.
fn merge_signed_deltas(deltas: &[Weights]) -> Weights {
    if deltas.is_empty() {
        panic!("no deltas to merge");
    }
    let n = deltas.len() as f32;

    // Initialize merged as zeros
    let template = &deltas[0];
    let zero_layer = LayerWeights {
        wq: Array2::zeros(template.layers[0].wq.raw_dim()),
        wk: Array2::zeros(template.layers[0].wk.raw_dim()),
        wv: Array2::zeros(template.layers[0].wv.raw_dim()),
        wo: Array2::zeros(template.layers[0].wo.raw_dim()),
        w1: Array2::zeros(template.layers[0].w1.raw_dim()),
        w2: Array2::zeros(template.layers[0].w2.raw_dim()),
        w3: Array2::zeros(template.layers[0].w3.raw_dim()),
        attn_norm: ndarray::Array1::zeros(template.layers[0].attn_norm.raw_dim()),
        ffn_norm: ndarray::Array1::zeros(template.layers[0].ffn_norm.raw_dim()),
        // MoE extensions
        moe_shared_w1: None,
        moe_shared_w2: None,
        moe_shared_w3: None,
        moe_expert_w1: None,
        moe_expert_w2: None,
        moe_expert_w3: None,
        moe_router_weight: None,
        moe_router_bias: None,
        // MLA extensions
        mla_wq_a: None,
        mla_wq_b: None,
        mla_wk_v_a: None,
        mla_wk_b: None,
        mla_wv_b: None,
        mla_wq_rope: None,
        mla_wo: None,
        // MoE extensions (GGUF)
        moe_gate_exps: None,
        moe_down_exps: None,
        moe_up_exps: None,
        moe_gate_inp: None,
        moe_shared_gate: None,
        moe_shared_down: None,
        moe_shared_up: None,
        // MLA extensions (GGUF)
        mla_q: None,
        mla_kv_a_mqa: None,
        mla_kv_b: None,
        mla_kv_a_norm: None,
    };

    fn sign_vote_2d(merged: &mut Array2<f32>, deltas: &[&Array2<f32>], _n: f32) {
        // For each element: count positive deltas, negative deltas
        // Majority sign = whichever has more votes
        // Merge = average of all deltas with the majority sign
        let _shape = merged.raw_dim();
        let total_elements = merged.len();
        let mut pos_count = vec![0u32; total_elements];
        let mut neg_count = vec![0u32; total_elements];
        let mut pos_sum = vec![0.0f32; total_elements];
        let mut neg_sum = vec![0.0f32; total_elements];

        for delta in deltas {
            for (i, &val) in delta.iter().enumerate() {
                if val > 0.0 {
                    pos_count[i] += 1;
                    pos_sum[i] += val;
                } else if val < 0.0 {
                    neg_count[i] += 1;
                    neg_sum[i] += val;
                }
            }
        }

        // Merged = majority sign average
        for i in 0..total_elements {
            if pos_count[i] > neg_count[i] && pos_count[i] > 0 {
                merged.as_slice_mut().unwrap()[i] = pos_sum[i] / pos_count[i] as f32;
            } else if neg_count[i] > pos_count[i] && neg_count[i] > 0 {
                merged.as_slice_mut().unwrap()[i] = neg_sum[i] / neg_count[i] as f32;
            } else if pos_count[i] > 0 && neg_count[i] > 0 && pos_count[i] == neg_count[i] {
                // Tie: take whichever has larger magnitude
                if pos_sum[i].abs() > neg_sum[i].abs() {
                    merged.as_slice_mut().unwrap()[i] = pos_sum[i] / pos_count[i] as f32;
                } else {
                    merged.as_slice_mut().unwrap()[i] = neg_sum[i] / neg_count[i] as f32;
                }
            }
            // else: all zeros → stays 0
        }
    }

    fn sign_vote_1d(merged: &mut ndarray::Array1<f32>, deltas: &[&ndarray::Array1<f32>], _n: f32) {
        let total = merged.len();
        let mut pos_count = vec![0u32; total];
        let mut neg_count = vec![0u32; total];
        let mut pos_sum = vec![0.0f32; total];
        let mut neg_sum = vec![0.0f32; total];

        for delta in deltas {
            for (i, &val) in delta.iter().enumerate() {
                if val > 0.0 {
                    pos_count[i] += 1;
                    pos_sum[i] += val;
                } else if val < 0.0 {
                    neg_count[i] += 1;
                    neg_sum[i] += val;
                }
            }
        }

        for i in 0..total {
            if pos_count[i] > neg_count[i] && pos_count[i] > 0 {
                merged.as_slice_mut().unwrap()[i] = pos_sum[i] / pos_count[i] as f32;
            } else if neg_count[i] > pos_count[i] && neg_count[i] > 0 {
                merged.as_slice_mut().unwrap()[i] = neg_sum[i] / neg_count[i] as f32;
            } else if pos_count[i] > 0 && neg_count[i] > 0 && pos_count[i] == neg_count[i] {
                if pos_sum[i].abs() > neg_sum[i].abs() {
                    merged.as_slice_mut().unwrap()[i] = pos_sum[i] / pos_count[i] as f32;
                } else {
                    merged.as_slice_mut().unwrap()[i] = neg_sum[i] / neg_count[i] as f32;
                }
            }
        }
    }

    let ref_deltas: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.embed).collect();
    let mut merged_embed = Array2::zeros(template.embed.raw_dim());
    sign_vote_2d(&mut merged_embed, &ref_deltas, n);

    let ref_output: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.output).collect();
    let mut merged_output = Array2::zeros(template.output.raw_dim());
    sign_vote_2d(&mut merged_output, &ref_output, n);

    let ref_norm: Vec<&ndarray::Array1<f32>> = deltas.iter().map(|d| &d.final_norm).collect();
    let mut merged_norm = ndarray::Array1::zeros(template.final_norm.raw_dim());
    sign_vote_1d(&mut merged_norm, &ref_norm, n);

    let mut layers = Vec::with_capacity(template.layers.len());
    for li in 0..template.layers.len() {
        let l = &zero_layer;
        let mut ml = l.clone();

        let wqs: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].wq).collect();
        sign_vote_2d(&mut ml.wq, &wqs, n);

        let wks: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].wk).collect();
        sign_vote_2d(&mut ml.wk, &wks, n);

        let wvs: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].wv).collect();
        sign_vote_2d(&mut ml.wv, &wvs, n);

        let wos: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].wo).collect();
        sign_vote_2d(&mut ml.wo, &wos, n);

        let w1s: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].w1).collect();
        sign_vote_2d(&mut ml.w1, &w1s, n);

        let w2s: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].w2).collect();
        sign_vote_2d(&mut ml.w2, &w2s, n);

        let w3s: Vec<&Array2<f32>> = deltas.iter().map(|d| &d.layers[li].w3).collect();
        sign_vote_2d(&mut ml.w3, &w3s, n);

        let ans: Vec<&ndarray::Array1<f32>> = deltas.iter().map(|d| &d.layers[li].attn_norm).collect();
        sign_vote_1d(&mut ml.attn_norm, &ans, n);

        let fns: Vec<&ndarray::Array1<f32>> = deltas.iter().map(|d| &d.layers[li].ffn_norm).collect();
        sign_vote_1d(&mut ml.ffn_norm, &fns, n);

        layers.push(ml);
    }

    Weights {
        embed: merged_embed,
        layers,
        final_norm: merged_norm,
        output: merged_output,
    }
}

fn add_delta_weights(base: &mut Weights, delta: &Weights) {
    add_delta(&mut base.embed, &delta.embed);
    add_delta(&mut base.output, &delta.output);
    add_delta_1d(&mut base.final_norm, &delta.final_norm);
    for (li, layer) in base.layers.iter_mut().enumerate() {
        add_delta(&mut layer.wq, &delta.layers[li].wq);
        add_delta(&mut layer.wk, &delta.layers[li].wk);
        add_delta(&mut layer.wv, &delta.layers[li].wv);
        add_delta(&mut layer.wo, &delta.layers[li].wo);
        add_delta(&mut layer.w1, &delta.layers[li].w1);
        add_delta(&mut layer.w2, &delta.layers[li].w2);
        add_delta(&mut layer.w3, &delta.layers[li].w3);
        add_delta_1d(&mut layer.attn_norm, &delta.layers[li].attn_norm);
        add_delta_1d(&mut layer.ffn_norm, &delta.layers[li].ffn_norm);
    }
}

// ---------------------------------------------------------------------------
// DARE (Drop And REscale)
// ---------------------------------------------------------------------------

/// DARE: randomly drop p% of delta parameters, rescale survivors by 1/(1-p).
fn dare_weights(weights: &mut [Weights], drop_prob: f32) -> Weights {
    let n = weights.len();
    if n < 2 {
        return weights[0].clone();
    }

    let base = weights[0].clone();
    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - drop_prob);

    // For each other model, compute delta, drop, rescale, add to base
    let mut result = base.clone();

    for w in weights.iter().skip(1) {
        let delta = compute_delta(&weights[0], w);
        let masked_delta = apply_dare_mask(&delta, drop_prob, scale, &mut rng);
        add_delta_weights(&mut result, &masked_delta);
    }

    // If >2 models, divide by number of merged deltas
    let n_deltas = (n - 1) as f32;
    if n_deltas > 1.0 {
        result.embed.mapv_inplace(|x| x / n_deltas);
        result.output.mapv_inplace(|x| x / n_deltas);
        result.final_norm.mapv_inplace(|x| x / n_deltas);
        for layer in &mut result.layers {
            layer.wq.mapv_inplace(|x| x / n_deltas);
            layer.wk.mapv_inplace(|x| x / n_deltas);
            layer.wv.mapv_inplace(|x| x / n_deltas);
            layer.wo.mapv_inplace(|x| x / n_deltas);
            layer.w1.mapv_inplace(|x| x / n_deltas);
            layer.w2.mapv_inplace(|x| x / n_deltas);
            layer.w3.mapv_inplace(|x| x / n_deltas);
            layer.attn_norm.mapv_inplace(|x| x / n_deltas);
            layer.ffn_norm.mapv_inplace(|x| x / n_deltas);
        }
    }

    result
}

fn apply_dare_mask(delta: &Weights, drop_prob: f32, scale: f32, rng: &mut impl Rng) -> Weights {
    fn mask_2d(w: &Array2<f32>, p: f32, s: f32, rng: &mut impl Rng) -> Array2<f32> {
        let mut result = w.clone();
        for elem in result.iter_mut() {
            if rng.gen::<f32>() < p {
                *elem = 0.0;
            } else {
                *elem *= s;
            }
        }
        result
    }
    fn mask_1d(w: &ndarray::Array1<f32>, p: f32, s: f32, rng: &mut impl Rng) -> ndarray::Array1<f32> {
        let mut result = w.clone();
        for elem in result.iter_mut() {
            if rng.gen::<f32>() < p {
                *elem = 0.0;
            } else {
                *elem *= s;
            }
        }
        result
    }

    let layers: Vec<LayerWeights> = delta.layers.iter().map(|l| LayerWeights {
        wq: mask_2d(&l.wq, drop_prob, scale, rng),
        wk: mask_2d(&l.wk, drop_prob, scale, rng),
        wv: mask_2d(&l.wv, drop_prob, scale, rng),
        wo: mask_2d(&l.wo, drop_prob, scale, rng),
        w1: mask_2d(&l.w1, drop_prob, scale, rng),
        w2: mask_2d(&l.w2, drop_prob, scale, rng),
        w3: mask_2d(&l.w3, drop_prob, scale, rng),
        attn_norm: mask_1d(&l.attn_norm, drop_prob, scale, rng),
        ffn_norm: mask_1d(&l.ffn_norm, drop_prob, scale, rng),
        // MoE extensions
        moe_shared_w1: None,
        moe_shared_w2: None,
        moe_shared_w3: None,
        moe_expert_w1: None,
        moe_expert_w2: None,
        moe_expert_w3: None,
        moe_router_weight: None,
        moe_router_bias: None,
        // MLA extensions
        mla_wq_a: None,
        mla_wq_b: None,
        mla_wk_v_a: None,
        mla_wk_b: None,
        mla_wv_b: None,
        mla_wq_rope: None,
        mla_wo: None,
        // MoE extensions (GGUF)
        moe_gate_exps: None,
        moe_down_exps: None,
        moe_up_exps: None,
        moe_gate_inp: None,
        moe_shared_gate: None,
        moe_shared_down: None,
        moe_shared_up: None,
        // MLA extensions (GGUF)
        mla_q: None,
        mla_kv_a_mqa: None,
        mla_kv_b: None,
        mla_kv_a_norm: None,
    }).collect();

    Weights {
        embed: mask_2d(&delta.embed, drop_prob, scale, rng),
        layers,
        final_norm: mask_1d(&delta.final_norm, drop_prob, scale, rng),
        output: mask_2d(&delta.output, drop_prob, scale, rng),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Merge multiple GGUF models into one using the specified method.
///
/// # Arguments
/// * `paths` — List of paths to GGUF files. First model is the base.
/// * `config` — Merge configuration (method, lambda, etc.)
/// * `output_path` — Where to write the merged GGUF
///
/// # Returns
/// The merged weights and config (for further use or analysis)
pub fn merge_models(
    paths: &[&str],
    config: &MergeConfig,
    output_path: &str,
) -> Result<(Weights, Config), MergeError> {
    if paths.len() < 2 {
        return Err(MergeError::NoModels);
    }

    // Load all models
    let mut models: Vec<(Weights, Config, HashMap<String, u32>, Vec<String>)> = Vec::new();
    for path in paths {
        let (w, c, tt, v) = load_model(path)?;
        models.push((w, c, tt, v));
    }

    let cfgs: Vec<&Config> = models.iter().map(|(_, c, _, _)| c).collect();
    let vocabs: Vec<&Vec<String>> = models.iter().map(|(_, _, _, v)| v).collect();

    // Verify compatibility
    if config.verify {
        verify_compatible(&cfgs, &vocabs)?;
    }

    let ref_cfg = cfgs[0].clone();
    let ref_tt = models[0].2.clone();
    let ref_vocab = models[0].3.clone();

    // Extract just the weights for merging
    let mut all_weights: Vec<Weights> = models.into_iter().map(|(w, _, _, _)| w).collect();

    // Merge
    let merged = match config.method {
        MergeMethod::Soup => {
            println!("  Merging {} models with Soup (λ={})", paths.len(), config.lambda);
            soup_weights(&mut all_weights, config.lambda)
        }
        MergeMethod::Ties => {
            println!("  Merging {} models with TIES (trim={})", paths.len(), config.trim_fraction);
            ties_weights(&mut all_weights, config.trim_fraction)
        }
        MergeMethod::Dare => {
            println!("  Merging {} models with DARE (drop={})", paths.len(), config.drop_prob);
            dare_weights(&mut all_weights, config.drop_prob)
        }
    };

    // Save merged model (preserve the first model's tokenizer vocab)
    save_merged(&merged, &ref_cfg, &ref_tt, output_path, &ref_vocab)?;

    Ok((merged, ref_cfg))
}

/// Save merged weights as GGUF, preserving the original tokenizer vocabulary.
fn save_merged(
    model: &Weights,
    cfg: &Config,
    tensor_types: &HashMap<String, u32>,
    output_path: &str,
    vocab_list: &[String],
) -> Result<(), MergeError> {
    let tt = tensor_types.clone();

    let meta: Vec<(String, crate::loader::GgufMeta)> = vec![
        ("general.architecture".into(), crate::loader::GgufMeta::Str("llama".into())),
        ("general.name".into(), crate::loader::GgufMeta::Str("kai-merged".into())),
        ("llama.block_count".into(), crate::loader::GgufMeta::Num(cfg.n_layers as f64)),
        ("llama.embedding_length".into(), crate::loader::GgufMeta::Num(cfg.dim as f64)),
        ("llama.attention.head_count".into(), crate::loader::GgufMeta::Num(cfg.n_heads as f64)),
        ("llama.attention.head_count_kv".into(), crate::loader::GgufMeta::Num(cfg.n_kv_heads as f64)),
        ("llama.attention.layer_norm_rms_epsilon".into(), crate::loader::GgufMeta::Num(1e-5)),
        ("llama.feed_forward_length".into(), crate::loader::GgufMeta::Num(cfg.intermediate as f64)),
        ("llama.rope.freq_base".into(), crate::loader::GgufMeta::Num(cfg.rope_theta as f64)),
        ("llama.vocab_size".into(), crate::loader::GgufMeta::Num(cfg.vocab_size as f64)),
        ("llama.context_length".into(), crate::loader::GgufMeta::Num(cfg.max_seq as f64)),
        ("tokenizer.ggml.model".into(), crate::loader::GgufMeta::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), crate::loader::GgufMeta::StrArr(vocab_list.to_vec())),
        ("tokenizer.ggml.scores".into(), crate::loader::GgufMeta::Arr(vec![0.0; vocab_list.len()])),
        ("tokenizer.ggml.bos_token_id".into(), crate::loader::GgufMeta::Num(1.0)),
        ("tokenizer.ggml.eos_token_id".into(), crate::loader::GgufMeta::Num(2.0)),
        ("tokenizer.ggml.unknown_token_id".into(), crate::loader::GgufMeta::Num(0.0)),
    ];

    write_gguf(output_path, &meta, model, cfg, &tt)
        .map_err(|e| MergeError::SaveError(e.to_string()))
}

/// Write a GGUF file from merged weights
fn write_gguf(
    path: &str,
    meta: &[(String, crate::loader::GgufMeta)],
    model: &Weights,
    cfg: &Config,
    tensor_types: &HashMap<String, u32>,
) -> Result<(), String> {
    use crate::gguf_write as gen_internal;
    use std::fs::File;
    use std::io::Write;

    let mut buf: Vec<u8> = Vec::new();
    let dk = cfg.dim_kv();

    // GGUFv3 header
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    gen_internal::put_u32(&mut buf, 3);

    // Count tensors
    let n_tensors = 1 + cfg.n_layers * 9 + 2; // embed + layers*9 + final_norm + output
    gen_internal::put_u64(&mut buf, n_tensors as u64);
    gen_internal::put_u64(&mut buf, meta.len() as u64);

    // Write metadata
    for (k, v) in meta {
        gen_internal::put_str(&mut buf, k);
        gen_internal::put_value(&mut buf, v);
    }

    // Collect tensor names, shapes, data
    struct TensorInfo {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    let mut tensors: Vec<TensorInfo> = Vec::new();

    tensors.push(TensorInfo {
        name: "token_embd.weight".to_string(),
        shape: vec![cfg.dim, cfg.vocab_size],
        data: model.embed.iter().copied().collect(),
    });

    for li in 0..cfg.n_layers {
        let l = &model.layers[li];
        tensors.push(TensorInfo {
            name: format!("blk.{li}.attn_q.weight"),
            shape: vec![cfg.dim, cfg.dim],
            data: l.wq.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.attn_k.weight"),
            shape: vec![dk, cfg.dim],
            data: l.wk.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.attn_v.weight"),
            shape: vec![dk, cfg.dim],
            data: l.wv.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.attn_output.weight"),
            shape: vec![cfg.dim, cfg.dim],
            data: l.wo.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.ffn_gate.weight"),
            shape: vec![cfg.intermediate, cfg.dim],
            data: l.w1.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.ffn_up.weight"),
            shape: vec![cfg.intermediate, cfg.dim],
            data: l.w3.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.ffn_down.weight"),
            shape: vec![cfg.dim, cfg.intermediate],
            data: l.w2.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.attn_norm.weight"),
            shape: vec![cfg.dim],
            data: l.attn_norm.iter().copied().collect(),
        });
        tensors.push(TensorInfo {
            name: format!("blk.{li}.ffn_norm.weight"),
            shape: vec![cfg.dim],
            data: l.ffn_norm.iter().copied().collect(),
        });
    }

    tensors.push(TensorInfo {
        name: "output_norm.weight".to_string(),
        shape: vec![cfg.dim],
        data: model.final_norm.iter().copied().collect(),
    });

    tensors.push(TensorInfo {
        name: "output.weight".to_string(),
        shape: vec![cfg.vocab_size, cfg.dim],
        data: model.output.iter().copied().collect(),
    });

    // Write tensor info headers
    let mut off2 = 0usize;
    for t in &tensors {
        gen_internal::put_str(&mut buf, &t.name);
        gen_internal::put_u32(&mut buf, t.shape.len() as u32);
        for s in &t.shape {
            gen_internal::put_u64(&mut buf, *s as u64);
        }
        // Use the tensor type from the original model, default to F16 (1)
        let ggml_type = tensor_types.get(&t.name).copied().unwrap_or(1);
        gen_internal::put_u32(&mut buf, ggml_type);
        gen_internal::put_u64(&mut buf, off2 as u64);
        let nbytes = t.data.len() * 2; // F16 = 2 bytes per element
        off2 = gen_internal::align32(off2 + nbytes);
    }

    // Pad to 32-aligned data start
    let data_start = gen_internal::align32(buf.len());
    while buf.len() < data_start {
        buf.push(0);
    }

    // Write tensor data (convert f32 to f16)
    for t in &tensors {
        let f16_bytes: Vec<u8> = t.data.iter()
            .flat_map(|&x| gen_internal::f32_to_f16(x).to_le_bytes().to_vec())
            .collect();
        buf.extend_from_slice(&f16_bytes);
        let nbytes = t.data.len() * 2;
        let _padded = gen_internal::align32(nbytes);
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
    }

    let mut f = File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    f.write_all(&buf).map_err(|e| format!("write {path}: {e}"))?;
    println!("wrote merged GGUF -> {path} ({:.2} MB)", buf.len() as f64 / 1e6);

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_config_defaults() {
        let cfg = MergeConfig::default();
        assert_eq!(cfg.method, MergeMethod::Soup);
        assert!((cfg.lambda - 0.5).abs() < 1e-6);
        assert!((cfg.trim_fraction - 0.2).abs() < 1e-6);
        assert!((cfg.drop_prob - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_verify_compatible_same() {
        let cfg = Config {
            dim: 512, n_layers: 8, n_heads: 8, n_kv_heads: 4,
            vocab_size: 4096, intermediate: 1024,
            rope_theta: 10000.0, max_seq: 512,
            tau: 1.0, e: 1.0, age: 0, cycles: 0,
            h: 0.5, base_ms: 1000.0, phi: 0.0,
            attn_policy: crate::config::AttnPolicy::Global(crate::config::AttnKind::MHA),
            mlp_kind: crate::config::MlpKind::Dense,
            moe: crate::config::MoEConfig::default(),
            mla: crate::config::MLAConfig::default(),
            vision: crate::config::VisionConfig::default(),
            expert_intermediate: 0,
            leading_dense_blocks: 0,
        };
        let vocab = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        let result = verify_compatible(&[&cfg, &cfg], &[&vocab, &vocab]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_compatible_dim_mismatch() {
        fn make_cfg(dim: usize) -> Config {
            Config {
                dim, n_layers: 8, n_heads: 8, n_kv_heads: 4,
                vocab_size: 4096, intermediate: 1024,
                rope_theta: 10000.0, max_seq: 512,
                tau: 1.0, e: 1.0, age: 0, cycles: 0,
                h: 0.5, base_ms: 1000.0, phi: 0.0,
                attn_policy: crate::config::AttnPolicy::Global(crate::config::AttnKind::MHA),
                mlp_kind: crate::config::MlpKind::Dense,
                moe: crate::config::MoEConfig::default(),
                mla: crate::config::MLAConfig::default(),
                vision: crate::config::VisionConfig::default(),
            expert_intermediate: 0,
            leading_dense_blocks: 0,
            }
        }
        let cfg_a = make_cfg(512);
        let cfg_b = make_cfg(1024);
        let vocab = vec!["<unk>".to_string()];
        let result = verify_compatible(&[&cfg_a, &cfg_b], &[&vocab, &vocab]);
        assert!(result.is_err());
    }
}
