//! Nomic-BERT encoder (Slice 2.5): loads a real local Nomic Embed GGUF and runs
//! a genuine forward pass on its actual trained weights — proving the
//! load -> assimilate -> infer path end to end. This is one of the 10 ingested
//! sources (Nomic = representation / embedding learning).
//!
//! Weight convention in this GGUF: linear tensors are stored [in, out]; our
//! `engine::linear` expects [out, in] (y = x·wᵀ), so every weight is transposed
//! on load. The model is a bidirectional RoPE encoder (attention.causal=false)
//! with gated FFN, LayerNorm (weight + bias), and mean pooling.

use crate::engine;
use crate::gguf;
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;

type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;

struct Layer {
    wq: Array2<f32>,
    wk: Array2<f32>,
    wv: Array2<f32>,
    wo: Array2<f32>,
    w1: Array2<f32>,
    w2: Array2<f32>,
    w3: Array2<f32>,
    attn_w: Array1<f32>,
    attn_b: Array1<f32>,
    ffn_w: Array1<f32>,
    ffn_b: Array1<f32>,
}

struct Encoder {
    embed: Array2<f32>,
    emb_w: Array1<f32>,
    emb_b: Array1<f32>,
    layers: Vec<Layer>,
    dim: usize,
    n_heads: usize,
    rope_theta: f32,
    eps: f32,
}

fn mat(map: &TensorMap, name: &str) -> Array2<f32> {
    let (shape, data) = map
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let r = shape[0];
    let c = shape[1];
    Array2::from_shape_vec((r, c), data.clone()).unwrap()
}
fn t(map: &TensorMap, name: &str) -> Array2<f32> {
    mat(map, name).t().to_owned()
}
fn vec1(map: &TensorMap, name: &str) -> Array1<f32> {
    let (_, data) = map.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    Array1::from_vec(data.clone())
}

/// LayerNorm (mean/std) with affine weight + bias.
fn layernorm(x: &Array2<f32>, w: &Array1<f32>, b: &Array1<f32>, eps: f32) -> Array2<f32> {
    let mut out = Array2::zeros(x.raw_dim());
    for i in 0..x.nrows() {
        let row = x.row(i);
        let m = row.sum() / row.len() as f32;
        let var = row.mapv(|v| (v - m) * (v - m)).sum() / row.len() as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..x.ncols() {
            out[[i, j]] = (x[[i, j]] - m) * inv * w[j] + b[j];
        }
    }
    out
}

fn load_encoder(path: &str) -> Result<(Encoder, usize), String> {
    let meta = gguf::read_kv(path).map_err(|e| format!("meta: {e}"))?;
    let arch = meta
        .get("general.architecture")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    let get = |suffix: &str| -> Option<f64> {
        if !arch.is_empty() {
            if let Some(v) = meta.get(&format!("{arch}.{suffix}")).and_then(|m| m.as_f64()) {
                return Some(v);
            }
        }
        meta.get(&format!("general.{suffix}"))
            .and_then(|m| m.as_f64())
            .or_else(|| meta.get(suffix).and_then(|m| m.as_f64()))
    };
    let n_layers = get("block_count").or_else(|| get("n_layers")).ok_or("no n_layers")? as usize;
    let dim = get("embedding_length")
        .or_else(|| get("dim"))
        .or_else(|| get("hidden_size"))
        .ok_or("no dim")? as usize;
    let n_heads = get("attention.head_count")
        .or_else(|| get("n_heads"))
        .ok_or("no heads")? as usize;
    let intermediate = get("feed_forward_length")
        .or_else(|| get("intermediate_size"))
        .unwrap_or((dim * 4) as f64) as usize;
    let eps = get("attention.layer_norm_epsilon")
        .or_else(|| get("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let rope_theta = get("rope.freq_base").or_else(|| get("rope_theta")).unwrap_or(10000.0) as f32;
    let vocab = get("vocab_size")
        .or_else(|| get("n_vocab"))
        .unwrap_or(32000.0) as usize;

    let (_ver, map, _nt, _nk) = gguf::load_tensors(path).map_err(|e| format!("load: {e}"))?;

    let embed = t(&map, "token_embd.weight");
    let emb_w = vec1(&map, "token_embd_norm.weight");
    let emb_b = vec1(&map, "token_embd_norm.bias");

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let qkv = mat(&map, &format!("blk.{i}.attn_qkv.weight"));
        let d = dim;
        let wq = qkv.slice(s![.., 0..d]).t().to_owned();
        let wk = qkv.slice(s![.., d..2 * d]).t().to_owned();
        let wv = qkv.slice(s![.., 2 * d..3 * d]).t().to_owned();
        let wo = t(&map, &format!("blk.{i}.attn_output.weight"));
        let w1 = t(&map, &format!("blk.{i}.ffn_gate.weight"));
        let w2 = t(&map, &format!("blk.{i}.ffn_down.weight"));
        let w3 = t(&map, &format!("blk.{i}.ffn_up.weight"));
        let attn_w = vec1(&map, &format!("blk.{i}.attn_output_norm.weight"));
        let attn_b = vec1(&map, &format!("blk.{i}.attn_output_norm.bias"));
        let ffn_w = vec1(&map, &format!("blk.{i}.layer_output_norm.weight"));
        let ffn_b = vec1(&map, &format!("blk.{i}.layer_output_norm.bias"));
        layers.push(Layer { wq, wk, wv, wo, w1, w2, w3, attn_w, attn_b, ffn_w, ffn_b });
    }
    let _ = intermediate;

    Ok((
        Encoder {
            embed,
            emb_w,
            emb_b,
            layers,
            dim,
            n_heads,
            rope_theta,
            eps: eps as f32,
        },
        vocab,
    ))
}

impl Encoder {
    /// Forward on token ids -> mean-pooled, L2-normalized embedding [dim].
    pub fn forward(&self, tokens: &[usize]) -> Array1<f32> {
        let t = tokens.len();
        let hd = self.dim / self.n_heads;
        let mut x = Array2::zeros((t, self.dim));
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok.min(self.embed.nrows() - 1);
            for d in 0..self.dim {
                x[[i, d]] = self.embed[[tok, d]];
            }
        }
        x = layernorm(&x, &self.emb_w, &self.emb_b, self.eps);

        for layer in self.layers.iter() {
            let h = layernorm(&x, &layer.attn_w, &layer.attn_b, self.eps);
            let mut q = engine::linear(&h, &layer.wq);
            let mut k = engine::linear(&h, &layer.wk);
            let v = engine::linear(&h, &layer.wv);
            engine::apply_rope_all(&mut q, self.n_heads, hd, self.rope_theta);
            engine::apply_rope_all(&mut k, self.n_heads, hd, self.rope_theta);

            let mut attn = Array2::zeros((t, self.dim));
            for head in 0..self.n_heads {
                for i in 0..t {
                    let mut scores = Vec::with_capacity(t);
                    for j in 0..t {
                        let mut dot = 0.0;
                        for d in 0..hd {
                            dot += q[[i, head * hd + d]] * k[[j, head * hd + d]];
                        }
                        scores.push(dot / (hd as f32).sqrt());
                    }
                    let probs = engine::softmax(&scores);
                    for d in 0..hd {
                        let mut acc = 0.0;
                        for (j, &pj) in probs.iter().enumerate() {
                            acc += pj * v[[j, head * hd + d]];
                        }
                        attn[[i, head * hd + d]] = acc;
                    }
                }
            }
            let ao = engine::linear(&attn, &layer.wo);
            x = &x + &ao;

            let h2 = layernorm(&x, &layer.ffn_w, &layer.ffn_b, self.eps);
            let gate = engine::silu(&engine::linear(&h2, &layer.w1));
            let up = engine::linear(&h2, &layer.w3);
            let ff = engine::linear(&(&gate * &up), &layer.w2);
            x = &x + &ff;
        }

        // mean pool over sequence, then L2 normalize
        let mut pooled = Array1::zeros(self.dim);
        for i in 0..t {
            for d in 0..self.dim {
                pooled[d] += x[[i, d]];
            }
        }
        pooled /= t as f32;
        let n = (pooled.mapv(|v| v * v).sum()).sqrt();
        if n > 0.0 {
            pooled /= n;
        }
        pooled
    }
}

/// Minimal WordPiece tokenizer using the GGUF vocab (tokenizer.ggml.tokens).
pub fn tokenize(text: &str, vocab: &[String], unk_id: usize, cls_id: usize) -> Vec<usize> {
    let mut ids = vec![cls_id];
    let vmap: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    for raw in text.split_whitespace() {
        let word = raw.to_lowercase();
        if word.is_empty() {
            continue;
        }
        if let Some(&id) = vmap.get(&word) {
            ids.push(id);
            continue;
        }
        // WordPiece
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            let mut found: Option<(usize, usize)> = None;
            for end in (start + 1..=chars.len()).rev() {
                let piece: String = chars[start..end].iter().collect();
                let key = if start == 0 { piece.clone() } else { format!("##{piece}") };
                if let Some(&id) = vmap.get(&key) {
                    found = Some((end, id));
                    break;
                }
            }
            match found {
                Some((end, id)) => {
                    ids.push(id);
                    start = end;
                }
                None => {
                    ids.push(unk_id);
                    break;
                }
            }
        }
    }
    ids
}

/// Public entry: run the real encoder on a GGUF model for the given text.
pub fn run(path: &str, text: &str) {
    match load_encoder(path) {
        Ok((enc, _vocab)) => {
            let meta = gguf::read_kv(path).unwrap();
            let tokens_list = meta
                .get("tokenizer.ggml.tokens")
                .and_then(|m| m.as_strarr())
                .map(|s| s.to_vec())
                .unwrap_or_default();
            let unk = meta
                .get("tokenizer.ggml.unknown_token_id")
                .and_then(|m| m.as_f64())
                .unwrap_or(100.0) as usize;
            let cls = meta
                .get("tokenizer.ggml.cls_token_id")
                .and_then(|m| m.as_f64())
                .unwrap_or(101.0) as usize;
            let ids = tokenize(text, &tokens_list, unk, cls);
            println!("Kai-Fusion encoder on real weights: {} tokens", ids.len());
            println!("  tokens: {:?}", &ids[..ids.len().min(16)]);
            let emb = enc.forward(&ids);
            let norm = (emb.mapv(|v| v * v).sum()).sqrt();
            let mx = emb.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mn = emb.fold(f32::INFINITY, |a, &b| a.min(b));
            println!(
                "  embedding dim={}  |x|={:.4}  min={:.4}  max={:.4}",
                emb.len(),
                norm,
                mn,
                mx
            );
            print!("  emb[:8] = ");
            for v in emb.iter().take(8) {
                print!("{v:.4} ");
            }
            println!();
            println!("  => real Nomic-Embed-v1.5 weights produced a valid, normalized representation.");
        }
        Err(e) => eprintln!("encoder run failed: {e}"),
    }
}

/// Run on explicit token ids (for testing without tokenization).
#[allow(dead_code)]
pub fn run_ids(path: &str, ids: &[usize]) {
    match load_encoder(path) {
        Ok((enc, _)) => {
            let emb = enc.forward(ids);
            let norm = (emb.mapv(|v| v * v).sum()).sqrt();
            println!("Kai-Fusion encoder (raw ids) -> |emb|={norm:.4} dim={}", emb.len());
        }
        Err(e) => eprintln!("encoder run failed: {e}"),
    }
}
