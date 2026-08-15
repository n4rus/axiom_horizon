//! Dev tool: write a tiny llama-format GGUF (F16) with deterministic weights +
//! a small vocab, so the decoder pipeline can be exercised end-to-end on a real
//! GGUF produced by our own reader. Not a trained model — a correctness harness
//! for the load -> map -> generate path (the same path a real llama GGUF uses).

use crate::config::{Config, AttnPolicy, AttnKind, MlpKind, MoEConfig, MLAConfig, VisionConfig};
use crate::gguf::GgufMeta;
use std::fs::File;
use std::io::Write;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}
fn put_value(b: &mut Vec<u8>, v: &GgufMeta) {
    match v {
        GgufMeta::Num(f) => {
            put_u32(b, 6);
            put_f32(b, *f as f32);
        }
        GgufMeta::Str(s) => {
            put_u32(b, 8);
            put_str(b, s);
        }
        GgufMeta::Arr(a) => {
            put_u32(b, 9);
            put_u32(b, 6);
            put_u64(b, a.len() as u64);
            for x in a {
                put_f32(b, *x as f32);
            }
        }
        GgufMeta::StrArr(a) => {
            put_u32(b, 9);
            put_u32(b, 8);
            put_u64(b, a.len() as u64);
            for s in a {
                put_str(b, s);
            }
        }
        GgufMeta::Bool(bv) => {
            put_u32(b, 7);
            b.push(if *bv { 1 } else { 0 });
        }
    }
}

fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = (x >> 23) & 0xff;
    let mant = x & 0x7fffff;
    if exp == 255 {
        return sign | 0x7c00 | ((mant >> 13) as u16);
    }
    if exp == 0 {
        return sign;
    }
    let e = (exp as i32) - 127;
    if e > 15 {
        return sign | 0x7c00;
    }
    if e >= -14 {
        let ee = (e + 15) as u32;
        let m = mant >> 13;
        return sign | ((ee << 10) as u16) | (m as u16);
    }
    let shift = (-(e + 1)) as u32;
    if shift >= 24 {
        return sign;
    }
    let v = 0x800000u32 | mant;
    let m = (v >> shift) as u16;
    sign | m
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.0 >> 33) as f32;
        (bits / (1u32 << 31) as f32) - 1.0
    }
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f32_to_f16(x).to_le_bytes().to_vec()).collect()
}
fn align32(n: usize) -> usize {
    (n + 31) & !31usize
}
fn small(r: &mut Rng) -> f32 {
    r.next_f32() * 0.05
}
fn ones(_r: &mut Rng) -> f32 {
    1.0f32
}

/// Generate a tiny llama-format GGUF at `path`.
pub fn gen(path: &str, dim: usize, layers: usize, vocab: usize, words: &[&str]) {
    let words: Vec<String> = if words.is_empty() {
        (0..vocab).map(|i| i.to_string()).collect()
    } else {
        words.iter().map(|s| s.to_string()).collect()
    };
    let words: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
    let cfg = Config {
        dim,
        n_layers: layers,
        n_heads: 8,
        n_kv_heads: 4,
        vocab_size: vocab,
        intermediate: dim * 2,
        rope_theta: 10000.0,
        max_seq: 256,
        attn_policy: AttnPolicy::Global(AttnKind::MHA),
        mlp_kind: MlpKind::Dense,
        moe: MoEConfig::default(),
        mla: MLAConfig::default(),
        vision: VisionConfig::default(),
        tau: 1.0,
        e: 1.0,
        age: 0,
        cycles: 0,
        h: 0.5,
        base_ms: 1000.0,
        phi: 0.0,
        leading_dense_blocks: 0,
        expert_intermediate: 0,
    };
    let dk = cfg.dim_kv();
    let mut rng = Rng(0x1234_5678);
    let mut tensors: Vec<(String, usize, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut add = |name: &str, shape: Vec<usize>, fill: &dyn Fn(&mut Rng) -> f32| {
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(fill(&mut rng));
        }
        tensors.push((name.to_string(), 1u8 as usize, shape, data));
    };
    add("token_embd.weight", vec![dim, vocab], &small);
    for i in 0..layers {
        add(&format!("blk.{i}.attn_q.weight"), vec![dim, dim], &small);
        add(&format!("blk.{i}.attn_k.weight"), vec![dk, dim], &small);
        add(&format!("blk.{i}.attn_v.weight"), vec![dk, dim], &small);
        add(&format!("blk.{i}.attn_output.weight"), vec![dim, dim], &small);
        add(&format!("blk.{i}.ffn_gate.weight"), vec![cfg.intermediate, dim], &small);
        add(&format!("blk.{i}.ffn_up.weight"), vec![cfg.intermediate, dim], &small);
        add(&format!("blk.{i}.ffn_down.weight"), vec![dim, cfg.intermediate], &small);
        add(&format!("blk.{i}.attn_norm.weight"), vec![dim], &ones);
        add(&format!("blk.{i}.ffn_norm.weight"), vec![dim], &ones);
    }
    add("output_norm.weight", vec![dim], &ones);
    add("output.weight", vec![vocab, dim], &small);

    // vocab: 0=<unk>, 1=<s>(bos), 2=</s>(eos), then words, then filler
    let mut vocab_list: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    for w in words {
        vocab_list.push(w.to_string());
    }
    let mut i = vocab_list.len();
    while vocab_list.len() < vocab {
        vocab_list.push(format!("<t{i}>"));
        i += 1;
    }

    // --- assemble GGUF v3 ---
    let meta: Vec<(String, GgufMeta)> = vec![
        ("general.architecture".into(), GgufMeta::Str("llama".into())),
        ("general.name".into(), GgufMeta::Str("kai-fusion-gen".into())),
        ("llama.block_count".into(), GgufMeta::Num(layers as f64)),
        ("llama.embedding_length".into(), GgufMeta::Num(dim as f64)),
        ("llama.attention.head_count".into(), GgufMeta::Num(cfg.n_heads as f64)),
        ("llama.attention.head_count_kv".into(), GgufMeta::Num(cfg.n_kv_heads as f64)),
        ("llama.attention.layer_norm_rms_epsilon".into(), GgufMeta::Num(1e-5)),
        ("llama.feed_forward_length".into(), GgufMeta::Num(cfg.intermediate as f64)),
        ("llama.rope.freq_base".into(), GgufMeta::Num(cfg.rope_theta as f64)),
        ("llama.vocab_size".into(), GgufMeta::Num(vocab as f64)),
        ("llama.context_length".into(), GgufMeta::Num(cfg.max_seq as f64)),
        ("tokenizer.ggml.model".into(), GgufMeta::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), GgufMeta::StrArr(vocab_list.clone())),
        ("tokenizer.ggml.scores".into(), GgufMeta::Arr(vec![0.0; vocab])),
        ("tokenizer.ggml.bos_token_id".into(), GgufMeta::Num(1.0)),
        ("tokenizer.ggml.eos_token_id".into(), GgufMeta::Num(2.0)),
        ("tokenizer.ggml.unknown_token_id".into(), GgufMeta::Num(0.0)),
    ];

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    put_u32(&mut buf, 3); // version
    put_u64(&mut buf, tensors.len() as u64);
    put_u64(&mut buf, meta.len() as u64);
    for (k, v) in &meta {
        put_str(&mut buf, k);
        put_value(&mut buf, v);
    }
    // tensor infos: offsets are relative to the (32-aligned) data start.
    let mut off2 = 0usize;
    for (name, _, shape, _) in &tensors {
        put_str(&mut buf, name);
        put_u32(&mut buf, shape.len() as u32);
        for s in shape {
            put_u64(&mut buf, *s as u64);
        }
        put_u32(&mut buf, 1); // ggml_type F16
        put_u64(&mut buf, off2 as u64);
        let nbytes = shape.iter().product::<usize>() * 2;
        off2 = align32(off2 + nbytes);
    }
    // pad to 32-aligned data start
    let data_start = align32(buf.len());
    while buf.len() < data_start {
        buf.push(0);
    }
    // write tensor data
    for (_, _, shape, data) in &tensors {
        let bytes = f16_bytes(data);
        buf.extend_from_slice(&bytes);
        let nbytes = shape.iter().product::<usize>() * 2;
        let padded = align32(nbytes);
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        let _ = padded;
    }

    let mut f = File::create(path).expect("create gguh");
    f.write_all(&buf).expect("write gguh");
    println!(
        "wrote {} tensors, vocab={}, dim={}, layers={} -> {}",
        tensors.len(),
        vocab,
        dim,
        layers,
        path
    );
}
