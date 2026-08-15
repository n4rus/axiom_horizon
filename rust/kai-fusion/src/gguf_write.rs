//! GGUF serialization helpers — shared by cli.rs and merge.rs.

use crate::loader::GgufMeta;
use crate::model::Weights;
use crate::tok::Tokenizer;
use std::collections::HashMap;

pub fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
pub fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}
pub fn put_value(b: &mut Vec<u8>, v: &GgufMeta) {
    match v {
        GgufMeta::Num(f) => { put_u32(b, 12); b.extend_from_slice(&f.to_le_bytes()); }
        GgufMeta::Str(s) => { put_u32(b, 8); put_str(b, s); }
        GgufMeta::Arr(a) => {
            put_u32(b, 9); put_u32(b, 6); // f32 array (standard for tokenizer.ggml.scores)
            put_u64(b, a.len() as u64);
            for x in a { b.extend_from_slice(&(*x as f32).to_le_bytes()); }
        }
        GgufMeta::StrArr(a) => {
            put_u32(b, 9); put_u32(b, 8);
            put_u64(b, a.len() as u64);
            for s in a { put_str(b, s); }
        }
        GgufMeta::Bool(bv) => { put_u32(b, 7); b.push(if *bv { 1 } else { 0 }); }
    }
}
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7fffff;
    // NaN or Infinity
    if exp == 255 { return sign | 0x7c00u16 | ((mant >> 13) as u16); }
    // Zero / subnormal → flush to zero (preserve sign)
    if exp == 0 { return sign; }
    // Normal number
    let e = exp - 127 + 15;
    if e >= 31 { return sign | 0x7c00u16; } // overflow to inf
    if e <= 0 { return sign; } // underflow to zero
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}
pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f32_to_f16(x).to_le_bytes()).collect()
}
pub fn align32(n: usize) -> usize { (n + 31) & !31usize }

pub fn save_gguf(model: &Weights, cfg: &crate::config::Config, tok: &Tokenizer, output_path: &str, _tensor_types: &HashMap<String, u32>) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF
    put_u32(&mut buf, 3); // version
    let n_tensors = 1 + cfg.n_layers * 9 + 2; // embed + layers*9 + final_norm + output
    put_u64(&mut buf, n_tensors as u64);

    let mut meta: Vec<(String, crate::loader::GgufMeta)> = Vec::new();
    meta.push(("general.architecture".into(), crate::loader::GgufMeta::Str("llama".into())));
    meta.push(("general.name".into(), crate::loader::GgufMeta::Str("kai-trained".into())));
    meta.push(("llama.block_count".into(), crate::loader::GgufMeta::Num(cfg.n_layers as f64)));
    meta.push(("llama.embedding_length".into(), crate::loader::GgufMeta::Num(cfg.dim as f64)));
    meta.push(("llama.attention.head_count".into(), crate::loader::GgufMeta::Num(cfg.n_heads as f64)));
    meta.push(("llama.attention.head_count_kv".into(), crate::loader::GgufMeta::Num(cfg.n_kv_heads as f64)));
    meta.push(("llama.attention.layer_norm_rms_epsilon".into(), crate::loader::GgufMeta::Num(1e-5)));
    meta.push(("llama.feed_forward_length".into(), crate::loader::GgufMeta::Num(cfg.intermediate as f64)));
    meta.push(("llama.rope.freq_base".into(), crate::loader::GgufMeta::Num(cfg.rope_theta as f64)));
    meta.push(("llama.vocab_size".into(), crate::loader::GgufMeta::Num(cfg.vocab_size as f64)));
    meta.push(("llama.context_length".into(), crate::loader::GgufMeta::Num(cfg.max_seq as f64)));
    meta.push(("tokenizer.ggml.model".into(), crate::loader::GgufMeta::Str("llama".into())));
    meta.push(("tokenizer.ggml.tokens".into(), crate::loader::GgufMeta::StrArr(tok.vocab.clone())));
    meta.push(("tokenizer.ggml.scores".into(), crate::loader::GgufMeta::Arr(vec![0.0; cfg.vocab_size])));
    meta.push(("tokenizer.ggml.bos_token_id".into(), crate::loader::GgufMeta::Num(tok.bos as f64)));
    meta.push(("tokenizer.ggml.eos_token_id".into(), crate::loader::GgufMeta::Num(tok.eos as f64)));
    meta.push(("tokenizer.ggml.unknown_token_id".into(), crate::loader::GgufMeta::Num(tok.unk as f64)));
    let merges = tok.get_merges();
    if !merges.is_empty() {
        meta.push(("tokenizer.ggml.merges".into(), crate::loader::GgufMeta::StrArr(merges)));
    }
    meta.push(("tokenizer.ggml.pre".into(), crate::loader::GgufMeta::Str("default".into())));

    // Build tensor list
    let mut buf_meta = Vec::new();
    put_u64(&mut buf_meta, meta.len() as u64);
    for (k, v) in &meta {
        put_str(&mut buf_meta, k);
        put_value(&mut buf_meta, v);
    }
    buf.extend(buf_meta);

    // Tensor infos
    let mut off2 = 0usize;
    let mut tensor_infos = Vec::new();

    let embed_data: Vec<f32> = model.embed.iter().cloned().collect();
    let embed_shape = vec![cfg.dim, cfg.vocab_size];
    let nbytes = embed_shape.iter().product::<usize>() * 2;
    tensor_infos.push(("token_embd.weight".into(), embed_shape, off2, 1)); // F16 = 1
    off2 = align32(off2 + nbytes);

    for i in 0..cfg.n_layers {
        tensor_infos.push((format!("blk.{i}.attn_q.weight"), vec![cfg.dim, cfg.dim], off2, 1)); off2 = align32(off2 + cfg.dim * cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.attn_k.weight"), vec![cfg.dim_kv(), cfg.dim], off2, 1)); off2 = align32(off2 + cfg.dim_kv() * cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.attn_v.weight"), vec![cfg.dim_kv(), cfg.dim], off2, 1)); off2 = align32(off2 + cfg.dim_kv() * cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.attn_output.weight"), vec![cfg.dim, cfg.dim], off2, 1)); off2 = align32(off2 + cfg.dim * cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.ffn_gate.weight"), vec![cfg.intermediate, cfg.dim], off2, 1)); off2 = align32(off2 + cfg.intermediate * cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.ffn_down.weight"), vec![cfg.dim, cfg.intermediate], off2, 1)); off2 = align32(off2 + cfg.dim * cfg.intermediate * 2);
        tensor_infos.push((format!("blk.{i}.ffn_up.weight"), vec![cfg.intermediate, cfg.dim], off2, 1)); off2 = align32(off2 + cfg.intermediate * cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.attn_norm.weight"), vec![cfg.dim], off2, 1)); off2 = align32(off2 + cfg.dim * 2);
        tensor_infos.push((format!("blk.{i}.ffn_norm.weight"), vec![cfg.dim], off2, 1)); off2 = align32(off2 + cfg.dim * 2);
    }

    let fnorm_data: Vec<f32> = model.final_norm.iter().cloned().collect();
    let shape = vec![cfg.dim]; let n = shape.iter().product::<usize>() * 2;
    tensor_infos.push(("output_norm.weight".into(), shape, off2, 1)); off2 = align32(off2 + n);
    let output_data: Vec<f32> = model.output.iter().cloned().collect();
    let out_shape = vec![cfg.vocab_size, cfg.dim];
    let _out_nbytes = out_shape.iter().product::<usize>() * 2;
    tensor_infos.push(("output.weight".into(), out_shape, off2, 1));

    // GGUF v3 does NOT have a second tensor count before tensor infos.
    // The header already has the count; tensor infos follow immediately.
    for (name, shape, offset, ggml_type) in &tensor_infos {
        put_str(&mut buf, name);
        put_u32(&mut buf, shape.len() as u32);
        for s in shape { put_u64(&mut buf, *s as u64); }
        put_u32(&mut buf, *ggml_type);
        put_u64(&mut buf, *offset as u64);
    }

    // Pad to 32-byte alignment
    let data_start = align32(buf.len());
    while buf.len() < data_start { buf.push(0); }

    // Write tensor data (use the embedding_data collected above)
    {
        let data = f16_bytes(&embed_data);
        let len = data.len();
        buf.extend(data);
        buf.extend(vec![0u8; align32(len) - len]);
    }

    for i in 0..cfg.n_layers {
        let l = &model.layers[i];
        let flat = |a: &ndarray::Array2<f32>| a.as_standard_layout().as_slice().unwrap().to_vec();
        for arr in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w1, &l.w3, &l.w2] {
            let data = f16_bytes(&flat(arr));
            let len = data.len();
            buf.extend(data);
            buf.extend(vec![0u8; align32(len) - len]);
        }
        for arr1 in [&l.attn_norm, &l.ffn_norm] {
            let data = f16_bytes(arr1.as_slice().unwrap());
            let len = data.len();
            buf.extend(data);
            buf.extend(vec![0u8; align32(len) - len]);
        }
    }
    {
        let data = f16_bytes(&fnorm_data);
        let len = data.len();
        buf.extend(data);
        buf.extend(vec![0u8; align32(len) - len]);
    }
    {
        let data = f16_bytes(&output_data);
        let len = data.len();
        buf.extend(data);
        buf.extend(vec![0u8; align32(len) - len]);
    }

    std::fs::write(output_path, &buf).expect("write gguf");
}
