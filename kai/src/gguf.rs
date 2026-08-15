//! GGUF Reader — Minimal, fast, zero-dep parser for local model weights

use std::collections::HashMap;
use std::fs::File;
use memmap2::Mmap;
use ndarray::{Array1, Array2};
use crate::config::Config;

#[derive(Debug, Clone)]
pub enum GgufMeta {
    Num(f64),
    Str(String),
    Bool(bool),
    Arr(Vec<f64>),
    StrArr(Vec<String>),
    /// Raw bytes for large values (tokenizer data, etc.)
    Raw(Vec<u8>),
}

impl GgufMeta {
    pub fn as_f64(&self) -> Option<f64> { match self { Self::Num(v) => Some(*v), _ => None } }
    pub fn as_str(&self) -> Option<&str> { match self { Self::Str(v) => Some(v), _ => None } }
    #[allow(dead_code)]
    pub fn as_bool(&self) -> Option<bool> { match self { Self::Bool(v) => Some(*v), _ => None } }
}

fn rd_u32(d: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(d[*p..*p+4].try_into().unwrap());
    *p += 4; v
}
fn rd_u64(d: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(d[*p..*p+8].try_into().unwrap());
    *p += 8; v
}
fn rd_str(d: &[u8], p: &mut usize) -> String {
    let len = rd_u64(d, p) as usize;
    let s = String::from_utf8_lossy(&d[*p..*p+len]).into_owned();
    *p += len; s
}
fn rd_u8(d: &[u8], p: &mut usize) -> u8 {
    let v = d[*p];
    *p += 1; v
}
fn rd_u16(d: &[u8], p: &mut usize) -> u16 {
    let v = u16::from_le_bytes(d[*p..*p+2].try_into().unwrap());
    *p += 2; v
}
fn rd_i16(d: &[u8], p: &mut usize) -> i16 {
    let v = i16::from_le_bytes(d[*p..*p+2].try_into().unwrap());
    *p += 2; v
}
fn rd_i8(d: &[u8], p: &mut usize) -> i8 {
    let v = d[*p] as i8;
    *p += 1; v
}
fn rd_f32(d: &[u8], p: &mut usize) -> f32 {
    let v = f32::from_le_bytes(d[*p..*p+4].try_into().unwrap());
    *p += 4; v
}
fn rd_f64(d: &[u8], p: &mut usize) -> f64 {
    let v = f64::from_le_bytes(d[*p..*p+8].try_into().unwrap());
    *p += 8; v
}

fn peek_u32(d: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(d[p..p+4].try_into().unwrap())
}
fn peek_u64(d: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(d[p..p+8].try_into().unwrap())
}

/// Skip array elements in-place, updating p to point past the last element.
/// Returns true if all elements were successfully skipped.
fn skip_array_elements(d: &[u8], p: &mut usize, elem_type: u32, count: usize) -> bool {
    match elem_type {
        8 | 9 => {
            for _ in 0..count {
                if *p + 8 > d.len() { return false; }
                let slen = peek_u64(d, *p) as usize;
                if slen > 10_000_000 || *p + 8 + slen > d.len() { return false; }
                *p += 8 + slen;
            }
            true
        }
        _ => {
            let elem_size = match elem_type {
                0 | 1 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 7 | 11 | 12 => 8, _ => 4,
            };
            let total = count.checked_mul(elem_size).unwrap_or(0);
            if *p + total > d.len() { return false; }
            *p += total;
            true
        }
    }
}

fn rd_value(d: &[u8], p: &mut usize, t: u32) -> GgufMeta {
    match t {
        0 => GgufMeta::Num(rd_u8(d, p) as f64),
        1 => GgufMeta::Num(rd_i8(d, p) as f64),
        2 => GgufMeta::Num(rd_u16(d, p) as f64),
        3 => GgufMeta::Num(rd_i16(d, p) as f64),
        4 => GgufMeta::Num(rd_u32(d, p) as f64),
        5 => GgufMeta::Num(rd_u32(d, p) as i32 as f64),
        6 => GgufMeta::Num(rd_f32(d, p) as f64),
        7 => {
            // f64 value. Some GGUF files store a uint8/bool in the first byte
            // with the remaining 7 bytes overlapped into the next KV.
            let raw = &d[*p..*p+8];
            let fval = f64::from_le_bytes(raw.try_into().unwrap());
            if raw[0] <= 1 && raw[0] != 0 || fval != 0.0 {
                // Check if byte 1 looks like a valid key length for the next KV
                if raw[1] >= 1 && raw[1] < 100 && *p + 1 + 8 + raw[1] as usize <= d.len() {
                    // This is actually a uint8 value with the next KV embedded
                    let val = raw[0] as f64;
                    *p += 1; // only consume the actual value byte
                    GgufMeta::Num(val)
                } else {
                    *p += 8;
                    GgufMeta::Num(fval)
                }
            } else {
                *p += 8;
                GgufMeta::Num(fval)
            }
        }
        8 => {
            // GGUF type 8 = string (uint64 length + UTF-8 bytes)
            // Also used for bool (rare) — if length is 0 or 1, treat as bool.
            let len = rd_u64(d, p) as usize;
            if len <= 1 {
                GgufMeta::Bool(len != 0)
            } else if len < d.len().saturating_sub(*p) {
                let s = String::from_utf8_lossy(&d[*p..*p+len]).into_owned();
                *p += len;
                GgufMeta::Str(s)
            } else {
                // Plausible string — read it
                let max = d.len().saturating_sub(*p);
                let s = String::from_utf8_lossy(&d[*p..*p+max]).into_owned();
                *p += max;
                GgufMeta::Str(s)
            }
        }
        9 => {
            // GGUF type 9 = array: uint32 element_type + uint64 count + elements
            let elem_type = rd_u32(d, p);
            let count = rd_u64(d, p) as usize;
            if count == 0 {
                return GgufMeta::Arr(vec![]);
            }
            // For large arrays (>5000 elements), store raw bytes so tokenizer
            // extraction can parse them lazily without allocating millions of Strings.
            // Small arrays get stored as StrArr for convenience.
            let is_large = count > 5000;
            match elem_type {
                8 | 9 => {
                    if is_large {
                        // Store header + raw bytes for tokenizer to parse
                        let data_start = *p - 12; // include elem_type + count header
                        if skip_array_elements(d, p, elem_type, count) {
                            return GgufMeta::Raw(d[data_start..*p].to_vec());
                        }
                        // Fallback: skip and return raw from current pos
                        let remain = d.len().saturating_sub(*p);
                        let raw = d[*p..*p + remain.min(count * 32)].to_vec();
                        *p += raw.len();
                        return GgufMeta::Raw(raw);
                    }
                    // Small string array: store as StrArr
                    let mut strings = Vec::with_capacity(count);
                    for _ in 0..count {
                        if *p + 8 > d.len() { break; }
                        let slen = peek_u64(d, *p) as usize;
                        if slen > 1_000_000 || *p + 8 + slen > d.len() { break; }
                        strings.push(rd_str(d, p));
                    }
                    GgufMeta::StrArr(strings)
                }
                _ => {
                    // Numeric array — just skip elements
                    let elem_size = match elem_type {
                        0 | 1 => 1,  // u8, i8
                        2 | 3 => 2,  // u16, i16
                        4 | 5 | 6 => 4, // u32, i32, f32
                        7 | 11 | 12 => 8, // f64, u64, i64
                        _ => 4,
                    };
                    let total = count.checked_mul(elem_size).unwrap_or(0);
                    let consume = total.min(d.len().saturating_sub(*p));
                    *p += consume;
                    GgufMeta::Arr(vec![])
                }
            }
        }
        10 => {
            // GGUF type 10 (legacy/alternate array): uint32 elem_type + uint64 count
            let arr_type = rd_u32(d, p);
            let n = rd_u64(d, p) as usize;
            if arr_type == 9 || arr_type == 8 {
                if n > 5000 {
                    let data_start = *p - 12;
                    skip_array_elements(d, p, arr_type, n);
                    GgufMeta::Raw(d[data_start..*p].to_vec())
                } else {
                    let mut strings = Vec::with_capacity(n);
                    for _ in 0..n {
                        if *p + 8 > d.len() { break; }
                        let slen = peek_u64(d, *p) as usize;
                        if slen > 1_000_000 || *p + 8 + slen > d.len() { break; }
                        strings.push(rd_str(d, p));
                    }
                    GgufMeta::StrArr(strings)
                }
            } else {
                skip_array_elements(d, p, arr_type, n);
                GgufMeta::Arr(vec![])
            }
        }
        11 => GgufMeta::Num(rd_u64(d, p) as f64),
        12 => GgufMeta::Num(rd_u64(d, p) as i64 as f64),
        _ => { *p += 8; GgufMeta::Num(0.0) }
    }
}

pub fn read_kv(path: &str) -> Result<HashMap<String, GgufMeta>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    let mut p = 0;
    if &mmap[p..p+4] != b"GGUF" { return Err("Not a GGUF file".into()); }
    p += 4;
    let _ver = rd_u32(&mmap, &mut p);
    let _tensor_count = rd_u64(&mmap, &mut p);
    let kv_count = rd_u64(&mmap, &mut p) as usize;
    let mut meta = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = rd_str(&mmap, &mut p);
        let t = rd_u32(&mmap, &mut p);
        let val = rd_value(&mmap, &mut p, t);
        meta.insert(key, val);
    }
    Ok(meta)
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
    pub file_offset: u64,
}

pub fn read_tensors(path: &str) -> Result<(Mmap, HashMap<String, TensorInfo>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
    let mut p = 0;
    if &mmap[p..p+4] != b"GGUF" { return Err("Not a GGUF file".into()); }
    p += 4;
    let _ver = rd_u32(&mmap, &mut p);
    let tensor_count = rd_u64(&mmap, &mut p) as usize;
    let kv_count = rd_u64(&mmap, &mut p) as usize;
    
    // Skip KV metadata
    for _ in 0..kv_count {
        let _key = rd_str(&mmap, &mut p);
        let t = rd_u32(&mmap, &mut p);
        let _ = rd_value(&mmap, &mut p, t);
    }
    
    // Read tensor metadata
    let mut tensors_raw: Vec<(String, Vec<usize>, u32, u64)> = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = rd_str(&mmap, &mut p);
        let ndims = rd_u32(&mmap, &mut p) as usize;
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims { shape.push(rd_u64(&mmap, &mut p) as usize); }
        let ggml_type = rd_u32(&mmap, &mut p);
        let offset = rd_u64(&mmap, &mut p);
        tensors_raw.push((name, shape, ggml_type, offset));
    }
    
    // Compute tensor data start (32-byte aligned)
    let data_start = (p + 31) & !31;
    
    // Build final tensor info with absolute file offsets
    let mut tensors = HashMap::with_capacity(tensor_count);
    for (name, shape, ggml_type, offset) in tensors_raw {
        let file_offset = data_start as u64 + offset;
        tensors.insert(name, TensorInfo { shape, ggml_type, offset, file_offset });
    }
    
    Ok((mmap, tensors))
}

pub fn build_config(meta: &HashMap<String, GgufMeta>) -> Option<Config> {
    let arch = meta.get("general.architecture").and_then(|m| m.as_str()).unwrap_or("");
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

    let n_layers = get("n_layers").or_else(|| get("block_count"))? as usize;
    let dim = get("embedding_length").or_else(|| get("dim")).or_else(|| get("hidden_size"))? as usize;
    let n_heads = get("attention.head_count").or_else(|| get("n_heads"))? as usize;
    let n_kv_heads = get("attention.head_count_kv").map(|v| v as usize).unwrap_or(n_heads);
    let head_dim = dim / n_heads;
    let vocab_size = get("vocab_size").or_else(|| get("n_vocab")).unwrap_or(0.0) as usize;
    let intermediate = get("feed_forward_length").or_else(|| get("intermediate_size")).or_else(|| get("ffn_hidden_size")).unwrap_or((dim * 4) as f64) as usize;
    let rope_theta = get("rope.freq_base").or_else(|| get("rope_theta")).unwrap_or(10000.0) as f32;
    let rope_dim = get("rope.dimension_count")
        .or_else(|| get("rope_dim"))
        .map(|v| v as usize)
        .unwrap_or(head_dim);
    let max_seq = get("context_length").or_else(|| get("max_position_embeddings")).unwrap_or(4096.0) as usize;

    // Detect MLA/MoE
    let is_deepseek2 = arch == "deepseek2";
    let mla_enabled = is_deepseek2;
    let moe_enabled = is_deepseek2;

    Some(Config {
        dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        intermediate,
        rope_theta,
        rope_dim,
        max_seq,
        tau: 1.0,
        e: 1.0,
        age: 0,
        cycles: 0,
        h: 0.5,
        base_ms: 1000.0,
        phi: 0.0,
        attn_policy: if is_deepseek2 { crate::config::AttnPolicy::Global(crate::config::AttnKind::MLA) } else { crate::config::AttnPolicy::Global(crate::config::AttnKind::MHA) },
        mlp_kind: if is_deepseek2 { crate::config::MlpKind::MoE } else { crate::config::MlpKind::Dense },
        moe: crate::config::MoEConfig { enabled: moe_enabled, n_experts: 64, n_shared: 2, top_k: 6, capacity_factor: 1.25, router_bias: true, ..Default::default() },
        mla: crate::config::MLAConfig { enabled: mla_enabled, q_lora_rank: dim, kv_lora_rank: 512, qk_rope_head_dim: 64, v_head_dim: 128, ..Default::default() },
        vision: Default::default(),
        leading_dense_blocks: 0,
        expert_intermediate: 0,
    })
}

#[derive(Debug, Clone)]
pub struct Weights {
    pub embed: Array2<f32>,      // [vocab, dim]
    pub output: Array2<f32>,     // [vocab, dim] or tied to embed
    pub final_norm: Array1<f32>, // [dim]
    pub layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Array1<f32>,
    pub ffn_norm: Array1<f32>,
    pub wq: Array2<f32>,
    pub wk: Array2<f32>,
    pub wv: Array2<f32>,
    pub wo: Array2<f32>,
    pub w1: Array2<f32>, // gate
    pub w2: Array2<f32>, // down
    pub w3: Array2<f32>, // up
}

pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    pub merge_ranks: HashMap<(String, String), usize>,
    pub bos: usize,
    pub eos: usize,
    pub space_prefix: String, // "▁" for SentencePiece, "Ġ" for GPT-2
    pub add_bos: bool,        // whether to prepend BOS token
}

impl Tokenizer {
    pub fn from_gguf(meta: &HashMap<String, GgufMeta>) -> Option<Self> {
        let vocab = Self::extract_tokens(meta.get("tokenizer.ggml.tokens")?)?;
        let merges_str = Self::extract_merges(meta.get("tokenizer.ggml.merges")?)?;
        let bos = meta.get("tokenizer.ggml.bos_token_id")?.as_f64()? as usize;
        let eos = meta.get("tokenizer.ggml.eos_token_id")?.as_f64()? as usize;
        
        // Detect tokenizer type from metadata
        let model = meta.get("tokenizer.ggml.model")
            .and_then(|m| m.as_str())
            .unwrap_or("llama");
        let space_prefix = match model {
            "gpt2" => "\u{0120}".to_string(), // Ġ for GPT-2 / Qwen
            _      => "\u{2581}".to_string(), // ▁ for SentencePiece
        };
        // Check add_bos_token (default true for SentencePiece, false for GPT-2)
        let add_bos = meta.get("tokenizer.ggml.add_bos_token")
            .and_then(|m| match m {
                GgufMeta::Bool(b) => Some(*b),
                GgufMeta::Num(v) => Some(*v != 0.0),
                _ => None,
            })
            .unwrap_or(model != "gpt2");
        
        let vocab_map: HashMap<_, _> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let merge_ranks: HashMap<_, _> = merges_str.iter().filter_map(|s| {
            let parts: Vec<_> = s.split(' ').collect();
            if parts.len() == 2 { Some((parts[0].to_string(), parts[1].to_string())) } else { None }
        }).enumerate().map(|(i, (a, b))| ((a, b), i)).collect();
        
        Some(Self { vocab: vocab_map, merge_ranks, bos, eos, space_prefix, add_bos })
    }

    fn extract_tokens(meta: &GgufMeta) -> Option<Vec<String>> {
        match meta {
            GgufMeta::StrArr(v) => Some(v.clone()),
            GgufMeta::Raw(data) => {
                // Parse embedded array format
                if data.len() < 12 { return None; }
                let elem_type = u32::from_le_bytes(data[0..4].try_into().ok()?);
                if elem_type > 12 { return None; }
                let count = u64::from_le_bytes(data[4..12].try_into().ok()?) as usize;
                let mut p = 12;
                let mut tokens = Vec::with_capacity(count.min(100000));
                for _ in 0..count {
                    if p + 8 > data.len() { break; }
                    let len = u64::from_le_bytes(data[p..p+8].try_into().ok()?) as usize;
                    if len > 10000 || p + 8 + len > data.len() { break; }
                    p += 8;
                    let s = String::from_utf8_lossy(&data[p..p+len]).into_owned();
                    p += len;
                    tokens.push(s);
                }
                if tokens.len() >= count.saturating_sub(10) { Some(tokens) } else { None }
            }
            _ => None,
        }
    }

    fn extract_merges(meta: &GgufMeta) -> Option<Vec<String>> {
        match meta {
            GgufMeta::StrArr(v) => Some(v.clone()),
            GgufMeta::Raw(data) => {
                if data.len() < 12 { return None; }
                let elem_type = u32::from_le_bytes(data[0..4].try_into().ok()?);
                if elem_type > 12 { return None; }
                let count = u64::from_le_bytes(data[4..12].try_into().ok()?) as usize;
                let mut p = 12;
                let mut items = Vec::with_capacity(count.min(100000));
                for _ in 0..count {
                    if p + 8 > data.len() { break; }
                    let len = u64::from_le_bytes(data[p..p+8].try_into().ok()?) as usize;
                    if len > 10000 || p + 8 + len > data.len() { break; }
                    p += 8;
                    let s = String::from_utf8_lossy(&data[p..p+len]).into_owned();
                    p += len;
                    items.push(s);
                }
                if items.len() >= count.saturating_sub(10) { Some(items) } else { None }
            }
            _ => None,
        }
    }

    pub fn encode_debug(&self, text: &str) -> Vec<(usize, String)> {
        let mut result = Vec::new();
        let ids = self.encode(text);
        for &id in &ids {
            let s = self.vocab.iter().find(|(_, &v)| v == id).map(|(k,_)| k.clone()).unwrap_or_else(|| format!("<missing {}>", id));
            result.push((id, s));
        }
        result
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let trim = text.trim();
        let mut ids = if self.add_bos { vec![self.bos] } else { Vec::new() };
        // Pre-tokenize: split on whitespace, but preserve special tokens
        let words: Vec<&str> = trim.split_whitespace().collect();
        for (wi, word) in words.iter().enumerate() {
            // Direct vocabulary lookup first (handles special tokens like <|im_start|>)
            if let Some(&id) = self.vocab.get(*word) {
                ids.push(id);
                continue;
            }
            // Prefix every word except the first with space_prefix (▁ for SentencePiece, Ġ for GPT-2)
            let w = if wi == 0 { word.to_string() } else { format!("{}{}", self.space_prefix, word) };
            // Also check the prefixed word in vocab
            if let Some(&id) = self.vocab.get(&w) {
                ids.push(id);
                continue;
            }
            let token_ids = self.bpe_encode(&w);
            if token_ids.is_empty() {
                // Fallback: use byte tokens
                for b in word.bytes() {
                    let byte_tok = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.vocab.get(&byte_tok) {
                        ids.push(id);
                    }
                }
            } else {
                ids.extend(token_ids);
            }
        }
        ids
    }

    fn bpe_encode(&self, word: &str) -> Vec<usize> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if parts.len() <= 1 {
            return parts.iter().filter_map(|p| self.vocab.get(p)).copied().collect();
        }
        loop {
            let mut best: Option<(usize, usize, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let pair = (parts[i].clone(), parts[i+1].clone());
                if let Some(rank) = self.merge_ranks.get(&pair) {
                    if best.is_none() || *rank < best.unwrap().2 { best = Some((i, i+1, *rank)); }
                }
            }
            match best {
                Some((i, j, _)) => {
                    let merged = parts[i].clone() + &parts[j];
                    parts.splice(i..=j, [merged]);
                }
                None => break,
            }
        }
        parts.iter().filter_map(|p| self.vocab.get(p)).copied().collect()
    }
}

#[allow(dead_code)]
impl GgufMeta {
    fn as_str_arr(&self) -> Option<Vec<String>> {
        match self { Self::StrArr(v) => Some(v.clone()), _ => None }
    }
}