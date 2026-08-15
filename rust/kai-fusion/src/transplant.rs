//! Transplant weights from any llama-format GGUF into a Kai-sized model.
//! Uses GgufBuffer for streaming tensor reads (one tensor at a time)
//! to keep peak RAM low (~2GB + one dequantized tensor).
//!
//! Usage: kai transplant <source.gguf> <template.gguf> -o <output.gguf>
//!   --embed-map <f>  optional JSON mapping: kai_token_id -> source_token_id
//!
//! Without embed-map: copies source tokens 0..vocab_size directly (identity mapping).

use crate::loader::{GgufBuffer, read_kv, build_config};

/// Slice a 2D weight matrix along both dimensions, taking top-left submatrix.
fn slice_2d(data: &[f32], src_rows: usize, src_cols: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    let take_rows = rows.min(src_rows);
    let take_cols = cols.min(src_cols);
    for r in 0..take_rows {
        let src_row_start = r * src_cols;
        out.extend_from_slice(&data[src_row_start..src_row_start + take_cols]);
        for _ in take_cols..cols { out.push(0.0); }
    }
    for _ in take_rows..rows {
        for _ in 0..cols { out.push(0.0); }
    }
    out
}

fn slice_1d(data: &[f32], n: usize) -> Vec<f32> {
    data.iter().take(n).cloned().collect()
}

/// Transpose row-major f32 data from [rows, cols] to [cols, rows].
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Reorder embedding rows from source to target vocab using a mapping.
/// Source data must be in [VOCAB, DIM] layout (row-major, each row is one vocab entry).
/// mapping[kai_token_id] = source_token_id.
fn reorder_embed_rows(src_data: &[f32], src_stride: usize, src_vocab: usize,
                      target_dim: usize, target_vocab: usize,
                      mapping: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; target_vocab * target_dim];
    let take_dim = target_dim.min(src_stride);
    for target_id in 0..target_vocab.min(mapping.len()) {
        let src_id = mapping[target_id].min(src_vocab - 1);
        let src_off = src_id * src_stride;
        let dst_off = target_id * target_dim;
        for d in 0..take_dim {
            out[dst_off + d] = src_data[src_off + d];
        }
    }
    out
}

pub fn transplant(source_path: &str, template_path: &str, output_path: &str,
                  embed_map_path: Option<&str>, target_arch: &str) -> Result<(), String> {
    // Open source as GgufBuffer (reads raw bytes only, doesn't dequantize)
    let src = GgufBuffer::open(source_path)?;
    eprintln!("transplant: opened source ({} tensors)", src.tensors.len());

    // Load target config from template
    let meta = read_kv(template_path).map_err(|e| format!("reading template: {e}"))?;
    let mut cfg = build_config(&meta).ok_or("could not build config from template")?;

    // Build the tokenizer from the SOURCE model's metadata (real vocabulary).
    use std::collections::HashMap;
    let src_meta: &HashMap<String, crate::loader::GgufMeta> = &src.meta;

    // Get source vocab size and override template's vocab_size
    let src_vocab_size = crate::tok::Tokenizer::from_gguf(&src_meta)
        .map(|t| t.vocab.len())
        .unwrap_or(cfg.vocab_size);
    eprintln!("transplant: source vocab_size = {}", src_vocab_size);
    cfg.vocab_size = src_vocab_size;

    let dk = cfg.dim_kv();
    eprintln!("transplant: target config dim={} layers={} vocab={} inter={} dk={}",
              cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.intermediate, dk);

    // Configure target architecture
    match target_arch {
        "moe" => {
            cfg.moe.n_experts = 64; // DeepSeek-V2 default
            cfg.moe.n_shared = 2;
            cfg.mlp_kind = crate::config::MlpKind::MoE;
            eprintln!("transplant: target arch = MoE (experts={})", cfg.moe.n_experts);
        }
        "mla" => {
            cfg.mla.q_lora_rank = 1536;
            cfg.mla.kv_lora_rank = 512;
            cfg.mla.qk_rope_head_dim = 64;
            cfg.mla.v_head_dim = 128;
            cfg.attn_policy = crate::config::AttnPolicy::Global(crate::config::AttnKind::MLA);
            eprintln!("transplant: target arch = MLA");
        }
        "moe+mla" => {
            cfg.moe.n_experts = 64;
            cfg.moe.n_shared = 2;
            cfg.mlp_kind = crate::config::MlpKind::MoE;
            cfg.mla.q_lora_rank = 1536;
            cfg.mla.kv_lora_rank = 512;
            cfg.mla.qk_rope_head_dim = 64;
            cfg.mla.v_head_dim = 128;
            cfg.attn_policy = crate::config::AttnPolicy::Global(crate::config::AttnKind::MLA);
            eprintln!("transplant: target arch = MoE+MLA (experts={})", cfg.moe.n_experts);
        }
        _ => {
            eprintln!("transplant: target arch = dense MHA (default)");
        }
    }

    // Load embed mapping if provided
    let embed_map: Option<Vec<usize>> = embed_map_path.map(|p| {
        let content = std::fs::read_to_string(p).map_err(|e| format!("read {p}: {e}")).unwrap();
        serde_json::from_str(&content).map_err(|e| format!("parse {p}: {e}")).unwrap()
    });

    let use_embed = embed_map.is_some();

    let mut target_map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

    // Recompute dk after potential config changes
    let dk = cfg.dim_kv();

    /// Helper: dequant a 2D tensor from source, slice it, add to target_map.
    macro_rules! copy_2d {
        ($name:expr, $shape:expr) => {{
            let name = $name;
            let target_shape: Vec<usize> = $shape;
            eprintln!("  dequantizing {}...", name);
            let arr = src.dequant_arr2(name).map_err(|e| format!("dequant {name}: {e}"))?;
            let (src_rows, src_cols) = (arr.shape()[0], arr.shape()[1]);
            let sliced = slice_2d(arr.as_slice().unwrap(), src_rows, src_cols,
                                  target_shape[0], target_shape[1]);
            eprintln!("    {}: [{}x{}] -> [{}x{}] ({} elems)",
                      name, src_rows, src_cols, target_shape[0], target_shape[1], sliced.len());
            target_map.insert(name.to_string(), (target_shape, sliced));
        }};
    }

    /// Helper: dequant a 1D tensor from source, slice it, add to target_map.
    macro_rules! copy_1d {
        ($name:expr, $shape:expr) => {{
            let name = $name;
            let target_shape: Vec<usize> = $shape;
            eprintln!("  dequantizing {}...", name);
            let data = src.dequant(name).map_err(|e| format!("dequant {name}: {e}"))?;
            let sliced = slice_1d(&data, target_shape[0]);
            eprintln!("    {}: [{}] -> [{}] ({} elems)",
                      name, data.len(), target_shape[0], sliced.len());
            target_map.insert(name.to_string(), (target_shape, sliced));
        }};
    }

    // === Embedding ===
    // GGUF stores token_embd.weight as [dim, vocab].
    // For reordering we need [vocab, dim]; save back as [dim, vocab].
    if let Some(map) = &embed_map {
        eprintln!("  dequantizing token_embd.weight...");
        let arr = src.dequant_arr2("token_embd.weight")
            .map_err(|e| format!("dequant token_embd.weight: {e}"))?;
        let (src_d, src_v) = (arr.shape()[0], arr.shape()[1]);
        // Transpose [dim, vocab] -> [vocab, dim] for reordering
        let row_major = transpose(arr.as_slice().unwrap(), src_d, src_v);
        drop(arr); // free f32 data
        let reordered = reorder_embed_rows(&row_major, src_d, src_v,
                                            cfg.dim, cfg.vocab_size, map);
        // Transpose back to [dim, vocab] for GGUF storage
        let result = transpose(&reordered, cfg.vocab_size, cfg.dim);
        eprintln!("  token_embd.weight (mapped): -> [{}x{}] ({} elems)",
                  cfg.dim, cfg.vocab_size, result.len());
        target_map.insert("token_embd.weight".to_string(),
                          (vec![cfg.dim, cfg.vocab_size], result));
    } else {
        // Identity mapping: copy rows 0..vocab_size directly
        eprintln!("  dequantizing token_embd.weight (identity)...");
        let arr = src.dequant_arr2("token_embd.weight")
            .map_err(|e| format!("dequant token_embd.weight: {e}"))?;
        let (src_d, src_v) = (arr.shape()[0], arr.shape()[1]);
        // Transpose [dim, vocab] -> [vocab, dim]
        let row_major = transpose(arr.as_slice().unwrap(), src_d, src_v);
        drop(arr);
        // Take first vocab_size rows
        let sliced = slice_2d(&row_major, src_v, src_d, cfg.vocab_size, cfg.dim);
        // Transpose back to [dim, vocab]
        let result = transpose(&sliced, cfg.vocab_size, cfg.dim);
        eprintln!("  token_embd.weight (identity): [{}x{}] -> [{}x{}]",
                  src_d, src_v, cfg.dim, cfg.vocab_size);
        target_map.insert("token_embd.weight".to_string(),
                          (vec![cfg.dim, cfg.vocab_size], result));
    }

    // === Per-layer weights ===
    for i in 0..cfg.n_layers {
        let src_layer = i; // Use corresponding layer (assumes source has enough layers)
        let ln = |suffix: &str| format!("blk.{}.{}", src_layer, suffix);
        copy_2d!(&ln("attn_q.weight"), vec![cfg.dim, cfg.dim]);
        copy_2d!(&ln("attn_k.weight"), vec![dk, cfg.dim]);
        copy_2d!(&ln("attn_v.weight"), vec![dk, cfg.dim]);
        copy_2d!(&ln("attn_output.weight"), vec![cfg.dim, cfg.dim]);
        copy_2d!(&ln("ffn_gate.weight"), vec![cfg.intermediate, cfg.dim]);
        copy_2d!(&ln("ffn_up.weight"), vec![cfg.intermediate, cfg.dim]);
        copy_2d!(&ln("ffn_down.weight"), vec![cfg.dim, cfg.intermediate]);
        copy_1d!(&ln("attn_norm.weight"), vec![cfg.dim]);
        copy_1d!(&ln("ffn_norm.weight"), vec![cfg.dim]);

        // === MoE extensions (DeepSeek-V2 style) ===
        // Shared experts
        if src.tensor(&ln("ffn_shared_gate.weight")).is_some() {
            copy_2d!(&ln("ffn_shared_gate.weight"), vec![cfg.dim, cfg.dim]);
            copy_2d!(&ln("ffn_shared_down.weight"), vec![cfg.dim, cfg.dim]);
            copy_2d!(&ln("ffn_shared_up.weight"), vec![cfg.dim, cfg.dim]);
        }
        // Router
        if src.tensor(&ln("ffn_router.weight")).is_some() {
            copy_2d!(&ln("ffn_router.weight"), vec![cfg.moe.n_experts, cfg.dim]);
            copy_1d!(&ln("ffn_router.bias"), vec![cfg.moe.n_experts]);
        }
        // Experts
        if src.tensor(&ln("ffn_expert.0.gate.weight")).is_some() {
            let n_experts = cfg.moe.n_experts;
            for e in 0..n_experts {
                copy_2d!(&ln(&format!("ffn_expert.{}.gate.weight", e)), vec![cfg.intermediate, cfg.dim]);
                copy_2d!(&ln(&format!("ffn_expert.{}.down.weight", e)), vec![cfg.dim, cfg.intermediate]);
                copy_2d!(&ln(&format!("ffn_expert.{}.up.weight", e)), vec![cfg.intermediate, cfg.dim]);
            }
        }

        // === MLA extensions (DeepSeek-V2 style) ===
        if src.tensor(&ln("attn_q_a.weight")).is_some() {
            copy_2d!(&ln("attn_q_a.weight"), vec![cfg.mla.q_lora_rank, cfg.dim]);
            copy_2d!(&ln("attn_q_b.weight"), vec![cfg.dim, cfg.mla.q_lora_rank]);
            copy_2d!(&ln("attn_kv_a.weight"), vec![cfg.mla.kv_lora_rank, cfg.dim]);
            copy_2d!(&ln("attn_k_b.weight"), vec![cfg.dim, cfg.mla.kv_lora_rank]);
            copy_2d!(&ln("attn_v_b.weight"), vec![cfg.dim, cfg.mla.kv_lora_rank]);
            copy_2d!(&ln("attn_q_rope.weight"), vec![cfg.dim, cfg.dim]);
            copy_2d!(&ln("attn_output.weight"), vec![cfg.dim, cfg.dim]);
        }
    }

    // === Output norm ===
    if src.tensor("output_norm.weight").is_some() {
        copy_1d!("output_norm.weight", vec![cfg.dim]);
    } else {
        // Use last block's attn_norm
        let max_blk = (0..100usize).rev()
            .find(|i| src.tensor(&format!("blk.{i}.attn_norm.weight")).is_some())
            .unwrap_or(0);
        eprintln!("  output_norm.weight (using blk.{max_blk}.attn_norm.weight)");
        let data = src.dequant(&format!("blk.{max_blk}.attn_norm.weight"))
            .map_err(|e| format!("dequant attn_norm: {e}"))?;
        let sliced = slice_1d(&data, cfg.dim);
        target_map.insert("output_norm.weight".to_string(), (vec![cfg.dim], sliced));
    }

    // === Output weight ===
    if use_embed {
        if let Some(map) = &embed_map {
            if src.tensor("output.weight").is_some() {
                eprintln!("  dequantizing output.weight...");
                let arr = src.dequant_arr2("output.weight")
                    .map_err(|e| format!("dequant output.weight: {e}"))?;
                let (src_rows, src_cols) = (arr.shape()[0], arr.shape()[1]);
                let (src_v, src_d) = if src_rows >= src_cols {
                    (src_rows, src_cols) // [vocab, dim]
                } else {
                    (src_cols, src_rows) // [dim, vocab]
                };
                let data_vd = if src_rows >= src_cols {
                    arr.as_slice().unwrap().to_vec()
                } else {
                    transpose(arr.as_slice().unwrap(), src_rows, src_cols)
                };
                drop(arr);
                let reordered = reorder_embed_rows(&data_vd, src_d, src_v,
                                                    cfg.dim, cfg.vocab_size, map);
                eprintln!("  output.weight (mapped): -> [{}x{}] ({} elems)",
                          cfg.vocab_size, cfg.dim, reordered.len());
                target_map.insert("output.weight".to_string(),
                                  (vec![cfg.vocab_size, cfg.dim], reordered));
            } else {
                eprintln!("  (no output.weight, will tie to embedding)");
            }
        }
    } else {
        // Identity: copy rows 0..vocab_size
        if src.tensor("output.weight").is_some() {
            eprintln!("  dequantizing output.weight (identity)...");
            let arr = src.dequant_arr2("output.weight")
                .map_err(|e| format!("dequant output.weight: {e}"))?;
            let (src_rows, src_cols) = (arr.shape()[0], arr.shape()[1]);
            if src_rows >= src_cols {
                // [vocab, dim]
                let sliced = slice_2d(arr.as_slice().unwrap(), src_rows, src_cols,
                                      cfg.vocab_size, cfg.dim);
                target_map.insert("output.weight".to_string(),
                                  (vec![cfg.vocab_size, cfg.dim], sliced));
            } else {
                // [dim, vocab] -> transpose -> [vocab, dim]
                let interim = slice_2d(arr.as_slice().unwrap(), src_rows, src_cols,
                                       cfg.dim, cfg.vocab_size);
                let sliced = transpose(&interim, cfg.dim, cfg.vocab_size);
                target_map.insert("output.weight".to_string(),
                                  (vec![cfg.vocab_size, cfg.dim], sliced));
            }
            eprintln!("  output.weight (identity): -> [{}x{}] ({} elems)",
                      cfg.vocab_size, cfg.dim,
                      target_map.get("output.weight").map(|v| v.1.len()).unwrap_or(0));
        } else {
            // No output.weight in source: create from embedding (first vocab_size rows in [vocab,dim])
            eprintln!("  output.weight: creating from embedding (tied)");
            let (embed_shape, embed_data) = target_map.get("token_embd.weight")
                .ok_or("missing token_embd.weight for tied output")?;
            // embed_shape is [dim, vocab], data is row-major [dim, vocab]
            // We need [vocab, dim], so transpose
            let dim = embed_shape[0];
            let vocab = embed_shape[1];
            let take_dim = dim.min(cfg.dim);
            let take_vocab = vocab.min(cfg.vocab_size);
            // Transpose [dim, vocab] -> [vocab, dim]
            let mut out = vec![0.0f32; take_vocab * take_dim];
            for v in 0..take_vocab {
                for d in 0..take_dim {
                    out[v * take_dim + d] = embed_data[d * vocab + v];
                }
            }
            target_map.insert("output.weight".to_string(),
                              (vec![take_vocab, take_dim], out));
        }
    }

    // Build model from target map
    let model = crate::model::Weights::from_gguf(&target_map, &cfg)
        .map_err(|e| format!("building model from sliced weights: {e}"))?;

    // Build tokenizer from SOURCE metadata (has real vocabulary)
    let tok = crate::tok::Tokenizer::from_gguf(&src_meta)
        .ok_or("no tokenizer in source GGUF")?;

    let mut tensor_types = HashMap::new();
    if let Ok((_ver, info, _data_start)) = crate::loader::parse(template_path) {
        for t in &info {
            tensor_types.insert(t.name.clone(), t.ggml_type);
        }
    }

    crate::gguf_write::save_gguf(&model, &cfg, &tok, output_path, &tensor_types);
    eprintln!("transplant: saved -> {}", output_path);
    Ok(())
}
