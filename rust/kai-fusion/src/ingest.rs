//! Knowledge Ingestion (Phase 5.3): loads text files into BodyEngram memory.
//! Chunks text into overlapping segments, embeds each with Nomic-BERT,
//! and stores (text, embedding) pairs for later retrieval.

use crate::bert;
use crate::body::BodyEngram;
use std::path::Path;

/// Default chunk size in tokens (words, since BERT tokenizer is word-level here).
const DEFAULT_CHUNK_TOKENS: usize = 128;

/// Default overlap between chunks in tokens.
const DEFAULT_OVERLAP: usize = 32;

/// Ingest a text file into the engram memory.
///
/// Reads the file, splits into overlapping chunks, embeds each with the
/// Nomic-BERT encoder loaded from `bert_gguf`, and stores into `engram`.
///
/// Returns the number of chunks ingested.
pub fn ingest_file(
    engram: &BodyEngram,
    bert_gguf: &str,
    file_path: &str,
) -> Result<usize, String> {
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| format!("read {file_path}: {e}"))?;

    let source_name = Path::new(file_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let chunks = chunk_text(&content, DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP);
    if chunks.is_empty() {
        return Ok(0);
    }

    // Load the BERT encoder once for all chunks
    let (encoder, _vocab) = bert::load_encoder(bert_gguf)
        .map_err(|e| format!("load encoder: {e}"))?;

    let meta = crate::loader::read_kv(bert_gguf)
        .map_err(|e| format!("meta: {e}"))?;
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

    let mut ingested = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let ids = bert::tokenize(chunk, &tokens_list, unk, cls);
        if ids.len() <= 2 {
            // Only CLS + maybe UNK, skip
            continue;
        }
        let emb = encoder.forward(&ids);
        let emb_vec: Vec<f32> = emb.to_vec();

        // Normalize for cosine similarity
        let norm: f32 = emb_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let normed: Vec<f32> = if norm > 0.0 {
            emb_vec.iter().map(|v| v / norm).collect()
        } else {
            emb_vec
        };

        let concept = format!("ingest::{source_name}::{}", i);
        engram.store_chunk(&concept, chunk, normed);
        ingested += 1;
    }

    Ok(ingested)
}

/// Ingest a directory recursively (all .txt and .md files).
pub fn ingest_directory(
    engram: &BodyEngram,
    bert_gguf: &str,
    dir_path: &str,
) -> Result<usize, String> {
    let mut total = 0;
    let dir = std::path::Path::new(dir_path);
    if !dir.is_dir() {
        return Err(format!("{dir_path} is not a directory"));
    }
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read_dir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            total += ingest_directory(engram, bert_gguf, path.to_str().unwrap_or(""))?;
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "txt" || ext == "md" || ext == "markdown" {
                let n = ingest_file(engram, bert_gguf, path.to_str().unwrap_or(""))?;
                println!("  ingested {} chunks from {}", n, path.display());
                total += n;
            }
        }
    }
    Ok(total)
}

/// Ingest a single Markdown string directly (for testing / inline use).
#[allow(dead_code)]
pub fn ingest_text(
    engram: &BodyEngram,
    bert_gguf: &str,
    source_name: &str,
    text: &str,
) -> Result<usize, String> {
    let chunks = chunk_text(text, DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP);
    if chunks.is_empty() {
        return Ok(0);
    }

    let (encoder, _vocab) = bert::load_encoder(bert_gguf)
        .map_err(|e| format!("load encoder: {e}"))?;

    let meta = crate::loader::read_kv(bert_gguf)
        .map_err(|e| format!("meta: {e}"))?;
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

    let mut ingested = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let ids = bert::tokenize(chunk, &tokens_list, unk, cls);
        if ids.len() <= 2 {
            continue;
        }
        let emb = encoder.forward(&ids);
        let emb_vec: Vec<f32> = emb.to_vec();
        let norm: f32 = emb_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let normed: Vec<f32> = if norm > 0.0 {
            emb_vec.iter().map(|v| v / norm).collect()
        } else {
            emb_vec
        };
        let concept = format!("ingest::{source_name}::{}", i);
        engram.store_chunk(&concept, chunk, normed);
        ingested += 1;
    }
    Ok(ingested)
}

/// Split text into overlapping chunks of roughly `chunk_tokens` words.
/// Chunks are split on paragraph boundaries when possible.
fn chunk_text(text: &str, chunk_tokens: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    // First split into paragraphs
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_words = 0;

    for para in paragraphs {
        let para_words: usize = para.split_whitespace().count();
        if para_words == 0 {
            continue;
        }

        if current_words + para_words <= chunk_tokens {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
            current_words += para_words;
        } else {
            // Flush current chunk
            if !current.trim().is_empty() {
                chunks.push(current.trim().to_string());
            }

            if para_words > chunk_tokens {
                // Large paragraph: split into word windows
                let words: Vec<&str> = para.split_whitespace().collect();
                let mut start = 0;
                while start < words.len() {
                    let end = (start + chunk_tokens).min(words.len());
                    let chunk: String = words[start..end].join(" ");
                    chunks.push(chunk);
                    if end >= words.len() {
                        break;
                    }
                    start += chunk_tokens - overlap;
                }
                current = String::new();
                current_words = 0;
            } else {
                // Start new chunk with overlap from end of previous
                if !current.is_empty() {
                    let words: Vec<&str> = current.split_whitespace().collect();
                    let overlap_start = if words.len() > overlap {
                        words.len() - overlap
                    } else {
                        0
                    };
                    current = words[overlap_start..].join(" ");
                    current_words = current.split_whitespace().count();
                }
                if !current.is_empty() {
                    current.push_str("\n\n");
                }
                current.push_str(para);
                current_words += para_words;
            }
        }
    }

    // Flush last chunk
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_simple() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_text(text, 4, 1);
        assert!(!chunks.is_empty());
        // All text should be covered
        let combined: String = chunks.join(" ");
        assert!(combined.contains("First"));
        assert!(combined.contains("Third"));
    }

    #[test]
    fn test_chunk_text_empty() {
        assert!(chunk_text("", 128, 32).is_empty());
        assert!(chunk_text("   ", 128, 32).is_empty());
    }

    #[test]
    fn test_chunk_text_single_para() {
        let text = "One paragraph only.";
        let chunks = chunk_text(text, 128, 32);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunk_text_large_para_splits() {
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = chunk_text(&text, 50, 10);
        assert!(chunks.len() > 1);
        // Every word should appear in at least one chunk
        for w in &words {
            let found = chunks.iter().any(|c| c.contains(w.as_str()));
            assert!(found, "{w} missing from all chunks");
        }
    }
}
