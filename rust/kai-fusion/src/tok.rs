//! Minimal tokenizer built from a GGUF vocab (tokenizer.ggml.tokens + ids).
//! Sufficient for the generated/dev models and a reasonable approximation for
//! real llama vocabs (whitespace split, lowercase + `▁` lookup, unk fallback).

use crate::gguf::GgufMeta;
use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: Vec<String>,
    id_of: HashMap<String, usize>,
    pub bos: usize,
    pub eos: usize,
    pub unk: usize,
}

impl Tokenizer {
    pub fn from_gguf(meta: &HashMap<String, GgufMeta>) -> Option<Tokenizer> {
        let tokens = meta.get("tokenizer.ggml.tokens")?.as_strarr()?.to_vec();
        let mut id_of = HashMap::new();
        for (i, s) in tokens.iter().enumerate() {
            id_of.insert(s.clone(), i);
        }
        let bos = meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|m| m.as_f64())
            .unwrap_or(1.0) as usize;
        let eos = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|m| m.as_f64())
            .unwrap_or(2.0) as usize;
        let unk = meta
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|m| m.as_f64())
            .unwrap_or(0.0) as usize;
        Some(Tokenizer { vocab: tokens, id_of, bos, eos, unk })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.bos];
        for w in text.split_whitespace() {
            let key = w.to_lowercase();
            if let Some(&id) = self.id_of.get(&key) {
                ids.push(id);
            } else if let Some(&id) = self.id_of.get(&format!("▁{key}")) {
                ids.push(id);
            } else {
                ids.push(self.unk);
            }
        }
        ids.push(self.eos);
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut s = String::new();
        for &id in ids {
            if id == self.bos || id == self.eos {
                continue;
            }
            if let Some(t) = self.vocab.get(id) {
                s.push_str(&t.replace('▁', " "));
            }
        }
        s
    }
}
