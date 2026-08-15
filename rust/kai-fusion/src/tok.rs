//! BPE tokenizer (GPT-2 / llama-bpe style) built from GGUF metadata.
//! Uses byte-level encoding with merge rules from tokenizer.ggml.merges.

use crate::loader::GgufMeta;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: Vec<String>,
    id_of: HashMap<String, usize>,
    merge_ranks: HashMap<(String, String), usize>,
    cache: RefCell<HashMap<String, Vec<usize>>>,
    pub bos: usize,
    pub eos: usize,
    pub unk: usize,
}

impl Tokenizer {
    pub fn from_gguf(meta: &HashMap<String, crate::loader::GgufMeta>) -> Option<Tokenizer> {
        let tokens = meta.get("tokenizer.ggml.tokens")?.as_strarr()?.to_vec();
        let mut id_of = HashMap::new();
        for (i, s) in tokens.iter().enumerate() {
            id_of.insert(s.clone(), i);
        }

        let merge_ranks = if let Some(GgufMeta::StrArr(merges)) = meta.get("tokenizer.ggml.merges") {
            let mut ranks = HashMap::new();
            for (rank, merge_str) in merges.iter().enumerate() {
                if let Some((left, right)) = merge_str.split_once(' ') {
                    ranks.insert((left.to_string(), right.to_string()), rank);
                }
            }
            ranks
        } else {
            HashMap::new()
        };

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

        Some(Tokenizer {
            vocab: tokens,
            id_of,
            merge_ranks,
            cache: RefCell::new(HashMap::new()),
            bos, eos, unk,
        })
    }

    /// Simple pre-tokenizer: split text on whitespace, add Ġ prefix for word-initial tokens.
    fn pretokenize(text: &str) -> Vec<String> {
        let mut words: Vec<String> = Vec::new();
        let mut current = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            } else if ch.is_ascii_punctuation() || ch.is_ascii_digit() {
                // Punctuation and digits form their own tokens
                if !current.is_empty() && !current.chars().all(|c| c.is_ascii_punctuation() || c.is_ascii_digit()) {
                    words.push(std::mem::take(&mut current));
                }
                current.push(ch);
            } else {
                // Letters and other chars
                if !current.is_empty() && !current.chars().all(|c| c.is_alphabetic()) {
                    words.push(std::mem::take(&mut current));
                }
                current.push(ch);
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        
        // Add Ġ prefix for all words except the first one to indicate word boundaries
        if !words.is_empty() {
            let first = words.remove(0);
            words.insert(0, first);
            for i in 1..words.len() {
                words[i] = format!("\u{0120}{}", words[i]);
            }
        }
        
        words
    }

    /// Encode a single pre-token using BPE merge rules.
    fn bpe_encode(&self, word: &str) -> Vec<usize> {
        // Check cache
        if let Some(cached) = self.cache.borrow().get(word) {
            return cached.clone();
        }

        // Start with individual characters
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        
        if symbols.len() <= 1 {
            let result = self.symbols_to_ids(&symbols);
            if word.len() <= 20 {
                self.cache.borrow_mut().insert(word.to_string(), result.clone());
            }
            return result;
        }

        // Iteratively merge the highest-priority pair
        loop {
            let mut best_info: Option<(usize, usize, usize)> = None;
            
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    let better = best_info.map_or(true, |(_, _, best_rank)| rank < best_rank);
                    if better {
                        best_info = Some((i, i + 1, rank));
                    }
                }
            }
            
            match best_info {
                Some((left, right, _)) => {
                    let merged = format!("{}{}", symbols[left], symbols[right]);
                    symbols.splice(left..=right, [merged]);
                }
                None => break,
            }
        }

        let result = self.symbols_to_ids(&symbols);
        if word.len() <= 20 {
            self.cache.borrow_mut().insert(word.to_string(), result.clone());
        }
        result
    }

    fn symbols_to_ids(&self, symbols: &[String]) -> Vec<usize> {
        symbols.iter().map(|s| {
            self.id_of.get(s).copied().unwrap_or(self.unk)
        }).collect()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.bos];
        let words = Self::pretokenize(text);
        for word in &words {
            ids.extend(self.bpe_encode(word));
        }
        ids.push(self.eos);
        ids
    }

    /// Get merge rules sorted by rank.
    pub fn get_merges(&self) -> Vec<String> {
        let mut merges: Vec<(&(String, String), &usize)> = self.merge_ranks.iter().collect();
        merges.sort_by(|a, b| a.1.cmp(b.1));
        merges.iter().map(|(pair, _)| format!("{} {}", pair.0, pair.1)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut s = String::new();
        for &id in ids {
            if id == self.bos || id == self.eos {
                continue;
            }
            if let Some(t) = self.vocab.get(id) {
                // GPT-2 BPE uses Ġ (U+0120) for space prefix
                s.push_str(&t.replace('\u{0120}', " "));
            }
        }
        s
    }
}
