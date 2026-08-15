//! Attractor Memory — Persistent fixed-point convergence
//!
//! Stores (concept, tokens, embedding) triples. The embedding is normalized
//! so cosine similarity is a simple dot product.  Convergence iterates push
//! each stored vector toward its nearest neighbours, forming fixed-point
//! attractors — the core memory substrate for VFE prior biasing.

use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

/// A single attractor entry: human-readable text + token IDs + embedding.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub concept: String,
    pub tokens: Vec<usize>,
    pub embedding: Array1<f32>,
}

pub struct Attractor {
    path: String,
    entries: HashMap<String, MemoryEntry>,
    alpha: f32,
    dim: usize,
}

impl Attractor {
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        std::fs::create_dir_all(path)?;
        let path = format!("{}/attractor.bin", path);
        let mut a = Self {
            path,
            entries: HashMap::new(),
            alpha: 0.1,
            dim: 768,
        };
        a.load()?;
        Ok(a)
    }

    /// Store a new memory.  `embedding` is normalised internally.
    /// Dimensions are sliced to `self.dim` for consistency.
    pub fn push(
        &mut self,
        name: &str,
        concept: &str,
        tokens: &[usize],
        embedding: &ArrayView1<f32>,
    ) -> Result<(), std::io::Error> {
        let dim = self.dim.min(embedding.len());
        if dim == 0 {
            return Ok(()); // skip empty embeddings
        }
        let mut vec = embedding.slice(ndarray::s![..dim]).to_owned();
        normalize_mut(&mut vec);

        // Convergence blend with existing entry of the same name
        if let Some(old) = self.entries.get(name) {
            let blend_dim = dim.min(old.embedding.len());
            if blend_dim > 0 {
                let old_slice = old.embedding.slice(ndarray::s![..blend_dim]);
                let new_slice = vec.slice(ndarray::s![..blend_dim]);
                let blended = &old_slice * (1.0 - self.alpha) + &new_slice * self.alpha;
                let mut blended = blended.to_owned();
                normalize_mut(&mut blended);
                vec = blended;
            }
        }

        self.entries.insert(
            name.to_string(),
            MemoryEntry {
                concept: concept.to_string(),
                tokens: tokens.to_vec(),
                embedding: vec,
            },
        );
        self.save(name)?;
        Ok(())
    }

    /// Query by embedding; returns up to `k` entries sorted by similarity.
    /// Defensively handles dimension mismatches by slicing to the minimum.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<(&MemoryEntry, f32)> {
        let dim = self.dim.min(query.len());
        if dim == 0 {
            return vec![];
        }
        let q = Array1::from_vec(query[..dim].to_vec());
        let q_norm = q.dot(&q).sqrt().max(1e-8);
        let q = q / q_norm;

        let mut results: Vec<(&MemoryEntry, f32)> = self
            .entries
            .values()
            .map(|entry| {
                let use_dim = q.len().min(entry.embedding.len());
                if use_dim == 0 {
                    return (entry, 0.0);
                }
                let qs = q.slice(ndarray::s![..use_dim]);
                let es = entry.embedding.slice(ndarray::s![..use_dim]);
                let sim = qs.dot(&es);
                (entry, sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }

    /// Build a token-level prior distribution from attractor memories.
    ///
    /// For each retrieved memory, scatter probability mass to its stored
    /// tokens proportional to `similarity^2`.  Smooth with `uniform_weight`
    /// to keep the distribution non-degenerate.
    pub fn build_token_prior(
        &self,
        query_emb: &[f32],
        vocab_size: usize,
        k: usize,
        uniform_weight: f32,
    ) -> Vec<f32> {
        let memories = self.query(query_emb, k);
        if memories.is_empty() {
            // Fall back to uniform
            let unif = 1.0 / vocab_size as f32;
            return vec![unif; vocab_size];
        }

        let mut prior = vec![0.0f32; vocab_size];
        let mut total = 0.0f32;

        for (entry, sim) in &memories {
            if *sim <= 0.0 {
                continue;
            }
            let weight = sim * sim; // squared similarity — sharper peak
            for &tid in &entry.tokens {
                if tid < vocab_size {
                    prior[tid] += weight;
                    total += weight;
                }
            }
        }

        // Normalise and blend with uniform
        let uniform = 1.0 / vocab_size as f32;
        if total > 0.0 {
            for p in prior.iter_mut() {
                *p = *p / total * (1.0 - uniform_weight) + uniform * uniform_weight;
            }
        } else {
            prior.iter_mut().for_each(|p| *p = uniform);
        }

        prior
    }

    /// Run fixed-point convergence iterations on all stored vectors.
    /// Uses safe dimension slicing so vectors of varying lengths still converge.
    pub fn converge(&mut self, iterations: usize) -> Result<(f32, usize), String> {
        let names: Vec<String> = self.entries.keys().cloned().collect();
        if names.is_empty() {
            return Ok((1.0, 0));
        }

        for it in 0..iterations {
            let mut max_shift = 0.0f32;
            for name in &names {
                let entry = self.entries.get(name).cloned();
                if let Some(entry) = entry {
                    // Find top-3 neighbours (excluding self)
                    let mut sims: Vec<(String, f32)> = self
                        .entries
                        .iter()
                        .filter(|(n, _)| *n != name)
                        .map(|(n, e)| {
                            let d = entry.embedding.len().min(e.embedding.len());
                            if d == 0 { return (n.clone(), 0.0); }
                            let a = entry.embedding.slice(ndarray::s![..d]);
                            let b = e.embedding.slice(ndarray::s![..d]);
                            (n.clone(), a.dot(&b))
                        })
                        .collect();
                    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    if sims.is_empty() {
                        continue;
                    }

                    // Average top-3 (slice to common dimension)
                    let min_dim = sims.iter().take(3).fold(self.dim, |d, (n, _)| {
                        d.min(self.entries.get(n).map(|e| e.embedding.len()).unwrap_or(d))
                    }).min(entry.embedding.len());
                    if min_dim == 0 { continue; }
                    let neighbors: Vec<Array1<f32>> = sims
                        .iter()
                        .take(3)
                        .map(|(n, _)| {
                            let e = &self.entries[n].embedding;
                            e.slice(ndarray::s![..min_dim]).to_owned()
                        })
                        .collect();
                    if neighbors.is_empty() {
                        continue;
                    }

                    let zero: Array1<f32> = Array1::zeros(min_dim);
                    let mean = neighbors.iter().fold(zero, |a, b| a + b) / neighbors.len() as f32;
                    let emb_slice = entry.embedding.slice(ndarray::s![..min_dim]);
                    let new_emb = &emb_slice * (1.0 - self.alpha) + &mean * self.alpha;
                    let shift = (&emb_slice - &new_emb).mapv(|x| x * x).sum().sqrt();
                    max_shift = max_shift.max(shift);

                    if let Some(e) = self.entries.get_mut(name) {
                        e.embedding = new_emb;
                    }
                }
            }
            if max_shift < 1e-4 {
                return Ok((max_shift, it + 1));
            }
        }
        Ok((0.0, iterations))
    }

    // ── persistence ──────────────────────────────────────────────────

    fn save(&mut self, name: &str) -> Result<(), std::io::Error> {
        let file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        let mut file = file;
        if let Some(entry) = self.entries.get(name) {
            let tokens_str = entry
                .tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let vec_str = entry
                .embedding
                .iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                file,
                "{}:{}:{}:{}",
                name, entry.concept, tokens_str, vec_str
            )?;
        }
        Ok(())
    }

    fn load(&mut self) -> Result<(), std::io::Error> {
        if let Ok(file) = File::open(&self.path) {
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                // format: name:concept:tid1,tid2,...:vec0,vec1,...
                let parts: Vec<&str> = line.splitn(4, ':').collect();
                if parts.len() >= 4 {
                    let name = parts[0].to_string();
                    let concept = parts[1].to_string();
                    let tokens: Vec<usize> =
                        parts[2].split(',').filter_map(|s| s.parse().ok()).collect();
                    let vec: Array1<f32> =
                        parts[3].split(',').filter_map(|s| s.parse().ok()).collect();
                    if !vec.is_empty() && !tokens.is_empty() {
                        let mut v = vec;
                        normalize_mut(&mut v);
                        self.entries.insert(
                            name,
                            MemoryEntry {
                                concept,
                                tokens,
                                embedding: v,
                            },
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Borrow the internal dim so callers can query it.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Set the embedding dimension (call after loading config).
    pub fn set_dim(&mut self, d: usize) {
        self.dim = d;
    }
}

fn normalize_mut(v: &mut Array1<f32>) {
    let norm = v.dot(v).sqrt().max(1e-8);
    if norm > 1e-8 {
        v.mapv_inplace(|x| x / norm);
    }
}
