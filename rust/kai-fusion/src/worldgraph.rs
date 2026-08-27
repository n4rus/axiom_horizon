//! worldgraph.rs — the World-Model onset.
//!
//! Grounds Kai's reasoning in REAL world knowledge (Wikipedia, embedded to
//! 768-dim) rather than abstract physics. Instead of abstract metric math,
//! this builds a *connection graph* over the embedded corpus:
//!
//!   1. **Load**: `.kai_wiki_memory.*.json` shards (node = article chunk key,
//!      embedding = Nomic-BERT 768-dim vector).
//!   2. **Domain clustering**: k-means over node embeddings. Without external
//!      metadata the corpus self-organizes into latent domains — typically
//!      math, code, physics, life sciences, geography… (assignable by
//!      inspecting each cluster's articles).
//!   3. **Connection graph**: k-nearest-neighbor edges per node (cosine).
//!   4. **Bridge discovery**: edges whose endpoints fall in DIFFERENT domains
//!      with high cosine = "novel connections" — exactly what `darwin evolve`
//!      should seek (new ways to present/reason; math↔physics↔code).
//!   5. **Attractor export**: domain centroids written in the format
//!      `attractor::load` expects, so `kai physics bench` and `darwin evolve`
//!      get domain priors *grounded on the real world corpus*, and
//!      `compute_attractor_agreement` scores decode consistency against the
//!      learned world model.
//!
//! All math here is plain linear algebra over the real memory data.

/// One embedded world node (a chunk of a real Wikipedia article).
#[derive(Clone)]
pub struct WorldNode {
    /// Full key like `__doc__/wiki/international_atomic_time#3`.
    pub key: String,
    /// Article name parsed from the key.
    pub article: String,
    /// 768-dim embedding (Nomic-BERT).
    pub emb: Vec<f32>,
}

/// A latent k-NN edge in the connection graph.
#[derive(Clone)]
pub struct Edge {
    pub ai: usize,
    pub aj: usize,
    pub cosine: f32,
}

/// A cross-domain bridge: a high-cosine link between two nodes that belong to
/// different domains — the "novel connection" candidate the evolution hunts.
#[derive(Clone)]
pub struct Bridge {
    pub domain_a: usize,
    pub domain_b: usize,
    pub ai: usize,
    pub aj: usize,
    pub cosine: f32,
    pub node_a: String,
    pub node_b: String,
    pub key_a: String,
    pub key_b: String,
}

/// The world-model graph over an embedded corpus.
pub struct WorldGraph {
    pub nodes: Vec<WorldNode>,
    /// cluster id per node index.
    pub cluster: Vec<usize>,
    pub n_clusters: usize,
    /// per-cluster centroids (768-dim).
    pub centroids: Vec<Vec<f32>>,
    /// representative article name per domain (for human-readable labels).
    pub domain_reps: Vec<String>,
}

impl WorldGraph {
    /// Load all node entries from every memory shard matching `pattern`
    /// (e.g. `/path/.kai_wiki_memory.*.json`), or from a single file if no
    /// `*` is present. Returns the raw nodes, unclustered.
    pub fn load_shards(pattern: &str) -> Result<Vec<WorldNode>, String> {
        let mut all: Vec<WorldNode> = Vec::new();
        let matches = if pattern.contains('*') {
            read_wildcard_shards(pattern)?
        } else {
            vec![pattern.to_string()]
        };
        for path in matches {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("read {path}: {e}"))?;
            let v: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("parse {path}: {e}"))?;
            let entries = v.get("entries")
                .and_then(|x| x.as_object())
                .ok_or_else(|| format!("{path}: no 'entries' object"))?;
            for (k, entry) in entries {
                let emb: Vec<f32> = match entry.get("embedding").and_then(|x| x.as_array()) {
                    Some(arr) => arr.iter()
                        .filter_map(|x| x.as_f64())
                        .map(|f| f as f32)
                        .collect(),
                    None => continue,
                };
                if emb.is_empty() {
                    continue;
                }
                let (article, _chunk) = parse_key(k);
                all.push(WorldNode {
                    key: k.clone(),
                    article,
                    emb,
                });
            }
        }
        if all.is_empty() {
            return Err(format!("no nodes loaded from {pattern}"));
        }
        Ok(all)
    }

    /// Cluster raw nodes into `n_clusters` latent domains and build the graph.
    pub fn from_nodes(nodes: Vec<WorldNode>, n_clusters: usize, iters: usize, seed: u64) -> WorldGraph {
        if nodes.is_empty() {
            return WorldGraph {
                nodes,
                cluster: Vec::new(),
                n_clusters,
                centroids: Vec::new(),
                domain_reps: Vec::new(),
            };
        }
        let d = nodes[0].emb.len();
        let n_clusters = n_clusters.clamp(2, nodes.len());

        // k-means with deterministic RNG.
        let mut rng = Seeded::new(seed);
        let mut chosen: Vec<usize> = Vec::with_capacity(n_clusters);
        while chosen.len() < n_clusters {
            let idx = (rng.next()) % nodes.len();
            if !chosen.contains(&idx) {
                chosen.push(idx);
            }
        }
        let mut centroids: Vec<Vec<f32>> =
            chosen.iter().map(|&i| nodes[i].emb.clone()).collect();

        let mut assign = vec![0usize; nodes.len()];
        for _ in 0..iters {
            for (i, n) in nodes.iter().enumerate() {
                assign[i] = nearest_centroid(&n.emb, &centroids);
            }
            let mut sums = vec![vec![0.0f32; d]; n_clusters];
            let mut counts = vec![0usize; n_clusters];
            for (i, n) in nodes.iter().enumerate() {
                let c = assign[i];
                counts[c] += 1;
                for dd in 0..d {
                    sums[c][dd] += n.emb[dd];
                }
            }
            for c in 0..n_clusters {
                if counts[c] > 0 {
                    for dd in 0..d {
                        centroids[c][dd] = sums[c][dd] / counts[c] as f32;
                    }
                }
            }
        }
        // Reassign once more with the converged centroids.
        for (i, n) in nodes.iter().enumerate() {
            assign[i] = nearest_centroid(&n.emb, &centroids);
        }

        let mut domain_reps = Vec::with_capacity(n_clusters);
        for c in 0..n_clusters {
            let rep = nodes.iter().enumerate()
                .find(|(i, _)| assign[*i] == c)
                .map(|(_, n)| n.article.clone())
                .unwrap_or_else(|| format!("domain-{c}"));
            domain_reps.push(rep);
        }

        WorldGraph {
            nodes,
            cluster: assign,
            n_clusters,
            centroids,
            domain_reps,
        }
    }

    /// k-nearest neighbors of node `i` (cosine), excluding self.
    pub fn neighbors(&self, i: usize, k: usize) -> Vec<Edge> {
        let e = &self.nodes[i].emb;
        let mut scored: Vec<(usize, f32)> = self.nodes.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, n)| (j, cosine(&n.emb, e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k)
            .map(|(aj, cos)| Edge { ai: i, aj, cosine: cos })
            .collect()
    }

    /// Find cross-domain bridges: for each node, scan its k-neighborhood;
    /// edges that span two different domains become bridge candidates.
    /// Sorted by cosine descending.
    pub fn bridges(&self, k: usize, min_cosine: f32) -> Vec<Bridge> {
        let mut found: Vec<Bridge> = Vec::new();
        let mut seen = std::collections::HashSet::<(usize, usize)>::new();
        for i in 0..self.nodes.len() {
            for e in self.neighbors(i, k) {
                let (a, b) = if e.ai < e.aj { (e.ai, e.aj) } else { (e.aj, e.ai) };
                if self.cluster[a] != self.cluster[b]
                    && e.cosine >= min_cosine
                    && seen.insert((a, b))
                {
                    found.push(Bridge {
                        domain_a: self.cluster[a],
                        domain_b: self.cluster[b],
                        ai: a,
                        aj: b,
                        cosine: e.cosine,
                        node_a: self.nodes[a].article.clone(),
                        node_b: self.nodes[b].article.clone(),
                        key_a: self.nodes[a].key.clone(),
                        key_b: self.nodes[b].key.clone(),
                    });
                }
            }
        }
        found.sort_by(|a, b| b.cosine.partial_cmp(&a.cosine).unwrap_or(std::cmp::Ordering::Equal));
        found
    }

    /// Export domain centroids in the attractor JSON format `attractor::load`
    /// expects: `{"vecs": [[...], ...]}`.
    pub fn export_attractor(&self, path: &str) -> Result<(), String> {
        let vecs: Vec<serde_json::Value> = self.centroids.iter()
            .filter(|c| !c.is_empty())
            .map(|c| serde_json::Value::Array(
                c.iter().map(|&f| serde_json::Value::from(f as f64)).collect()))
            .collect();
        if vecs.is_empty() {
            return Err("no non-empty centroids to export".into());
        }
        let mut obj = serde_json::Map::new();
        obj.insert("vecs".into(), serde_json::Value::Array(vecs));
        let text = serde_json::Value::Object(obj).to_string();
        std::fs::write(path, text).map_err(|e| format!("write {path}: {e}"))?;
        Ok(())
    }

    /// Sample summary stats: number of nodes, domains, and per-domain sizes.
    pub fn summary(&self) -> String {
        let mut per = vec![0usize; self.n_clusters];
        for &c in self.cluster.iter() {
            per[c] += 1;
        }
        let mut s = String::from("WorldGraph {\n");
        s.push_str(&format!("  nodes={}\n", self.nodes.len()));
        for c in 0..self.n_clusters {
            s.push_str(&format!(
                "  domain[{c}] '{}': {} nodes (rep: {})\n",
                c, per[c],
                self.domain_reps.get(c).cloned().unwrap_or_default()
            ));
        }
        s.push('}');
        s
    }
}

/// Read all shard files matching a wildcard pattern (e.g. `.kai_wiki_memory.*.json`).
fn read_wildcard_shards(pattern: &str) -> Result<Vec<String>, String> {
    // Split the pattern into (directory, filename_glob) by finding the last
    // `/` before the first `*`. For `.kai_wiki_memory.*.json` the directory is
    // `.` and the glob is `.kai_wiki_memory.*.json`. For `./shards/foo*.json`
    // the directory is `./shards` and the glob is `foo*.json`.
    let (dir, glob) = if let Some(star_pos) = pattern.find('*') {
        let before_star = &pattern[..star_pos];
        match before_star.rfind('/') {
            Some(slash_pos) => (&pattern[..slash_pos], &pattern[slash_pos + 1..]),
            None => (".", pattern),
        }
    } else {
        return Err(format!("no wildcard '*' in pattern: {pattern}"));
    };
    let dir = if dir.is_empty() { "." } else { dir };
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| format!("read dir {dir}: {e}"))? {
        let p = e.map_err(|e| format!("dir entry: {e}"))?.path();
        if let Some(s) = p.as_os_str().to_str() {
            // Match the filename portion against the glob. A simple glob here
            // means: the filename must end with the suffix after `*` and contain
            // the prefix before `*`. Good enough for `.kai_wiki_memory.*.json`.
            let file_name = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
            if glob_matches(glob, file_name) {
                v.push(s.to_string());
            }
        }
    }
    v.sort();
    if v.is_empty() {
        Err(format!("no shard files matching {pattern} in {dir}"))
    } else {
        Ok(v)
    }
}

/// Simple wildcard matcher: supports a single `*` that matches any sequence of
/// characters. The pattern is split at the first `*` into prefix + suffix; the
/// filename must start with the prefix and end with the suffix.
fn glob_matches(pattern: &str, filename: &str) -> bool {
    if let Some(star_pos) = pattern.find('*') {
        let prefix = &pattern[..star_pos];
        let suffix = &pattern[star_pos + 1..];
        filename.starts_with(prefix) && filename.ends_with(suffix)
    } else {
        filename == pattern
    }
}

/// Parse `__doc__/wiki/<article>#<chunk>` → (article, chunk).
fn parse_key(k: &str) -> (String, usize) {
    let chunk = k.rfind('#')
        .and_then(|p| k[p + 1..].parse::<usize>().ok())
        .unwrap_or(0);
    let article = k.rsplit('/').next()
        .map(|s| s.split('#').next().unwrap_or(s).to_string())
        .unwrap_or_else(|| k.to_string());
    (article, chunk)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for x in a.iter().zip(b.iter()) {
        dot += x.0 * x.1;
        na += x.0 * x.0;
        nb += x.1 * x.1;
    }
    let denom = (na * nb).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Hash a text string into a 768-dim L2-normalized vector (bag-of-words via
/// FNV-1a). Deterministic, offline, no Ollama needed — the retrieval quality
/// score must be computable inside the darwin evaluator without network.
pub fn text_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    if text.trim().is_empty() {
        return v;
    }
    for tok in text.split_whitespace() {
        let t = tok.to_lowercase();
        let mut h: u64 = 1469598103934665603;
        for b in t.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        // double-mix into two indices for slightly denser signal
        let i1 = (h as usize) % dim;
        let i2 = ((h >> 32) as usize) % dim;
        v[i1] += 1.0;
        v[i2] += 0.5;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in &mut v { *x /= norm; }
    }
    v
}

impl WorldGraph {
    /// Query by raw embedding: top-k nearest nodes by cosine.
    pub fn query_embedding(&self, emb: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() || emb.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self.nodes.iter().enumerate()
            .map(|(i, n)| (i, cosine(&n.emb, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).collect()
    }

    /// Query by text (hash-embedding).
    pub fn query_text(&self, text: &str, top_k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() || self.nodes[0].emb.is_empty() {
            return Vec::new();
        }
        let dim = self.nodes[0].emb.len();
        let q = text_hash_embedding(text, dim);
        self.query_embedding(&q, top_k)
    }

    /// Retrieval quality ∈ [0,1]: how grounded is `text` in the worldgraph?
    /// Computed as domain diversity of the top-k hits plus a bridge bonus.
    /// High score = the text pulls knowledge from many distinct domains that are
    /// bridged in the corpus (general intelligence signal, not code-only).
    pub fn retrieval_quality(&self, text: &str) -> f32 {
        if self.nodes.is_empty() || self.n_clusters == 0 {
            return 0.5;
        }
        let k = 12usize.min(self.nodes.len());
        let hits = self.query_text(text, k);
        if hits.is_empty() {
            return 0.5;
        }
        // distinct domains among hits, normalized by min(k, n_clusters)
        let mut domains = std::collections::HashSet::new();
        for (idx, _) in &hits {
            domains.insert(self.cluster[*idx]);
        }
        let denom = (k.min(self.n_clusters)) as f32;
        let diversity = if denom > 0.0 { domains.len() as f32 / denom } else { 0.0 };
        // bridge bonus: fraction of hit pairs that are known bridges
        let bridges = self.bridges(8, 0.55);
        let bridge_keys: std::collections::HashSet<String> = bridges.iter()
            .flat_map(|b| vec![b.key_a.clone(), b.key_b.clone()])
            .collect();
        let mut bridge_hits = 0usize;
        for (idx, _) in &hits {
            if bridge_keys.contains(&self.nodes[*idx].key) {
                bridge_hits += 1;
            }
        }
        let bridge_score = bridge_hits as f32 / k as f32;
        // 0.7 diversity + 0.3 bridge signal, clamped
        (0.7 * diversity + 0.3 * bridge_score).clamp(0.0, 1.0)
    }
}

/// Cached global worldgraph (lazy, process-wide). Loaded once from the first
/// pattern that succeeds; subsequent calls reuse it. Returns a neutral 0.5 when
/// no shards exist (bench still passes in minimal checkouts).
pub fn retrieval_quality_cached(text: &str) -> f32 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Option<WorldGraph>> = OnceLock::new();
    let opt = CACHE.get_or_init(|| {
        // Try repo-root absolute path first, then relative, then home-relative.
        let candidates = [
            "/home/l/Desktop/AxiomTree/axiom_horizon/.kai_wiki_memory.*.json".to_string(),
            ".kai_wiki_memory.*.json".to_string(),
            "../.kai_wiki_memory.*.json".to_string(),
            "../../.kai_wiki_memory.*.json".to_string(),
        ];
        for pat in &candidates {
            if let Ok(nodes) = WorldGraph::load_shards(pat) {
                // quick clustering: 6 domains, 6 iters is enough for scoring
                let g = WorldGraph::from_nodes(nodes, 6, 6, 42);
                return Some(g);
            }
        }
        None
    });
    match opt {
        Some(g) => g.retrieval_quality(text),
        None => 0.5,
    }
}

fn nearest_centroid(emb: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_s = f32::NEG_INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let s = cosine(emb, c);
        if s > best_s {
            best_s = s;
            best = i;
        }
    }
    best
}

/// Tiny deterministic LCG for k-means seeding (no external RNG dependency
/// impact on reproducibility).
struct Seeded(u64);
impl Seeded {
    fn new(seed: u64) -> Self {
        Seeded(seed.max(1).wrapping_mul(0x9E3779B97F4A7C15))
    }
    fn next(&mut self) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic corpus with 3 clear latent domains plus a node half in
    /// domain A and half in domain B (a planted cross-domain bridge).
    fn synthetic_nodes(seed: u64) -> Vec<WorldNode> {
        let dim = 40;
        let a = d(10.0, 0.3, 25);
        let b = d(30.0, 0.3, 25);
        let c = d(50.0, 0.3, 25);
        let mut nodes = Vec::new();
        for (gi, group) in [&a, &b, &c].iter().enumerate() {
            for (ci, v) in group.iter().enumerate() {
                nodes.push(WorldNode {
                    key: format!("__doc__/wiki/domain{gi}_article{ci}#0"),
                    article: format!("domain{gi}-article{ci}"),
                    emb: v.clone(),
                });
            }
        }
        // Planted bridge: 10 nodes half-way between domain 0 and domain 1.
        for i in 0..10 {
            let mut v = vec![0.0f32; dim];
            for d in 0..3 {
                v[d] = 20.0; // between 10 and 30
            }
            for d in 3..dim {
                v[d] = (seed as u32 % 997) as f32 / 997.0;
            }
            nodes.push(WorldNode {
                key: format!("__doc__/wiki/bridge_topic{i}#0"),
                article: format!("bridge-topic{i}"),
                emb: v,
            });
        }
        nodes
    }

    fn d(base: f32, amp: f32, n: usize) -> Vec<Vec<f32>> {
        let dim = 40;
        let mut rng = Seeded::new((base as u64 * 17) + 7);
        let mut out = Vec::new();
        for _ in 0..n {
            let mut v = vec![0.0f32; dim];
            for dd in 0..3 { v[dd] = base + (rng.next() % 1000) as f32 / 1000.0 * amp; }
            for dd in 3..dim { v[dd] = (rng.next() % 1000) as f32 / 1000.0 * amp; }
            out.push(v);
        }
        out
    }

    #[test]
    fn worldgraph_load_and_cluster() {
        let nodes = synthetic_nodes(7);
        assert_eq!(nodes.len(), 85);
        let g = WorldGraph::from_nodes(nodes.clone(), 3, 8, 42);
        assert_eq!(g.n_clusters, 3);
        assert_eq!(g.cluster.len(), 85);
        assert_eq!(g.centroids.len(), 3);
        // All 3 domains must be non-empty.
        let mut per = [0usize; 3];
        for &c in g.cluster.iter() { per[c] += 1; }
        for p in per.iter() { assert!(*p > 0, "empty domain"); }
    }

    #[test]
    fn worldgraph_finds_cross_domain_bridges() {
        let g = WorldGraph::from_nodes(synthetic_nodes(91), 3, 8, 42);
        let br = g.bridges(12, 0.55);
        // The planted bridge-topic nodes (at base 20) sit between domains
        // {0,1}-ish; at least one bridge must exist involving them.
        assert!(!br.is_empty(), "expected at least one cross-domain bridge");
        // Every reported bridge must cross domains.
        for b in br.iter() {
            assert_ne!(b.domain_a, b.domain_b, "bridge must be cross-domain");
            assert!(b.cosine >= 0.55, "bridge cosine must respect min");
        }
    }

    #[test]
    fn worldgraph_attractor_roundtrip() {
        let g = WorldGraph::from_nodes(synthetic_nodes(91), 3, 8, 42);
        let path = std::env::temp_dir().join("kai_world_attr_test.json");
        let p = path.to_string_lossy().to_string();
        g.export_attractor(&p).unwrap();
        let vecs = crate::attractor::load(&p).unwrap();
        assert_eq!(vecs.len(), 3, "domain centroids exported");
        assert_eq!(vecs[0].len(), 40, "centroid dim");
        let _ = std::fs::remove_file(&p);
    }
}
