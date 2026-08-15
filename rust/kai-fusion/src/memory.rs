//! Vector memory for Kai — stores (text, embedding) pairs with cosine similarity search.
//! Persisted as SQLite database (`.kai_memory.db`).
//!
//! Phase 1.1: Semantic Memory (Vector Embeddings). Every turn is embedded and stored.
//! On retrieval, find top-k similar past events and inject into context.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A single memory entry: user text + model embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: u64,
    pub text: String,
    pub embedding: Vec<f32>,
    pub timestamp: String,
    /// Optional tags for filtering (e.g. "user", "model", "system")
    pub tags: Vec<String>,
}

/// SQLite-backed vector store with JSON fallback path for migration.
pub struct MemoryStore {
    conn: Connection,
    #[allow(dead_code)]
    path: String,
    /// Maximum memories before evicting oldest (0 = no limit)
    pub max_memories: usize,
}

impl MemoryStore {
    /// Create a new empty store at the given path (SQLite).
    pub fn new(path: &str, max_memories: usize) -> Self {
        let conn = Connection::open(path).expect("open sqlite");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
            "#,
        ).expect("create memories table");

        MemoryStore {
            conn,
            path: path.to_string(),
            max_memories,
        }
    }

    /// Load from disk, or create empty if file doesn't exist.
    /// If JSON file exists at old path, migrate it to SQLite.
    #[allow(dead_code)]
    pub fn load_or_new(path: &str, max_memories: usize) -> Self {
        let sqlite_path = path.replace(".json", ".db");
        
        // If SQLite already exists, use it
        if Path::new(&sqlite_path).exists() {
            return Self::new(&sqlite_path, max_memories);
        }

        // If old JSON exists, migrate it
        if Path::new(path).exists() {
            if let Ok(json) = fs::read_to_string(path) {
                if let Ok(memories) = serde_json::from_str::<Vec<Memory>>(&json) {
                    let mut store = Self::new(&sqlite_path, max_memories);
                    for m in memories {
                        let _ = store.add(&m.text, &m.embedding, m.tags);
                    }
                    // Rename old JSON as backup
                    let _ = fs::rename(path, format!("{}.backup", path));
                    return store;
                }
            }
        }

        Self::new(&sqlite_path, max_memories)
    }

    /// Add a memory. Returns the assigned id.
    pub fn add(&mut self, text: &str, embedding: &[f32], tags: Vec<String>) -> u64 {
        // Serialize embedding as binary (little-endian f32)
        let emb_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        let tags_json = serde_json::to_string(&tags).unwrap_or_default();
        let ts = chrono_now();

        self.conn.execute(
            "INSERT INTO memories (text, embedding, timestamp, tags) VALUES (?1, ?2, ?3, ?4)",
            params![text, emb_bytes, ts, tags_json],
        ).expect("insert memory");

        let id = self.conn.last_insert_rowid() as u64;

        // Evict oldest if over limit
        if self.max_memories > 0 {
            let count: u64 = self.conn.query_row(
                "SELECT COUNT(*) FROM memories", [], |r| r.get(0)
            ).unwrap_or(0);
            if count > self.max_memories as u64 {
                let excess = count - self.max_memories as u64;
                self.conn.execute(
                    "DELETE FROM memories WHERE id IN (SELECT id FROM memories ORDER BY timestamp ASC LIMIT ?1)",
                    params![excess],
                ).expect("evict old memories");
            }
        }

        id
    }

    /// Search for top-k memories by cosine similarity to query_embedding.
    /// Returns Vec of (memory, score) sorted descending by score.
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Vec<(Memory, f32)> {
        if query_embedding.is_empty() {
            return Vec::new();
        }

        let mut stmt = self.conn.prepare(
            "SELECT id, text, embedding, timestamp, tags FROM memories"
        ).expect("prepare search");

        let rows = stmt.query_map([], |row| {
            let id: u64 = row.get(0)?;
            let text: String = row.get(1)?;
            let emb_blob: Vec<u8> = row.get(2)?;
            let timestamp: String = row.get(3)?;
            let tags_json: String = row.get(4)?;
            
            // Deserialize embedding from bytes
            let embedding: Vec<f32> = emb_blob
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

            Ok(Memory { id, text, embedding, timestamp, tags })
        }).expect("query memories");

        let mut scored: Vec<(Memory, f32)> = rows
            .filter_map(|r| r.ok())
            .map(|m| {
                let sim = cosine_similarity(query_embedding, &m.embedding);
                (m, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// List the n most recent memories.
    pub fn list(&self, n: usize) -> Vec<Memory> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, embedding, timestamp, tags FROM memories ORDER BY timestamp DESC LIMIT ?1"
        ).expect("prepare list");

        let rows = stmt.query_map(params![n], |row| {
            let id: u64 = row.get(0)?;
            let text: String = row.get(1)?;
            let emb_blob: Vec<u8> = row.get(2)?;
            let timestamp: String = row.get(3)?;
            let tags_json: String = row.get(4)?;
            
            let embedding: Vec<f32> = emb_blob
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

            Ok(Memory { id, text, embedding, timestamp, tags })
        }).expect("query list");

        rows.filter_map(|r| r.ok()).collect()
    }

    /// Clear all memories.
    pub fn clear(&mut self) {
        self.conn.execute("DELETE FROM memories", []).expect("clear memories");
    }

    /// Total stored memories.
    pub fn len(&self) -> usize {
        self.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0)
    }
}

/// Cosine similarity between two vectors.
/// Returns value in [-1, 1] (or 0.0 if either vector is zero).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).max(-1.0).min(1.0)
    }
}

/// Simple ISO-8601 timestamp without external dep.
fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();

    // Days since Unix epoch (1970-01-01)
    let days_from_epoch = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let secs = time_secs % 60;

    // Days from 1970-01-01 to 2026-01-01
    // = (2026-1970)*365 + leap_days(1970..2025)
    // = 56*365 + 14 = 20440 + 14 = 20454
    let days_to_2026: u64 = 20454;

    let mut remaining = if days_from_epoch >= days_to_2026 {
        days_from_epoch - days_to_2026
    } else {
        0
    };
    let mut year = 2026u64;

    // Walk forward years
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    // Convert day-of-year (0-indexed) to month/day
    let month_days = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1usize;
    for &md in month_days.iter() {
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }
    let day = remaining + 1;
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hours, mins, secs
    )
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_memory_store_add_search() {
        let mut store = MemoryStore::new("/tmp/test_kai_memory.db", 0);
        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0];
        let e3 = vec![1.0, 1.0, 0.0];

        store.add("hello world", &e1, vec![]);
        store.add("goodbye world", &e2, vec![]);
        store.add("hello there", &e3, vec![]);

        // Search with query close to e1
        let query = vec![1.0, 0.1, 0.0];
        let results = store.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.text, "hello world");
        assert!((results[0].1 - 1.0).abs() < 0.1);

        // Clean up
        let _ = std::fs::remove_file("/tmp/test_kai_memory.db");
    }

    #[test]
    fn test_memory_store_list() {
        let mut store = MemoryStore::new("/tmp/test_kai_memory_list.db", 0);
        store.add("first", &[1.0, 0.0], vec![]);
        store.add("second", &[0.0, 1.0], vec![]);
        store.add("third", &[1.0, 1.0], vec![]);

        let list = store.list(2);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].text, "third");
        assert_eq!(list[1].text, "second");

        let _ = std::fs::remove_file("/tmp/test_kai_memory_list.db");
    }

    #[test]
    fn test_timestamp_format() {
        let ts = chrono_now();
        assert!(ts.len() >= 19); // "YYYY-MM-DD HH:MM:SS"
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
    }
}