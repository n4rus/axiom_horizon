//! Kai personal-companion core: chat sessions (Tier 1), factual memory (Tier 2),
//! episodic recall bridge (Tier 3 — delegates to memory::MemoryStore), and
//! Darwin/VFE state sync with Kai-Android (Tier 4).
//!
//! Mirrors the Android `MemoryDb.kt` contract so `kai_state_export.json`
//! moves freely between phone and desktop.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ==================== Tier 1: Chat Sessions ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRow {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRow {
    pub id: String,
    pub chat_id: String,
    pub role: String, // USER | KAI | KAI_RECURSIVE
    pub text: String,
    pub vfe: Option<f32>,
    pub curvature: Option<f32>,
    pub temp: Option<f32>,
    pub model: String,
    pub ts: i64,
}

pub struct ChatStore {
    conn: Connection,
}

impl ChatStore {
    pub fn load_or_new(path: &str) -> Self {
        if let Some(dir) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let conn = Connection::open(path).expect("open kai chat db");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, role TEXT NOT NULL,
                text TEXT NOT NULL, vfe REAL, curvature REAL, temp REAL,
                model TEXT NOT NULL, ts INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, ts);
            "#,
        ).expect("create chat tables");
        ChatStore { conn }
    }

    pub fn upsert_chat(&self, c: &ChatRow) {
        let _ = self.conn.execute(
            "INSERT INTO chats (id,title,model,created_at,updated_at) VALUES (?1,?2,?3,?4,?5)
             ON CONFLICT(id) DO UPDATE SET title=?2, model=?3, updated_at=?5",
            params![c.id, c.title, c.model, c.created_at, c.updated_at],
        );
    }

    pub fn insert_message(&self, m: &MessageRow) {
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO messages (id,chat_id,role,text,vfe,curvature,temp,model,ts)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![m.id, m.chat_id, m.role, m.text, m.vfe, m.curvature, m.temp, m.model, m.ts],
        );
    }

    pub fn messages(&self, chat_id: &str) -> Vec<MessageRow> {
        let mut stmt = self.conn.prepare(
            "SELECT id,chat_id,role,text,vfe,curvature,temp,model,ts FROM messages WHERE chat_id=?1 ORDER BY ts"
        ).expect("prepare");
        stmt.query_map(params![chat_id], |r| Ok(MessageRow {
            id: r.get(0)?, chat_id: r.get(1)?, role: r.get(2)?, text: r.get(3)?,
            vfe: r.get(4)?, curvature: r.get(5)?, temp: r.get(6)?, model: r.get(7)?, ts: r.get(8)?,
        })).expect("query").filter_map(Result::ok).collect()
    }

    pub fn latest_chat(&self) -> Option<ChatRow> {
        let mut stmt = self.conn.prepare(
            "SELECT id,title,model,created_at,updated_at FROM chats ORDER BY updated_at DESC LIMIT 1"
        ).expect("prepare");
        stmt.query_row([], |r| Ok(ChatRow {
            id: r.get(0)?, title: r.get(1)?, model: r.get(2)?, created_at: r.get(3)?, updated_at: r.get(4)?,
        })).ok()
    }
}

// ==================== Tier 2: Factual Memory ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub fact: String,
    pub source: String, // user-stated | kai-inferred | desktop-import
    pub created_at: i64,
    pub recall_count: i64,
}

pub struct FactStore {
    conn: Connection,
}

impl FactStore {
    pub fn load_or_new(path: &str) -> Self {
        if let Some(dir) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let conn = Connection::open(path).expect("open fact db");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL DEFAULT 'user-stated',
                created_at INTEGER NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0);
            "#,
        ).expect("create facts table");
        FactStore { conn }
    }

    /// Mirrors Android MemoryDb.extractFact patterns
    pub fn extract_fact(text: &str) -> Option<String> {
        let t = text.trim();
        let lower = t.to_lowercase();
        let take = |s: &str, n: usize| -> String { s.chars().take(n).collect() };
        if lower.starts_with("remember ") { return Some(take(t.substring(9.min(t.len())), 300)); }
        if lower.starts_with("note that ") { return Some(take(t.substring(10.min(t.len())), 300)); }
        if lower.starts_with("my name is ") { return Some(format!("user's name is {}", take(t.substring(11.min(t.len())), 100))); }
        if lower.starts_with("call me ") { return Some(format!("user prefers to be called {}", take(t.substring(8.min(t.len())), 100))); }
        if lower.starts_with("i am ") || lower.starts_with("i'm ") {
            let rest = if lower.starts_with("i am ") { t.substring(5.min(t.len())) } else { t.substring(4.min(t.len())) };
            if rest.len() >= 3 && rest.len() <= 200 { return Some(format!("user is {}", rest)); }
            return None;
        }
        if lower.starts_with("i like ") { return Some(format!("user likes {}", take(t.substring(7.min(t.len())), 200))); }
        if lower.starts_with("i prefer ") { return Some(format!("user prefers {}", take(t.substring(9.min(t.len())), 200))); }
        if lower.starts_with("i work ") { return Some(format!("user works {}", take(t.substring(7.min(t.len())), 200))); }
        if lower.starts_with("i live ") { return Some(format!("user lives {}", take(t.substring(7.min(t.len())), 200))); }
        None
    }

    pub fn store(&self, fact: &str, source: &str) -> bool {
        self.conn.execute(
            "INSERT OR IGNORE INTO facts (fact,source,created_at) VALUES (?1,?2,?3)",
            params![fact, source, chrono_now()],
        ).map(|n| n > 0).unwrap_or(false)
    }

    pub fn all(&self) -> Vec<Fact> {
        let mut stmt = self.conn.prepare(
            "SELECT fact,source,created_at,recall_count FROM facts ORDER BY created_at DESC"
        ).expect("prepare");
        stmt.query_map([], |r| Ok(Fact {
            fact: r.get(0)?, source: r.get(1)?, created_at: r.get(2)?, recall_count: r.get(3)?,
        })).expect("query").filter_map(Result::ok).collect()
    }

    pub fn count(&self) -> i64 {
        self.conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0)).unwrap_or(0)
    }
}

// helper: safe substring (char-boundary)
trait SubstringSafe {
    fn substring(&self, start: usize) -> &str;
}
impl SubstringSafe for str {
    fn substring(&self, start: usize) -> &str {
        self.get(start..).unwrap_or("")
    }
}

fn chrono_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    let n = a.len().min(b.len());
    for i in 0..n { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if na == 0f32 || nb == 0f32 { 0f32 } else { dot / (na.sqrt() * nb.sqrt()) }
}

// ==================== Tier 3: recall bridge ====================

/// Inject memory context into a prompt before sending to the model.
/// `facts` = FactStore.all(); `query_embedding` = embedded query (caller embeds via engine).
/// Returns "[Memory]\n- fact1\n- fact2\n[/Memory]\n\n" prefix or "" if nothing relevant.
pub fn memory_context_block(facts: &[Fact], query_embedding: &[f32], fact_embeddings: &[(String, Vec<f32>)], k: usize) -> String {
    let mut scored: Vec<(f32, &Fact)> = Vec::new();
    for f in facts {
        if let Some((_, emb)) = fact_embeddings.iter().find(|(fact, _)| fact == &f.fact) {
            let s = cosine_sim(query_embedding, emb);
            if s > 0.15 { scored.push((s, f)); }
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<&Fact> = scored.into_iter().take(k).map(|(_, f)| f).collect();
    if top.is_empty() { return String::new(); }
    let mut out = String::from("[Memory]\n");
    for f in &top { out.push_str(&format!("- {}\n", f.fact)); }
    out.push_str("[/Memory]\n\n");
    out
}

// ==================== Tier 4: Darwin/VFE state sync ====================

#[derive(Debug, Serialize, Deserialize)]
pub struct KaiStateExport {
    pub kai_state_version: u32,
    pub exported_at: i64,
    pub physics: PhysicsState,
    pub memories: Vec<Fact>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PhysicsState {
    pub vfe_base: f32,
    pub curvature_alpha: f32,
}

pub struct DarwinSync;

impl DarwinSync {
    pub fn export(path: &str, facts: &[Fact], physics: PhysicsState) -> std::io::Result<()> {
        let state = KaiStateExport {
            kai_state_version: 1,
            exported_at: chrono_now(),
            physics,
            memories: facts.to_vec(),
        };
        std::fs::write(path, serde_json::to_string_pretty(&state)?)
    }

    pub fn import(path: &str, facts: &FactStore) -> std::io::Result<usize> {
        let data = std::fs::read_to_string(path)?;
        let state: KaiStateExport = serde_json::from_str(&data)?;
        let mut n = 0;
        for m in &state.memories {
            if facts.store(&m.fact, "desktop-import") { n += 1; }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_extraction_patterns() {
        assert_eq!(FactStore::extract_fact("my name is Lucas"), Some("user's name is Lucas".into()));
        assert_eq!(FactStore::extract_fact("remember I like rust"), Some("I like rust".into()));
        assert_eq!(FactStore::extract_fact("i am a dev"), Some("user is a dev".into()));
        assert_eq!(FactStore::extract_fact("hello world"), None);
    }

    #[test]
    fn chat_persistence_roundtrip() {
        let s = ChatStore::load_or_new("/tmp/kai_test_chat.db");
        s.upsert_chat(&ChatRow { id: "c1".into(), title: "t".into(), model: "qwen".into(), created_at: 1, updated_at: 2 });
        s.insert_message(&MessageRow { id: "m1".into(), chat_id: "c1".into(), role: "USER".into(),
            text: "hi".into(), vfe: Some(2.0), curvature: Some(0.4), temp: Some(0.9), model: "qwen".into(), ts: 3 });
        assert_eq!(s.messages("c1").len(), 1);
        assert!(s.latest_chat().is_some());
        let _ = std::fs::remove_file("/tmp/kai_test_chat.db");
    }

    #[test]
    fn memory_context_block_injects() {
        let facts = vec![Fact { fact: "user's name is Lucas".into(), source: "user".into(), created_at: 1, recall_count: 0 }];
        let embs = vec![("user's name is Lucas".to_string(), vec![1.0f32, 0.0, 0.0])];
        let block = memory_context_block(&facts, &[0.9f32, 0.1, 0.0], &embs, 3);
        assert!(block.contains("[Memory]"));
        assert!(block.contains("Lucas"));
    }

    #[test]
    fn darwin_sync_roundtrip() {
        let fs = FactStore::load_or_new("/tmp/kai_test_facts.db");
        fs.store("test fact", "test");
        let path = "/tmp/kai_test_state.json";
        DarwinSync::export(path, &fs.all(), PhysicsState { vfe_base: 0.85, curvature_alpha: 0.4 }).unwrap();
        let fs2 = FactStore::load_or_new("/tmp/kai_test_facts2.db");
        let n = DarwinSync::import(path, &fs2).unwrap();
        assert!(n >= 1);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file("/tmp/kai_test_facts.db");
        let _ = std::fs::remove_file("/tmp/kai_test_facts2.db");
    }
}
