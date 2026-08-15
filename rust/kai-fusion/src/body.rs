//! Kai's Virtual Body — minimal sandbox for embodied interaction.
//! Provides filesystem, shell, and web access as Kai's "body".

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::collections::HashMap;
use std::sync::RwLock;
use crate::scm::EngramInterface;

/// Result of an action in the virtual sandbox
#[derive(Debug, Clone)]
pub struct ActionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Kai's virtual body — the interface to reality
pub struct VirtualBody {
    work_dir: PathBuf,
    allowed_dirs: Vec<PathBuf>,
    command_timeout_secs: u64,
}

impl VirtualBody {
    pub fn new(work_dir: impl AsRef<Path>) -> Self {
        let work_dir = work_dir.as_ref().canonicalize().unwrap_or_else(|_| work_dir.as_ref().to_path_buf());
        Self {
            work_dir: work_dir.clone(),
            allowed_dirs: vec![work_dir],
            command_timeout_secs: 30,
        }
    }

    #[allow(dead_code)]
    pub fn with_allowed_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.allowed_dirs = dirs.into_iter().map(|d| d.canonicalize().unwrap_or(d)).collect();
        self
    }

    #[allow(dead_code)]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.command_timeout_secs = secs;
        self
    }

    fn validate_path(&self, path: &Path) -> Result<PathBuf, String> {
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.work_dir.join(path)
        };
        // Canonicalize the parent if the path doesn't exist yet
        let canonical = if abs.exists() {
            abs.canonicalize().map_err(|e| format!("Path error: {e}"))?
        } else if let Some(parent) = abs.parent() {
            let canon_parent = parent.canonicalize().map_err(|e| format!("Parent error: {e}"))?;
            canon_parent.join(abs.file_name().unwrap())
        } else {
            return Err("Invalid path".to_string());
        };
        
        // Check if path is within allowed directories
        let allowed = self.allowed_dirs.iter().any(|d| {
            canonical.starts_with(d)
        });
        if !allowed {
            return Err("Path outside allowed directories".to_string());
        }
        Ok(canonical)
    }

    // ===== FILESYSTEM OPERATIONS =====

    pub fn read_file(&self, path: &str) -> ActionResult {
        let start = std::time::Instant::now();
        let path = match self.validate_path(Path::new(path)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        match fs::read_to_string(&path) {
            Ok(content) => ActionResult {
                success: true,
                output: content,
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
            },
            Err(e) => ActionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Read error: {e}")),
                duration_ms: start.elapsed().as_millis() as u64,
            },
        }
    }

    pub fn write_file(&self, path: &str, content: &str) -> ActionResult {
        let start = std::time::Instant::now();
        let path = match self.validate_path(Path::new(path)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return ActionResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Create dir error: {e}")),
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
        }
        match fs::write(&path, content) {
            Ok(_) => ActionResult {
                success: true,
                output: format!("Written {} bytes to {}", content.len(), path.display()),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
            },
            Err(e) => ActionResult {
                success: false,
                output: String::new(),
                error: Some(format!("Write error: {e}")),
                duration_ms: start.elapsed().as_millis() as u64,
            },
        }
    }

    pub fn list_dir(&self, path: &str) -> ActionResult {
        let start = std::time::Instant::now();
        let path = match self.validate_path(Path::new(path)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        match fs::read_dir(&path) {
            Ok(entries) => {
                let mut output = String::new();
                for entry in entries.flatten() {
                    let ft = entry.file_type().ok();
                    let prefix = match ft {
                        Some(t) if t.is_dir() => "[DIR] ",
                        Some(t) if t.is_file() => "[FILE] ",
                        Some(t) if t.is_symlink() => "[LINK] ",
                        _ => "[?] ",
                    };
                    output.push_str(&format!("{}{}\n", prefix, entry.file_name().to_string_lossy()));
                }
                ActionResult { success: true, output, error: None, duration_ms: start.elapsed().as_millis() as u64 }
            },
            Err(e) => ActionResult { success: false, output: String::new(), error: Some(format!("List error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
        }
    }

    pub fn delete_file(&self, path: &str) -> ActionResult {
        let start = std::time::Instant::now();
        let path = match self.validate_path(Path::new(path)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        let result = if path.is_dir() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
        match result {
            Ok(_) => ActionResult { success: true, output: format!("Deleted {}", path.display()), error: None, duration_ms: start.elapsed().as_millis() as u64 },
            Err(e) => ActionResult { success: false, output: String::new(), error: Some(format!("Delete error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
        }
    }

    // ===== FILESYSTEM TOOLS =====

    /// Recursively list directory tree up to max_depth (0 = unlimited).
    pub fn read_tree(&self, path: &str, max_depth: usize) -> ActionResult {
        let start = std::time::Instant::now();
        let path = match self.validate_path(Path::new(path)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        if !path.is_dir() {
            return ActionResult { success: false, output: String::new(), error: Some(format!("Not a directory: {}", path.display())), duration_ms: start.elapsed().as_millis() as u64 };
        }
        let mut output = String::new();
        self._tree_walk(&path, 0, max_depth, &mut output);
        ActionResult { success: true, output, error: None, duration_ms: start.elapsed().as_millis() as u64 }
    }

    fn _tree_walk(&self, dir: &Path, depth: usize, max_depth: usize, out: &mut String) {
        if max_depth > 0 && depth > max_depth { return; }
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => { out.push_str(&format!("{}[ERR] {}\n", "  ".repeat(depth), e)); return; }
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let ft = entry.file_type().ok();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let prefix = match ft.as_ref() {
                Some(t) if t.is_dir() => "[DIR] ",
                Some(t) if t.is_file() => "[FILE] ",
                Some(t) if t.is_symlink() => "[LINK] ",
                _ => "[?] ",
            };
            out.push_str(&format!("{}{}{}\n", "  ".repeat(depth), prefix, name_str));
            if let Some(t) = ft.as_ref() {
                if t.is_dir() {
                    self._tree_walk(&entry.path(), depth + 1, max_depth, out);
                }
            }
        }
    }

    /// Find files matching a glob-like pattern (supports * and ? wildcards).
    pub fn find_file(&self, pattern: &str, start_path: &str) -> ActionResult {
        let start = std::time::Instant::now();
        let start_path = match self.validate_path(Path::new(start_path)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        if !start_path.is_dir() {
            return ActionResult { success: false, output: String::new(), error: Some(format!("Not a directory: {}", start_path.display())), duration_ms: start.elapsed().as_millis() as u64 };
        }
        let pattern = pattern.to_string();
        let mut results = Vec::new();
        self._find_walk(&start_path, &pattern, &mut results);
        results.sort();
        let output = if results.is_empty() {
            "No matches found.".to_string()
        } else {
            results.join("\n")
        };
        ActionResult { success: true, output, error: None, duration_ms: start.elapsed().as_millis() as u64 }
    }

    fn _find_walk(&self, dir: &Path, pattern: &str, results: &mut Vec<String>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if Self::_matches_pattern(&name_str, pattern) {
                results.push(entry.path().to_string_lossy().to_string());
            }
            if entry.path().is_dir() {
                self._find_walk(&entry.path(), pattern, results);
            }
        }
    }

    fn _matches_pattern(name: &str, pattern: &str) -> bool {
        if pattern == "*" { return true; }
        let name_chars: Vec<char> = name.chars().collect();
        let pat_chars: Vec<char> = pattern.chars().collect();
        Self::_match_recursive(&name_chars, 0, &pat_chars, 0)
    }

    fn _match_recursive(name: &[char], ni: usize, pat: &[char], pi: usize) -> bool {
        // Both exhausted → match
        if pi == pat.len() { return ni == name.len(); }
        if ni > name.len() {
            // Only wildcard can match empty remaining
            return pat[pi..].iter().all(|c| *c == '*');
        }
        match pat[pi] {
            '*' => {
                // Try matching 0 or more chars
                for k in ni..=name.len() {
                    if Self::_match_recursive(name, k, pat, pi + 1) {
                        return true;
                    }
                }
                false
            }
            '?' => Self::_match_recursive(name, ni + 1, pat, pi + 1),
            c => {
                if ni < name.len() && name[ni] == c {
                    Self::_match_recursive(name, ni + 1, pat, pi + 1)
                } else {
                    false
                }
            }
        }
    }

    /// Unified diff between two files (unified format, 3 lines context).
    pub fn diff(&self, path1: &str, path2: &str) -> ActionResult {
        let start = std::time::Instant::now();
        let p1 = match self.validate_path(Path::new(path1)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        let p2 = match self.validate_path(Path::new(path2)) {
            Ok(p) => p,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(e), duration_ms: start.elapsed().as_millis() as u64 },
        };
        let content1 = match fs::read_to_string(&p1) {
            Ok(c) => c,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(format!("Read {}: {e}", p1.display())), duration_ms: start.elapsed().as_millis() as u64 },
        };
        let content2 = match fs::read_to_string(&p2) {
            Ok(c) => c,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(format!("Read {}: {e}", p2.display())), duration_ms: start.elapsed().as_millis() as u64 },
        };
        let diff = self._unified_diff(&content1, &content2, &p1.to_string_lossy(), &p2.to_string_lossy());
        ActionResult { success: true, output: diff, error: None, duration_ms: start.elapsed().as_millis() as u64 }
    }

    fn _unified_diff(&self, a: &str, b: &str, name_a: &str, name_b: &str) -> String {
        let a_lines: Vec<&str> = a.lines().collect();
        let b_lines: Vec<&str> = b.lines().collect();
        let mut out = format!("--- {}\n+++ {}\n", name_a, name_b);

        // Simple LCS-based diff
        let lcs = self._lcs(&a_lines, &b_lines);
        let mut ai = 0usize;
        let mut bi = 0usize;

        for (a_idx, b_idx) in &lcs {
            // Emit lines between last position and current match as context/deletions/additions
            while ai < *a_idx || bi < *b_idx {
                if ai < *a_idx && bi < *b_idx {
                    // Both have unconsumed lines → they differ
                    // Emit deletions from a
                    while ai < *a_idx {
                        out.push_str(&format!("-{}\n", a_lines[ai]));
                        ai += 1;
                    }
                    // Emit additions from b
                    while bi < *b_idx {
                        out.push_str(&format!("+{}\n", b_lines[bi]));
                        bi += 1;
                    }
                } else if ai < *a_idx {
                    out.push_str(&format!("-{}\n", a_lines[ai]));
                    ai += 1;
                } else if bi < *b_idx {
                    out.push_str(&format!("+{}\n", b_lines[bi]));
                    bi += 1;
                }
            }
            // Marked as unchanged (context)
            out.push_str(&format!(" {}\n", a_lines[*a_idx]));
            ai = *a_idx + 1;
            bi = *b_idx + 1;
        }

        // Remaining lines
        while ai < a_lines.len() {
            out.push_str(&format!("-{}\n", a_lines[ai]));
            ai += 1;
        }
        while bi < b_lines.len() {
            out.push_str(&format!("+{}\n", b_lines[bi]));
            bi += 1;
        }

        out
    }

    fn _lcs(&self, a: &[&str], b: &[&str]) -> Vec<(usize, usize)> {
        let n = a.len();
        let m = b.len();
        if n == 0 || m == 0 { return vec![]; }
        // Use Hirschberg's algorithm for space efficiency, but for simplicity use standard DP
        let mut dp = vec![vec![0usize; m + 1]; n + 1];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                if a[i] == b[j] {
                    dp[i][j] = dp[i + 1][j + 1] + 1;
                } else {
                    dp[i][j] = dp[i + 1][j].max(dp[i][j + 1]);
                }
            }
        }
        // Backtrack to find LCS pairs
        let mut result = Vec::new();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < n && j < m {
            if a[i] == b[j] {
                result.push((i, j));
                i += 1;
                j += 1;
            } else if dp[i + 1][j] >= dp[i][j + 1] {
                i += 1;
            } else {
                j += 1;
            }
        }
        result
    }

    // ===== SHELL COMMANDS =====

    pub fn run_command(&self, cmd: &str, args: &[&str]) -> ActionResult {
        let start = std::time::Instant::now();
        let child = match Command::new(cmd)
            .args(args)
            .current_dir(&self.work_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return ActionResult { success: false, output: String::new(), error: Some(format!("Spawn error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
        };

        let output = match std::thread::spawn(move || {
            child.wait_with_output()
        }).join() {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => return ActionResult { success: false, output: String::new(), error: Some(format!("Wait error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
            Err(_) => return ActionResult { success: false, output: String::new(), error: Some("Thread panic".to_string()), duration_ms: start.elapsed().as_millis() as u64 },
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let success = output.status.success();

        ActionResult {
            success,
            output: if stdout.is_empty() { stderr.clone() } else { stdout },
            error: if success { None } else { Some(stderr) },
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    pub fn shell(&self, command: &str) -> ActionResult {
        // Parse simple shell command
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return ActionResult { success: false, output: String::new(), error: Some("Empty command".to_string()), duration_ms: 0 };
        }
        self.run_command(parts[0], &parts[1..])
    }

    // ===== WEB / HTTP =====

    pub fn http_get(&self, url: &str) -> ActionResult {
        let start = std::time::Instant::now();
        match ureq::get(url).timeout(std::time::Duration::from_secs(self.command_timeout_secs)).call() {
            Ok(resp) => {
                let status = resp.status();
                match resp.into_string() {
                    Ok(text) => ActionResult {
                        success: status >= 200 && status < 300,
                        output: text,
                        error: if status >= 200 && status < 300 { None } else { Some(format!("HTTP {}", status)) },
                        duration_ms: start.elapsed().as_millis() as u64,
                    },
                    Err(e) => ActionResult { success: false, output: String::new(), error: Some(format!("Read error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
                }
            },
            Err(e) => ActionResult { success: false, output: String::new(), error: Some(format!("Request error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
        }
    }

    pub fn http_post(&self, url: &str, body: &str) -> ActionResult {
        let start = std::time::Instant::now();
        match ureq::post(url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.command_timeout_secs))
            .send_string(body)
        {
            Ok(resp) => {
                let status = resp.status();
                match resp.into_string() {
                    Ok(text) => ActionResult {
                        success: status >= 200 && status < 300,
                        output: text,
                        error: if status >= 200 && status < 300 { None } else { Some(format!("HTTP {}", status)) },
                        duration_ms: start.elapsed().as_millis() as u64,
                    },
                    Err(e) => ActionResult { success: false, output: String::new(), error: Some(format!("Read error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
                }
            },
            Err(e) => ActionResult { success: false, output: String::new(), error: Some(format!("Request error: {e}")), duration_ms: start.elapsed().as_millis() as u64 },
        }
    }
}

/// High-level action interface for the agent
impl VirtualBody {
    /// Execute a structured action
    pub fn act(&self, action: &str, params: &str) -> ActionResult {
        match action {
            "read" => self.read_file(params),
            "write" => {
                // params format: "path|content"
                let parts: Vec<&str> = params.splitn(2, '|').collect();
                if parts.len() != 2 {
                    return ActionResult { success: false, output: String::new(), error: Some("write format: path|content".to_string()), duration_ms: 0 };
                }
                self.write_file(parts[0], parts[1])
            },
            "list" => self.list_dir(params),
            "delete" => self.delete_file(params),
            "tree" => {
                let parts: Vec<&str> = params.splitn(2, '|').collect();
                let max_depth = parts.get(1).and_then(|d| d.parse::<usize>().ok()).unwrap_or(0);
                self.read_tree(parts[0], max_depth)
            },
            "find" => {
                let parts: Vec<&str> = params.splitn(2, '|').collect();
                if parts.len() != 2 {
                    ActionResult { success: false, output: String::new(), error: Some("find format: pattern|path".to_string()), duration_ms: 0 }
                } else {
                    self.find_file(parts[0], parts[1])
                }
            },
            "diff" => {
                let parts: Vec<&str> = params.splitn(2, '|').collect();
                if parts.len() != 2 {
                    ActionResult { success: false, output: String::new(), error: Some("diff format: path1|path2".to_string()), duration_ms: 0 }
                } else {
                    self.diff(parts[0], parts[1])
                }
            },
            "shell" => self.shell(params),
            "get" => self.http_get(params),
            "post" => {
                let parts: Vec<&str> = params.splitn(2, '|').collect();
                if parts.len() != 2 {
                    return ActionResult { success: false, output: String::new(), error: Some("post format: url|body".to_string()), duration_ms: 0 };
                }
                self.http_post(parts[0], parts[1])
            },
            _ => ActionResult { success: false, output: String::new(), error: Some(format!("Unknown action: {}", action)), duration_ms: 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_filesystem() {
        let dir = tempdir().unwrap();
        let body = VirtualBody::new(dir.path());
        
        // Write
        let r = body.write_file("test.txt", "hello world");
        assert!(r.success);
        
        // Read
        let r = body.read_file("test.txt");
        assert!(r.success);
        assert_eq!(r.output, "hello world");
        
        // List
        let r = body.list_dir(".");
        assert!(r.success);
        assert!(r.output.contains("test.txt"));
        
        // Delete
        let r = body.delete_file("test.txt");
        assert!(r.success);
    }

    #[test]
    fn test_shell() {
        let dir = tempdir().unwrap();
        let body = VirtualBody::new(dir.path());
        
        let r = body.shell("echo hello");
        assert!(r.success);
        assert!(r.output.contains("hello"));
    }
}

// ===== Engram persistence via VirtualBody =====

/// An Engram implementation that stores embeddings on disk via VirtualBody.
/// Each embedding is stored as a JSON file at `.axiom_state/engram/{concept}.json`.
pub struct BodyEngram {
    body: VirtualBody,
    cache: RwLock<HashMap<String, Vec<f32>>>,
    engram_dir: PathBuf,
}

impl BodyEngram {
    pub fn new(work_dir: impl AsRef<Path>) -> Self {
        let body = VirtualBody::new(work_dir.as_ref());
        let engram_dir = work_dir.as_ref().join(".axiom_state/engram");
        let _ = fs::create_dir_all(&engram_dir);
        Self {
            body,
            cache: RwLock::new(HashMap::new()),
            engram_dir,
        }
    }

    fn concept_path(&self, concept: &str) -> PathBuf {
        let sanitized: String = concept.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        self.engram_dir.join(format!("{}.json", sanitized))
    }
}

impl EngramInterface for BodyEngram {
    fn get_embedding(&self, concept: &str) -> Option<Vec<f32>> {
        // Check cache first
        if let Some(emb) = self.cache.read().unwrap().get(concept) {
            return Some(emb.clone());
        }
        // Load from disk
        let path = self.concept_path(concept);
        let result = self.body.read_file(path.to_str().unwrap_or(""));
        if result.success {
            if let Ok(emb) = serde_json::from_str::<Vec<f32>>(&result.output) {
                self.cache.write().unwrap().insert(concept.to_string(), emb.clone());
                return Some(emb);
            }
        }
        None
    }

    fn store_embedding(&self, concept: &str, embedding: Vec<f32>) {
        if let Ok(json) = serde_json::to_string(&embedding) {
            let path = self.concept_path(concept);
            let _ = self.body.write_file(path.to_str().unwrap_or(""), &json);
        }
        self.cache.write().unwrap().insert(concept.to_string(), embedding);
    }

    fn query_similar(&self, embedding: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = Vec::new();
        // Scan all files in engram directory
        if let Ok(entries) = fs::read_dir(&self.engram_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(other) = serde_json::from_str::<Vec<f32>>(&content) {
                            let sim = cosine_similarity(embedding, &other);
                            scores.push((name, sim));
                        }
                    }
                }
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}