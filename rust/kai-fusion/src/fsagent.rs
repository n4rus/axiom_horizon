//! Phase 3.4: File System Agent — structured access to the filesystem.
//!
//! From AGI_PLAN.md §3.4:
//! "Structured access to the file system beyond the codebase:
//!   - read_tree(path) → recursive directory listing
//!   - find_file(pattern) → glob search
//!   - read_file(path) → content
//!   - write_file(path, content) → success/error
//!   - diff(path1, path2) → unified diff"

#![allow(dead_code)]

use std::path::Path;

/// Result of a filesystem operation.
#[derive(Debug)]
#[allow(dead_code)]
pub enum FsResult {
    /// Operation succeeded with text output
    Ok(String),
    /// Operation failed
    Err(String),
}

/// Recursively list a directory tree (max depth limited).
pub fn read_tree(root: &str, max_depth: usize) -> FsResult {
    let path = Path::new(root);
    if !path.exists() {
        return FsResult::Err(format!("path not found: {root}"));
    }
    let mut lines = Vec::new();
    read_tree_inner(path, 0, max_depth, &mut lines);
    FsResult::Ok(lines.join("\n"))
}

fn read_tree_inner(path: &Path, depth: usize, max_depth: usize, lines: &mut Vec<String>) {
    if depth > max_depth { return; }
    let indent = "  ".repeat(depth);
    let name = path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "/".to_string());

    let prefix = if path.is_dir() { "d " } else { "f " };
    let size = if path.is_file() {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    } else { 0 };
    lines.push(format!("{indent}{prefix}{name} ({size} bytes)"));

    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            entries.sort_by_key(|e| e.file_name());
            for entry in &entries {
                read_tree_inner(&entry.path(), depth + 1, max_depth, lines);
            }
        }
    }
}

/// Find files matching a simple glob pattern (supports * wildcards).
pub fn find_file(root: &str, pattern: &str) -> FsResult {
    let path = Path::new(root);
    if !path.exists() {
        return FsResult::Err(format!("path not found: {root}"));
    }
    let mut matches = Vec::new();
    find_file_inner(path, pattern, &mut matches, 0, 10); // max depth 10
    if matches.is_empty() {
        FsResult::Ok("no matches found".to_string())
    } else {
        matches.sort();
        FsResult::Ok(matches.join("\n"))
    }
}

fn find_file_inner(dir: &Path, pattern: &str, matches: &mut Vec<String>, depth: usize, max_depth: usize) {
    if depth > max_depth { return; }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if glob_match(pattern, &name) {
                matches.push(path.display().to_string());
            }
            if path.is_dir() {
                find_file_inner(&path, pattern, matches, depth + 1, max_depth);
            }
        }
    }
}

/// Simple glob matching: * matches any characters, ? matches one.
fn glob_match(pattern: &str, text: &str) -> bool {
    glob_match_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_inner(pattern: &[u8], text: &[u8]) -> bool {
    let mut pi = 0;
    let mut ti = 0;
    let mut star_pi = usize::MAX;
    let mut star_ti = 0;

    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == b'?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == b'*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == b'*' {
        pi += 1;
    }
    pi == pattern.len()
}

/// Read file contents.
pub fn read_file(path: &str) -> FsResult {
    match std::fs::read_to_string(path) {
        Ok(content) => FsResult::Ok(content),
        Err(e) => FsResult::Err(format!("error reading {path}: {e}")),
    }
}

/// Write content to a file.
pub fn write_file(path: &str, content: &str) -> FsResult {
    match std::fs::write(path, content) {
        Ok(()) => FsResult::Ok(format!("wrote {} bytes to {path}", content.len())),
        Err(e) => FsResult::Err(format!("error writing {path}: {e}")),
    }
}

/// Compute a simple diff between two files (line-by-line).
pub fn diff(path1: &str, path2: &str) -> FsResult {
    let content1 = match std::fs::read_to_string(path1) {
        Ok(c) => c,
        Err(e) => return FsResult::Err(format!("error reading {path1}: {e}")),
    };
    let content2 = match std::fs::read_to_string(path2) {
        Ok(c) => c,
        Err(e) => return FsResult::Err(format!("error reading {path2}: {e}")),
    };
    simple_diff(&content1, &content2)
}

/// Simple line-by-line diff.
fn simple_diff(a: &str, b: &str) -> FsResult {
    let lines_a: Vec<&str> = a.lines().collect();
    let lines_b: Vec<&str> = b.lines().collect();
    let mut output = Vec::new();

    let max_len = lines_a.len().max(lines_b.len());
    let mut changes = 0;

    for i in 0..max_len {
        match (lines_a.get(i), lines_b.get(i)) {
            (Some(a), Some(b)) if a == b => {
                output.push(format!("  {a}"));
            }
            (Some(a), Some(b)) => {
                output.push(format!("- {a}"));
                output.push(format!("+ {b}"));
                changes += 1;
            }
            (Some(a), None) => {
                output.push(format!("- {a}"));
                changes += 1;
            }
            (None, Some(b)) => {
                output.push(format!("+ {b}"));
                changes += 1;
            }
            (None, None) => {}
        }
    }

    if changes == 0 {
        FsResult::Ok("files are identical".to_string())
    } else {
        output.insert(0, format!("--- {}/diff", changes));
        FsResult::Ok(output.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_tree_existing() {
        // Test with current directory
        let result = read_tree(".", 1);
        assert!(matches!(result, FsResult::Ok(_)));
    }

    #[test]
    fn test_read_tree_nonexistent() {
        let result = read_tree("/nonexistent/path/abc123", 1);
        assert!(matches!(result, FsResult::Err(_)));
    }

    #[test]
    fn test_find_file_glob() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(!glob_match("*.rs", "main.py"));
        assert!(glob_match("test_*", "test_hello"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
    }

    #[test]
    fn test_read_write_file() {
        let path = "/tmp/kai_test_fs_agent.txt";
        let result = write_file(path, "hello world");
        assert!(matches!(result, FsResult::Ok(_)));

        let result = read_file(path);
        match result {
            FsResult::Ok(content) => assert_eq!(content, "hello world"),
            _ => panic!("read_file failed"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_diff_identical() {
        let path = "/tmp/kai_test_diff_identical.txt";
        std::fs::write(path, "line1\nline2\n").unwrap();
        let result = diff(path, path);
        match result {
            FsResult::Ok(msg) => assert!(msg.contains("identical")),
            _ => panic!("diff failed"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_diff_different() {
        let p1 = "/tmp/kai_test_diff1.txt";
        let p2 = "/tmp/kai_test_diff2.txt";
        std::fs::write(p1, "line1\nline2\n").unwrap();
        std::fs::write(p2, "line1\nline3\n").unwrap();
        let result = diff(p1, p2);
        match result {
            FsResult::Ok(msg) => {
                assert!(msg.contains("- line2"));
                assert!(msg.contains("+ line3"));
            }
            _ => panic!("diff failed"),
        }
        let _ = std::fs::remove_file(p1);
        let _ = std::fs::remove_file(p2);
    }

    #[test]
    fn test_find_file_in_tmp() {
        let path = "/tmp/kai_test_find_me.rs";
        std::fs::write(path, "// test").unwrap();
        let result = find_file("/tmp", "kai_test_find_me*");
        match result {
            FsResult::Ok(matches) => assert!(matches.contains("kai_test_find_me.rs")),
            _ => panic!("find_file failed"),
        }
        let _ = std::fs::remove_file(path);
    }
}
