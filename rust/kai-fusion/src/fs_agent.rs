//! File System agent: Rust-native FS operations for opencode integration.
//!
//! Provides `kai fs` CLI with read/write/glob/grep/list/stat/edit/rm/mkdir/find/cat commands.

use std::fs;
use std::path::Path;

/// Execute an FS command. Returns (output_string, exit_code).
pub fn fs_cmd(cmd: &str, args: &[&str]) -> (String, i32) {
    match cmd {
        "read" | "cat" => cmd_read(args),
        "write" => cmd_write(args),
        "append" => cmd_append(args),
        "glob" => cmd_glob(args),
        "grep" => cmd_grep(args),
        "list" | "ls" => cmd_list(args),
        "stat" => cmd_stat(args),
        "edit" => cmd_edit(args),
        "rm" => cmd_rm(args),
        "mkdir" => cmd_mkdir(args),
        "find" => cmd_find(args),
        "help" | "--help" | "-h" => cmd_help(),
        _ => cmd_help(),
    }
}

fn cmd_help() -> (String, i32) {
    (r#"kai fs <command> [args]

Commands:
  read <path>              — read file contents
  cat <path>               — same as read (concatenate)
  write <path> <text>      — write text to file (overwrites)
  append <path> <text>     — append text to file
  edit <path> <old> <new>  — replace first occurrence in-place
  glob <pattern>           — list files matching glob pattern (e.g. "*.rs")
  find <dir> <pattern>     — recursively find files containing pattern in name
  grep <pattern> [path]    — search text in file or directory recursively
  list [dir]               — list directory contents
  stat <path>              — file/directory metadata
  rm <path>                — remove file or empty dir (use rmdir for dirs)
  rmdir <path>             — remove empty directory
  mkdir <path>             — create directory (with parents)
  help                     — show this help"#.to_string(), 0)
}

fn cmd_read(args: &[&str]) -> (String, i32) {
    if args.is_empty() {
        return ("Usage: kai fs read <path>".to_string(), 1);
    }
    match fs::read_to_string(args[0]) {
        Ok(content) => (content, 0),
        Err(e) => (format!("Error reading {}: {e}", args[0]), 1),
    }
}

fn cmd_write(args: &[&str]) -> (String, i32) {
    if args.len() < 2 {
        return ("Usage: kai fs write <path> <text>".to_string(), 1);
    }
    let text = args[1..].join(" ");
    match fs::write(args[0], text) {
        Ok(()) => (format!("Wrote to {}", args[0]), 0),
        Err(e) => (format!("Error writing {}: {e}", args[0]), 1),
    }
}

fn cmd_append(args: &[&str]) -> (String, i32) {
    if args.len() < 2 {
        return ("Usage: kai fs append <path> <text>".to_string(), 1);
    }
    let text = args[1..].join(" ");
    use std::io::Write;
    let mut f = match fs::OpenOptions::new().create(true).append(true).open(args[0]) {
        Ok(f) => f,
        Err(e) => return (format!("Error opening {}: {e}", args[0]), 1),
    };
    match writeln!(f, "{}", text) {
        Ok(()) => (format!("Appended to {}", args[0]), 0),
        Err(e) => (format!("Error appending to {}: {e}", args[0]), 1),
    }
}

fn cmd_glob(args: &[&str]) -> (String, i32) {
    if args.is_empty() {
        return ("Usage: kai fs glob <pattern>".to_string(), 1);
    }
    let pattern = args[0];
    let paths = match glob::glob(pattern) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("glob error: {e}");
            return (format!("Invalid glob pattern '{pattern}'"), 1);
        }
    };
    let mut results: Vec<String> = Vec::new();
    for entry in paths {
        match entry {
            Ok(p) => results.push(p.display().to_string()),
            Err(e) => eprintln!("glob entry error: {e}"),
        }
    }
    if results.is_empty() {
        (format!("No files matching '{pattern}'"), 0)
    } else {
        results.sort();
        let output = results.join("\n");
        (format!("{output}\n({} file{})", results.len(), if results.len() == 1 { "" } else { "s" }), 0)
    }
}

fn cmd_find(args: &[&str]) -> (String, i32) {
    if args.len() < 2 {
        return ("Usage: kai fs find <dir> <pattern>".to_string(), 1);
    }
    let dir = Path::new(args[0]);
    let pattern = args[1];
    if !dir.exists() {
        return (format!("Directory not found: {}", args[0]), 1);
    }
    let mut results: Vec<String> = Vec::new();
    find_recursive(dir, pattern, &mut results);
    if results.is_empty() {
        (format!("No files matching '{pattern}' in {}", args[0]), 0)
    } else {
        results.sort();
        let output = results.join("\n");
        (format!("{output}\n({} file{})", results.len(), if results.len() == 1 { "" } else { "s" }), 0)
    }
}

fn find_recursive(dir: &Path, pattern: &str, results: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.contains(pattern) {
                results.push(path.display().to_string());
            }
            if path.is_dir() {
                find_recursive(&path, pattern, results);
            }
        }
    }
}

fn cmd_grep(args: &[&str]) -> (String, i32) {
    if args.is_empty() {
        return ("Usage: kai fs grep <pattern> [file_or_dir]".to_string(), 1);
    }
    let pattern = args[0];
    let target = args.get(1).copied().unwrap_or(".");
    let path = Path::new(target);

    let mut output = String::new();
    let mut count = 0usize;

    if path.is_file() {
        if let Ok(content) = fs::read_to_string(path) {
            for (line_no, line) in content.lines().enumerate() {
                if line.contains(pattern) {
                    output.push_str(&format!("{}:{}: {}\n", path.display(), line_no + 1, line));
                    count += 1;
                }
            }
        }
    } else if path.is_dir() {
        collect_grep(path, pattern, &mut output, &mut count);
    } else {
        return (format!("Not found: {target}"), 1);
    }

    if count == 0 {
        (format!("No matches for '{pattern}' in {target}"), 0)
    } else {
        let plural = if count == 1 { "" } else { "es" };
        (format!("{output}({count} match{plural})"), 0)
    }
}

fn collect_grep(dir: &Path, pattern: &str, output: &mut String, count: &mut usize) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Ok(content) = fs::read_to_string(&path) {
                    for (line_no, line) in content.lines().enumerate() {
                        if line.contains(pattern) {
                            output.push_str(&format!("{}:{}: {}\n", path.display(), line_no + 1, line));
                            *count += 1;
                        }
                    }
                }
            } else if path.is_dir() {
                collect_grep(&path, pattern, output, count);
            }
        }
    }
}

fn cmd_list(args: &[&str]) -> (String, i32) {
    let target = args.first().copied().unwrap_or(".");
    let path = Path::new(target);
    if !path.exists() {
        return (format!("Not found: {target}"), 1);
    }
    if !path.is_dir() {
        return cmd_stat(args);
    }
    let mut entries: Vec<String> = Vec::new();
    if let Ok(read_dir) = fs::read_dir(path) {
        for entry in read_dir.flatten() {
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let name = entry.file_name().to_string_lossy().to_string();
            let kind = if meta.is_dir() { "DIR" } else { "FILE" };
            let size = meta.len();
            entries.push(format!("{kind:>4} {size:>10} {name}"));
        }
    }
    entries.sort();
    if entries.is_empty() {
        (format!("(empty directory: {target})"), 0)
    } else {
        (format!("{}\n({} entries)", entries.join("\n"), entries.len()), 0)
    }
}

fn cmd_stat(args: &[&str]) -> (String, i32) {
    if args.is_empty() {
        return ("Usage: kai fs stat <path>".to_string(), 1);
    }
    let path = Path::new(args[0]);
    match fs::metadata(path) {
        Ok(meta) => {
            let kind = if meta.is_dir() { "directory" } else { "file" };
            let size = meta.len();
            let modified = meta.modified().map(|t| format!("{t:?}")).unwrap_or_else(|_| "unknown".to_string());
            (format!("{kind}: {}\n  size: {} bytes\n  modified: {modified}", args[0], size), 0)
        }
        Err(e) => (format!("Error: {e}"), 1),
    }
}

fn cmd_edit(args: &[&str]) -> (String, i32) {
    if args.len() < 3 {
        return ("Usage: kai fs edit <path> <old_text> <new_text>".to_string(), 1);
    }
    let path = Path::new(args[0]);
    let old_text = args[1];
    let new_text = args[2];

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return (format!("Error reading {}: {e}", args[0]), 1),
    };

    let new_content = content.replacen(old_text, new_text, 1);
    if new_content == content {
        return (format!("No occurrences of '{}' found in {}", old_text, args[0]), 1);
    }

    match fs::write(path, new_content) {
        Ok(()) => (format!("Replaced first occurrence in {}", args[0]), 0),
        Err(e) => (format!("Error writing {}: {e}", args[0]), 1),
    }
}

fn cmd_rm(args: &[&str]) -> (String, i32) {
    if args.is_empty() {
        return ("Usage: kai fs rm <path>".to_string(), 1);
    }
    let path = Path::new(args[0]);
    if path.is_dir() {
        match fs::remove_dir(path) {
            Ok(()) => (format!("Removed empty dir {}", args[0]), 0),
            Err(e) => (format!("Error removing {}: {e} (use rmdir for non-empty or fs rm -r <path>)", args[0]), 1),
        }
    } else {
        match fs::remove_file(path) {
            Ok(()) => (format!("Removed file {}", args[0]), 0),
            Err(e) => (format!("Error removing {}: {e}", args[0]), 1),
        }
    }
}

fn cmd_mkdir(args: &[&str]) -> (String, i32) {
    if args.is_empty() {
        return ("Usage: kai fs mkdir <path>".to_string(), 1);
    }
    match fs::create_dir_all(Path::new(args[0])) {
        Ok(()) => (format!("Created directory {}", args[0]), 0),
        Err(e) => (format!("Error creating {}: {e}", args[0]), 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> String {
    let pid = std::process::id();
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let d = format!("/tmp/kai_fs_test_{}_{}", pid, n);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

    #[test]
    fn test_fs_read_missing() {
        let (out, code) = fs_cmd("read", &["/tmp/nonexistent_file_xyz.txt"]);
        assert_ne!(code, 0);
        assert!(out.contains("Error reading"));
    }

    #[test]
    fn test_fs_write_read() {
        let d = temp_dir();
        let path = format!("{d}/test.txt");
        let (wout, wcode) = fs_cmd("write", &[&path, "hello world"]);
        assert_eq!(wcode, 0);
        assert!(wout.contains("Wrote"));

        let (rout, rcode) = fs_cmd("read", &[&path]);
        assert_eq!(rcode, 0);
        assert_eq!(rout.trim(), "hello world");
    }

    #[test]
    fn test_fs_append() {
        let d = temp_dir();
        let path = format!("{d}/append_test.txt");
        let _ = fs_cmd("write", &[&path, "hello"]);
        let (_aout, acode) = fs_cmd("append", &[&path, " world"]);
        assert_eq!(acode, 0);
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content.trim(), "hello world");
    }

    #[test]
    fn test_fs_glob() {
        let d = temp_dir();
        let _ = fs::write(format!("{d}/a.txt"), "");
        let _ = fs::write(format!("{d}/b.rs"), "");
        let (out, _code) = fs_cmd("glob", &[&format!("{d}/*.txt")]);
        assert!(out.contains("a.txt"));
        assert!(!out.contains("b.rs"));
    }

    #[test]
    fn test_fs_find() {
        let d = temp_dir();
        let sub = format!("{d}/sub");
        fs::create_dir_all(&sub).unwrap();
        let _ = fs::write(format!("{sub}/found.rs"), "");
        let _ = fs::write(format!("{d}/other.txt"), "");
        let (out, _code) = fs_cmd("find", &[&d, "found"]);
        assert!(out.contains("found.rs"));
        assert!(!out.contains("other.txt"));
    }

    #[test]
    fn test_fs_grep() {
        let d = temp_dir();
        let path = format!("{d}/test.txt");
        let _ = fs::write(&path, "line one\nhello world\nanother line\nhello again\n");
        let (out, _code) = fs_cmd("grep", &["hello", &path]);
        assert!(out.contains("hello world"));
        assert!(out.contains("hello again"));
    }

    #[test]
    fn test_fs_list() {
        let d = temp_dir();
        let _ = fs::write(format!("{d}/a.txt"), "");
        let _ = fs::create_dir_all(format!("{d}/subdir"));
        let (out, _code) = fs_cmd("list", &[&d]);
        assert!(out.contains("a.txt"));
        assert!(out.contains("subdir"));
    }

    #[test]
    fn test_fs_stat() {
        let d = temp_dir();
        let path = format!("{d}/stat_test.txt");
        fs::write(&path, "content").unwrap();
        let (out, _code) = fs_cmd("stat", &[&path]);
        assert!(out.contains("file"));
        assert!(out.contains("7 bytes"));
    }

    #[test]
    fn test_fs_edit() {
        let d = temp_dir();
        let path = format!("{d}/edit_test.txt");
        fs::write(&path, "hello world").unwrap();
        let (out, code) = fs_cmd("edit", &[&path, "world", "rust"]);
        assert_eq!(code, 0);
        assert!(out.contains("Replaced"));
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello rust");
    }

    #[test]
    fn test_fs_rm_file() {
        let d = temp_dir();
        let path = format!("{d}/rm_test.txt");
        fs::write(&path, "temp").unwrap();
        let (out, code) = fs_cmd("rm", &[&path]);
        assert_eq!(code, 0);
        assert!(out.contains("Removed file"));
        assert!(!Path::new(&path).exists());
    }

    #[test]
    fn test_fs_mkdir() {
        let d = temp_dir();
        let path = format!("{d}/new_dir");
        let (out, code) = fs_cmd("mkdir", &[&path]);
        assert_eq!(code, 0);
        assert!(out.contains("Created directory"));
        assert!(Path::new(&path).is_dir());
    }
}
