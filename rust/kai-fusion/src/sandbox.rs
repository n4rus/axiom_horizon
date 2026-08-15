//! Code execution sandbox — safe, resource-limited execution for self-play and testing.
//!
//! Phase 3 (World Interaction): The sandbox runs arbitrary code (Python, Shell, Rust)
//! in Docker containers with enforced resource limits (30s timeout, 256MB memory, no network).
//! Falls back to direct `std::process::Command` execution when Docker is unavailable.
//!
//! Usage:
//!   kai sandbox exec python "print('hello')"
//!   kai sandbox exec sh "ls -la"
//!   kai sandbox file /tmp/script.py

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Counter for unique temp file names (parallel-test-safe).
static SCRIPT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Default timeout for sandboxed execution (seconds).
const DEFAULT_TIMEOUT_S: u64 = 30;

/// Default memory limit for Docker containers (MB).
const DEFAULT_MEMORY_MB: u64 = 256;

/// Default CPU limit for Docker containers.
const DEFAULT_CPU_SHARES: u64 = 512; // ~0.5 CPU

// ── Result Types ──────────────────────────────────────────────────────────

/// Result of a sandboxed execution.
#[derive(Debug, Clone)]
pub struct SandboxResult {
    /// Standard output (stdout)
    pub stdout: String,
    /// Standard error (stderr)
    pub stderr: String,
    /// Exit code (0 = success)
    pub exit_code: i32,
    /// Wall-clock duration in milliseconds
    pub duration_ms: u64,
    /// Whether the execution was timed out
    pub timed_out: bool,
}

// ── Config ────────────────────────────────────────────────────────────────

/// Configuration for the code execution sandbox.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Timeout in seconds (default: 30)
    pub timeout_s: u64,
    /// Memory limit in MB (default: 256)
    pub memory_mb: u64,
    /// CPU shares (default: 512 ≈ 0.5 cores)
    pub cpu_shares: u64,
    /// Docker image to use (default: "python:3.11-slim")
    pub docker_image: String,
    /// Whether to disable network (default: true)
    pub no_network: bool,
    /// Use Docker if available (default: true)
    pub prefer_docker: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_s: DEFAULT_TIMEOUT_S,
            memory_mb: DEFAULT_MEMORY_MB,
            cpu_shares: DEFAULT_CPU_SHARES,
            docker_image: "python:3.11-slim".to_string(),
            no_network: true,
            prefer_docker: true,
        }
    }
}

// ── Language Detection ────────────────────────────────────────────────────

/// Map a language name to the command used to execute code.
/// Returns (executable, file_extension, docker_image_override).
fn lang_config(language: &str) -> (&'static str, &'static str, &'static str) {
    match language.to_lowercase().as_str() {
        "python" | "py" | "python3" => ("python3", ".py", "python:3.11-slim"),
        "shell" | "sh" | "bash"     => ("bash", ".sh", "ubuntu:22.04"),
        "rust" | "rs"               => ("rustc", ".rs", "rust:1.75-slim"),
        "node" | "js" | "javascript" => ("node", ".js", "node:20-slim"),
        "c"                          => ("gcc", ".c", "gcc:13-bookworm"),
        "cpp" | "c++" | "cxx"       => ("g++", ".cpp", "gcc:13-bookworm"),
        "ruby" | "rb"               => ("ruby", ".rb", "ruby:3.2-slim"),
        "go" | "golang"             => ("go", ".go", "golang:1.21-bookworm"),
        _                            => ("sh", ".sh", "ubuntu:22.04"),
    }
}

/// Detect language from a shebang line.
fn detect_language_from_code(code: &str) -> Option<&'static str> {
    let first_line = code.lines().next()?;
    if first_line.starts_with("#!/") {
        let path = first_line.trim_start_matches("#!/");
        let bin = path.split('/').last().unwrap_or(path);
        // Handle `env python3` case (#!/usr/bin/env python3)
        let words: Vec<&str> = bin.split_whitespace().collect();
        let interpreter = if words.first() == Some(&"env") {
            // After `env`, the next word is the actual interpreter
            words.get(1).unwrap_or(&"env")
        } else {
            words.first().unwrap_or(&"")
        };
        // Map common interpreters
        let lang = match *interpreter {
            "python" | "python3" => "python",
            "bash" | "sh" | "zsh" => "sh",
            "node" => "js",
            "ruby" => "ruby",
            "perl" => "perl",
            _ => return None,
        };
        Some(lang)
    } else {
        None
    }
}

// ── Docker Availability Check ────────────────────────────────────────────

/// Check if Docker is available on the system.
fn docker_available() -> bool {
    Command::new("docker")
        .arg("info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ── Core Execution ────────────────────────────────────────────────────────

/// Execute code in the sandbox.
///
/// `language` can be "python", "sh", "rust", "js", "c", "cpp", "ruby", "go".
/// If `language` is "auto", it will be detected from shebang or file extension.
pub fn exec(code: &str, language: &str) -> SandboxResult {
    let cfg = SandboxConfig::default();
    exec_with_config(code, language, &cfg)
}

/// Execute code with a custom config.
pub fn exec_with_config(code: &str, language: &str, cfg: &SandboxConfig) -> SandboxResult {
    let start = Instant::now();

    // Detect language if "auto"
    let lang = if language == "auto" {
        detect_language_from_code(code).unwrap_or("sh")
    } else {
        language
    };

    let (runner, ext, docker_img) = lang_config(lang);
    // Use configured docker_image override if non-default
    let effective_img = if cfg.docker_image != "python:3.11-slim" {
        &cfg.docker_image
    } else {
        docker_img
    };
    // If Docker is available and preferred, use Docker
    if cfg.prefer_docker && docker_available() {
        exec_docker(code, runner, ext, effective_img, cfg)
    } else {
        exec_direct(code, runner, ext, cfg)
    }
    .map(|mut result| {
        result.duration_ms = start.elapsed().as_millis() as u64;
        result
    })
    .unwrap_or_else(|e| SandboxResult {
        stdout: String::new(),
        stderr: e,
        exit_code: -1,
        duration_ms: start.elapsed().as_millis() as u64,
        timed_out: false,
    })
}

/// Execute code directly via `Command` — uses `wait_with_output` for clean pipe handling,
/// with PID-based kill for timeout.
fn exec_direct(code: &str, runner: &str, ext: &str, cfg: &SandboxConfig) -> Result<SandboxResult, String> {
    let tmp_dir = std::env::temp_dir().join("kai_sandbox");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("mkdir: {e}"))?;

    let counter = SCRIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = format!("script_{}{}", counter, ext);
    let file_path = tmp_dir.join(&file_name);
    std::fs::write(&file_path, code).map_err(|e| format!("write: {e}"))?;

    // Spawn child with piped output
    let child = Command::new(runner)
        .arg(&file_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn: {e}"))?;

    let pid = child.id();
    let timeout = Duration::from_secs(cfg.timeout_s);
    let start = Instant::now();

    // Channel to receive the completed output
    let (tx, rx) = mpsc::channel();

    // Thread: do blocking wait_with_output
    thread::spawn(move || {
        let output = child.wait_with_output();
        let _ = tx.send(output);
    });

    // Main thread: wait with timeout
    match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let exit_code = output.status.code().unwrap_or(-1);
            Ok(SandboxResult {
                stdout,
                stderr,
                exit_code,
                duration_ms: start.elapsed().as_millis() as u64,
                timed_out: false,
            })
        }
        Ok(Err(e)) => Err(format!("exec error: {e}")),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Kill by PID
            let _ = Command::new("kill").args(&["-9", &pid.to_string()]).status();
            // Wait a moment for process to die
            thread::sleep(Duration::from_millis(100));
            Ok(SandboxResult {
                stdout: String::new(),
                stderr: format!("[TIMEOUT] execution exceeded {} seconds", cfg.timeout_s),
                exit_code: -1,
                duration_ms: start.elapsed().as_millis() as u64,
                timed_out: true,
            })
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("child thread disconnected".to_string())
        }
    }
}

/// Execute code inside a Docker container.
fn exec_docker(code: &str, runner: &str, ext: &str, docker_img: &str, cfg: &SandboxConfig) -> Result<SandboxResult, String> {
    let tmp_dir = std::env::temp_dir().join("kai_sandbox");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("mkdir: {e}"))?;

    let counter = SCRIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = format!("script_{}{}", counter, ext);
    let container_path = format!("/tmp/{}", file_name);
    let host_path = tmp_dir.join(&file_name);
    std::fs::write(&host_path, code).map_err(|e| format!("write: {e}"))?;

    // Build docker run args
    let mut args: Vec<String> = vec![
        "run".to_string(),
        "--rm".to_string(),
        "-i".to_string(),
        format!("--memory={}m", cfg.memory_mb),
        format!("--cpu-shares={}", cfg.cpu_shares),
    ];
    if cfg.no_network {
        args.push("--network".to_string());
        args.push("none".to_string());
    }
    args.push("--read-only".to_string());
    args.push("-v".to_string());
    args.push(format!("{}:/tmp/{}:ro", host_path.display(), file_name));
    args.push(docker_img.to_string());
    args.push(runner.to_string());
    args.push(container_path);

    let child = Command::new("docker")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("docker spawn: {e}"))?;

    let pid = child.id();
    let timeout = Duration::from_secs(cfg.timeout_s);
    let start = Instant::now();
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let output = child.wait_with_output();
        let _ = tx.send(output);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let exit_code = output.status.code().unwrap_or(-1);
            Ok(SandboxResult {
                stdout,
                stderr,
                exit_code,
                duration_ms: start.elapsed().as_millis() as u64,
                timed_out: false,
            })
        }
        Ok(Err(e)) => Err(format!("docker exec error: {e}")),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = Command::new("kill").args(&["-9", &pid.to_string()]).status();
            thread::sleep(Duration::from_millis(200));
            Ok(SandboxResult {
                stdout: String::new(),
                stderr: format!("[TIMEOUT] Docker execution exceeded {} seconds", cfg.timeout_s),
                exit_code: -1,
                duration_ms: start.elapsed().as_millis() as u64,
                timed_out: true,
            })
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("docker thread disconnected".to_string())
        }
    }
}

/// Execute a file from disk in the sandbox.
pub fn exec_file(path: &str) -> SandboxResult {
    let code = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            return SandboxResult {
                stdout: String::new(),
                stderr: format!("read file error: {e}"),
                exit_code: -1,
                duration_ms: 0,
                timed_out: false,
            };
        }
    };

    // Detect language from extension
    let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("");
    let language = match ext {
        "py" => "python",
        "sh" | "bash" => "sh",
        "rs" => "rust",
        "js" | "mjs" => "js",
        "c" => "c",
        "cpp" | "cc" | "cxx" => "cpp",
        "rb" => "ruby",
        "go" => "go",
        _ => "auto",
    };

    exec_with_config(&code, language, &SandboxConfig::default())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_config_default() {
        let cfg = SandboxConfig::default();
        assert_eq!(cfg.timeout_s, 30);
        assert_eq!(cfg.memory_mb, 256);
        assert!(cfg.prefer_docker);
        assert!(cfg.no_network);
    }

    #[test]
    fn test_lang_config_python() {
        let (runner, ext, img) = lang_config("python");
        assert_eq!(runner, "python3");
        assert_eq!(ext, ".py");
        assert_eq!(img, "python:3.11-slim");
    }

    #[test]
    fn test_lang_config_shell() {
        let (runner, ext, img) = lang_config("sh");
        assert_eq!(runner, "bash");
        assert_eq!(ext, ".sh");
        assert_eq!(img, "ubuntu:22.04");
    }

    #[test]
    fn test_lang_config_rust() {
        let (runner, ext, img) = lang_config("rust");
        assert_eq!(runner, "rustc");
        assert_eq!(ext, ".rs");
        assert_eq!(img, "rust:1.75-slim");
    }

    #[test]
    fn test_lang_config_unknown_defaults_to_sh() {
        let (runner, ext, img) = lang_config("foobar");
        assert_eq!(runner, "sh");
        assert_eq!(ext, ".sh");
        assert_eq!(img, "ubuntu:22.04");
    }

    #[test]
    fn test_detect_language_from_shebang_python() {
        let code = "#!/usr/bin/env python3\nprint('hello')";
        assert_eq!(detect_language_from_code(code), Some("python"));
    }

    #[test]
    fn test_detect_language_from_shebang_bash() {
        let code = "#!/bin/bash\necho hello";
        assert_eq!(detect_language_from_code(code), Some("sh"));
    }

    #[test]
    fn test_detect_language_from_shebang_node() {
        let code = "#!/usr/bin/node\nconsole.log('hi')";
        assert_eq!(detect_language_from_code(code), Some("js"));
    }

    #[test]
    fn test_detect_language_no_shebang() {
        let code = "print('hello')";
        assert_eq!(detect_language_from_code(code), None);
    }

    #[test]
    fn test_detect_language_empty() {
        let code = "";
        assert_eq!(detect_language_from_code(code), None);
    }

    #[test]
    fn test_sandbox_result_creation() {
        let r = SandboxResult {
            stdout: "hello".to_string(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 42,
            timed_out: false,
        };
        assert_eq!(r.stdout, "hello");
        assert_eq!(r.exit_code, 0);
        assert_eq!(r.duration_ms, 42);
        assert!(!r.timed_out);
    }

    #[test]
    fn test_exec_direct_python_hello() {
        // Direct execution of a simple Python one-liner (no Docker)
        let cfg = SandboxConfig {
            prefer_docker: false,
            ..SandboxConfig::default()
        };
        // Only run if python3 is available
        if Command::new("python3").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok() {
            let result = exec_with_config("print('hello from kai')", "python", &cfg);
            assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
            assert!(result.stdout.contains("hello from kai"), "stdout: {}", result.stdout);
        }
    }

    #[test]
    fn test_exec_direct_shell_echo() {
        let cfg = SandboxConfig {
            prefer_docker: false,
            ..SandboxConfig::default()
        };
        if Command::new("bash").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok() {
            let result = exec_with_config("echo 'sandbox works'", "sh", &cfg);
            assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
            assert!(result.stdout.contains("sandbox works"), "stdout: {}", result.stdout);
        }
    }

    #[test]
    fn test_exec_direct_timeout() {
        let cfg = SandboxConfig {
            prefer_docker: false,
            timeout_s: 1, // 1 second timeout
            ..SandboxConfig::default()
        };
        // Sleep for 5 seconds — should be killed after 1s
        let result = exec_with_config("import time; time.sleep(5)", "python", &cfg);
        assert!(result.timed_out, "should have timed out: {:?}", result);
        assert!(result.stderr.contains("TIMEOUT"), "stderr: {}", result.stderr);
    }

    #[test]
    fn test_exec_file_not_found() {
        let result = exec_file("/tmp/nonexistent_script_42.sh");
        assert_eq!(result.exit_code, -1);
        assert!(result.stderr.contains("read file error"));
    }

    #[test]
    fn test_sandbox_result_defaults() {
        let r = SandboxResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            duration_ms: 0,
            timed_out: false,
        };
        assert_eq!(r.stdout, "");
        assert_eq!(r.stderr, "");
        assert_eq!(r.exit_code, 0);
    }
}
