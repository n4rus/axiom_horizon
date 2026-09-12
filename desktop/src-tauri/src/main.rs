//! Axiom Local v0.1 — Tauri shell around the local-first AI stack.
//!
//! Manages Ollama-backed chat plus the two Python services shipped in the
//! repo root (`axiom_mcp_server.py` on :8000, `kai_bridge.py` on :8765).
//! All state is local. Processes spawn via std (no shell plugin needed).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Emitter, Manager, State, Window};

struct Procs(Mutex<HashMap<String, Child>>);

/// Locate the repo root: $AXIOM_HOME, else walk up from the executable
/// looking for axiom_mcp_server.py (dev layout: desktop/src-tauri/target/..).
fn repo_root() -> Result<PathBuf, String> {
    if let Some(home) = std::env::var_os("AXIOM_HOME") {
        let p = PathBuf::from(home);
        if p.join("axiom_mcp_server.py").exists() {
            return Ok(p);
        }
    }
    let mut dir = std::env::current_exe().map_err(|e| e.to_string())?;
    for _ in 0..6 {
        if !(dir.pop()) {
            break;
        }
        if dir.join("axiom_mcp_server.py").exists() {
            return Ok(dir);
        }
    }
    Err("axiom_mcp_server.py not found — set AXIOM_HOME to the repo root".into())
}

/// Bundled-resource lookup first (installed .deb), then the dev fallbacks.
fn repo_root_via(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(res) = app.path().resource_dir() {
        if res.join("axiom_mcp_server.py").exists() {
            return Ok(res);
        }
    }
    repo_root()
}

fn ollama_models() -> Vec<String> {
    let agent: ureq::Agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(5)).build();
    let resp = agent.get("http://127.0.0.1:11434/api/tags").call();
    let Ok(resp) = resp else { return vec![] };
    let Ok(v) = resp.into_json::<serde_json::Value>() else { return vec![] };
    v.get("models")
        .and_then(|m| m.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[derive(serde::Serialize)]
struct Status {
    ollama: bool,
    models: Vec<String>,
    mcp: bool,
    bridge: bool,
    repo: String,
}

#[tauri::command]
fn get_status(app: tauri::AppHandle, procs: State<Procs>) -> Status {
    let models = ollama_models();
    let map = procs.0.lock().unwrap();
    let alive = |k: &str| map.get(k).is_some();
    Status {
        ollama: !models.is_empty(),
        models,
        mcp: alive("mcp"),
        bridge: alive("bridge"),
        repo: repo_root_via(&app).map(|p| p.display().to_string()).unwrap_or_default(),
    }
}

fn spawn(app: &tauri::AppHandle, key: &str, script: &str, procs: State<Procs>) -> Result<String, String> {
    let root = repo_root_via(app)?;
    let mut map = procs.0.lock().unwrap();
    if map.contains_key(key) {
        return Ok(format!("{key} already running"));
    }
    let child = Command::new("python3")
        .arg(root.join(script))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn {script}: {e}"))?;
    map.insert(key.to_string(), child);
    Ok(format!("{key} started (pid tracked)"))
}

fn stop(key: &str, procs: State<Procs>) -> Result<String, String> {
    let mut map = procs.0.lock().unwrap();
    match map.remove(key) {
        Some(mut child) => {
            child.kill().map_err(|e| e.to_string())?;
            Ok(format!("{key} stopped"))
        }
        None => Ok(format!("{key} was not running")),
    }
}

#[tauri::command]
fn start_mcp(app: tauri::AppHandle, procs: State<Procs>) -> Result<String, String> {
    spawn(&app, "mcp", "axiom_mcp_server.py", procs)
}

#[tauri::command]
fn stop_mcp(procs: State<Procs>) -> Result<String, String> {
    stop("mcp", procs)
}

#[tauri::command]
fn start_bridge(app: tauri::AppHandle, procs: State<Procs>) -> Result<String, String> {
    spawn(&app, "bridge", "kai_bridge.py", procs)
}

#[tauri::command]
fn stop_bridge(procs: State<Procs>) -> Result<String, String> {
    stop("bridge", procs)
}

#[tauri::command]
fn chat(model: String, prompt: String) -> Result<String, String> {    let agent: ureq::Agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(300)).build();
    let resp = agent
        .post("http://127.0.0.1:11434/api/generate")
        .send_json(serde_json::json!({"model": model, "prompt": prompt, "stream": false}))
        .map_err(|e| format!("ollama: {e}"))?;
    let v: serde_json::Value = resp.into_json().map_err(|e| e.to_string())?;
    v.get("response")
        .and_then(|r| r.as_str())
        .map(str::to_string)
        .ok_or_else(|| "no response field".to_string())
}

/// Streaming chat (Android parity): NDJSON tokens forwarded as
/// `chat-token` window events, closed with `chat-done`. Spawns a thread
/// so the UI stays responsive; mirrors StreamingGenerate pacing.
#[tauri::command]
fn chat_stream(window: Window, model: String, prompt: String) -> Result<(), String> {
    std::thread::spawn(move || {
        let agent: ureq::Agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(300))
            .build();
        let body = serde_json::json!({"model": model, "prompt": prompt, "stream": true});
        let done = (|| -> Result<(), String> {
            let resp = agent
                .post("http://127.0.0.1:11434/api/generate")
                .send_json(body)
                .map_err(|e| format!("ollama: {e}"))?;
            let reader = std::io::BufReader::new(resp.into_reader());
            for line in reader.lines().map_while(Result::ok) {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value =
                    serde_json::from_str(&line).map_err(|e| e.to_string())?;
                if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                    break;
                }
                if let Some(tok) = v.get("response").and_then(|r| r.as_str()) {
                    let _ = window.emit("chat-token", tok.to_string());
                }
            }
            Ok(())
        })();
        let _ = window.emit(
            "chat-done",
            done.err().unwrap_or_default(),
        );
    });
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(Procs(Mutex::new(HashMap::new())))
        .invoke_handler(tauri::generate_handler![
            get_status,
            start_mcp,
            stop_mcp,
            start_bridge,
            stop_bridge,
            chat,
            chat_stream
        ])
        .run(tauri::generate_context!())
        .expect("axiom-local failed to start");
}
