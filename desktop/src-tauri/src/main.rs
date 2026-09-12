//! Axiom Local v0.1 — Tauri shell around the local-first AI stack.
//!
//! Manages Ollama-backed chat plus the two Python services shipped in the
//! repo root (`axiom_mcp_server.py` on :8000, `kai_bridge.py` on :8765).
//! All state is local. Processes spawn via std (no shell plugin needed).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
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
/// Searches recursively: bundle staging may nest files (e.g. _up_/_up_/).
fn repo_root_via(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(res) = app.path().resource_dir() {
        let mut stack = vec![res.clone()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for dir in stack.drain(..) {
                if dir.join("axiom_mcp_server.py").exists() {
                    return Ok(dir);
                }
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    for e in entries.flatten() {
                        if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            next.push(e.path());
                        }
                    }
                }
            }
            stack = next;
            if stack.is_empty() {
                break;
            }
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
fn chat(model: String, prompt: String, app: tauri::AppHandle, procs: State<Procs>) -> Result<String, String> {
    ensure_mcp(&app, procs)?;
    kai_ask(&model, &prompt)
}

/// Real Kai pipeline (no preprogrammed answers): the MCP HTTP endpoint runs
/// AxiomAlien.ask with memory, attractor, VFE and persistent sessions.
/// Desktop never talks to raw Ollama for chat — only through this.
fn kai_ask(model: &str, prompt: &str) -> Result<String, String> {
    let agent: ureq::Agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(300))
        .build();
    let resp = agent
        .post("http://127.0.0.1:8000/v1/chat/completions")
        .send_json(serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }))
        .map_err(|e| format!("kai mcp: {e}"))?;
    let v: serde_json::Value = resp.into_json().map_err(|e| e.to_string())?;
    if let Some(err) = v.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()) {
        return Err(format!("kai mcp: {err}"));
    }
    v.pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .map(str::to_string)
        .ok_or_else(|| "kai mcp: empty reply".to_string())
}

/// Ensure the MCP service is up (start it if needed), then return.
fn ensure_mcp(app: &tauri::AppHandle, procs: State<Procs>) -> Result<(), String> {
    if mcp_alive() {
        return Ok(());
    }
    spawn(app, "mcp", "axiom_mcp_server.py", procs)?;
    for _ in 0..60 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if mcp_alive() {
            return Ok(());
        }
    }
    Err("kai mcp did not come up (check python3 + ollama)".into())
}

fn mcp_alive() -> bool {
    let agent: ureq::Agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    agent
        .get("http://127.0.0.1:8000/v1/models")
        .call()
        .map(|r| r.status() == 200)
        .unwrap_or(false)
}

/// Conversation history from the machine's own memory DB (via MCP).
/// Returns raw rows; the frontend groups them into date sessions.
#[tauri::command]
fn history(app: tauri::AppHandle, procs: State<Procs>) -> Result<serde_json::Value, String> {
    ensure_mcp(&app, procs)?;
    let agent: ureq::Agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build();
    let resp = agent
        .get("http://127.0.0.1:8000/v1/history")
        .call()
        .map_err(|e| format!("kai mcp: {e}"))?;
    resp.into_json().map_err(|e| e.to_string())
}

/// Streaming chat (Android parity): real Kai answer via the MCP pipeline,
/// delivered through the same token events. The MCP endpoint is
/// single-shot, so the full answer arrives as one event — display pacing
/// only, never canned text. Frontend unchanged.
#[tauri::command]
fn chat_stream(window: Window, model: String, prompt: String, app: tauri::AppHandle, procs: State<Procs>) -> Result<(), String> {
    ensure_mcp(&app, procs)?;
    std::thread::spawn(move || {
        let done = (|| -> Result<(), String> {
            let text = kai_ask(&model, &prompt)?;
            let _ = window.emit("chat-token", text);
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
            chat_stream,
            history
        ])
        .run(tauri::generate_context!())
        .expect("axiom-local failed to start");
}
