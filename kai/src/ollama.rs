//! Ollama API client — GPU-accelerated inference via llama.cpp backend
//!
//! Delegates generation to Ollama's `/api/generate` and `/api/chat` endpoints
//! using `curl` subprocess + `serde_json` (no extra deps).

use serde::Serialize;
use serde_json::Value;
use std::process::Command;

/// A single chat message.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Result of an Ollama generation call.
#[derive(Debug, Clone)]
pub struct OllamaResult {
    pub text: String,
    pub eval_count: usize,
    pub eval_duration_ns: u64,
    pub prompt_eval_count: usize,
    pub prompt_eval_duration_ns: u64,
    pub tokens_per_second: f32,
}

/// Generate text via Ollama's `/api/generate` (non-streaming).
///
/// # Arguments
/// * `model`   — Ollama model name (e.g. `"tinyllama"`, `"qwen2.5-coder:7b"`)
/// * `prompt`  — input text
/// * `max_new` — maximum tokens to generate
/// * `temperature` — sampling temperature (0.0 = greedy)
///
/// Returns `OllamaResult` or an error string.
pub fn generate(
    model: &str,
    prompt: &str,
    max_new: usize,
    temperature: f32,
) -> Result<OllamaResult, String> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": temperature,
            "num_predict": max_new,
        }
    });

    let output = Command::new("curl")
        .args([
            "-s",
            "http://localhost:11434/api/generate",
            "-d",
            &body.to_string(),
        ])
        .output()
        .map_err(|e| format!("curl subprocess failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("curl exited {}: {}", output.status, stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: Value =
        serde_json::from_str(&stdout).map_err(|e| format!("JSON parse error: {}", e))?;

    // Check for Ollama error field
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(format!("Ollama error: {}", err));
    }

    let text = v["response"].as_str().unwrap_or("").to_string();

    let eval_count = v["eval_count"].as_u64().unwrap_or(0) as usize;
    let eval_duration_ns = v["eval_duration"].as_u64().unwrap_or(1).max(1);
    let prompt_eval_count = v["prompt_eval_count"].as_u64().unwrap_or(0) as usize;
    let prompt_eval_duration_ns = v["prompt_eval_duration"].as_u64().unwrap_or(1).max(1);

    let total_tokens = eval_count.max(1);
    let tps = total_tokens as f64 / (eval_duration_ns as f64 / 1_000_000_000.0);

    Ok(OllamaResult {
        text,
        eval_count,
        eval_duration_ns,
        prompt_eval_count,
        prompt_eval_duration_ns,
        tokens_per_second: tps as f32,
    })
}

/// Send a chat message via Ollama's `/api/chat` (multi-turn capable).
///
/// `messages` — full conversation history, e.g.:
///   `[{"role":"system","content":"..."}, {"role":"user","content":"Hello"}]`
pub fn chat(
    model: &str,
    messages: &[Message],
    max_new: usize,
    temperature: f32,
) -> Result<OllamaResult, String> {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "options": {
            "temperature": temperature,
            "num_predict": max_new,
        }
    });

    let output = Command::new("curl")
        .args([
            "-s",
            "http://localhost:11434/api/chat",
            "-d",
            &body.to_string(),
        ])
        .output()
        .map_err(|e| format!("curl subprocess failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("curl exited {}: {}", output.status, stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: Value =
        serde_json::from_str(&stdout).map_err(|e| format!("JSON parse error: {}", e))?;

    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(format!("Ollama error: {}", err));
    }

    let text = v["message"]["content"].as_str().unwrap_or("").to_string();

    let eval_count = v["eval_count"].as_u64().unwrap_or(0) as usize;
    let eval_duration_ns = v["eval_duration"].as_u64().unwrap_or(1).max(1);
    let prompt_eval_count = v["prompt_eval_count"].as_u64().unwrap_or(0) as usize;
    let prompt_eval_duration_ns = v["prompt_eval_duration"].as_u64().unwrap_or(1).max(1);

    let total_tokens = eval_count.max(1);
    let tps = total_tokens as f64 / (eval_duration_ns as f64 / 1_000_000_000.0);

    Ok(OllamaResult {
        text,
        eval_count,
        eval_duration_ns,
        prompt_eval_count,
        prompt_eval_duration_ns,
        tokens_per_second: tps as f32,
    })
}

/// Quick check that Ollama is reachable.
pub fn ping() -> Result<(), String> {
    let output = Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:11434/api/tags"])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let code = String::from_utf8_lossy(&output.stdout);
    if code.trim() == "200" {
        Ok(())
    } else {
        Err(format!("Ollama returned HTTP {}", code.trim()))
    }
}
