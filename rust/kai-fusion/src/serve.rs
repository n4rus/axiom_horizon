//! `kai serve` — OpenAI-compatible API server for Kai-Fusion inference.
//!
//! Phase 3 (World Interaction): Serves the loaded GGUF model as an HTTP API
//! compatible with the OpenAI chat completions format, so opencode and other
//! tools can use Kai as an inference provider directly.
//!
//! Endpoints:
//!   GET  /v1/models              → list available models
//!   POST /v1/chat/completions    → generate completion
//!   GET  /health                 → health check
//!   GET  /v1/                    → Kai-specific status (VFE, attractor, tau)
//!
//! Usage:
//!   kai serve <gguf> [--port 8080] [--attractor <path>]

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server};

/// Default serve port.
const DEFAULT_PORT: u16 = 8080;

/// Max request body size (10 MB).
const MAX_BODY_SIZE: usize = 10_485_760;

/// Server configuration.
pub struct ServeConfig {
    pub port: u16,
    pub model_path: String,
    pub attractor_path: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            port: DEFAULT_PORT,
            model_path: String::new(),
            attractor_path: None,
            max_tokens: 512,
            temperature: 0.7,
        }
    }
}

/// Run the Kai API server. Blocks until interrupted.
pub fn run_server(cfg: &ServeConfig, stop: Arc<AtomicBool>) -> Result<(), String> {
    let addr = format!("0.0.0.0:{}", cfg.port);
    let server = Server::http(&addr)
        .map_err(|e| format!("Failed to start server on {addr}: {e}"))?;

    eprintln!("[serve] Kai-Fusion API server starting...");
    eprintln!("[serve]   model:     {}", cfg.model_path);
    eprintln!("[serve]   endpoint:  http://localhost:{}/v1/chat/completions", cfg.port);
    eprintln!("[serve]   opencode:  configure provider as http://localhost:{}/v1", cfg.port);
    if let Some(ref a) = cfg.attractor_path {
        eprintln!("[serve]   attractor: {a}");
    }
    eprintln!("[serve]   max_tokens: {}", cfg.max_tokens);
    eprintln!("[serve]   temperature: {}", cfg.temperature);
    eprintln!("[serve] Ready. Press Ctrl+C to stop.\n");

    // Pre-load model metadata
    let meta = match crate::gguf::read_kv(&cfg.model_path) {
        Ok(m) => m,
        Err(e) => return Err(format!("failed to read model: {e}")),
    };
    let model_cfg = match crate::gguf::build_config(&meta) {
        Some(c) => c,
        None => return Err("could not build config from model".to_string()),
    };
    let tok = match crate::tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => return Err("no tokenizer in model".to_string()),
    };
    let (_ver, map, _nt, _nk) = match crate::gguf::load_tensors(&cfg.model_path) {
        Ok(x) => x,
        Err(e) => return Err(format!("tensor load: {e}")),
    };
    let model = match crate::model::Weights::from_gguf(&map, &model_cfg) {
        Ok(m) => m,
        Err(e) => return Err(format!("model init: {e}")),
    };

    eprintln!("[serve] Model loaded: {} layers, {} heads, dim={}",
        model_cfg.n_layers, model_cfg.n_heads, model_cfg.dim);
    eprintln!("[serve] Listening on http://0.0.0.0:{}", cfg.port);

    // Main request loop (synchronous — fine for single-user)
    loop {
        if stop.load(Ordering::Relaxed) {
            eprintln!("\n[serve] Shutting down...");
            break;
        }

        match server.recv() {
            Ok(request) => {
                handle_request(request, &model, &model_cfg, &tok, &cfg.model_path,
                              cfg.attractor_path.as_deref(), cfg.temperature, cfg.max_tokens);
            }
            Err(e) => {
                eprintln!("[serve] error: {e}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    Ok(())
}

/// Handle an incoming HTTP request.
fn handle_request(
    request: Request,
    model: &crate::model::Weights,
    model_cfg: &crate::config::Config,
    tok: &crate::tok::Tokenizer,
    model_path: &str,
    attractor_path: Option<&str>,
    default_temp: f32,
    default_max_tokens: usize,
) {
    let url = request.url().to_string();
    let method = request.method();

    // CORS headers for opencode web UI
    let cors_headers = vec![
        Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
        Header::from_bytes(&b"Access-Control-Allow-Methods"[..], &b"GET, POST, OPTIONS"[..]).unwrap(),
        Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"Content-Type, Authorization"[..]).unwrap(),
    ];

    match (method, url.as_str()) {
        // OPTIONS — CORS preflight
        (Method::Options, _) => {
            respond_json(request, 200, r#"{"ok":true}"#, &cors_headers);
        }

        // GET /health — simple health check
        (Method::Get, "/health") => {
            let body = r#"{"status":"ok","service":"kai-fusion"}"#;
            respond_json(request, 200, body, &cors_headers);
        }

        // GET /v1/models — model listing
        (Method::Get, "/v1/models") => {
            let body = format!(r#"{{
                "object": "list",
                "data": [{{
                    "id": "kai",
                    "object": "model",
                    "created": {},
                    "owned_by": "kai-fusion",
                    "permission": []
                }}]
            }}"#, std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs());
            respond_json(request, 200, &body, &cors_headers);
        }

        // POST /v1/chat/completions — main inference endpoint
        (Method::Post, "/v1/chat/completions") | (Method::Post, "/v1/chat/completions/") => {
            handle_chat_completion(request, model, model_cfg, tok, model_path,
                                   attractor_path, default_temp, default_max_tokens, &cors_headers);
        }

        // GET /v1/ — Kai status endpoint
        (Method::Get, "/v1/") => {
            // Return Kai-specific status (VFE, attractor, tau)
            let body = r#"{
                "service": "kai-fusion",
                "version": "0.1.0",
                "features": ["vfe", "attractor", "routing", "bracket", "darwin"],
                "endpoints": {
                    "chat": "/v1/chat/completions",
                    "models": "/v1/models",
                    "health": "/health"
                }
            }"#;
            respond_json(request, 200, body, &cors_headers);
        }

        // 404
        _ => {
            let body = format!(r#"{{"error":"not found","path":"{}","method":"{}"}}"#, url, method);
            respond_json(request, 404, &body, &cors_headers);
        }
    }
}

/// Handle the chat completions endpoint.
fn handle_chat_completion(
    mut request: Request,
    model: &crate::model::Weights,
    model_cfg: &crate::config::Config,
    tok: &crate::tok::Tokenizer,
    _model_path: &str,
    attractor_path: Option<&str>,
    default_temp: f32,
    default_max_tokens: usize,
    cors_headers: &[Header],
) {
    // Read request body
    let mut body = String::new();
    let mut reader = request.as_reader().take(MAX_BODY_SIZE as u64);
    if reader.read_to_string(&mut body).is_err() {
        respond_json(request, 400, r#"{"error":"cannot read request body"}"#, cors_headers);
        return;
    }

    // Parse JSON
    let parsed: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            respond_json(request, 400, &format!(r#"{{"error":"invalid JSON: {e}"}}"#), cors_headers);
            return;
        }
    };

    // Extract messages
    let messages = match parsed.get("messages").and_then(|m| m.as_array()) {
        Some(arr) => arr,
        None => {
            respond_json(request, 400, r#"{"error":"missing 'messages' array"}"#, cors_headers);
            return;
        }
    };

    // Build prompt from messages
    let prompt = messages_to_prompt(messages);
    if prompt.is_empty() {
        respond_json(request, 400, r#"{"error":"empty prompt"}"#, cors_headers);
        return;
    }

    // Extract parameters
    let temperature = parsed.get("temperature")
        .and_then(|t| t.as_f64())
        .map(|t| t as f32)
        .unwrap_or(default_temp);
    let max_tokens = parsed.get("max_tokens")
        .and_then(|t| t.as_u64())
        .map(|t| t as usize)
        .unwrap_or(default_max_tokens);
    let _stream = parsed.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // TODO: streaming support via SSE (future enhancement)

    // Generate completion
    let start = Instant::now();
    let (tokens, generated_text) = generate_completion(model, model_cfg, tok, &prompt, max_tokens, temperature);
    let elapsed = start.elapsed().as_secs_f64();

    // Get VFE stats if available
    let vfe_info = get_vfe_info(attractor_path);

    // Build OpenAI-compatible response
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = format!(r#"{{
        "id": "chatcmpl-{}",
        "object": "chat.completion",
        "created": {},
        "model": "kai",
        "choices": [{{
            "index": 0,
            "message": {{
                "role": "assistant",
                "content": {}
            }},
            "finish_reason": "stop"
        }}],
        "usage": {{
            "prompt_tokens": {},
            "completion_tokens": {},
            "total_tokens": {}
        }},
        "kai_meta": {{
            "inference_time_s": {:.3},
            "tokens_per_sec": {:.1},
            {}
        }}
    }}"#,
        created,
        created,
        serde_json::to_string(&generated_text).unwrap_or_else(|_| "\"\"".to_string()),
        prompt.split_whitespace().count(),            // rough prompt token count
        tokens,                                         // generated tokens
        prompt.split_whitespace().count() + tokens,    // total
        elapsed,
        tokens as f64 / elapsed.max(0.001),
        vfe_info,
    );

    respond_json(request, 200, &response, cors_headers);
}

/// Convert OpenAI message array to a single prompt string.
fn messages_to_prompt(messages: &[serde_json::Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        match role {
            "system" => prompt.push_str(&format!("[System: {}]\n", content)),
            "user" => prompt.push_str(&format!("User: {}\n", content)),
            "assistant" => prompt.push_str(&format!("Assistant: {}\n", content)),
            _ => prompt.push_str(&format!("{}: {}\n", role, content)),
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

/// Generate a completion using the loaded model.
fn generate_completion(
    model: &crate::model::Weights,
    model_cfg: &crate::config::Config,
    tok: &crate::tok::Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> (usize, String) {
    let ids = tok.encode(prompt);
    let mut all_ids = ids.clone();
    let eos = tok.eos;
    let mut generated = 0usize;

    for _step in 0..max_tokens {
        let (logits, _xf) = model.forward(model_cfg, &all_ids);
        let last = logits.row(logits.nrows() - 1);

        let next = if temperature > 0.0 {
            // Temperature-scaled sampling
            let mut scaled: Vec<f32> = last.iter().map(|&x| (x / temperature).exp()).collect();
            let sum: f32 = scaled.iter().sum();
            if sum > 0.0 {
                for v in &mut scaled { *v /= sum; }
                let mut rng = 42u64;
                let mut r = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    rng as f32 / u64::MAX as f32 };
                let p = r();
                let mut cum = 0.0f32;
                let mut selected = 0;
                for (i, &v) in scaled.iter().enumerate() {
                    cum += v;
                    if p <= cum { selected = i; break; }
                }
                selected
            } else { 0 }
        } else {
            // Greedy
            last.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        let next = next.min(model_cfg.vocab_size - 1);
        all_ids.push(next);
        generated += 1;

        if next == eos {
            break;
        }
    }

    // Decode response
    let text = tok.decode(&all_ids[ids.len()..]);
    (generated, text.trim().to_string())
}

/// Get VFE/attractor status info for the response metadata.
fn get_vfe_info(attractor_path: Option<&str>) -> String {
    match attractor_path {
        Some(path) if std::path::Path::new(path).exists() => {
            match crate::attractor::convergence(path, 10) {
                Ok((conv, n)) => {
                    format!(r#""attractor_convergence": {:.4},"attractor_vectors": {}"#, conv, n)
                }
                Err(_) => r#""attractor": "unavailable""#.to_string(),
            }
        }
        _ => r#""attractor": "disabled""#.to_string(),
    }
}

/// Send a JSON response.
fn respond_json(request: Request, status: i32, body: &str, extra_headers: &[Header]) {
    let content_type = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();

    let mut response = Response::from_string(body)
        .with_status_code(status.try_into().unwrap_or(200))
        .with_header(content_type);
    for h in extra_headers {
        response = response.with_header((*h).clone());
    }
    let _ = request.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_config_default() {
        let cfg = ServeConfig::default();
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.max_tokens, 512);
        assert_eq!(cfg.temperature, 0.7);
    }

    #[test]
    fn test_messages_to_prompt_single() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
        ];
        let prompt = messages_to_prompt(&msgs);
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_messages_to_prompt_system() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hi"}),
        ];
        let prompt = messages_to_prompt(&msgs);
        assert!(prompt.contains("[System: You are helpful.]"));
        assert!(prompt.contains("User: Hi"));
    }

    #[test]
    fn test_messages_to_prompt_multi_turn() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "What is 2+2?"}),
            serde_json::json!({"role": "assistant", "content": "4"}),
            serde_json::json!({"role": "user", "content": "Thanks"}),
        ];
        let prompt = messages_to_prompt(&msgs);
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("Assistant: 4"));
        assert!(prompt.contains("User: Thanks"));
    }

    #[test]
    fn test_messages_to_prompt_empty() {
        let msgs = vec![];
        let prompt = messages_to_prompt(&msgs);
        assert_eq!(prompt, "Assistant: ");
    }

    #[test]
    fn test_get_vfe_info_no_attractor() {
        let info = get_vfe_info(None);
        assert!(info.contains("disabled"));
    }

    #[test]
    fn test_get_vfe_info_nonexistent_path() {
        let info = get_vfe_info(Some("/tmp/nonexistent_attractor.json"));
        assert!(info.contains("disabled") || info.contains("unavailable"));
    }
}
