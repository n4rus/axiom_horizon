//! # kai-standalone
//!
//! The "standalone Kai AGI" milestone: a pure-Rust binary that ingests the live
//! Kai checkpoint and runs the autonomous learning loop (world-model forward
//! pass, variational free-energy, attractor update, self-supervised training)
//! entirely in Rust via `kai-core` — **no Python orchestration**.
//!
//! **Local-model ingestion (all of them):**
//! - *Ollama* — enumerate every local model, route general reasoning to a chat
//!   model and embeddings to `nomic-embed-text`; the attractor is grounded by
//!   real local-model output.
//! - *opencode* — a coding/agent loop: generate code with the local coder model,
//!   compile-check it locally (no execution), and ingest the verification result
//!   as a learning signal. The mind thus operates as **both** reasoning (Ollama)
//!   and coding (opencode) together — the unified, mutated standalone code.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use kai_core::{compute_vfe, cosine_similarity, predict, train_step, WorldModelParams};
use kai_mlir::{AttributeSet, DialectRegistry, Value};
use ndarray::{Array1, Array2};
use physics_dialect::KaiPhysicsDialect;
use serde::Deserialize;
use serde_json::json;
use serde_json::Value as JsonValue;

const OLLAMA: &str = "http://localhost:11434";
const EMBED_MODEL: &str = "nomic-embed-text";

/// Topics for the Ollama reasoning/grounding loop.
const REASON_CURRICULUM: &[&str] = &[
    "How does attention geometry (g_ij = 1 - a_ij) encode learned structure?",
    "Why does minimizing variational free energy maximize model evidence?",
    "How can a rolling attractor keep long-horizon context without unbounded growth?",
    "What is the relationship between tonal collapse and a fixed point?",
    "How does time dilation (tau) reflect processing speed under load?",
    "In what sense is self-supervised world-model training a reducible learning signal?",
    "How can deeper multi-step reasoning improve active-inference decisions?",
    "What distinguishes exploration from consolidation in an autonomy loop?",
];

/// Coding tasks for the opencode-style generate+verify loop.
const CODE_CURRICULUM: &[&str] = &[
    "Write a Rust function `cosine_similarity(a: &[f32], b: &[f32]) -> f32` using only std.",
    "Write a Python function returning the running variance of a list of floats.",
    "Write a Rust struct for a 2-layer world model with a tanh hidden layer and a `predict` method.",
    "Write a Python function that lowercases text and removes punctuation using only str methods.",
];

/// Philosophy / logic corpus Kai ingests as grounding knowledge (local, offline).
const PHILOSOPHY_CURRICULUM: &[&str] = &[
    "Logic: valid inference preserves truth from premises to conclusion; modus ponens is its simplest law.",
    "Hegel's dialectic: a thesis and its antithesis resolve into a synthesis that becomes a new thesis.",
    "Spinoza: mind and body are two attributes of one substance; thought and extension are one.",
    "Russell and Whitehead: mathematics reduces to logic via type theory and the theory of descriptions.",
    "Kant: the categories of understanding structure experience; the noumenon lies beyond phenomena.",
    "Godel: any consistent formal system rich enough for arithmetic contains true but undecidable statements.",
];

#[derive(Deserialize)]
struct WmJson {
    #[serde(rename = "W1")]
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    #[serde(rename = "W2")]
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    wm_steps: f64,
    #[allow(dead_code)]
    hidden: usize,
}

#[derive(Deserialize)]
struct AttractorJson {
    vecs: Vec<Vec<f32>>,
    #[serde(default)]
    #[allow(dead_code)]
    labels: Vec<String>,
}

/// Which local Ollama models to use for each role (resolved against what's installed).
struct Models {
    chat: String,
    coder: String,
    embed: String,
}

fn to_array2(m: &[Vec<f32>]) -> Array2<f32> {
    let rows = m.len();
    let cols = m.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f32> = m.iter().flatten().cloned().collect();
    Array2::from_shape_vec((rows, cols), flat).expect("W matrix shape")
}

fn variance(v: &Array1<f32>) -> f32 {
    let mean = v.mean().unwrap_or(0.0);
    v.mapv(|x| (x - mean) * (x - mean)).mean().unwrap_or(0.0)
}

/// Ingest every local Ollama model name (proves we see *all* local models).
fn ollama_tags() -> Vec<String> {
    let resp = ureq::get(&format!("{OLLAMA}/api/tags"))
        .timeout(Duration::from_secs(10))
        .call();
    let resp = match resp {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let v: serde_json::Value = match resp.into_json::<serde_json::Value>() {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    v.get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Pick the best available model for a role from the locally installed set.
fn resolve_models(tags: &[String]) -> Models {
    let has = |name: &str| tags.iter().any(|t| t == name || t.starts_with(name));
    let chat = if has("qwen2.5:7b") {
        "qwen2.5:7b"
    } else if has("llama3.2") {
        "llama3.2"
    } else {
        tags.first().map(|s| s.as_str()).unwrap_or("qwen2.5:7b")
    };
    let coder = if has("qwen3-coder:latest") {
        "qwen3-coder:latest"
    } else if has("deepseek-coder-v2:16b") {
        "deepseek-coder-v2:16b"
    } else if has("qwen2.5-coder:7b") {
        "qwen2.5-coder:7b"
    } else {
        chat
    };
    let embed = EMBED_MODEL;
    Models {
        chat: chat.to_string(),
        coder: coder.to_string(),
        embed: embed.to_string(),
    }
}

fn chat(model: &str, system: &str, user: &str) -> Option<String> {
    let body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": false,
    })
    .to_string();
    let resp = ureq::post(&format!("{OLLAMA}/api/chat"))
        .timeout(Duration::from_secs(60))
        .send_string(&body)
        .ok()?
        .into_json::<serde_json::Value>()
        .ok()?;
    resp.get("message")?.get("content")?.as_str().map(|s| s.trim().to_string())
}

fn embed(model: &str, text: &str) -> Option<Array1<f32>> {
    let body = json!({"model": model, "prompt": text}).to_string();
    let resp = ureq::post(&format!("{OLLAMA}/api/embeddings"))
        .timeout(Duration::from_secs(30))
        .send_string(&body)
        .ok()?
        .into_json::<serde_json::Value>()
        .ok()?;
    let arr = resp.get("embedding")?.as_array()?;
    let v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
    if v.len() == 768 {
        Some(Array1::from(v))
    } else {
        None
    }
}

/// Extract a fenced code block (```lang ... ```), preferring rust/python. If the
/// model returns code without a fence, fall back to the whole response when it
/// clearly looks like source.
fn extract_code(text: &str) -> Option<(String, &'static str)> {
    for lang in ["rust", "python", "rs", "py"] {
        let fence = format!("```{lang}");
        if let Some(start) = text.find(&fence) {
            let after = start + fence.len();
            let rest = &text[after..];
            if let Some(end) = rest.find("```") {
                let code = rest[..end].trim().to_string();
                let lang = if lang.starts_with("rust") || lang == "rs" {
                    "rust"
                } else {
                    "python"
                };
                return Some((code, lang));
            }
        }
    }
    let looks_code = text.contains("fn ")
        || text.contains("def ")
        || text.contains("impl ")
        || text.contains("class ")
        || text.contains("func ")
        || text.contains("=>");
    if looks_code {
        let lang = if text.contains("def ")
            || text.contains("print(")
            || text.contains("import ")
        {
            "python"
        } else {
            "rust"
        };
        Some((text.trim().to_string(), lang))
    } else {
        None
    }
}

/// opencode-style: compile-check generated code locally (NO execution) and return
/// (ok, detail). Safe — only type-checks; never runs model-generated code.
fn verify_code(src: &str, lang: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join("kai_standalone_code");
    let _ = fs::create_dir_all(&dir);
    if lang == "rust" {
        let path = dir.join("snippet.rs");
        if fs::write(&path, src).is_err() {
            return (false, "write failed".into());
        }
        let out = Command::new("rustc")
            .args([
                "--edition", "2021", "--crate-type", "lib",
                "-o", "/dev/null", "--emit", "metadata",
                path.to_str().unwrap(),
            ])
            .output();
        match out {
            Ok(o) => {
                let ok = o.status.success();
                let detail = String::from_utf8_lossy(&o.stderr).lines().take(3).collect::<Vec<_>>().join(" ");
                (ok, if ok { "compiles".into() } else { detail })
            }
            Err(e) => (false, format!("rustc unavailable: {e}")),
        }
    } else {
        let path = dir.join("snippet.py");
        if fs::write(&path, src).is_err() {
            return (false, "write failed".into());
        }
        let out = Command::new("python3")
            .args(["-m", "py_compile", path.to_str().unwrap()])
            .output();
        match out {
            Ok(o) => {
                let ok = o.status.success();
                let detail = String::from_utf8_lossy(&o.stderr).lines().take(3).collect::<Vec<_>>().join(" ");
                (ok, if ok { "compiles".into() } else { detail })
            }
            Err(e) => (false, format!("python3 unavailable: {e}")),
        }
    }
}

/// Run the world-model forward pass *through the IR* (`kai_physics.world_model_physics`).
/// The IR op uses f64; we convert the f32 weights/activations on the boundary.
fn predict_via_ir(
    reg: &DialectRegistry,
    state: &[f32],
    action: &[f32],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
) -> Vec<f32> {
    let operands = [
        Value::tensor(state.iter().map(|x| *x as f64).collect()),
        Value::tensor(action.iter().map(|x| *x as f64).collect()),
        Value::tensor(w1.to_vec()),
        Value::tensor(b1.to_vec()),
        Value::tensor(w2.to_vec()),
        Value::tensor(b2.to_vec()),
    ];
    let out = reg
        .evaluate("kai_physics", "world_model_physics", &operands, &AttributeSet::new())
        .expect("ir world_model_physics");
    out.as_tensor().unwrap().iter().map(|x| *x as f32).collect()
}

/// Compute variational free energy *through the IR* (`kai_physics.calculate_vfe`).
fn vfe_via_ir(
    reg: &DialectRegistry,
    pred: &[f32],
    actual: &[f32],
    variance: f32,
    novelty: f32,
) -> f32 {
    let operands = [
        Value::tensor(pred.iter().map(|x| *x as f64).collect()),
        Value::tensor(actual.iter().map(|x| *x as f64).collect()),
        Value::Scalar(variance as f64),
        Value::Scalar(novelty as f64),
    ];
    let out = reg
        .evaluate("kai_physics", "calculate_vfe", &operands, &AttributeSet::new())
        .expect("ir calculate_vfe");
    out.as_scalar().unwrap() as f32
}

/// Lower the loaded world-model graph to upstream MLIR and verify it with the
/// installed `mlir-opt` (LLVM 23). The `.mlir` file is written regardless so the
/// lowering is inspectable; execution is still done in Rust (no mlir-cpu-runner).
fn lower_and_verify_mlir(input_dim: usize, hidden: usize, output_dim: usize) {
    let mlir = physics_dialect::emit_world_model_mlir(input_dim, hidden, output_dim);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.axiom_state/kai_world_model.mlir");
    if std::fs::write(&path, &mlir).is_ok() {
        eprintln!("  [mlir] wrote upstream MLIR -> {}", path.display());
    }
    match Command::new("mlir-opt")
        .arg("-o")
        .arg("/dev/null")
        .stdin(Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(mut child) => {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(mlir.as_bytes());
            }
            match child.wait() {
                Ok(s) if s.success() => eprintln!("  [mlir] upstream MLIR verified by mlir-opt (LLVM 23)"),
                Ok(s) => eprintln!("  [mlir] mlir-opt exited {s}"),
                Err(e) => eprintln!("  [mlir] mlir-opt wait error: {e}"),
            }
        }
        Err(e) => eprintln!(
            "  [mlir] mlir-opt unavailable: {e} (lowering still emitted to {})",
            path.display()
        ),
    }
}

/// Embed a text and fold it into the attractor as an ingested knowledge signal.
fn ingest_text(label: &str, text: &str, models: &Models, vecs: &mut Vec<Array1<f32>>) {
    if let Some(ev) = embed(&models.embed, text) {
        vecs.push(ev);
        eprintln!(
            "  [ingest:{}] +{} chars embedded (|V|={})",
            label,
            text.len(),
            vecs.len()
        );
    } else {
        eprintln!("  [ingest:{}] embed failed (skipped)", label);
    }
}

/// Ingest the philosophy/logic curriculum (and any `PHILOSOPHY_DIR` texts).
fn ingest_philosophy(models: &Models, vecs: &mut Vec<Array1<f32>>) {
    for (i, t) in PHILOSOPHY_CURRICULUM.iter().take(4).enumerate() {
        ingest_text(&format!("philosophy{i}"), t, models, vecs);
    }
    if let Ok(dir) = std::env::var("PHILOSOPHY_DIR") {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
                    if ext == "txt" || ext == "md" {
                        if let Ok(txt) = std::fs::read_to_string(&p) {
                            let lbl = format!(
                                "phil:{}",
                                p.file_name().unwrap_or_default().to_string_lossy()
                            );
                            ingest_text(&lbl, &txt, models, vecs);
                        }
                    }
                }
            }
        }
    }
}

/// Ingest MATLAB `.mat` files (local knowledge DBs). Reads each via the pure-Python
/// `tools/mat_reader.py` and embeds a summary of its variables into the attractor.
fn ingest_matlab(models: &Models, vecs: &mut Vec<Array1<f32>>) {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");
    let mut dirs = vec![
        repo.join(".axiom_state/matladb"),
        repo.join("content/matlab"),
    ];
    if let Ok(d) = std::env::var("MATLDB_DIR") {
        dirs.push(PathBuf::from(d));
    }
    let reader = repo.join("tools/mat_reader.py");

    let mut files = Vec::new();
    for d in &dirs {
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("mat") {
                    files.push(p);
                }
            }
        }
    }
    if files.is_empty() {
        eprintln!("  [ingest:matlab] no .mat files found (dirs: {dirs:?}); skipping");
        return;
    }
    for f in &files {
        let out = Command::new("python3")
            .arg(&reader)
            .arg(f)
            .output();
        match out {
            Ok(o) if o.status.success() => {
                let txt = String::from_utf8_lossy(&o.stdout);
                match serde_json::from_str::<JsonValue>(&txt) {
                    Ok(json) => {
                        if let Some(vars) = json
                            .get(0)
                            .and_then(|j| j.get("vars"))
                            .and_then(|v| v.as_array())
                        {
                            let mut desc =
                                format!("MATLAB data from {}: ", f.file_name().unwrap_or_default().to_string_lossy());
                            for var in vars {
                                if let Some(err) = var.get("error") {
                                    desc.push_str(&format!("[error: {err}] "));
                                } else if let Some(val) = var.get("value") {
                                    desc.push_str(&format!(
                                        "[{}: {:?}] ",
                                        var.get("name").unwrap_or(&JsonValue::Null),
                                        val
                                    ));
                                } else if let Some(st) = var.get("stats") {
                                    desc.push_str(&format!(
                                        "[{} shape={:?} min={} max={} mean={:.3}] ",
                                        var.get("name").unwrap_or(&JsonValue::Null),
                                        var.get("shape"),
                                        st.get("min").unwrap_or(&JsonValue::Null),
                                        st.get("max").unwrap_or(&JsonValue::Null),
                                        st.get("mean").unwrap_or(&JsonValue::Null),
                                    ));
                                }
                            }
                            ingest_text("matlab", &desc, models, vecs);
                        }
                    }
                    Err(e) => eprintln!("  [ingest:matlab] parse error {}: {}", f.display(), e),
                }
            }
            Ok(o) => eprintln!(
                "  [ingest:matlab] reader failed {}: {}",
                f.display(),
                String::from_utf8_lossy(&o.stderr).lines().take(1).collect::<Vec<_>>().join(" ")
            ),
            Err(e) => eprintln!("  [ingest:matlab] cannot run reader: {e}"),
        }
    }
}

/// Ingest *every* local Ollama model: probe each with a short prompt and fold its
/// reply into the attractor, so Kai is grounded by the full local model ensemble.
fn ingest_all_models(tags: &[String], models: &Models, vecs: &mut Vec<Array1<f32>>) {
    let probe = "Identify yourself and your single strongest capability in one sentence.";
    for m in tags {
        if let Some(r) = chat(m, "You are a local AI model. Reply concisely.", probe) {
            ingest_text(
                &format!("model:{m}"),
                &format!("Local model {m} says: {r}"),
                models,
                vecs,
            );
        } else {
            eprintln!("  [ingest:model] {m} unreachable (skipped)");
        }
    }
}

/// Persist the self-supervised weights learned during the standalone run so the
/// next session resumes from a trained (not just loaded) world-model.
fn save_learned(params: &WorldModelParams, wm_steps: f64, path: &PathBuf) {
    let w1: Vec<Vec<f32>> = params.w1.outer_iter().map(|r| r.to_vec()).collect();
    let w2: Vec<Vec<f32>> = params.w2.outer_iter().map(|r| r.to_vec()).collect();
    let learned = json!({
        "W1": w1,
        "b1": params.b1.to_vec(),
        "W2": w2,
        "b2": params.b2.to_vec(),
        "wm_steps": wm_steps,
        "hidden": params.w1.nrows(),
    });
    match fs::write(path, serde_json::to_string_pretty(&learned).unwrap_or_default()) {
        Ok(()) => eprintln!("  [save] learned weights -> {}", path.display()),
        Err(e) => eprintln!("  [save] failed: {e}"),
    }
}

/// Persist the (possibly grown) attractor so the next session resumes the full mind.
fn save_attractor(vecs: &[Array1<f32>], path: &PathBuf) {
    let arr: Vec<Vec<f32>> = vecs.iter().map(|v| v.to_vec()).collect();
    let j = json!({ "vecs": arr });
    match fs::write(path, serde_json::to_string_pretty(&j).unwrap_or_default()) {
        Ok(()) => eprintln!("  [save] attractor -> {}", path.display()),
        Err(e) => eprintln!("  [save] attractor failed: {e}"),
    }
}

fn main() {
    let n_cycles: usize = env::var("KAI_STANDALONE_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.axiom_state");
    let learned_wm = base.join("kai_wm_learned.json");
    let learned_attr = base.join("kai_attractor_learned.json");
    let base_wm = base.join("kai_wm.json");
    let base_attr = base.join("kai_attractor.json");

    // Resume from a previously trained checkpoint when both learned files exist,
    // so Kai is a *persistent* self-improving mind across runs (not restarted fresh).
    let (wm, wm_src): (WmJson, &str) = if learned_wm.exists() {
        match serde_json::from_str(&fs::read_to_string(&learned_wm).unwrap_or_default()) {
            Ok(w) => (w, "kai_wm_learned.json"),
            Err(_) => (
                serde_json::from_str(&fs::read_to_string(&base_wm).expect("read kai_wm.json"))
                    .expect("parse kai_wm.json"),
                "kai_wm.json",
            ),
        }
    } else {
        (
            serde_json::from_str(&fs::read_to_string(&base_wm).expect("read kai_wm.json"))
                .expect("parse kai_wm.json"),
            "kai_wm.json",
        )
    };

    let (attr, attr_src): (AttractorJson, &str) = if learned_attr.exists() {
        match serde_json::from_str(&fs::read_to_string(&learned_attr).unwrap_or_default()) {
            Ok(a) => (a, "kai_attractor_learned.json"),
            Err(_) => (
                serde_json::from_str(
                    &fs::read_to_string(&base_attr).expect("read kai_attractor.json"),
                )
                .expect("parse kai_attractor.json"),
                "kai_attractor.json",
            ),
        }
    } else {
        (
            serde_json::from_str(
                &fs::read_to_string(&base_attr).expect("read kai_attractor.json"),
            )
            .expect("parse kai_attractor.json"),
            "kai_attractor.json",
        )
    };

    let resumed = wm_src == "kai_wm_learned.json" && attr_src == "kai_attractor_learned.json";

    let mut params = WorldModelParams {
        w1: to_array2(&wm.w1),
        b1: Array1::from(wm.b1.clone()),
        w2: to_array2(&wm.w2),
        b2: Array1::from(wm.b2.clone()),
    };
    params.validate().expect("world-model shapes");

    let mut vecs: Vec<Array1<f32>> = attr
        .vecs
        .iter()
        .map(|v| Array1::from(v.clone()))
        .collect();

    // ── IR executor wiring ────────────────────────────────────────────────────
    // The autonomous loop's forward pass + free-energy now run *through the IR*
    // (physics-dialect), not by calling kai-core directly. Build the registry +
    // f64 weight views once, and prove the IR matches the trusted kai-core math.
    let mut registry = DialectRegistry::new();
    registry.register(KaiPhysicsDialect);
    let w1f: Vec<f64> = params.w1.iter().map(|x| *x as f64).collect();
    let b1f: Vec<f64> = params.b1.iter().map(|x| *x as f64).collect();
    let w2f: Vec<f64> = params.w2.iter().map(|x| *x as f64).collect();
    let b2f: Vec<f64> = params.b2.iter().map(|x| *x as f64).collect();

    {
        let s0 = vecs.last().expect("attractor non-empty");
        let a0 = Array1::<f32>::zeros(s0.len());
        let ir = predict_via_ir(&registry, s0.as_slice().unwrap(), a0.as_slice().unwrap(), &w1f, &b1f, &w2f, &b2f);
        let core = predict(s0, &a0, &params).expect("core predict");
        let maxdiff = ir
            .iter()
            .zip(core.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(maxdiff < 1e-2, "IR executor diverged from kai-core predict: {maxdiff}");

        let var0 = variance(&core);
        let nov0 = 1.0 - cosine_similarity(&core, &a0).unwrap_or(0.0);
        let vfe_ir = vfe_via_ir(&registry, &ir, a0.as_slice().unwrap(), var0, nov0);
        let vfe_core = compute_vfe(&core, &a0, var0, nov0).expect("core vfe");
        assert!(
            (vfe_ir - vfe_core).abs() < 1e-2,
            "IR executor diverged from kai-core vfe: {}",
            (vfe_ir - vfe_core).abs()
        );
        eprintln!(
            "  [parity] IR executor matches kai-core (predict Δ={maxdiff:.2e}, vfe Δ={:.2e})",
            (vfe_ir - vfe_core).abs()
        );
    }

    // ── Upstream MLIR lowering (Phase 3) ──────────────────────────────────────
    let input_dim = vecs[0].len() * 2; // state ‖ action, both embed-dim
    let hidden = params.w1.nrows();
    let output_dim = params.w2.nrows();
    lower_and_verify_mlir(input_dim, hidden, output_dim);

    // ── Ingest ALL local models ─────────────────────────────────────────────
    let tags = ollama_tags();
    let models = resolve_models(&tags);
    eprintln!(
        "KAI-STANDALONE (ingest all local models)\n  local Ollama models: {:?}\n  chat={} coder={} embed={}\n  W1 {}x{} W2 {}x{} attractor={} wm_steps={:.0} cycles={}",
        tags,
        models.chat,
        models.coder,
        models.embed,
        params.w1.nrows(),
        params.w1.ncols(),
        params.w2.nrows(),
        params.w2.ncols(),
         vecs.len(),
         wm.wm_steps,
         n_cycles,
    );
    eprintln!(
        "  checkpoint: wm={} attractor={} resumed={}",
        wm_src, attr_src, resumed,
    );

    // ── Knowledge grounding: philosophy + MATLAB .mat (only on a fresh mind;
    //    a resumed checkpoint already holds this knowledge) ──────────────────
    if !resumed {
        ingest_philosophy(&models, &mut vecs);
        ingest_matlab(&models, &mut vecs);
        // ── Ingest the ENTIRE local model ensemble (opt-in, slow) ────────────
        if std::env::var("KAI_INGEST_ALL_MODELS").is_ok() {
            ingest_all_models(&tags, &models, &mut vecs);
        }
    }

    let lr = 0.01_f32 / (1.0 + wm.wm_steps as f32 * 0.001);
    let prime_goal = "Minimize variational free energy; become a self-improving standalone Kai AGI.";

    let mut prev_actual: Option<Array1<f32>> = None;
    let mut ci = 0usize; // code-curriculum index
    let mut train_steps: u64 = 0; // self-supervised updates applied

    for cycle in 0..n_cycles {
        // Alternate: opencode coding task, else Ollama reasoning grounding.
        if cycle % 3 == 0 && ci < CODE_CURRICULUM.len() {
            // ── opencode: generate + compile-check code, ingest verification ──
            let task = CODE_CURRICULUM[ci];
            ci += 1;
            let sys = "You are a local coding agent. Output ONLY a fenced code block, no prose.";
            let gen = match chat(&models.coder, sys, task) {
                Some(g) => g,
                None => {
                    eprintln!("  C{cycle:>3} [code] Ollama unreachable");
                    continue;
                }
            };
            let (src, lang) = match extract_code(&gen) {
                Some(x) => x,
                None => {
                    eprintln!("  C{cycle:>3} [code] no code block produced");
                    continue;
                }
            };
            let (ok, detail) = verify_code(&src, lang);
            let verdict = if ok { "OK" } else { "FAIL" };
            let signal = format!("code task: {task} -> {verdict}: {detail}");
            eprintln!("  C{cycle:>3} [code] {signal}");
            if let Some(ev) = embed(&models.embed, &signal) {
                vecs.push(ev.clone());
                prev_actual = Some(ev);
            }
            continue;
        }

        // ── Ollama reasoning grounding ──────────────────────────────────────
        let topic = REASON_CURRICULUM[cycle % REASON_CURRICULUM.len()];
        let reasoning = match chat(&models.chat, prime_goal, topic) {
            Some(r) => r,
            None => {
                eprintln!("  C{cycle:>3} [reason] Ollama unreachable");
                continue;
            }
        };
        let action_vec = match embed(&models.embed, &reasoning) {
            Some(v) => v,
            None => {
                eprintln!("  C{cycle:>3} [reason] embed failed");
                continue;
            }
        };
        let state = vecs.last().cloned().expect("attractor non-empty");
        let pred = predict_via_ir(
            &registry,
            state.as_slice().unwrap(),
            action_vec.as_slice().unwrap(),
            &w1f, &b1f, &w2f, &b2f,
        );
        let pred_arr = Array1::from(pred.clone());
        let actual = prev_actual.take().unwrap_or_else(|| action_vec.clone());
        let nov = 1.0 - cosine_similarity(&pred_arr, &actual).unwrap_or(0.0);
        let vfe = vfe_via_ir(
            &registry,
            &pred,
            actual.as_slice().unwrap(),
            variance(&pred_arr),
            nov,
        );
        let mse = train_step(
            state.view(),
            action_vec.view(),
            actual.view(),
            params.w1.view_mut(),
            params.b1.view_mut(),
            params.w2.view_mut(),
            params.b2.view_mut(),
            lr,
        )
        .expect("train_step");
        train_steps += 1;
        vecs.push(actual.clone());
        prev_actual = Some(action_vec);

        let mut cons = 0.0_f32;
        let m = vecs.len();
        let k = 8.min(m - 1);
        for i in (m - k)..m {
            cons += cosine_similarity(&vecs[i - 1], &vecs[i]).unwrap_or(0.0);
        }
        cons /= k as f32;
        eprintln!(
            "  C{cycle:>3} [reason] vfe={vfe:+.4} mse={mse:.4} novelty={nov:.3} consol={cons:.4} |V|={}",
            vecs.len(),
        );
    }

    save_learned(
        &params,
        wm.wm_steps + train_steps as f64,
        &learned_wm,
    );
    save_attractor(&vecs, &learned_attr);

    eprintln!("KAI-STANDALONE done — mind ran standalone in Rust, ingesting local Ollama + opencode.");
}
