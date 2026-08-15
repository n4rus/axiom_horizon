//! Darwin Evolution — Real source patches + physics-parameter evolution
//!
//! Two mutation modes:
//! 1. Runtime-only: mutate PhysicsParams, evaluate in-process (fast)
//! 2. Source-patch: mutate config.rs defaults, rebuild, run new binary (slow but persistent)

use std::process::Command;
use std::fs;

use crate::config::PhysicsParams;
use crate::streaming::generate_with_phys;
use crate::gguf;
use crate::ollama;

#[derive(Debug, Clone)]
pub struct DarwinConfig {
    pub generations: usize,
    pub population: usize,
    pub mutation_rate: f32,
    pub benchmark_prompt: String,
    pub benchmark_max_new: usize,
    /// If true, also mutate source defaults and rebuild (persistent evolution).
    pub source_evolve: bool,
    /// Path to the kai project root (for `cargo build`).
    pub project_root: String,
    /// If true, use Ollama GPU backend instead of local CPU inference.
    pub use_ollama: bool,
}

impl Default for DarwinConfig {
    fn default() -> Self {
        Self {
            generations: 10,
            population: 4,
            mutation_rate: 0.1,
            benchmark_prompt: "The capital of France is".into(),
            benchmark_max_new: 16,
            source_evolve: false,
            project_root: ".".into(),
            use_ollama: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub source_patch: String,      // source code diff (evolution of the engine itself)
    pub physics: PhysicsParams,    // evolved physics parameters
    pub score: f32,
}

pub fn darwin_evolve(
    _source_path: &str,
    model_path: &str,
    config: DarwinConfig,
) -> Result<Candidate, String> {
    let mut best = Candidate {
        source_patch: String::new(),
        physics: PhysicsParams::default(),
        score: f32::NEG_INFINITY,
    };

    for gen in 0..config.generations {
        eprintln!("Darwin: Generation {}", gen + 1);

        let mut candidates = vec![best.clone()];
        for _ in 1..config.population {
            let mut c = mutate_runtime(&PhysicsParams::default());
            if config.source_evolve {
                // Also produce a source-level mutation
                c.source_patch = mutate_source_defaults(&config.project_root)?;
            }
            candidates.push(c);
        }

        for c in &mut candidates {
            if config.source_evolve && !c.source_patch.is_empty() {
                c.score = evaluate_rebuilt(model_path, &config, &c.source_patch)?;
            } else if config.use_ollama {
                c.score = evaluate_ollama(model_path, &config, &c.physics)?;
            } else {
                c.score = evaluate_inproc(model_path, &config, &c.physics)?;
            }
            eprintln!("  score={:.3}", c.score);
        }

        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        best = candidates[0].clone();
        eprintln!("  best={:.3}", best.score);

        if config.source_evolve && !best.source_patch.is_empty() {
            eprintln!("  best source patch: {} bytes", best.source_patch.len());
        }
    }

    Ok(best)
}

// ── Runtime-only mutation (fast, in-process) ────────────────────────────

fn mutate_runtime(_base: &PhysicsParams) -> Candidate {
    Candidate {
        source_patch: String::new(),
        physics: PhysicsParams {
            base_temperature: 0.3 + rand::random::<f32>() * 0.7,
            top_p: 0.7 + rand::random::<f32>() * 0.25,
            novelty_scale: 0.2 + rand::random::<f32>() * 0.5,
            vfe_tau_rate: 0.05 + rand::random::<f32>() * 0.15,
            tau_min: 0.2 + rand::random::<f32>() * 0.5,
            tau_max: 1.0 + rand::random::<f32>() * 2.0,
            assim_iters: (rand::random::<f32>() * 3.0) as usize,
            assim_lr: 0.005 + rand::random::<f32>() * 0.02,
        },
        score: 0.0,
    }
}

fn evaluate_ollama(model_name: &str, config: &DarwinConfig, physics: &PhysicsParams) -> Result<f32, String> {
    ollama::ping()?;
    let max_new = config.benchmark_max_new;
    let result = ollama::generate(model_name, &config.benchmark_prompt, max_new, physics.base_temperature)?;
    // Throughput component: shorter+higher-tps is not intrinsically better on its
    // own; token count gives confidence it produced a real answer.
    let throughput = (result.eval_count as f32).sqrt() * result.tokens_per_second;
    // Attractor-agreement / physics-sanity bonus. Reward candidates whose VFE
    // controller parameters live in the "agentic" band — enough novelty to
    // explore, tight enough tau window to stay coherent. This steers evolution
    // toward good physics, not just raw tok/s. Degenerate extremes are penalized.
    let tau_mid = (physics.tau_min + physics.tau_max) * 0.5;
    let tau_band_bonus = if physics.tau_max > physics.tau_min {
        1.0 + (1.0 - (tau_mid - 0.75).abs()) * 0.15
    } else {
        0.6
    };
    let novelty_bonus = if (0.1..=0.9).contains(&physics.novelty_scale) {
        1.0
    } else {
        0.7
    };
    let agreement = tau_band_bonus * novelty_bonus;
    // Blend: 40% throughput, 60% physics-agreement so evolution primarily
    // optimizes for agentic-understanding-consistent parameters.
    let score = throughput * 0.1 * agreement * 0.6
              + physics_tau_penalty(physics) * 1.0;
    let score = if score.is_finite() && score >= 0.0 { score } else { 0.0 };
    Ok(score)
}

/// Extra penalty when tau window is absurd (max _spread_ huge) — a degenerate
/// controller that would make inference incoherent cannot be a "best".
fn physics_tau_penalty(p: &PhysicsParams) -> f32 {
    let spread = p.tau_max - p.tau_min;
    if spread > 2.0 || p.tau_min < 0.0 { 0.0 } else { 1.0 }
}

fn evaluate_inproc(model_path: &str, config: &DarwinConfig, physics: &PhysicsParams) -> Result<f32, String> {
    let meta = gguf::read_kv(model_path)?;
    let cfg = gguf::build_config(&meta)
        .ok_or_else(|| "Failed to build config from GGUF metadata".to_string())?;

    let tokens = generate_with_phys(
        model_path,
        &config.benchmark_prompt,
        &cfg,
        config.benchmark_max_new,
        physics,
        true,
    )?;

    if tokens == 0 {
        return Ok(0.0);
    }
    Ok((tokens as f32).sqrt() * 10.0)
}

// ── Source-patch evolution (persistent, rebuild required) ───────────────

/// Replace a `const NAME: f32 = value;` literal in source text with a mutated value.
/// Returns the mutated source on success.
fn mutate_f32_const(source: &str, name: &str, lo: f32, hi: f32) -> Option<String> {
    let pattern = format!("const {}: f32 = ", name);
    let pos = source.find(&pattern)?;
    let val_start = pos + pattern.len();
    let val_end = source[val_start..]
        .find(|c: char| c == ';' || c == '\n')
        .map(|end| val_start + end)
        .unwrap_or(source.len());
    let old = source[val_start..val_end].trim();
    if old.parse::<f32>().is_ok() {
        let new_val = lo + rand::random::<f32>() * (hi - lo);
        let mut s = source.to_string();
        s.replace_range(val_start..val_end, &format!("{:.6e}", new_val));
        Some(s)
    } else {
        None
    }
}

/// Replace a `name: value,` inside a `Self { ... }` block with a mutated value.
fn mutate_self_field(source: &str, field: &str, is_float: bool, lo: f32, hi: f32) -> Option<String> {
    let pattern = format!("{}: ", field);
    let pos = source.find(&pattern)?;
    let val_start = pos + pattern.len();
    let val_end = source[val_start..]
        .find(|c: char| c == ',' || c == '\n' || c == '}')
        .map(|end| val_start + end)
        .unwrap_or(source.len());
    let old = source[val_start..val_end].trim().to_string();
    let valid = if is_float { old.parse::<f32>().is_ok() } else { old.parse::<usize>().is_ok() };
    if valid {
        let new_val = if is_float {
            format!("{:.4}", lo + rand::random::<f32>() * (hi - lo))
        } else {
            ((rand::random::<f32>() * hi as f32) as usize).to_string()
        };
        let mut s = source.to_string();
        s.replace_range(val_start..val_end, &new_val.to_string());
        Some(s)
    } else {
        None
    }
}

/// Back up file to `.bak` then mutate config.rs (PhysicsParams defaults) and
/// streaming.rs (EPS constant).  Returns the backup snapshot (for revert).
fn mutate_source_defaults(project_root: &str) -> Result<String, String> {
    let config_path = format!("{}/src/config.rs", project_root);
    let stream_path = format!("{}/src/streaming.rs", project_root);

    // Back up both files
    let config_bak = fs::read_to_string(&config_path).map_err(|e| format!("read config.rs: {}", e))?;
    let stream_bak = fs::read_to_string(&stream_path).map_err(|e| format!("read stream.rs: {}", e))?;

    // Store backups for revert
    fs::write(&format!("{}.bak", config_path), &config_bak).ok();
    fs::write(&format!("{}.bak", stream_path), &stream_bak).ok();

    // ── Mutate config.rs PhysicsParams defaults ───────────────────────
    let impl_tag = "impl Default for PhysicsParams";
    let impl_pos = config_bak.find(impl_tag)
        .ok_or_else(|| "Could not find PhysicsParams Default impl".to_string())?;
    let after_impl = &config_bak[impl_pos..];
    let self_brace = after_impl.find("Self {")
        .map(|p| impl_pos + p)
        .ok_or_else(|| "Could not find Self { in PhysicsParams impl".to_string())?;
    let self_open = config_bak[self_brace..].find('{')
        .map(|p| self_brace + p)
        .ok_or_else(|| "Could not find opening brace of Self block".to_string())?;
    let mut depth = 1u32;
    let self_close = config_bak[self_open + 1..]
        .char_indices()
        .find(|&(_, c)| {
            match c { '{' => depth += 1, '}' => depth -= 1, _ => {} }
            depth == 0
        })
        .map(|(p, _)| self_open + 1 + p)
        .ok_or_else(|| "Could not find closing brace of Self block".to_string())?;

    let self_block: String = config_bak[self_open..=self_close].to_string();
    let mut mutated = self_block;

    for &(name, lo, hi) in &[
        ("base_temperature", 0.3, 1.0),
        ("top_p", 0.7, 0.95),
        ("novelty_scale", 0.1, 0.9),
        ("vfe_tau_rate", 0.02, 0.25),
        ("tau_min", 0.2, 0.9),
        ("tau_max", 1.0, 4.0),
        ("assim_lr", 0.001, 0.03),
    ] {
        if let Some(m) = mutate_self_field(&mutated, name, true, lo, hi) {
            mutated = m;
        }
    }
    if let Some(m) = mutate_self_field(&mutated, "assim_iters", false, 0.0, 4.0) {
        mutated = m;
    }

    let patched_config = config_bak[..self_open].to_string() + &mutated + &config_bak[self_close + 1..];
    fs::write(&config_path, &patched_config).map_err(|e| format!("write config.rs: {}", e))?;

    // ── Mutate streaming.rs EPS constant ─────────────────────────────
    if let Some(mutated_stream) = mutate_f32_const(&stream_bak, "EPS", 1e-7, 1e-3) {
        fs::write(&stream_path, &mutated_stream).ok();
    }

    // Return the snapshot so caller can revert both
    Ok(config_bak)
}

/// Rebuild the project and run benchmark, scoring the result.
/// If rebuild fails, reverts the patch and returns 0.
fn evaluate_rebuilt(model_path: &str, config: &DarwinConfig, patched_source: &str) -> Result<f32, String> {
    // Rebuild
    eprintln!("  rebuilding...");
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&config.project_root)
        .status()
        .map_err(|e| format!("cargo build: {}", e))?;

    if !status.success() {
        eprintln!("  build failed, reverting patch");
        revert_source(&config.project_root, patched_source)?;
        return Ok(0.0);
    }

    // Run benchmark with the rebuilt binary
    let binary = format!("{}/target/release/kai", config.project_root);
    let output = Command::new(&binary)
        .args([
            "bench",
            model_path,
            "--max-new",
            &config.benchmark_max_new.to_string(),
        ])
        .output()
        .map_err(|e| format!("run benchmark: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("  bench failed: {}", stderr);
        revert_source(&config.project_root, patched_source)?;
        return Ok(0.0);
    }

    // Parse bench output: "Bench: N tokens in Xs = Y tok/s"
    let stdout = String::from_utf8_lossy(&output.stdout);
    let bench_line = stdout.lines().find(|l| l.starts_with("Bench:"));
    let score = if let Some(line) = bench_line {
        // Simple: higher tok/s = better. Extract the tok/s value.
        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() >= 2 {
            let tps_str = parts[1].trim().trim_end_matches(" tok/s");
            tps_str.parse::<f32>().unwrap_or(0.0) * 10.0
        } else {
            // Fallback: count "Bench:" as success
            5.0
        }
    } else {
        5.0
    };

    Ok(score)
}

pub fn revert_source(project_root: &str, _original: &str) -> Result<(), String> {
    let config_path = format!("{}/src/config.rs", project_root);
    let stream_path = format!("{}/src/streaming.rs", project_root);
    // Restore from .bak files (written before mutation)
    if let Ok(bak) = fs::read_to_string(format!("{}.bak", config_path)) {
        fs::write(&config_path, &bak).ok();
    }
    if let Ok(bak) = fs::read_to_string(format!("{}.bak", stream_path)) {
        fs::write(&stream_path, &bak).ok();
    }
    Ok(())
}
