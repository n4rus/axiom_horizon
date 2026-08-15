//! Kai — Minimal AGI Core
//!
//! A physics-wired transformer inference engine with VFE adaptive control,
//! streaming layer-by-layer execution, attractor memory, and Darwin evolution.

#![allow(dead_code)]

mod config;
mod gguf;
mod streaming;
mod vfe;
mod attractor;
mod darwin;
mod ollama;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    // Debug mode: show tokenization
    if args.len() >= 3 && args[1] == "tokenize" {
        let meta = gguf::read_kv(&args[2])?;
        let tok = gguf::Tokenizer::from_gguf(&meta)
            .ok_or_else(|| "Failed to build tokenizer".to_string())?;
        let text = if args.len() >= 4 { &args[3] } else { "Hello, world!" };
        let ids = tok.encode_debug(text);
        for (id, s) in &ids {
            println!("  {:6} -> {}", id, s.escape_debug());
        }
        return Ok(());
    }

    // kai darwin evolve [--ollama] <model.gguf|model> [--generations N] [--population N] [--prompt "..." ] [--source-evolve]
    if args.len() >= 4 && args[1] == "darwin" && args[2] == "evolve" {
        let use_ollama = args.iter().any(|a| a == "--ollama");
        let model_path: &str;
        let flag_start: usize;
        if use_ollama && args[3] == "--ollama" {
            model_path = &args[4];
            flag_start = 5;
        } else {
            model_path = &args[3];
            flag_start = 4;
        }
        let mut generations = 5usize;
        let mut population = 3usize;
        let mut prompt = "The capital of France is".to_string();
        let mut source_evolve = false;
        let mut i = flag_start;
        while i < args.len() {
            match args[i].as_str() {
                "--generations" if i+1 < args.len() => { generations = args[i+1].parse().map_err(|e| format!("--generations: {}", e))?; i += 2; }
                "--population" if i+1 < args.len() => { population = args[i+1].parse().map_err(|e| format!("--population: {}", e))?; i += 2; }
                "--max-new" if i+1 < args.len() => { i += 2; } // ignored, kept for compat
                "--prompt" if i+1 < args.len() => { prompt = args[i+1].clone(); i += 2; }
                "--source-evolve" => { source_evolve = true; i += 1; }
                _ => { i += 1; }
            }
        }
        let cfg = darwin::DarwinConfig {
            generations,
            population,
            mutation_rate: 0.1,
            benchmark_prompt: prompt,
            benchmark_max_new: 16,
            source_evolve,
            project_root: ".".to_string(),
            use_ollama,
        };
        let mode = if source_evolve { "source" } else { if use_ollama { "ollama" } else { "runtime" } };
        eprintln!("Kai Darwin: evolving {} ({} gen, {} pop)", mode, generations, population);
        let best = darwin::darwin_evolve("src/config.rs", model_path, cfg)?;
        eprintln!("Best score: {:.3}", best.score);
        if !best.source_patch.is_empty() {
            eprintln!("Best source patch (config.rs) applied — next run uses mutated defaults");
        }
        eprintln!("Best physics params: base_temp={:.3} top_p={:.2} novelty_scale={:.2} vfe_tau_rate={:.3} tau_min={:.2} tau_max={:.2} assim_iters={} assim_lr={:.4}",
            best.physics.base_temperature,
            best.physics.top_p,
            best.physics.novelty_scale,
            best.physics.vfe_tau_rate,
            best.physics.tau_min,
            best.physics.tau_max,
            best.physics.assim_iters,
            best.physics.assim_lr,
        );
        return Ok(());
    }

    // kai bench <model.gguf|--ollama model> [--max-new N]
    if args.len() >= 3 && args[1] == "bench" {
        let use_ollama = args.iter().any(|a| a == "--ollama");
        let model_arg: &str;
        let flag_start: usize;

        if use_ollama && args.len() >= 4 && args[2] == "--ollama" {
            model_arg = &args[3];
            flag_start = 4;
        } else {
            model_arg = &args[2];
            flag_start = 3;
        }

        let mut max_new = 16usize;
        let mut i = flag_start;
        while i < args.len() {
            match args[i].as_str() {
                "--max-new" if i+1 < args.len() => { max_new = args[i+1].parse().map_err(|e| format!("--max-new: {}", e))?; i += 2; }
                _ => { i += 1; }
            }
        }

        if use_ollama {
            // ── Ollama benchmark ──────────────────────────
            ollama::ping()?;
            eprintln!("Kai Bench — ollama {} (GPU)", model_arg);
            let start = std::time::Instant::now();
            let result = ollama::generate(model_arg, "Benchmark test", max_new, 0.7)?;
            let elapsed = start.elapsed();
            eprintln!("Bench: {} tokens in {:.2}s = {:.1} tok/s",
                result.eval_count, elapsed.as_secs_f32(), result.tokens_per_second);
        } else {
            // ── Local GGUF CPU benchmark ──────────────────
            let meta = gguf::read_kv(model_arg)?;
            let cfg = gguf::build_config(&meta)
                .ok_or_else(|| "Failed to build config from GGUF metadata".to_string())?;
            eprintln!("Kai Bench — dim={} layers={} heads={}", cfg.dim, cfg.n_layers, cfg.n_heads);
            let start = std::time::Instant::now();
            let tokens = streaming::generate(model_arg, "Benchmark test", &cfg, max_new, 0.7, true)?;
            let elapsed = start.elapsed();
            let tps = if elapsed.as_secs_f32() > 0.0 { tokens as f32 / elapsed.as_secs_f32() } else { 0.0 };
            eprintln!("Bench: {} tokens in {:.2}s = {:.1} tok/s", tokens, elapsed.as_secs_f32(), tps);
        }
        return Ok(());
    }

    if args.len() >= 3 && args[1] == "run" {
        // Pre-scan for --ollama flag (may appear before or after model name)
        let use_ollama = args.iter().any(|a| a == "--ollama");

        // Determine model_arg and prompt_start based on --ollama position
        let model_arg: &str;
        let prompt: &str;
        let flag_start: usize;

        if use_ollama && args.len() >= 5 && args[2] == "--ollama" {
            model_arg = &args[3];
            prompt = &args[4];
            flag_start = 5;
        } else {
            model_arg = &args[2];
            prompt = &args[3];
            flag_start = 4;
        }

        // Parse flags
        let mut max_new = 64usize;
        let mut temperature = 0.7f32;
        let mut no_engram = false;

        let mut i = flag_start;
        while i < args.len() {
            match args[i].as_str() {
                "--max-new" if i+1 < args.len() => { max_new = args[i+1].parse().map_err(|e| format!("--max-new: {}", e))?; i += 2; }
                "--temp" if i+1 < args.len() => { temperature = args[i+1].parse().map_err(|e| format!("--temp: {}", e))?; i += 2; }
                "--no-engram" => { no_engram = true; i += 1; }
                _ => { i += 1; }
            }
        }

        if use_ollama {
            // ── Ollama GPU-accelerated path with VFE physics ───
            ollama::ping()?;
            eprintln!("Kai — ollama {} (GPU via llama.cpp backend)", model_arg);

            // Physics params for VFE control
            let phys = config::PhysicsParams {
                base_temperature: temperature,
                top_p: 0.9,
                ..config::PhysicsParams::default()
            };
            let mut vfe_state = vfe::VFEState::default();
            vfe_state.tau = 1.0;

            // Attractor memory for context & prior
            let mut engram = if !no_engram {
                attractor::Attractor::new(".").ok()
            } else {
                None
            };

            // Build augmented prompt from attractor context
            let augmented = if let Some(ref attr) = engram {
                let emb_vec: Vec<f32> = prompt.bytes().map(|b| b as f32 / 255.0).collect();
                let results = attr.query(&emb_vec, 3);
                if !results.is_empty() {
                    let ctx: Vec<&str> = results.iter().map(|(e, _)| e.concept.as_str()).collect();
                    eprintln!("  attractor: {} memories", results.len());
                    format!("[context: {}]\n{}", ctx.join("; "), prompt)
                } else {
                    prompt.to_string()
                }
            } else {
                prompt.to_string()
            };

            // Surrogate VFE from text properties (Ollama doesn't expose logits)
            let novelty = text_novelty(prompt);
            vfe_state.novelty = novelty;
            vfe_state.vfe = novelty * 2.0; // heuristic: VFE ≈ 2× novelty

            // Adaptive temperature: base * (1 + novelty_scale*(novelty-0.5)) / τ
            let adapted_temp = if phys.base_temperature > 0.0 {
                let boost = phys.novelty_scale * (novelty - 0.5);
                (phys.base_temperature * (1.0 + boost) / vfe_state.tau.max(0.1)).max(0.01)
            } else {
                0.0
            };

            if (adapted_temp - temperature).abs() > 0.01 {
                eprintln!("  VFE: τ={:.3} η={:.3} temp={:.3}→{:.3}",
                    vfe_state.tau, novelty, temperature, adapted_temp);
            }

            // Call Ollama with physics-adapted temperature
            let result = ollama::generate(model_arg, &augmented, max_new, adapted_temp)?;
            eprintln!("  ({} tok/s, {} tokens, {} prompt tokens, τ={:.3})",
                result.tokens_per_second, result.eval_count, result.prompt_eval_count, vfe_state.tau);

            // Post-hoc VFE: tau responds to response novelty
            let resp_novelty = text_novelty(&result.text);
            let vfe_after = resp_novelty * 2.0;
            let vfe_factor = (1.0 - phys.vfe_tau_rate * vfe_after).max(0.0).min(1.0);
            let tau_after = (vfe_state.tau * vfe_factor.sqrt())
                .clamp(phys.tau_min, phys.tau_max);
            eprintln!("  VFE→τ={:.3} (η_resp={:.3})", tau_after, resp_novelty);

            // Store response in attractor memory for future context
            if let Some(ref mut attr) = engram {
                let emb_vec: Vec<f32> = prompt.bytes().map(|b| b as f32 / 255.0).collect();
                let arr = ndarray::Array1::from_vec(emb_vec);
                if arr.len() >= attr.dim() {
                    let sub = arr.slice(ndarray::s![..attr.dim()]);
                    let _ = attr.push("ollama_gen", prompt, &[], &sub);
                } else {
                    let _ = attr.push("ollama_gen", prompt, &[], &arr.view());
                }
            }

            print!("{}", result.text);
            println!();
        } else {
            // ── Local GGUF CPU path ────────────────────────
            let meta = gguf::read_kv(model_arg)?;
            let cfg = gguf::build_config(&meta)
                .ok_or_else(|| "Failed to build config from GGUF metadata".to_string())?;

            eprintln!("Kai — dim={} layers={} heads={} vocab={} est_gb={:.1}",
                cfg.dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size, cfg.estimated_f32_gb());

            let _n = streaming::generate(model_arg, prompt, &cfg, max_new, temperature, no_engram)?;
        }

        return Ok(());
    }

    // kai chat --ollama <model> [--max-new N] [--temp T] [--no-engram]
    if args.len() >= 3 && args[1] == "chat" {
        let use_ollama = args.iter().any(|a| a == "--ollama");
        if !use_ollama {
            return Err("kai chat requires --ollama <model>".to_string());
        }
        let model: &str;
        let flag_start: usize;
        if args.len() >= 5 && args[2] == "--ollama" {
            model = &args[3];
            flag_start = 4;
        } else if args.len() >= 4 {
            // model name at position 2 with --ollama somewhere later
            let pos = args.iter().position(|a| a == "--ollama").unwrap_or(3);
            model = &args[pos + 1];
            flag_start = args.len(); // parse nothing further for now
        } else {
            return Err("Usage: kai chat --ollama <model> [--max-new N]".to_string());
        }

        let mut max_new = 64usize;
        let mut temperature = 0.7f32;
        let mut no_engram = false;
        let mut i = flag_start;
        while i < args.len() {
            match args[i].as_str() {
                "--max-new" if i+1 < args.len() => { max_new = args[i+1].parse().map_err(|e| format!("--max-new: {}", e))?; i += 2; }
                "--temp" if i+1 < args.len() => { temperature = args[i+1].parse().map_err(|e| format!("--temp: {}", e))?; i += 2; }
                "--no-engram" => { no_engram = true; i += 1; }
                _ => { i += 1; }
            }
        }

        ollama::ping()?;
        eprintln!("Kai Chat — ollama {} (VFE physics, Ctrl+D to exit)", model);
        eprintln!();

        // Physics params
        let phys = config::PhysicsParams {
            base_temperature: temperature,
            top_p: 0.9,
            ..config::PhysicsParams::default()
        };
        let mut vfe_state = vfe::VFEState::default();
        vfe_state.tau = 1.0;

        // Attractor memory
        let mut engram = if !no_engram {
            attractor::Attractor::new(".").ok()
        } else {
            None
        };

        // Message history
        let mut messages: Vec<ollama::Message> = Vec::new();

        // Interactive REPL
        let stdin = std::io::stdin();
        let mut turn: usize = 0;
        loop {
            turn += 1;
            eprint!("[{}] You> ", turn);
            use std::io::Write;
            std::io::stderr().flush().ok();

            let mut input = String::new();
            if stdin.read_line(&mut input).map_err(|e| format!("stdin: {}", e))? == 0 {
                eprintln!(); // EOF
                break;
            }
            let input = input.trim().to_string();
            if input.is_empty() {
                continue;
            }
            if input == "/exit" || input == "/quit" {
                break;
            }

            // Query attractor for context
            let augmented = if let Some(ref attr) = engram {
                let emb_vec: Vec<f32> = input.bytes().map(|b| b as f32 / 255.0).collect();
                let results = attr.query(&emb_vec, 2);
                if !results.is_empty() {
                    let ctx: Vec<&str> = results.iter().map(|(e, _)| e.concept.as_str()).collect();
                    format!("[context: {}]\n{}", ctx.join("; "), input)
                } else {
                    input.clone()
                }
            } else {
                input.clone()
            };

            // VFE adaptive temperature
            let novelty = text_novelty(&input);
            vfe_state.novelty = novelty;
            vfe_state.vfe = novelty * 2.0;
            let adapted_temp = if phys.base_temperature > 0.0 {
                let boost = phys.novelty_scale * (novelty - 0.5);
                (phys.base_temperature * (1.0 + boost) / vfe_state.tau.max(0.1)).max(0.01)
            } else {
                0.0
            };

            // Build messages with system prompt
            let mut chat_msgs = vec![ollama::Message {
                role: "system".to_string(),
                content: format!(
                    "You are Kai, a physics-wired AGI. τ={:.3} η={:.3}",
                    vfe_state.tau, novelty
                ),
            }];
            for msg in &messages {
                chat_msgs.push(msg.clone());
            }
            chat_msgs.push(ollama::Message {
                role: "user".to_string(),
                content: augmented,
            });

            // Call Ollama
            let result = ollama::chat(model, &chat_msgs, max_new, adapted_temp)?;

            eprintln!("  ({} tok/s, {} tok, τ={:.3})",
                result.tokens_per_second, result.eval_count, vfe_state.tau);

            // VFE post-hoc update
            let resp_novelty = text_novelty(&result.text);
            let vfe_after = resp_novelty * 2.0;
            let vfe_factor = (1.0 - phys.vfe_tau_rate * vfe_after).max(0.0).min(1.0);
            vfe_state.tau = (vfe_state.tau * vfe_factor.sqrt())
                .clamp(phys.tau_min, phys.tau_max);

            // Print assistant response
            println!("Kai> {}", result.text);

            // Store in attractor
            if let Some(ref mut attr) = engram {
                let emb_vec: Vec<f32> = input.bytes().map(|b| b as f32 / 255.0).collect();
                let arr = ndarray::Array1::from_vec(emb_vec);
                if arr.len() >= attr.dim() {
                    let sub = arr.slice(ndarray::s![..attr.dim()]);
                    let _ = attr.push("chat_gen", &input, &[], &sub);
                } else {
                    let _ = attr.push("chat_gen", &input, &[], &arr.view());
                }
            }

            // Add to message history (keep last 8 for context window)
            messages.push(ollama::Message {
                role: "user".to_string(),
                content: input,
            });
            messages.push(ollama::Message {
                role: "assistant".to_string(),
                content: result.text,
            });
            if messages.len() > 16 {
                messages.drain(0..2); // drop oldest pair
            }
        }

        return Ok(());
    }

    print_usage();
    Ok(())
}

/// Surrogate text novelty [0,1] for VFE when token logits aren't available.
fn text_novelty(text: &str) -> f32 {
    if text.is_empty() {
        return 1.0;
    }
    let chars: Vec<char> = text.chars().collect();
    let unique: std::collections::HashSet<char> = chars.iter().copied().collect();
    let diversity = unique.len() as f32 / chars.len() as f32;
    let len_factor = 1.0 - (chars.len() as f32 / 256.0).min(1.0);
    0.5 * diversity + 0.5 * len_factor
}

fn print_usage() {
    eprintln!("Kai — Physics-wired AGI Core");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  kai run <model.gguf|--ollama model> \"prompt\" [--max-new N] [--temp T] [--no-engram] [--ollama]");
    eprintln!("  kai chat --ollama <model> [--max-new N] [--temp T] [--no-engram]");
    eprintln!("  kai bench <model.gguf|--ollama model> [--max-new N] [--ollama]");
    eprintln!("  kai darwin evolve <model.gguf> [--generations N] [--population N] [--prompt \"...\"] [--source-evolve]");
    eprintln!("  kai tokenize <model.gguf> [text]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --max-new N    Max new tokens to generate (default: 64)");
    eprintln!("  --temp T       Temperature (default: 0.7)");
    eprintln!("  --no-engram    Skip semantic memory (faster, lower memory)");
    eprintln!("  --ollama       Use Ollama GPU backend (model arg is an Ollama model name)");
}