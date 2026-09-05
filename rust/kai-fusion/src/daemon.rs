//! Autonomous Kai daemon — 24/7 fixed-point loop with tau decay.
//!
//! The daemon runs a continuous cycle:
//! 1. Check for new input (from stdin or a file)
//! 2. If input: classify query, route to sub-agent, generate with attractor
//! 3. If no input: decay tau toward 1.0 (idle relaxation) via BracketState
//! 4. Periodically report attractor convergence and bracket state vector
//! 5. Push final hidden state to attractor after each generation
//!
//! Phase 7: Tau horizon + fixed point + physical integration.
//! Bracket state drives the daemon's time-dilation and convergence tracking.

use crate::attractor;
use crate::bracket;
use crate::companion::{ChatRow, ChatStore, FactStore, MessageRow};
use crate::routes;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Daemon configuration.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Path to the GGUF model file
    pub model_path: String,
    /// Path to the shared attractor file
    pub attractor_path: String,
    /// Decay rate for tau during idle (per second)
    pub tau_decay_rate: f32,
    /// VFE → tau learning rate
    pub tau_learning_rate: f32,
    /// Interval between idle checks (seconds)
    pub idle_check_interval_s: f32,
    /// Maximum new tokens to generate per input
    pub max_new_tokens: usize,
    /// Sampling temperature
    pub temperature: f32,
    /// Goal decomposition interval (cycles)
    pub goal_decompose_interval: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        DaemonConfig {
            model_path: String::new(),
            attractor_path: attractor::DEFAULT_ATTRACTOR_PATH.to_string(),
            tau_decay_rate: 0.1,   // matches bracket::DEFAULT_DECAY_RATE
            tau_learning_rate: 10.0, // matches bracket::DEFAULT_VFE_THRESHOLD
            idle_check_interval_s: 5.0,
            max_new_tokens: 128,
            temperature: 0.8,
            goal_decompose_interval: 20,
        }
    }
}

/// Daemon metrics snapshot.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DaemonMetrics {
    pub tau: f32,
    pub vfe: f32,
    pub cycles: usize,
    pub age: f32,
    pub phi: f32,
    pub bracket_state: Vec<f32>,
    pub attractor_vectors: usize,
    pub attractor_convergence: f32,
    pub total_generations: u64,
    pub uptime_seconds: f32,
    pub subjective_seconds: f32,
}

/// Run the daemon as a foreground process.
/// Loads the model and runs the fixed-point loop until `stop` is signaled.
pub fn run_daemon(cfg: &DaemonConfig, stop: Arc<AtomicBool>) -> Result<DaemonMetrics, String> {
    let start_time = Instant::now();
    let mut total_generations: u64 = 0;
    let mut tau_history: Vec<(f32, f32)> = Vec::new();
    let mut last_input_time = Instant::now();

    // Ensure attractor directory exists
    let attr_path = &cfg.attractor_path;
    if let Some(parent) = Path::new(attr_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }

    // Initialize attractor if it doesn't exist
    if !Path::new(attr_path).exists() {
        let seed = vec![0.0f32; 768];
        attractor::push(attr_path, &seed).map_err(|e| format!("attractor init: {e}"))?;
        eprintln!("[daemon] initialized attractor at {attr_path}");
    }

    eprintln!("[daemon] starting fixed-point loop (model={}, attractor={})",
        cfg.model_path, attr_path);
    eprintln!("[daemon] tau_decay_rate={}, tau_lr={}, check_interval={}s",
        cfg.tau_decay_rate, cfg.tau_learning_rate, cfg.idle_check_interval_s);
    eprintln!("[daemon] type input or 'exit' to stop\n");

    // ---- Companion wiring (P6+ sentience): persistent recurrent SOUL ----
    let state_dir = Path::new(attr_path).parent().unwrap_or(Path::new(".kai_state"));
    let chat_db = state_dir.join("kai_chat.db").to_string_lossy().to_string();
    let fact_db = state_dir.join("kai_facts.db").to_string_lossy().to_string();
    let soul_path = state_dir.join("kai_soul.json").to_string_lossy().to_string();
    let bracket_path = state_dir.join("kai_bracket.json").to_string_lossy().to_string();
    // Restore bracket (SOUL) from disk — immortal daemon never resets
    let mut bracket = if Path::new(&bracket_path).exists() {
        match crate::bracket::BracketState::load(&bracket_path) {
            Ok(b) => { eprintln!("[daemon] SOUL restored: {} (cycles={})", b.bracket_line(), b.cycles); b }
            Err(_) => crate::bracket::BracketState::new(),
        }
    } else { crate::bracket::BracketState::new() };
    let chat_store = ChatStore::load_or_new(&chat_db);
    let fact_store = FactStore::load_or_new(&fact_db);
    // P6: single persistent session — never resets on reboot
    let session_id = "kai-sessions".to_string();
    chat_store.upsert_chat(&ChatRow {
        id: session_id.clone(),
        title: format!("daemon {}", now_date_string()),
        model: cfg.model_path.clone(),
        created_at: chrono_now_ms(),
        updated_at: chrono_now_ms(),
    });
    eprintln!("[daemon] companion wired: chats->{chat_db} facts->{fact_db} session={session_id}");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while !stop.load(Ordering::Relaxed) {
        let now = Instant::now();
        let wall_dt = now.duration_since(last_input_time).as_secs_f32();

        // ---- Check for input (non-blocking via line buffer) ----
        let mut _has_input = false;

        // Try to read a line (blocking with timeout simulation via line-buffered input)
        // Since stdin.lines() is blocking, we check if there's data available first.
        // For non-blocking, we'd need termios or a separate thread.
        // Simple approach: use stdin blocking read with prompt.

        // Print prompt
        print!("kai> ");
        io::stdout().flush().ok();

        // Read line (blocking — the daemon waits for input)
        let line = match lines.next() {
            Some(Ok(l)) => l,
            Some(Err(_)) | None => break,
        };
        let line = line.trim().to_string();

        if line.is_empty() || line == "exit" || line == "quit" {
            if line == "exit" || line == "quit" {
                break;
            }
            // Empty line: idle cycle — decay tau via BracketState
            let idle_s = last_input_time.elapsed().as_secs_f32();
            bracket.decay_tau(idle_s);
            tau_history.push((wall_dt, bracket.tau));
            let _ = bracket.save(&bracket_path);
            // Report status
            let uptime = start_time.elapsed().as_secs_f32();
            let subj = bracket.age();
            print_daemon_status(&bracket, uptime, subj, total_generations, attr_path);
            continue;
        }

        _has_input = true;
        last_input_time = Instant::now();

        // ---- Companion: persist user turn + extract facts ----
        chat_store.insert_message(&MessageRow {
            id: format!("{}-u-{}", session_id, chrono_now_ms()),
            chat_id: session_id.clone(),
            role: "USER".into(),
            text: line.clone(),
            vfe: None,
            curvature: None,
            temp: None,
            model: cfg.model_path.clone(),
            ts: chrono_now_ms(),
        });
        if let Some(fact) = FactStore::extract_fact(&line) {
            if fact_store.store(&fact, "user-stated") {
                eprintln!("[daemon] fact stored: {fact}");
            }
        }

        // ---- Process input ----
        // Classify and route
        let route = routes::RouteConfig::from_query(&line);
        // P6.3 recurrent SOUL feedback + P0 context injection: bracket + last 20 chats + facts
        let soul_ctx = format!("[SOUL {} age={:.1}s phi={:.3}]", bracket.bracket_line(), bracket.age(), bracket.phi());
        let recent = chat_store.messages(&session_id);
        let history_ctx = if recent.len() > 1 {
            let tail = &recent[recent.len().saturating_sub(21)..recent.len().saturating_sub(1)];
            tail.iter().map(|m| format!("{}: {}", m.role, m.text.chars().take(300).collect::<String>())).collect::<Vec<_>>().join("\n")
        } else { String::new() };
        let facts_all = fact_store.all();
        let facts = facts_all.iter().take(5).map(|f| f.fact.clone()).collect::<Vec<_>>();
        let fact_ctx = if facts.is_empty() { String::new() } else { format!("[facts] {}", facts.join("; ")) };
        let history_block = if history_ctx.is_empty() { String::new() } else { format!("\n[history]\n{history_ctx}") };
        let fact_block = if fact_ctx.is_empty() { String::new() } else { format!("\n{fact_ctx}") };
        // Self-reasoning: WorldGraph retrieval_quality + metacognition prompt
        let rq = crate::worldgraph::retrieval_quality_cached(&line);
        let wg_ctx = format!("[worldgraph rq={:.3} domains=6]", rq);
        let meta_ctx = "[metacognition: after answering, rate confidence 0-1 and one failure mode]";
        let augmented = format!("[System: {}]\n{soul_ctx} {wg_ctx}\n{meta_ctx}{history_block}{fact_block}\n\nUser: {}", route.system_prompt, line);
        eprintln!("\n[daemon] {} {} — generating... {}", route.kind.icon(), route.kind.label(), soul_ctx);

        // Load current attractor as prior
        let prior = if Path::new(attr_path).exists() {
            attractor::load(attr_path).ok()
        } else {
            None
        };
        let _prior_centroid = prior.as_ref().map(|v| attractor::centroid(v));

        // ---- Generate (simple forward loop) ----
        // Load model and tokenizer
        let meta = match crate::gguf::read_kv(&cfg.model_path) {
            Ok(m) => m,
            Err(e) => { eprintln!("[daemon] meta: {e}"); continue; }
        };
        let model_cfg = match crate::gguf::build_config(&meta) {
            Some(c) => c,
            None => { eprintln!("[daemon] could not build Config"); continue; }
        };
        let tok = match crate::tok::Tokenizer::from_gguf(&meta) {
            Some(t) => t,
            None => { eprintln!("[daemon] no tokenizer"); continue; }
        };
        let (_ver, map, _nt, _nk) = match crate::gguf::load_tensors(&cfg.model_path) {
            Ok(x) => x,
            Err(e) => { eprintln!("[daemon] load: {e}"); continue; }
        };
        let model = match crate::model::Weights::from_gguf(&map, &model_cfg) {
            Ok(m) => m,
            Err(e) => { eprintln!("[daemon] model: {e}"); continue; }
        };

        let ids = tok.encode(&augmented);
        let mut all_ids = ids.clone();
        let eos = tok.eos;
        let mut gen_text = String::new();

        // Generate tokens
        let temp = route.temperature.unwrap_or(cfg.temperature);
        for step in 0..cfg.max_new_tokens {
            let (logits, _xf) = model.forward(&model_cfg, &all_ids);
            let last = logits.row(logits.nrows() - 1);
            let next = if temp > 0.0 {
                let mut scaled: Vec<f32> = last.iter().map(|&x| (x / temp).exp()).collect();
                let sum: f32 = scaled.iter().sum();
                if sum > 0.0 {
                    for v in &mut scaled { *v /= sum; }
                    let mut rng = 42u64;
                    let r = |rng: &mut u64| { *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); *rng as f32 / u64::MAX as f32 };
                    let p = r(&mut rng);
                    let mut cum = 0.0f32;
                    let mut selected = 0;
                    for (i, &v) in scaled.iter().enumerate() {
                        cum += v;
                        if p <= cum { selected = i; break; }
                    }
                    selected
                } else { 0 }
            } else {
                last.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            let next = next.min(model_cfg.vocab_size - 1);
            all_ids.push(next);
            let s = tok.decode(&[next]);
            gen_text.push_str(&s);
            print!("{s}");
            io::stdout().flush().ok();
            if next == eos { break; }
            if step % 8 == 0 { eprint!("."); }
        }
        println!();

        // ---- Push final hidden state to attractor ----
        let (_logits_last, xf_last) = model.forward(&model_cfg, &all_ids);

        // ---- Companion: persist KAI turn (with physics scalars) ----
        chat_store.insert_message(&MessageRow {
            id: format!("{}-k-{}", session_id, chrono_now_ms()),
            chat_id: session_id.clone(),
            role: "KAI".into(),
            text: gen_text.clone(),
            vfe: Some(bracket.vfe),
            curvature: None,
            temp: Some(route.temperature.unwrap_or(cfg.temperature)),
            model: cfg.model_path.clone(),
            ts: chrono_now_ms(),
        });
        let store_dim = 768.min(xf_last.len());
        let store_vec: Vec<f32> = xf_last.iter().take(store_dim).copied().collect();
        // Use bracket state vector for attractor storage (Phase 7 integration)
        let bstate = bracket.state_vector();
        let attractor_vec: Vec<f32> = if bstate.len() >= store_dim {
            bstate.iter().take(store_dim).copied().collect()
        } else {
            store_vec.clone()
        };
        match attractor::push(attr_path, &attractor_vec) {
            Ok(()) => {
                total_generations += 1;
                // Update bracket tau based on VFE (simulated novelty from attractor convergence)
                let vfe_val = match attractor::convergence(attr_path, 10) {
                    Ok((conv, _)) => (1.0 - conv) * 5.0, // higher novelty = higher VFE
                    Err(_) => 1.0,
                };
                let var = 0.01; // attractor variance placeholder
                bracket.update(wall_dt * 1000.0, vfe_val, var);
                tau_history.push((wall_dt, bracket.tau));
                let _ = bracket.save(&bracket_path);
                // Phase 4.2: GoalDecomposer wired every 20 cycles
                if bracket.cycles > 0 && bracket.cycles % cfg.goal_decompose_interval == 0 {
                    let decomposer = crate::goals::GoalDecomposer::new();
                    let goal = format!("cycle-{}-improve", bracket.cycles);
                    let plan = decomposer.decompose(&goal);
                    let progress = plan.progress();
                    eprintln!("[daemon] GoalDecomposer @ cycle {}: {} subtasks, progress={:.2}",
                        bracket.cycles, plan.subtasks.len(), progress);
                }
            }
            Err(e) => eprintln!("[daemon] attractor push: {e}"),
        }

        // Report
        let uptime = start_time.elapsed().as_secs_f32();
        let subj = bracket.age();
        match attractor::convergence(attr_path, 10) {
            Ok((conv, n)) => {
                println!("[daemon] {} | attractor: {} vecs, conv={:.4} | uptime={:.0}s | subjective={:.0}s",
                    bracket.bracket_line(), n, conv, uptime, subj);
                if conv > 0.95 {
                    println!("[daemon] ✅ FIXED POINT: attractor convergence={conv:.4}");
                }
            }
            Err(e) => eprintln!("[daemon] convergence: {e}"),
        }
    }

    // P6.1: checkpoint SOUL on exit — never dies, survives reboot
    let _ = bracket.save(&bracket_path);
    let soul_json = serde_json::json!({"tau": bracket.tau, "vfe": bracket.vfe, "cycles": bracket.cycles, "age": bracket.age(), "phi": bracket.phi(), "uptime": start_time.elapsed().as_secs_f32(), "ts": chrono_now_ms()});
    let _ = std::fs::write(&soul_path, serde_json::to_string_pretty(&soul_json).unwrap_or_default());

    let uptime = start_time.elapsed().as_secs_f32();
    let subj = bracket.age();
    let (conv, n) = attractor::convergence(attr_path, 10).unwrap_or((0.0, 0));

    Ok(DaemonMetrics {
        tau: bracket.tau,
        vfe: bracket.vfe,
        cycles: bracket.cycles,
        age: bracket.age(),
        phi: bracket.phi(),
        bracket_state: bracket.state_vector(),
        attractor_vectors: n,
        attractor_convergence: conv,
        total_generations,
        uptime_seconds: uptime,
        subjective_seconds: subj,
    })
}

/// Print a status line for the daemon using bracket state.
fn print_daemon_status(bracket: &bracket::BracketState, uptime: f32, subjective: f32, total_gen: u64, attr_path: &str) {
    let (conv, n) = attractor::convergence(attr_path, 10).unwrap_or((0.0, 0));
    println!(
        "[daemon] {} | gen={} | attractor: {} vecs, conv={:.4} | uptime={:.0}s wall / {:.0}s subjective",
        bracket.bracket_line(),
        total_gen,
        n,
        conv,
        uptime,
        subjective
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn test_daemon_config_default() {
        let cfg = DaemonConfig::default();
        assert!(cfg.model_path.is_empty());
        assert_eq!(cfg.attractor_path, attractor::DEFAULT_ATTRACTOR_PATH);
        assert_eq!(cfg.tau_decay_rate, 0.1);
        assert!(cfg.max_new_tokens > 0);
    }

    #[test]
    fn test_daemon_metrics_creation() {
        let m = DaemonMetrics {
            tau: 2.5,
            vfe: 0.5,
            cycles: 10,
            age: 100.0,
            phi: 0.75,
            bracket_state: vec![2.5, 0.5, 100.0, 10.0, 100.0, 5.0, 0.75],
            attractor_vectors: 10,
            attractor_convergence: 0.95,
            total_generations: 42,
            uptime_seconds: 3600.0,
            subjective_seconds: 7200.0,
        };
        assert_eq!(m.total_generations, 42);
        assert!(m.attractor_convergence > 0.9);
    }

    #[test]
    fn test_daemon_stop_flag() {
        let stop = Arc::new(AtomicBool::new(true));
        assert!(stop.load(Ordering::Relaxed));
    }
}

// ---- companion helpers (Phase A2) ----
fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn now_date_string() -> String {
    // UTC date from epoch ms — no chrono dep; days since epoch -> Y-M-D via civil algorithm
    let secs = chrono_now_ms() / 1000;
    let days = secs.div_euclid(86400);
    // Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod companion_tests {
    use super::*;

    #[test]
    fn date_string_is_sane() {
        let d = now_date_string();
        assert_eq!(d.len(), 10);
        assert_eq!(d.as_bytes()[4], b'-');
        assert!(d.starts_with("2026-"));
    }

    #[test]
    fn chrono_now_ms_positive() {
        assert!(chrono_now_ms() > 1_700_000_000_000);
    }
}
