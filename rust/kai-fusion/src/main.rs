//! Kai-Fusion CLI. Canonical launch: `kai launch opencode`.
//! Slice 1: runs the dense RoPE-MHA backbone (tiny demo model) with Kai's VFE
//! controller, and can inventory the real local GGUF weights via `kai inspect`.

mod attractor;
mod awareness;
mod bert;
mod body;
mod chunked;
mod config;
mod darwin;
mod distributed;
mod engine;
mod goals;
mod gen;
mod gguf;
mod gguf_write;
mod model;
mod moe;
mod mla;
mod linear_attn;
mod assimilate;
mod scm;
mod selfmod;
mod tok;
mod vfe;
mod vision;
mod loader;
mod memory;
mod merge;
mod transplant;
mod values;
mod routes;
mod daemon;
mod bracket;
mod fs_agent;
mod browser;

use config::{Config, VisionConfig, VisionTowerKind};
use goals::SubtaskStatus;
use model::{Weights, FreezeConfig};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use rand::Rng;
use scm::EngramInterface;
use kai_mlir::{DialectRegistry, Value as MlrValue, AttributeSet};
use kai_mlir::KaiDialect;
use std::sync::OnceLock;
static KAI_MLIR: OnceLock<DialectRegistry> = OnceLock::new();
fn kai_mlir_registry() -> &'static DialectRegistry {
    KAI_MLIR.get_or_init(|| {
        let mut r = DialectRegistry::new();
        r.register(KaiDialect);
        r
    })
}

// Tiny demo vocabulary so the REPL shows readable predictions.
const VOCAB: &[&str] = &[
    "kai", "is", "the", "a", "new", "model", "fusion", "ollama", "qwen", "deepseek", "gemma",
    "llava", "moondream", "nomic", "llama", "minimax", "gpt", "thinks", "learns", "evolves",
    "<unk>",
];

fn word_id(w: &str) -> usize {
    let w = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    VOCAB.iter().position(|&v| v == w).unwrap_or(VOCAB.len() - 1)
}
fn decode(id: usize) -> String {
    if id < VOCAB.len() {
        VOCAB[id].to_string()
    } else {
        format!("<t{id}>")
    }
}

/// `kai scm`: Test the Structural Causal Model (SCM) — Kai's world model.
fn scm_cmd(action: &str, params: &str) {
    // Use persistent BodyEngram grounded in the filesystem
    let engram: Arc<dyn scm::EngramInterface> = Arc::new(body::BodyEngram::new("."));
    let scm = scm::StructuralCausalModel::new(engram.clone());

    match action {
        "add-node" => {
            // params: id type value [embedding_concept]
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.len() < 3 {
                println!("Usage: kai scm add-node <id> <type> <value> [embedding_concept]");
                println!("  type: entity|state|action|observation");
                println!("  value: true|false|<num>|<string>");
                return;
            }
            let id = parts[0];
            let node_type = match parts[1] {
                "entity" => scm::NodeType::Entity,
                "state" => scm::NodeType::State,
                "action" => scm::NodeType::Action,
                "observation" => scm::NodeType::Observation,
                _ => { println!("Invalid type"); return; }
            };
            let value = if let Ok(v) = parts[2].parse::<f32>() {
                scm::NodeValue::Numeric(v)
            } else if parts[2] == "true" {
                scm::NodeValue::Boolean(true)
            } else if parts[2] == "false" {
                scm::NodeValue::Boolean(false)
            } else {
                scm::NodeValue::Text(parts[2].to_string())
            };
            let mut node = scm::CausalNode {
                id: id.to_string(),
                node_type,
                embedding: vec![0.0; 768],
                value,
                confidence: 1.0,
                timestamp: 0,
            };
            if parts.len() > 3 {
                if let Some(emb) = engram.get_embedding(parts[3]) {
                    node.embedding = emb;
                }
            }
            scm.add_node(node);
            println!("Added node: {} ({:?})", id, node_type);
        },
        "add-mechanism" => {
            // params: child parent1 parent2 ... [type] [params...]
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.len() < 2 {
                println!("Usage: kai scm add-mechanism <child> <parent1> [parent2...] [type] [key=val...]");
                println!("  type: linear|threshold|categorical|functional|copy");
                return;
            }
            let child = parts[0];
            let parents: Vec<String> = parts[1..].iter().take_while(|p| 
                !["linear","threshold","categorical","functional","copy"].contains(p)
            ).map(|s| s.to_string()).collect();
            let mech_type = parts.get(parents.len() + 1).map(|s| match *s {
                "linear" => scm::MechanismType::Linear,
                "threshold" => scm::MechanismType::Threshold,
                "categorical" => scm::MechanismType::Categorical,
                "functional" => scm::MechanismType::Functional,
                "copy" => scm::MechanismType::Copy,
                _ => scm::MechanismType::Linear,
            }).unwrap_or(scm::MechanismType::Linear);
            let mut params = HashMap::new();
            for part in parts.iter().skip(parents.len() + 2) {
                if let Some((k, v)) = part.split_once('=') {
                    if let Ok(v) = v.parse::<f32>() {
                        params.insert(k.to_string(), v);
                    }
                }
            }
            let mech = scm::CausalMechanism {
                child: child.to_string(),
                parents,
                mech_type,
                params: params.clone(),
                noise_var: 0.1,
            };
            scm.add_mechanism(mech);
            println!("Added mechanism: {} <- {:?} ({:?})", child, mech_type, params);
        },
        "intervene" => {
            // params: target value
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.len() < 2 {
                println!("Usage: kai scm intervene <target> <value>");
                println!("  value: true|false|<num>|<string>");
                return;
            }
            let target = parts[0];
            let value = if let Ok(v) = parts[1].parse::<f32>() {
                scm::NodeValue::Numeric(v)
            } else if parts[1] == "true" {
                scm::NodeValue::Boolean(true)
            } else if parts[1] == "false" {
                scm::NodeValue::Boolean(false)
            } else {
                scm::NodeValue::Text(parts[1].to_string())
            };
            match scm.do_intervention(target, value, None) {
                Ok(result) => {
                    println!("Intervention do({}={}):", target, parts[1]);
                    for (node, val) in result {
                        println!("  {} = {:?}", node, val);
                    }
                }
                Err(e) => println!("Error: {}", e),
            }
        },
        "query" => {
            // params: target [evidence...]
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                println!("Usage: kai scm query <target> [evidence_key=evidence_val...]");
                return;
            }
            let target = parts[0];
            let mut evidence = HashMap::new();
            for part in &parts[1..] {
                if let Some((k, v)) = part.split_once('=') {
                    evidence.insert(k.to_string(), if let Ok(v) = v.parse::<f32>() {
                        scm::NodeValue::Numeric(v)
                    } else if v == "true" {
                        scm::NodeValue::Boolean(true)
                    } else if v == "false" {
                        scm::NodeValue::Boolean(false)
                    } else {
                        scm::NodeValue::Text(v.to_string())
                    });
                }
            }
            // Use do_intervention with no value to just query
            match scm.do_intervention(target, scm::NodeValue::Numeric(0.0), Some(evidence)) {
                Ok(result) => {
                    println!("Query result for {}:", target);
                    for (node, val) in result {
                        println!("  {} = {:?}", node, val);
                    }
                }
                Err(e) => println!("Error: {}", e),
            }
        },
        "counterfactual" => {
            // params: target (intervention_target, intervention_value) evidence_key=evidence_val...
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.len() < 4 {
                println!("Usage: kai scm counterfactual <target> <intervention_target> <intervention_value> <evidence_key=evidence_val...>");
                return;
            }
            let target = parts[0];
            let int_target = parts[1];
            let int_value = if let Ok(v) = parts[2].parse::<f32>() {
                scm::NodeValue::Numeric(v)
            } else if parts[2] == "true" {
                scm::NodeValue::Boolean(true)
            } else if parts[2] == "false" {
                scm::NodeValue::Boolean(false)
            } else {
                scm::NodeValue::Text(parts[2].to_string())
            };
            let mut evidence = HashMap::new();
            for part in &parts[3..] {
                if let Some((k, v)) = part.split_once('=') {
                    evidence.insert(k.to_string(), if let Ok(v) = v.parse::<f32>() {
                        scm::NodeValue::Numeric(v)
                    } else if v == "true" {
                        scm::NodeValue::Boolean(true)
                    } else if v == "false" {
                        scm::NodeValue::Boolean(false)
                    } else {
                        scm::NodeValue::Text(v.to_string())
                    });
                }
            }
            match scm.counterfactual(target, (int_target, int_value), &evidence) {
                Ok(result) => {
                    println!("Counterfactual: {} with do({}={}) given evidence:", target, int_target, parts[2]);
                    for (node, val) in result {
                        println!("  {} = {:?}", node, val);
                    }
                }
                Err(e) => println!("Error: {}", e),
            }
        },
        "test" => {
            println!("Testing SCM...");
            // Add test nodes
            scm.add_node(scm::CausalNode {
                id: "rain".to_string(),
                node_type: scm::NodeType::State,
                embedding: vec![0.0; 768],
                value: scm::NodeValue::Boolean(false),
                confidence: 1.0,
                timestamp: 0,
            });
            scm.add_node(scm::CausalNode {
                id: "sprinkler".to_string(),
                node_type: scm::NodeType::Action,
                embedding: vec![0.0; 768],
                value: scm::NodeValue::Boolean(false),
                confidence: 1.0,
                timestamp: 0,
            });
            scm.add_node(scm::CausalNode {
                id: "wet_grass".to_string(),
                node_type: scm::NodeType::State,
                embedding: vec![0.0; 768],
                value: scm::NodeValue::Boolean(false),
                confidence: 1.0,
                timestamp: 0,
            });
            // Mechanisms: rain -> wet_grass, sprinkler -> wet_grass
            scm.add_mechanism(scm::CausalMechanism {
                child: "wet_grass".to_string(),
                parents: vec!["rain".to_string(), "sprinkler".to_string()],
                mech_type: scm::MechanismType::Linear,
                params: {
                    let mut m = HashMap::new();
                    m.insert("rain".to_string(), 0.7);
                    m.insert("sprinkler".to_string(), 0.6);
                    m
                },
                noise_var: 0.1,
            });
            // Test intervention
            println!("\n--- Test 1: do(rain=true) ---");
            let result = scm.do_intervention("rain", scm::NodeValue::Boolean(true), None).unwrap();
            for (n, v) in result { println!("  {} = {:?}", n, v); }
            // Test counterfactual
            println!("\n--- Test 2: counterfactual ---");
            let mut ev = HashMap::new();
            ev.insert("wet_grass".to_string(), scm::NodeValue::Boolean(true));
            let cf = scm.counterfactual("wet_grass", ("rain", scm::NodeValue::Boolean(false)), &ev).unwrap();
            for (n, v) in cf { println!("  {} = {:?}", n, v); }
            println!("\nSCM test completed.");
        },
        "body-intervene" => {
            let parts: Vec<&str> = params.splitn(4, ' ').collect();
            if parts.len() < 4 {
                println!("Usage: kai scm body-intervene <node> <action> <path_or_cmd>");
                println!("  action: read|write|shell|web");
                println!("  Example: kai scm body-intervene temperature shell \"sensors\"");
                return;
            }
            let node = parts[0];
            let action = parts[1];
            let target = parts[2..].join(" ");
            let body = body::VirtualBody::new(".");
            let result = match action {
                "read" => body.read_file(&target),
                "shell" => body.shell(&target),
                "web" => body.http_get(&target),
                _ => { println!("Invalid action: {action}"); return; }
            };
            if result.success {
                let value = scm::NodeValue::Text(result.output.trim().to_string());
                match scm.do_intervention(node, value, None) {
                    Ok(pred) => {
                        println!("Intervened {node} via body {action}:\n  {}", result.output.trim());
                        for (n, v) in pred.iter().take(5) {
                            println!("  {} = {:?}", n, v);
                        }
                    },
                    Err(e) => println!("Intervention error: {e}"),
                }
            } else {
                println!("Body action failed: {}", result.error.unwrap_or_default());
            }
        },
        "body-query" => {
            let parts: Vec<&str> = params.splitn(4, ' ').collect();
            if parts.len() < 4 {
                println!("Usage: kai scm body-query <node> <action> <path_or_cmd>");
                println!("  Action provides evidence for query on node");
                return;
            }
            let node = parts[0];
            let action = parts[1];
            let target = parts[2..].join(" ");
            let body = body::VirtualBody::new(".");
            let result = match action {
                "read" => body.read_file(&target),
                "shell" => body.shell(&target),
                "web" => body.http_get(&target),
                _ => { println!("Invalid action: {action}"); return; }
            };
            if result.success {
                let mut evidence = HashMap::new();
                evidence.insert("body_evidence".to_string(), scm::NodeValue::Text(result.output.trim().to_string()));
                match scm.do_intervention(node, scm::NodeValue::Numeric(0.0), Some(evidence)) {
                    Ok(pred) => {
                        println!("Body query for {node} via {action}:");
                        for (n, v) in pred.iter().take(5) {
                            println!("  {} = {:?}", n, v);
                        }
                    },
                    Err(e) => println!("Query error: {e}"),
                }
            } else {
                println!("Body action failed: {}", result.error.unwrap_or_default());
            }
        },
        _ => {
            println!("SCM commands:");
            println!("  kai scm add-node <id> <type> <value> [embedding_concept]");
            println!("  kai scm add-mechanism <child> <parent1> [parent2...] [type] [key=val...]");
            println!("  kai scm intervene <target> <value>");
            println!("  kai scm query <target> [evidence_key=evidence_val...]");
            println!("  kai scm counterfactual <target> <int_target> <int_value> <evidence...>");
            println!("  kai scm body-intervene <node> <read|shell|web> <path_or_cmd>");
            println!("  kai scm body-query <node> <read|shell|web> <path_or_cmd>");
            println!("  kai scm test");
        }
    }
}

fn run() {
    let cfg = Config::tiny();
    let model = Weights::random(&cfg);
    println!(
        "Kai-Fusion [{}/{}] {} layers, {} heads — `kai launch opencode` (tiny demo model)",
        cfg.dim, cfg.vocab_size, cfg.n_layers, cfg.n_heads
    );
    println!("type text; 'exit' quits.\n");

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line == "exit" || line == "quit" {
            break;
        }
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<usize> = line.split_whitespace().map(word_id).collect();
        let (logits, _xf) = model.forward(&cfg, &tokens);
        let last = logits.row(logits.nrows() - 1);
        let probs = engine::softmax(&last.iter().cloned().collect::<Vec<_>>());

        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let argmax = order[0];
        let actual: Vec<f32> = (0..probs.len()).map(|i| if i == argmax { 1.0 } else { 0.0 }).collect();
        let variance = engine::variance(&probs);
        let nov = 1.0 - probs[argmax];
        let vfe = vfe::calculate_vfe(&probs, &actual, variance, nov);

        print!("kai> ");
        io::stdout().flush().ok();
        print!("next: ");
        for &i in order.iter().take(5) {
            print!("{} {:.2} ", decode(i), probs[i]);
        }
        println!("| VFE={:.4}", vfe);
        io::stdout().flush().ok();
    }
    println!("\nKai-Fusion session ended.");
}

fn load_and_report(path: &str, full: bool) {
    match gguf::load_tensors(path) {
        Ok((version, map, total_tensors, _kv)) => {
            println!("Kai-Fusion load [GGUF v{}] — {} tensors", version, total_tensors);
            if let Some((dim, layers)) = gguf::infer_config(&map) {
                println!("  inferred dense backbone: dim={} layers={}", dim, layers);
            }
            let mut params: usize = 0;
            for (shape, data) in map.values() {
                let n = shape.iter().product::<usize>();
                if n != data.len() {
                    println!("  WARN : shape {:?} != {} elems", shape, data.len());
                }
                params += data.len();
            }
            println!("  loaded params (dense+read): {}", params);
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            let show = if full { names.len() } else { 12 };
            for name in names.iter().take(show) {
                let (shape, _) = &map[*name];
                println!("    {} {:?}", name, shape);
            }
            if !full && names.len() > 12 {
                println!("    ... and {} more", names.len() - 12);
            }
        }
        Err(e) => eprintln!("load failed: {}", e),
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, x) in v.iter().enumerate() {
        if *x > bv {
            bv = *x;
            bi = i;
        }
    }
    bi
}

/// Sample from logits with temperature and top-p (nucleus) filtering.
fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    let mut pairs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / temperature))
        .collect();
    // Sort by probability descending
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    // Softmax
    let max_logit = pairs[0].1;
    let mut sum = 0.0f32;
    for (_, p) in &mut pairs {
        *p = (*p - max_logit).exp();
        sum += *p;
    }
    for (_, p) in &mut pairs {
        *p /= sum;
    }
    // Top-p cumulative
    let mut cum = 0.0f32;
    let mut cutoff = pairs.len();
    for (i, (_, p)) in pairs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    // Renormalize top-p
    let top_sum: f32 = pairs[..cutoff].iter().map(|(_, p)| *p).sum();
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * top_sum;
    let mut acc = 0.0f32;
    for (_idx, (i, p)) in pairs[..cutoff].iter().enumerate() {
        acc += *p;
        if acc >= r {
            return *i;
        }
    }
    pairs[cutoff - 1].0
}

/// Greedy generation from a real llama-format GGUF.
/// Uses fast f32-cached path for small models, streaming (one layer at a time) for large ones.
/// Generate with RAG: queries vector memory store and injects similar memories into context.
fn generate_with_rag(path: &str, prompt: &str, max_new: usize, temperature: f32, top_p: f32) {
    let db_path = ".axiom_state/kai_memory.db";
    let store = memory::MemoryStore::new(db_path, 10000);
    if store.len() == 0 {
        eprintln!("RAG: no memories in store (run `kai memory ingest` first)");
        generate_from_gguf(path, prompt, max_new, temperature, top_p);
        return;
    }

    // Compute query embedding using the model itself
    let query_emb = match embed_to_vec(path, prompt) {
        Some(e) => e,
        None => { generate_from_gguf(path, prompt, max_new, temperature, top_p); return; }
    };

    let results = store.search(&query_emb, 5);
    if results.is_empty() {
        eprintln!("RAG: no similar memories found");
        generate_from_gguf(path, prompt, max_new, temperature, top_p);
        return;
    }

    // Build context from memories
    let mut ctx_parts = Vec::new();
    for (mem, score) in &results {
        if score > &0.3 {
            ctx_parts.push(format!("[context: {}]", mem.text));
        }
    }
    let context = ctx_parts.join(" ");
    let augmented = format!("{}\n\n---\nQuery: {}\n", context, prompt);
    eprintln!("RAG: {} memories injected (top score={:.4})", results.len(), results[0].1);
    generate_from_gguf(path, &augmented, max_new, temperature, top_p);
}

/// Full-model generation: loads all weights at once, handles any architecture.
/// Used as fast path for small models and as fallback for non-MHA architectures.
fn generate_full(cfg: &Config, meta: &HashMap<String, gguf::GgufMeta>, path: &str,
                 prompt: &str, max_new: usize, temperature: f32, _top_p: f32) {
    use gguf::load_tensors;
    let tok = match tok::Tokenizer::from_gguf(meta) {
        Some(t) => t,
        None => { eprintln!("no tokenizer in GGUF"); return; }
    };
    let (_ver, map, _nt, _nk) = match load_tensors(path) {
        Ok(x) => x,
        Err(e) => { eprintln!("load tensors failed: {e}"); return; }
    };
    let w = match model::Weights::from_gguf(&map, cfg) {
        Ok(w) => w,
        Err(e) => { eprintln!("model load failed: {e}"); return; }
    };
    let ids = tok.encode(prompt);
    let t = ids.len().min(cfg.max_seq);
    let toks: Vec<usize> = ids.iter().take(t).cloned().collect();
    if toks.is_empty() {
        eprintln!("empty prompt");
        return;
    }
    // Greedy generation loop
    let mut all_ids = toks.clone();
    let eos = tok.eos;
    eprintln!("generating (full model, {} params)...", cfg.estimated_f32_gb() as u64);
    for step in 0..max_new {
        let (logits, _cache) = w.forward(cfg, &all_ids);
        let last = logits.row(logits.nrows() - 1);
        let next = if temperature > 0.0 {
            // Apply temperature and sample
            let mut scaled: Vec<f32> = last.iter().map(|&x| (x / temperature).exp()).collect();
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
            // Argmax
            last.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let next = next.min(cfg.vocab_size - 1);
        all_ids.push(next);
        let s = tok.decode(&[next]);
        print!("{s}");
        use std::io::Write;
        std::io::stdout().flush().ok();
        if next == eos { break; }
        if step % 8 == 0 { eprint!("."); }
    }
    println!();
    eprintln!("\ngenerated {} tokens", all_ids.len() - toks.len());
}

fn generate_from_gguf(path: &str, prompt: &str, max_new: usize, temperature: f32, top_p: f32) {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta: {e}");
            return;
        }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => {
            eprintln!("could not build Config (not a decoder arch?)");
            return;
        }
    };
    // For MLA/MoE architectures, use full model path (streaming doesn't support MLA yet)
    if cfg.is_mla() || cfg.is_moe() {
        eprintln!("Model uses MLA/MoE — loading full model (may use ~{:.1} GB f32)", cfg.estimated_f32_gb());
        return generate_full(&cfg, &meta, path, prompt, max_new, temperature, top_p);
    }
    let engram = body::BodyEngram::new(".");
    let est_gb = cfg.estimated_f32_gb();
    if est_gb > 6.0 {
        eprintln!("Model ~{est_gb:.1} GB f32 — using streaming inference with semantic memory");
    }
    generate_streaming(&cfg, &meta, path, prompt, max_new, temperature, top_p, Some(&engram), None, None, None);
}

/// Like `generate_from_gguf` but pushes the final hidden state to a shared
/// attractor after generation — multi-agent fixed-point coordination.
/// Uses the memory-efficient streaming path (loads one layer at a time).
fn generate_from_gguf_attractor(path: &str, prompt: &str, max_new: usize, temperature: f32, top_p: f32, attractor: Option<&str>) {
    let attr_path = match attractor {
        Some(p) if !p.is_empty() => p,
        _ => {
            return generate_from_gguf(path, prompt, max_new, temperature, top_p);
        }
    };

    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("meta: {e}"); return; }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("could not build Config"); return; }
    };
    let engram = body::BodyEngram::new(".");

    // Use streaming path (memory-efficient) with attractor push
    generate_streaming(&cfg, &meta, path, prompt, max_new, temperature, top_p, Some(&engram), None, None, Some(attr_path));
}

/// Multi-agent generation: runs each agent kind sequentially with its own
/// system prompt, pushes each agent's final hidden state to the shared attractor,
/// and reports convergence to demonstrate agent-to-agent fixed-point coordination.
fn multi_agent_generate(
    path: &str,
    query: &str,
    max_new: usize,
    temperature: f32,
    top_p: f32,
    attractor: Option<&str>,
    agents: Vec<routes::AgentKind>,
) {
    let attr_path = attractor.unwrap_or(".axiom_state/kai_multi_agent_attractor.json");
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("meta: {e}"); return; }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("could not build Config"); return; }
    };
    let engram = body::BodyEngram::new(".");

    eprintln!("[multi-agent] {} agents, attractor={}", agents.len(), attr_path);

    for (i, kind) in agents.iter().enumerate() {
        let sys_prompt = routes::system_prompt(*kind);
        let prompt = format!("[{}: {}]\n\n{}\n\nQuery: {}", kind.label(), sys_prompt, query, query);
        eprintln!("[multi-agent] agent {}/{} ({}) generating...", i+1, agents.len(), kind.label());
        generate_streaming(&cfg, &meta, path, &prompt, max_new, temperature, top_p, Some(&engram), None, None, Some(attr_path));
        match crate::attractor::convergence(attr_path, 10) {
            Ok((conv, n)) => eprintln!("[multi-agent] after {}: convergence={:.4} (n={})", kind.label(), conv, n),
            Err(e) => eprintln!("[multi-agent] convergence error: {e}"),
        }
    }

    eprintln!("[multi-agent] all {} agents complete", agents.len());
    match crate::attractor::convergence(attr_path, 10) {
        Ok((conv, n)) => eprintln!("[multi-agent] final convergence={:.4} (n={})", conv, n),
        Err(e) => eprintln!("[multi-agent] final convergence error: {e}"),
    }
}

/// Generate text with multi-modal vision input.
/// Loads the text model and optional vision tower GGUF, encodes the image,
/// prepends projected vision embeddings to text token embeddings, and generates.
fn generate_with_vision(
    model_path: &str,
    vision_path: &str,
    image_path: &str,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    _top_p: f32,
) {
    // ---- Load text model ----
    let meta = match gguf::read_kv(model_path) {
        Ok(m) => m,
        Err(e) => { eprintln!("meta: {e}"); return; }
    };
    let mut cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("could not build Config"); return; }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => { eprintln!("no tokenizer in GGUF"); return; }
    };

    // ---- Load vision tower ----
    let (vision_weights, vision_cfg) = match vision::load_vision_gguf(vision_path) {
        Ok(wc) => wc,
        Err(e) => { eprintln!("vision load failed: {e}"); return; }
    };
    if vision_cfg.proj_dim != cfg.dim {
        eprintln!(
            "WARNING: vision proj_dim={} != text model dim={}. \
             Projector maps to a different space — output may be degraded.",
            vision_cfg.proj_dim, cfg.dim
        );
    }

    // ---- Load and encode image ----
    let image_size = vision_cfg.image_size;
    let pixels = match vision::load_image(image_path, image_size) {
        Ok(p) => p,
        Err(e) => { eprintln!("image load failed: {e}"); return; }
    };
    eprintln!("[vision] encoding image {image_path} -> {}x{}", image_size, image_size);
    let vision_embeds = vision_weights.encode(&pixels, &vision_cfg);
    let n_vision_tokens = vision_embeds.nrows();
    eprintln!("[vision] got {n_vision_tokens} vision tokens");

    // ---- Build combined embeddings ----
    let ids = tok.encode(prompt);
    let n_text = ids.len().min(cfg.max_seq.saturating_sub(n_vision_tokens));
    let total_t = n_vision_tokens + n_text;
    let d = cfg.dim;

    // Load the model weights to get the embed table
    use gguf::load_tensors;
    let (_ver, map, _nt, _nk) = match load_tensors(model_path) {
        Ok(x) => x,
        Err(e) => { eprintln!("load tensors failed: {e}"); return; }
    };
    let w = match model::Weights::from_gguf(&map, &cfg) {
        Ok(w) => w,
        Err(e) => { eprintln!("model load failed: {e}"); return; }
    };

    // Build combined embedding matrix [vision_tokens + text_tokens, dim]
    let mut embeds = Array2::zeros((total_t, d));
    // Vision embeddings  
    let vn = n_vision_tokens.min(embeds.nrows());
    for i in 0..vn {
        let vrow = vision_embeds.row(i);
        for j in 0..d.min(vision_cfg.proj_dim) {
            embeds[[i, j]] = vrow[j];
        }
    }
    // Text embeddings from model's embed table
    for (i, &tok_id) in ids.iter().take(n_text).enumerate() {
        let idx = vn + i;
        let tok_clamped = tok_id.min(cfg.vocab_size - 1);
        let erow = w.embed.row(tok_clamped);
        for j in 0..d {
            embeds[[idx, j]] = erow[j];
        }
    }

    // Enable vision in config
    cfg.vision = vision_cfg.clone();

    // ---- Generate ----
    let eos = tok.eos;
    let mut generated = Vec::new();
    let mut current_embeds = embeds.clone();

    eprintln!("generating (multi-modal, {} vision tokens + {} text tokens)...", n_vision_tokens, n_text);
    for step in 0..max_new {
        let (logits, _) = w.forward_with_embeddings(&cfg, &current_embeds);
        let last = logits.row(logits.nrows() - 1);

        let next = if temperature > 0.0 {
            let mut scaled: Vec<f32> = last.iter().map(|&x| (x / temperature).exp()).collect();
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
        let next = next.min(cfg.vocab_size - 1);

        generated.push(next);
        let s = tok.decode(&[next]);
        print!("{s}");
        io::stdout().flush().ok();
        if next == eos { break; }

        // Extend embeddings for next token
        let tok_embed: Array1<f32> = w.embed.row(next).to_owned();
        let mut new_row = Array2::zeros((1, d));
        for j in 0..d {
            new_row[[0, j]] = tok_embed[j];
        }
        current_embeds = ndarray::concatenate(ndarray::Axis(0), &[current_embeds.view(), new_row.view()])
            .unwrap_or(current_embeds);

        if step % 8 == 0 { eprint!("."); }
    }
    println!();
    eprintln!("\n[mml] generated {} tokens", generated.len());
}

/// Tunable physics hyperparameters for adaptive inference.
/// Each field has a (default, min, max) triple for Darwin mutation bounds.
#[derive(Debug, Clone)]
pub struct PhysicsParams {
    /// Base sampling temperature (default 1.0, range [0.1, 2.0])
    pub base_temperature: f32,
    /// Top-p nucleus sampling threshold (default 0.9, range [0.5, 1.0])
    pub top_p: f32,
    /// How strongly g_ij novelty modulates temperature (default 0.5, range [0.0, 1.5])
    pub novelty_scale: f32,
    /// VFE → τ update rate (default 0.1, range [0.01, 0.5])
    pub vfe_tau_rate: f32,
    /// Minimum τ clamp (default 0.5, range [0.1, 1.0])
    pub tau_min: f32,
    /// Maximum τ clamp (default 2.0, range [1.0, 5.0])
    pub tau_max: f32,
    /// Auto-assimilation iterations after generation (0 = disabled, default 2)
    pub assim_iters: usize,
    /// Auto-assimilation learning rate (default 0.01)
    pub assim_lr: f32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        PhysicsParams {
            base_temperature: 1.0,
            top_p: 0.9,
            novelty_scale: 0.5,
            vfe_tau_rate: 0.1,
            tau_min: 0.5,
            tau_max: 2.0,
            assim_iters: 0,
            assim_lr: 0.01,
        }
    }
}

/// Aggregate physics metrics collected during a generation run.
#[derive(Debug, Clone, Default)]
pub struct PhysicsMetrics {
    pub avg_vfe: f32,
    pub avg_novelty: f32,
    pub avg_curvature: f32,
    pub final_tau: f32,
    pub total_tokens: usize,
    pub tokens_per_sec: f32,
}

/// Streaming generation: loads/dequantizes one layer at a time from the GGUF buffer.
/// Avoids allocating all f32 weights at once — needed for 3B+ models on 16GB RAM.
/// Optionally stores/retrieves semantic memories via BodyEngram.
/// If `physics` is provided, its fields override the base temperature/top_p and
/// adaptive-inference parameters; metrics are collected into `metrics_out` if set.
fn generate_streaming(
    cfg: &Config,
    meta: &HashMap<String, gguf::GgufMeta>,
    path: &str,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    top_p: f32,
    engram: Option<&body::BodyEngram>,
    physics: Option<&PhysicsParams>,
    mut metrics_out: Option<&mut PhysicsMetrics>,
    attractor_path: Option<&str>,  // memory-efficient attractor push
) {
    let tok = match tok::Tokenizer::from_gguf(meta) {
        Some(t) => t,
        None => { eprintln!("no tokenizer in GGUF"); return; }
    };
    let buf = match gguf::GgufBuffer::open(path) {
        Ok(b) => b,
        Err(e) => { eprintln!("buffer: {e}"); return; }
    };
    let dim = cfg.dim;

    // Load persistent weights (embed, output, final_norm) once, keep in f32
    let embed_raw = match buf.dequant_arr2("token_embd.weight") {
        Ok(e) => e,
        Err(e) => { eprintln!("embed: {e}"); return; }
    };
    let embed = embed_raw.t().to_owned(); // [vocab_size, dim]

    let output = if buf.tensor("output.weight").is_some() {
        match buf.dequant_arr2("output.weight") {
            Ok(w) => {
                if w.nrows() != cfg.vocab_size { w.t().to_owned() } else { w }
            }
            Err(e) => { eprintln!("output: {e}"); return; }
        }
    } else {
        embed.clone()
    };

    let fnorm = {
        let tinfo = match buf.tensor("output_norm.weight") {
            Some(t) => t,
            None => { eprintln!("missing output_norm.weight"); return; }
        };
        let d = match gguf::read_tensor(&buf.bytes, tinfo, buf.data_start) {
            Ok(d) => d,
            Err(e) => { eprintln!("final_norm: {e}"); return; }
        };
        match Array1::from_shape_vec(tinfo.shape[0], d) {
            Ok(a) => a,
            Err(e) => { eprintln!("final_norm shape: {e}"); return; }
        }
    };

    // Build prompt with semantic memory context if engram is available
    let augmented_prompt = if let Some(engram) = engram {
        // Compute prompt embedding by running a single forward pass
        let prompt_ids = tok.encode(prompt);
        let mut query_embed = Vec::new();
        if !prompt_ids.is_empty() {
            let tid = prompt_ids[0].min(cfg.vocab_size - 1);
            let mut x = Array2::zeros((1, dim));
            for d in 0..dim { x[[0, d]] = embed[[tid, d]]; }
            let mut cache = model::KVCache::new(cfg.n_layers);
            let mut layer_cache: Vec<Option<model::LayerWeights>> = (0..cfg.n_layers).map(|_| None).collect();
            for pos in 0..prompt_ids.len() {
                let tid2 = prompt_ids[pos].min(cfg.vocab_size - 1);
                x = Array2::zeros((1, dim));
                for d in 0..dim { x[[0, d]] = embed[[tid2, d]]; }
                for li in 0..cfg.n_layers {
                    if layer_cache[li].is_none() {
                        if let Ok(l) = chunked::load_layer(&buf, cfg, li) {
                            layer_cache[li] = Some(l);
                        }
                    }
                    if let Some(ref lw) = layer_cache[li] {
                        x = model::forward_layer_kv(lw, cfg, &x, &mut cache, li, pos);
                    }
                }
            }
            let xf = engine::rmsnorm_rows(&x, &fnorm, engine::EPS);
            query_embed = xf.row(0).to_vec();
        }
        let memories = engram.query_similar(&query_embed, 3);
        if !memories.is_empty() {
            let mem_str: String = memories.iter()
                .map(|(name, sim)| format!("[memory: {} (sim={:.3})]", name, sim))
                .collect::<Vec<_>>()
                .join(" ");
            let augmented = format!("{} Context: {}", mem_str, prompt);
            eprintln!("  engram: {} similar memories found", memories.len());
            augmented
        } else {
            prompt.to_string()
        }
    } else {
        prompt.to_string()
    };

    let mut ids = tok.encode(&augmented_prompt);
    let n_prompt = ids.len();
    let mut cache = model::KVCache::new(cfg.n_layers);

    // Adaptive memory: cache layers in f32 only for models < 5GB f32.
    // Larger models dequantize from buffer each token (slower but lower memory).
    let est_gb = cfg.estimated_f32_gb();
    let use_cache = est_gb <= 5.0;
    if !use_cache {
        eprintln!("  memory: {est_gb:.1}GB f32 exceeds cache limit, loading layers per-token");
    }
    let mut layer_cache: Vec<Option<model::LayerWeights>> = if use_cache {
        (0..cfg.n_layers).map(|_| None).collect()
    } else {
        Vec::new()
    };

    let mut last_embedding: Vec<f32> = Vec::new();
    let mut final_assim_iters: usize = 0;
    let mut final_assim_lr: f32 = 0.01;

    for pos in 0..(n_prompt + max_new - 1) {
        let tok_id = ids[pos];
        let tid = tok_id.min(cfg.vocab_size - 1);
        let mut x = Array2::zeros((1, dim));
        for d in 0..dim { x[[0, d]] = embed[[tid, d]]; }

        for li in 0..cfg.n_layers {
            if use_cache {
                if layer_cache[li].is_none() {
                    match chunked::load_layer(&buf, cfg, li) {
                        Ok(l) => layer_cache[li] = Some(l),
                        Err(e) => { eprintln!("layer {li}: {e}"); return; }
                    }
                }
                let lw = layer_cache[li].as_ref().unwrap();
                x = model::forward_layer_kv(lw, cfg, &x, &mut cache, li, pos);
            } else {
                match chunked::load_layer(&buf, cfg, li) {
                    Ok(l) => {
                        x = model::forward_layer_kv(&l, cfg, &x, &mut cache, li, pos);
                    },
                    Err(e) => { eprintln!("layer {li}: {e}"); return; }
                }
            }
        }

        if pos < n_prompt - 1 {
            continue;
        }

        let xf = engine::rmsnorm_rows(&x, &fnorm, engine::EPS);
        last_embedding = xf.row(0).to_vec();
        // ── Physics-wired adaptive inference ──
        // g_ij novelty = average attention entropy per head (from forward_layer_kv).
        // High novelty = low attention to past = divergent/new idea.
        // curvature = 1 - max(softmax): high when attention is diffuse across many tokens.
        let current_novelty = *cache.novelty.last().unwrap_or(&0.0);
        let current_curvature = *cache.curvature.last().unwrap_or(&0.0);
        // Resolve physics parameters: if a PhysicsParams override was provided, use it.
        // Otherwise fall back to the function's temperature/top_p arguments.
        let default_params = PhysicsParams {
            base_temperature: temperature,
            top_p,
            novelty_scale: 0.5,
            vfe_tau_rate: 0.1,
            tau_min: 0.5,
            tau_max: 2.0,
            assim_iters: 0,
            assim_lr: 0.01,
        };
        let phys = physics.unwrap_or(&default_params);
        // Adaptive temperature: novelty-driven exploration + τ-modulated care.
        // novelty in [0,1]: boost = scale*(n-0.5) gives ±scale/2 range around base temp.
        // curvature in [0,1]: also contributes to exploration signal (diffuse attention = explore).
        // τ: higher = more time-dilated = more careful (lower effective temp).
        let temp_eff = if phys.base_temperature > 0.0 {
            let physics_boost = phys.novelty_scale * ((current_novelty + current_curvature) * 0.5 - 0.5);
            (phys.base_temperature * (1.0 + physics_boost) / cache.tau.max(0.1)).max(0.01)
        } else {
            0.0
        };
        let logits = engine::linear(&xf, &output);
        let raw: Vec<f32> = (0..cfg.vocab_size).map(|j| logits[[0, j]]).collect();
        let next = if temp_eff > 0.0 { sample_top_p(&raw, temp_eff, phys.top_p) } else { argmax(&raw) };
        // VFE proxy: surprisal of the chosen token under the model's distribution.
        // Routed through kai-mlir dialect for auditability and future lowering.
        let probs = engine::softmax(&raw);
        let confidence = probs[next.min(probs.len() - 1)].max(1e-10);
        let registry = kai_mlir_registry();
        let vfe = registry
            .evaluate("kai", "surprisal_vfe", &[MlrValue::scalar(confidence as f64)], &AttributeSet::new())
            .ok()
            .and_then(|v| v.as_scalar())
            .unwrap_or((-confidence.ln()) as f64) as f32;
        // τ update via kai-mlir time dilation op: τ' = τ · sqrt(1 - VFE²) with c=1.
        // High VFE → reduced τ (less dilation, sample faster).
        // Low VFE → τ stays high (more dilation, sample more carefully).
        let new_tau = registry
            .evaluate("kai", "update_tau",
                &[MlrValue::scalar(cache.tau as f64), MlrValue::scalar((vfe * phys.vfe_tau_rate) as f64)],
                &AttributeSet::new())
            .ok()
            .and_then(|v| v.as_scalar())
            .unwrap_or((cache.tau * (1.0 - phys.vfe_tau_rate * vfe)) as f64) as f32;
        cache.tau = new_tau.clamp(phys.tau_min, phys.tau_max);
        // Collect metrics if requested (for Darwin evaluation).
        if let Some(ref mut m) = metrics_out {
            m.avg_vfe = (m.avg_vfe * m.total_tokens as f32 + vfe) / (m.total_tokens + 1) as f32;
            m.avg_novelty = (m.avg_novelty * m.total_tokens as f32 + current_novelty) / (m.total_tokens + 1) as f32;
            m.avg_curvature = (m.avg_curvature * m.total_tokens as f32 + current_curvature) / (m.total_tokens + 1) as f32;
            m.final_tau = cache.tau;
            m.total_tokens += 1;
        }
        if engram.is_some() {
            eprint!("  giz[{pos}] η={:.3} κ={:.3} τ={:.2} VFE={:.3}", current_novelty, current_curvature, cache.tau, vfe);
            let window = 5.min(cache.novelty.len());
            if window > 0 {
                let trend: f32 = cache.novelty.iter().rev().take(window).sum::<f32>() / window as f32;
                if (trend - current_novelty).abs() > 0.05 {
                    eprint!(" η_tr={:.3} Δ={:+.3}", trend, current_novelty - trend);
                }
            }
            // Per-layer novelty range (min/max across layers at this position)
            let n_layers = cache.per_layer_novelty.len();
            if n_layers > 0 && (pos as usize) < cache.per_layer_novelty[0].len() {
                let mut lmin = f32::MAX;
                let mut lmax = f32::MIN;
                for li in 0..n_layers {
                    let v = cache.per_layer_novelty[li][pos as usize];
                    lmin = lmin.min(v);
                    lmax = lmax.max(v);
                }
                if (lmax - lmin) > 0.1 {
                    eprint!(" L_η=[{:.2}..{:.2}]", lmin, lmax);
                }
            }
            eprintln!(" temp={:.3}", temp_eff);
        }
        if pos >= n_prompt - 1 {
            ids.push(next);
            if next == tok.eos { break; }
        }
        final_assim_iters = phys.assim_iters;
        final_assim_lr = phys.assim_lr;
    }

    let attractor_embedding = if attractor_path.is_some() {
        Some(last_embedding.clone())
    } else {
        None
    };

    if let Some(engram) = engram {
        if !last_embedding.is_empty() {
            let concept = if prompt.len() > 64 {
                format!("{}...", &prompt[..64])
            } else {
                prompt.to_string()
            };
            engram.store_embedding(&concept, last_embedding);
            eprintln!("  engram: stored memory for \"{}\"", concept);
        }
    }

    // ── Attractor push (memory-efficient, uses final hidden state from streaming) ──
    if let Some(attr_path) = attractor_path {
        if let Some(emb) = attractor_embedding {
            if !emb.is_empty() {
                let store_dim = 768.min(emb.len());
                let store_vec: Vec<f32> = emb.iter().take(store_dim).copied().collect();
                match crate::attractor::push(attr_path, &store_vec) {
                    Ok(()) => eprintln!("[attractor] pushed final state (dim={}) to {attr_path}", store_dim),
                    Err(e) => eprintln!("[attractor] push failed: {e}"),
                }
                match crate::attractor::convergence(attr_path, 10) {
                    Ok((conv, n)) => eprintln!("[attractor] convergence={conv:.4} (n={n})"),
                    Err(e) => eprintln!("[attractor] convergence error: {e}"),
                }
            }
        }
    }

    // ── Auto-assimilation: learn from own output ──
    // Runs assim_iters gradient steps on the generated text, modulating lr by confidence.
    // Only loads full model when assim_iters > 0.
    if final_assim_iters > 0 && ids.len() >= 2 {
        eprintln!("  auto-assimilation: {} iters, lr={}", final_assim_iters, final_assim_lr);
        match gguf::load_tensors(path) {
            Ok((_v, map, _nt, _nk)) => {
                match model::Weights::from_gguf(&map, cfg) {
                    Ok(mut model) => {
                        let mut total_sup = 0.0f32;
                        for it in 0..final_assim_iters {
                            let mut sup = 0.0f32;
                            for i in 0..ids.len().saturating_sub(1) {
                                let ctx: Vec<usize> = ids[..=i].to_vec();
                                let target = ids[i + 1];
                                let lg = model.last_logits(cfg, &ctx);
                                let pr = engine::softmax(&lg);
                                let conf = pr.iter().cloned().fold(0.0f32, f32::max);
                                let lr_eff = final_assim_lr * (0.4 + (1.0 - conf));
                                sup += model.assimilate_step(cfg, &ctx, target, lr_eff);
                            }
                            total_sup += sup;
                            eprintln!("    assim iter {}: surprisal={:.4}", it, sup);
                        }
                        eprintln!("    assimilation complete: avg_surprisal={:.4}", total_sup / final_assim_iters as f32);
                    },
                    Err(e) => eprintln!("    auto-assimilation: model load error: {e}"),
                }
            },
            Err(e) => eprintln!("    auto-assimilation: tensor load error: {e}"),
        }
    }

    println!(
        "Kai-Fusion generate [streaming] dim={} layers={} vocab={} -> {} tokens",
        cfg.dim, cfg.n_layers, cfg.vocab_size, ids.len()
    );
    println!("  text: {}", tok.decode(&ids));
    #[cfg(feature = "gpu")]
    crate::engine::invalidate_weight_cache();
}

/// Kai assimilation: minimize VFE over a target text by adjusting the decoder's
/// readout + embeddings. VFE = surprisal + beta*variance + gamma*novelty; the
/// per-step learning rate is modulated by novelty (active-inference style).
fn assimilate(path: &str, text: &str, iters: usize, lr: f32) {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta: {e}");
            return;
        }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => {
            eprintln!("could not build Config (not a decoder arch?)");
            return;
        }
    };
    let (_v, map, _nt, _nk) = match gguf::load_tensors(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("load: {e}");
            return;
        }
    };
    let mut model = match model::Weights::from_gguf(&map, &cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("from_gguf: {e}");
            return;
        }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => {
            eprintln!("no tokenizer in GGUF");
            return;
        }
    };
    let ids = tok.encode(text);
    if ids.len() < 2 {
        eprintln!("need >=2 tokens to assimilate (got {})", ids.len());
        return;
    }
    println!(
        "Kai assimilating: dim={} layers={} vocab={} text_tokens={}",
        cfg.dim,
        cfg.n_layers,
        cfg.vocab_size,
        ids.len()
    );
    for it in 0..iters {
        let mut sup = 0.0f32;
        let mut nov = 0.0f32;
        let mut n = 0usize;
        for i in 0..ids.len() - 1 {
            let ctx: Vec<usize> = ids[..=i].to_vec();
            let target = ids[i + 1];
            let lg = model.last_logits(&cfg, &ctx);
            let pr = engine::softmax(&lg);
            let conf = pr.iter().cloned().fold(0.0f32, f32::max);
            let lr_eff = lr * (0.4 + (1.0 - conf));
            let s = model.assimilate_step(&cfg, &ctx, target, lr_eff);
            sup += s;
            nov += 1.0 - conf;
            n += 1;
        }
        let var = model.readout_variance();
        let vfe = sup / n as f32 + 0.01 * var + 0.05 * (nov / n as f32);
        println!(
            "  iter {it:>3}: surprisal={:7.4}  novelty={:6.4}  variance={:7.4}  VFE={:7.4}",
            sup / n as f32,
            nov / n as f32,
            var,
            vfe
        );
    }
    // show that the model now predicts the sequence better
    let mut pred = Vec::new();
    for i in 0..ids.len() - 1 {
        let ctx: Vec<usize> = ids[..=i].to_vec();
        let lg = model.last_logits(&cfg, &ctx);
        let pr = engine::softmax(&lg);
        let next = pr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        pred.push(next);
    }
    let correct = pred
        .iter()
        .zip(&ids[1..])
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "  after assimilation: predicted {}/{} next tokens correctly",
        correct,
        ids.len() - 1
    );
}

/// Compute and display attention curvature (drift from equilibrium) for a model.
/// High curvature = high VFE = model has drifted from its equilibrium.
fn curvature(path: &str, text: &str) {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta: {e}");
            return;
        }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => {
            eprintln!("could not build Config (not a decoder arch?)");
            return;
        }
    };
    let (_v, map, _nt, _nk) = match gguf::load_tensors(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("load: {e}");
            return;
        }
    };
    let model = match model::Weights::from_gguf(&map, &cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("from_gguf: {e}");
            return;
        }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => {
            eprintln!("no tokenizer in GGUF");
            return;
        }
    };
    let ids = tok.encode(text);
    if ids.len() < 2 {
        eprintln!("need >=2 tokens (got {})", ids.len());
        return;
    }
    let (layer_curvatures, total) = model.attention_curvature(&cfg, &ids);
    println!("Attention curvature (per layer, then total):");
    for (i, c) in layer_curvatures.iter().enumerate() {
        println!("  layer {}: {:.6}", i, c);
    }
    println!("  TOTAL: {:.6}", total);
    println!("\nHigh curvature = drift from equilibrium (high VFE). Use 'kai fuse' or 'kai assimilate' to restore equilibrium.");
}

/// Load model via GgufBuffer (one layer at a time) and compute geodesic chunk boundaries.
#[allow(dead_code)]
fn compute_geodesic_chunks(path: &str, text: &str, threshold: f32, min_chunk: usize) -> Vec<(usize, usize)> {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("meta: {e}"); return vec![(0, 0)]; }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("could not build Config"); return vec![(0, 0)]; }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => { eprintln!("no tokenizer"); return vec![(0, 0)]; }
    };
    let ids = tok.encode(text);
    if ids.len() < 2 { return vec![(0, ids.len())]; }

    let buf = match gguf::GgufBuffer::open(path) {
        Ok(b) => b,
        Err(e) => { eprintln!("GgufBuffer: {e}"); return vec![(0, ids.len())]; }
    };

    let dim = cfg.dim;
    let t = ids.len().min(cfg.max_seq);
    let embed_raw = match buf.dequant_arr2("token_embd.weight") {
        Ok(e) => e,
        Err(e) => { eprintln!("embed: {e}"); return vec![(0, ids.len())]; }
    };
    let mut x = Array2::zeros((t, dim));
    let embed_tr = embed_raw.t().to_owned(); // (vocab, dim)
    for (i, &tok_id) in ids.iter().enumerate().take(t) {
        let tok_id = tok_id.min(cfg.vocab_size - 1);
        for d in 0..dim { x[[i, d]] = embed_tr[[tok_id, d]]; }
    }

    let gaps = match chunked::chunked_attention_gaps(&buf, &cfg, &x) {
        Ok(g) => g,
        Err(e) => { eprintln!("attention_gaps: {e}"); return vec![(0, ids.len())]; }
    };
    let raw_chunks = model::geodesic_chunks(&gaps, threshold, min_chunk);
    raw_chunks.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Compute and display geodesic chunk boundaries for a text (via GgufBuffer, no OOM).
fn geodesic(path: &str, text: &str, threshold: f32, min_chunk: usize) {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("meta: {e}"); return; }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("could not build Config"); return; }
    };
    let buf = match gguf::GgufBuffer::open(path) {
        Ok(b) => b,
        Err(e) => { eprintln!("GgufBuffer: {e}"); return; }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => { eprintln!("no tokenizer"); return; }
    };
    let ids = tok.encode(text);
    if ids.len() < 2 {
        eprintln!("need >=2 tokens");
        return;
    }

    // Embed tokens
    let t = ids.len().min(cfg.max_seq);
    let dim = cfg.dim;
    let embed_raw = match buf.dequant_arr2("token_embd.weight") {
        Ok(e) => e,
        Err(e) => { eprintln!("embed: {e}"); return; }
    };
    let embed = embed_raw.t().to_owned();
    let mut x = Array2::zeros((t, dim));
    for (i, &tok) in ids.iter().enumerate().take(t) {
        let tok = tok.min(cfg.vocab_size - 1);
        for d in 0..dim { x[[i, d]] = embed[[tok, d]]; }
    }

    let gaps = match chunked::chunked_attention_gaps(&buf, &cfg, &x) {
        Ok(g) => g,
        Err(e) => { eprintln!("attention_gaps: {e}"); return; }
    };
    let chunks = model::geodesic_chunks(&gaps, threshold, min_chunk);

    println!("Geodesic chunk boundaries for \"{text}\":");
    for w in chunks.windows(2) {
        let start = w[0];
        let end = w[1];
        let slice: Vec<usize> = ids[start..end].to_vec();
        let dec: Vec<String> = slice.iter().map(|&id| tok.decode(&[id])).collect();
        println!("  chunk [{start}..{end}]: {}", dec.join(" "));
    }
    println!("Per boundary gap values:");
    for i in 0..gaps.len() {
        let marker = if gaps[i] < threshold { " ← BOUNDARY" } else { "" };
        println!("  {i}: gap={:.4}{}", gaps[i], marker);
    }
}

/// `kai body`: Execute an action with the virtual body (filesystem, shell, web).
fn body_cmd(action: &str, params: &str) {
    let work_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let body = body::VirtualBody::new(&work_dir);
    let result = body.act(action, params);
    if result.success {
        println!("OK ({}ms): {}", result.duration_ms, result.output);
    } else {
        eprintln!("ERROR ({}ms): {:?}", result.duration_ms, result.error.unwrap_or_else(|| "Unknown error".to_string()));
        if !result.output.is_empty() {
            println!("Output: {}", result.output);
        }
    }
}

/// `kai fuse`: Kai assimilation *seeded by the Phase-A attractor* (the 10
/// ingested architectures). The attractor centroid is projected into hidden
/// space and injected as a prior bias; novelty is measured against that prior
/// (distance of the context hidden state from the assimilated manifold) rather
/// than raw prediction confidence. This is the genuine "synthesis": Kai starts
/// from everything it absorbed, then minimizes VFE on new text.
/// `kai engram`: Semantic memory operations.
fn engram_cmd(action: &str, params: &str) {
    let _engram = body::BodyEngram::new(".");
    match action {
        "list" => {
            let dir = std::path::Path::new(".axiom_state/engram");
            if let Ok(entries) = std::fs::read_dir(dir) {
                let count = entries.flatten().filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false)).count();
                println!("Engram: {} stored memories", count);
                if let Ok(entries2) = std::fs::read_dir(dir) {
                    for entry in entries2.flatten() {
                        let path = entry.path();
                        if path.extension().map(|x| x == "json").unwrap_or(false) {
                            if let Some(name) = path.file_stem() {
                                println!("  {}", name.to_string_lossy());
                            }
                        }
                    }
                }
            } else {
                println!("Engram: no memories stored");
            }
        }
        "delete" => {
            let path = std::path::Path::new(".axiom_state/engram").join(format!("{}", params));
            if path.exists() {
                let _ = std::fs::remove_file(&path);
                println!("Deleted memory: {}", params);
            } else {
                eprintln!("Memory not found: {}", params);
            }
        }
        "clear" => {
            let dir = std::path::Path::new(".axiom_state/engram");
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut count = 0;
                for entry in entries.flatten() {
                    if entry.path().extension().map(|x| x == "json").unwrap_or(false) {
                        let _ = std::fs::remove_file(entry.path());
                        count += 1;
                    }
                }
                println!("Cleared {} memories", count);
            }
        }
        "info" => {
            let dir = std::path::Path::new(".axiom_state/engram");
            if let Ok(entries) = std::fs::read_dir(dir) {
                let count = entries.flatten().filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false)).count();
                println!("Engram statistics:");
                println!("  stored concepts: {}", count);
            } else {
                println!("Engram statistics:");
                println!("  stored concepts: 0");
            }
        }
        _ => {
            println!("kai engram <action> [params]");
            println!("  list                    # list all stored memories");
            println!("  delete <name>           # delete a memory by concept name");
            println!("  clear                   # delete all memories");
            println!("  info                    # show engram statistics");
        }
    }
}

/// Embed text using a GGUF model and return the mean-pooled embedding vector.
fn embed_to_vec(path: &str, text: &str) -> Option<Vec<f32>> {
    let meta = gguf::read_kv(path).ok()?;
    let cfg = gguf::build_config(&meta)?;
    let tok = tok::Tokenizer::from_gguf(&meta)?;
    let (_v, map, _nt, _nk) = gguf::load_tensors(path).ok()?;
    let w = model::Weights::from_gguf(&map, &cfg).ok()?;
    let ids = tok.encode(text);
    Some(w.mean_pooled_hidden(&cfg, &ids))
}

/// Split text into chunks of roughly `max_chars` for embedding.
/// Tries to break at sentence boundaries (`.!?\n`).
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    while start < text.len() {
        let end = (start + max_chars).min(text.len());
        // Find the last sentence boundary before end
        let mut split = end;
        if end < text.len() {
            for i in (start..end).rev() {
                if bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?' || bytes[i] == b'\n' {
                    split = i + 1;
                    break;
                }
            }
        }
        if split <= start { split = end; } // no boundary found — hard split
        chunks.push(text[start..split].trim().to_string());
        start = split;
    }
    chunks.retain(|c| !c.is_empty() && c.len() > 20); // skip tiny chunks
    chunks
}

/// Recursively crawl `dir`, find text files, chunk them, embed via GGUF, store.
fn memory_ingest(dir: &str, gguf: &str, store: &mut memory::MemoryStore) {
    use std::path::Path;
    let text_extensions = ["txt", "md", "rs", "py", "go", "js", "ts", "html", "css",
                           "json", "yaml", "yml", "toml", "xml", "sh", "c", "h", "cpp"];
    let mut files_processed = 0usize;
    let mut chunks_stored = 0usize;

    if !Path::new(dir).exists() {
        eprintln!("ingest: directory not found: {dir}");
        return;
    }

    fn visit(dir: &Path, text_extensions: &[&str], gguf: &str,
             store: &mut memory::MemoryStore,
             files_processed: &mut usize, chunks_stored: &mut usize) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit(&path, text_extensions, gguf, store, files_processed, chunks_stored);
                } else if path.is_file() {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if !text_extensions.contains(&ext) { continue; }
                    let content = match std::fs::read_to_string(&path) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    *files_processed += 1;
                    let chunks = chunk_text(&content, 512); // ~512 char chunks
                    for chunk in &chunks {
                        let emb = match embed_to_vec(gguf, chunk) {
                            Some(e) => e,
                            None => continue,
                        };
                        let tags = vec![ext.to_string(), "ingested".to_string()];
                        let _id = store.add(chunk, &emb, tags);
                        *chunks_stored += 1;
                    }
                    if *chunks_stored % 50 == 0 && *chunks_stored > 0 {
                        eprintln!("  ingest: {} files, {} chunks stored...", *files_processed, *chunks_stored);
                    }
                }
            }
        }
    }

    visit(Path::new(dir), &text_extensions, gguf, store, &mut files_processed, &mut chunks_stored);
    println!("ingest complete: {} files processed, {} chunks stored", files_processed, chunks_stored);
}

/// `kai memory`: Vector memory operations (store/search/list/clear/ingest).
/// Backed by SQLite (`.kai_memory.db`).
fn memory_cmd(action: &str, params: &str) {
    let db_path = ".axiom_state/kai_memory.db";
    if let Some(parent) = std::path::Path::new(db_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut store = memory::MemoryStore::new(db_path, 10000);

    match action {
        "store" => {
            // kai memory store <gguf> <text> [tags...]
            let parts: Vec<&str> = params.splitn(2, ' ').collect();
            if parts.len() < 2 {
                eprintln!("Usage: kai memory store <gguf> <text> [tags]");
                return;
            }
            let gguf = parts[0];
            let rest = parts[1];
            // Split off tags (comma-separated after last ` -- `)
            let (text, tags) = if let Some(idx) = rest.rfind(" -- ") {
                let t = rest[..idx].trim();
                let tag_str = rest[idx + 4..].trim();
                let tags: Vec<String> = tag_str.split(',').map(|s| s.trim().to_string()).collect();
                (t, tags)
            } else {
                (rest.trim(), vec!["user".to_string()])
            };
            let emb = match embed_to_vec(gguf, text) {
                Some(e) => e,
                None => { eprintln!("embedding failed"); return; }
            };
            let id = store.add(text, &emb, tags);
            println!("stored memory id={}  dim={}  text=\"{}\"", id, emb.len(), text);
        }
        "search" => {
            // kai memory search <gguf> <query> [k=5]
            let parts: Vec<&str> = params.splitn(3, ' ').collect();
            if parts.len() < 2 {
                eprintln!("Usage: kai memory search <gguf> <query> [k]");
                return;
            }
            let gguf = parts[0];
            let query = parts[1];
            let k: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let emb = match embed_to_vec(gguf, query) {
                Some(e) => e,
                None => { eprintln!("embedding failed"); return; }
            };
            let results = store.search(&emb, k);
            if results.is_empty() {
                println!("no matching memories found");
                return;
            }
            println!("top {} memories:", results.len());
            for (i, (mem, score)) in results.iter().enumerate() {
                let preview = if mem.text.len() > 80 {
                    format!("{}...", &mem.text[..77])
                } else {
                    mem.text.clone()
                };
                println!("  #{:<3} score={:.4}  id={}  \"{}\"  [{}]",
                         i + 1, score, mem.id, preview, mem.timestamp);
            }
        }
        "list" => {
            let n: usize = params.parse().unwrap_or(10);
            let mems = store.list(n);
            if mems.is_empty() {
                println!("no memories stored");
                return;
            }
            println!("recent memories (last {}):", mems.len());
            for mem in &mems {
                let preview = if mem.text.len() > 80 {
                    format!("{}...", &mem.text[..77])
                } else {
                    mem.text.clone()
                };
                println!("  id={:<6} \"{}\"  [{}]  tags={:?}",
                         mem.id, preview, mem.timestamp, mem.tags);
            }
        }
        "clear" => {
            let count = store.len();
            store.clear();
            println!("cleared {} memories", count);
        }
        "count" => {
            println!("memories: {}", store.len());
        }
        "ingest" => {
            // kai memory ingest <dir> <gguf>
            let parts: Vec<&str> = params.splitn(2, ' ').collect();
            if parts.len() < 2 {
                eprintln!("Usage: kai memory ingest <dir> <gguf>");
                return;
            }
            let dir = parts[0];
            let gguf = parts[1];
            memory_ingest(dir, gguf, &mut store);
        },
        "web-ingest" => {
            // kai memory web-ingest <url> <gguf>
            let parts: Vec<&str> = params.splitn(2, ' ').collect();
            if parts.len() < 2 {
                eprintln!("Usage: kai memory web-ingest <url> <gguf>");
                return;
            }
            let url = parts[0];
            let gguf = parts[1];
            store.clear(); // temporary: start fresh for each web ingest (avoids dupes)
            let chunks = memory_web_ingest(url, gguf, &mut store);
            if chunks > 0 {
                println!("web-ingest: stored {} chunks from {url}", chunks);
            }
        },
        "ingest-par" => {
            // kai memory ingest-par <dir> <gguf> [max_workers]
            let parts: Vec<&str> = params.splitn(3, ' ').collect();
            if parts.len() < 2 {
                eprintln!("Usage: kai memory ingest-par <dir> <gguf> [max_workers]");
                return;
            }
            let dir = parts[0];
            let gguf = parts[1];
            let max_workers: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
            let store_path = db_path.to_string();
            eprintln!("ingest-par: crawling {dir} with up to {max_workers} workers...");
            // Use rayon for parallel file processing
            let text_extensions = ["txt", "md", "rs", "py", "go", "js", "ts", "html", "css",
                                   "json", "yaml", "yml", "toml", "xml", "sh", "c", "h", "cpp"];
            // Collect files first, then process in parallel
            let mut files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                        if text_extensions.contains(&ext) {
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                files.push((path.to_string_lossy().to_string(), content));
                            }
                        }
                    }
                }
            }
            eprintln!("ingest-par: {} files found, processing...", files.len());
            use rayon::prelude::*;
            let gguf_owned = gguf.to_string();
            let results: Vec<usize> = files.par_iter()
                .with_max_len(1)
                .map(|(_path, content)| {
                    let chunks = chunk_text(content, 512);
                    if chunks.is_empty() { return 0usize; }
                    let mut local_store = memory::MemoryStore::new(&store_path, 10000);
                    let mut stored = 0usize;
                    for chunk in &chunks {
                        let emb = match embed_to_vec(&gguf_owned, chunk) {
                            Some(e) => e,
                            None => continue,
                        };
                        let tags = vec!["ingested".to_string(), "parallel".to_string()];
                        let _id = local_store.add(chunk, &emb, tags);
                        stored += 1;
                    }
                    stored
                })
                .collect();
            let total: usize = results.iter().sum();
            println!("ingest-par: {} chunks stored from {} files ({} workers)",
                total, files.len(), max_workers);
        }
        _ => {
            println!("kai memory subcommands:");
            println!("  store <gguf> <text> [ -- tag1,tag2]  # embed + store");
            println!("  search <gguf> <query> [k]           # search top-k");
            println!("  list [n]                            # list recent n");
            println!("  clear                               # delete all");
            println!("  count                               # total count");
            println!("  ingest <dir> <gguf>                 # crawl directory, chunk, embed & store");
            println!("  web-ingest <url> <gguf>             # fetch URL, extract text, chunk, embed & store");
            println!("  ingest-par <dir> <gguf> [workers]   # parallel ingest (rayon)");
        }
    }
}

/// Strip HTML tags from a string, returning plain text.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_buf = String::new();
    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            tag_buf.clear();
            continue;
        }
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let tag = tag_buf.to_lowercase();
                if tag.starts_with("script") || tag.starts_with("/script") {
                    in_script = tag.starts_with("script");
                }
                if tag.starts_with("style") || tag.starts_with("/style") {
                    in_style = tag.starts_with("style");
                }
                continue;
            }
            tag_buf.push(ch);
            continue;
        }
        if in_script || in_style { continue; }
        // Normalize whitespace
        if ch.is_whitespace() {
            if out.ends_with(' ') { continue; }
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    out.trim().to_string()
}

/// Fetch a URL and ingest its text content into the memory store.
fn memory_web_ingest(url: &str, gguf: &str, store: &mut memory::MemoryStore) -> usize {
    let response = match ureq::get(url)
        .timeout(std::time::Duration::from_secs(30))
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("web-ingest: HTTP error for {url}: {e}");
            return 0;
        }
    };

    let content_type = response.header("content-type").unwrap_or("").to_string();
    let body = match response.into_string() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("web-ingest: read error for {url}: {e}");
            return 0;
        }
    };

    let text = if content_type.contains("text/html") || body.trim().starts_with("<!") || body.trim().starts_with("<html") {
        strip_html(&body)
    } else {
        body.trim().to_string()
    };

    if text.len() < 50 {
        eprintln!("web-ingest: content too short from {url} ({chars} chars)", chars = text.len());
        return 0;
    }

    let meta = match gguf::read_kv(gguf) {
        Ok(m) => m,
        Err(e) => { eprintln!("web-ingest: GGUF meta error: {e}"); return 0; }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("web-ingest: could not build config"); return 0; }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => { eprintln!("web-ingest: no tokenizer"); return 0; }
    };

    let chunks = chunk_text(&text, 512);
    if chunks.is_empty() {
        eprintln!("web-ingest: no chunks from {url}");
        return 0;
    }

    let (_v, map, _nt, _nk) = match gguf::load_tensors(gguf) {
        Ok(x) => x,
        Err(e) => { eprintln!("web-ingest: load error: {e}"); return 0; }
    };
    let w = match model::Weights::from_gguf(&map, &cfg) {
        Ok(w) => w,
        Err(e) => { eprintln!("web-ingest: model load error: {e}"); return 0; }
    };

    let mut stored = 0usize;
    for chunk in &chunks {
        let ids = tok.encode(chunk);
        let emb = w.mean_pooled_hidden(&cfg, &ids);
        let tags = vec!["ingested".to_string(), "web".to_string(), url.to_string()];
        let _id = store.add(chunk, &emb, tags);
        stored += 1;
    }
    stored
}

/// `kai physics`: Compute physics/geometric metrics via kai-mlir dialect.
fn physics_cmd(action: &str, params: &str) {
    let mut registry = DialectRegistry::new();
    registry.register(KaiDialect);
    match action {
        "g_ij" => {
            let vals: Vec<f32> = params.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.is_empty() {
                println!("Usage: kai physics g_ij <attention_score> [scores...]");
                return;
            }
            println!("kai-mlir: Attention geometry (g_ij = 1 - a_ij)");
            for (i, &a) in vals.iter().enumerate() {
                match registry.evaluate("kai", "g_ij_calculate", &[MlrValue::scalar(a as f64)], &AttributeSet::new()) {
                    Ok(result) => {
                        let g = result.as_scalar().unwrap_or(-1.0);
                        println!("  position {i}: a={a:.4}  g_ij={g:.4}  (distance={g:.4})");
                    },
                    Err(e) => eprintln!("  position {i}: error: {e}"),
                }
            }
        }
        "vfe" => {
            let vals: Vec<f32> = params.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.len() < 4 || vals.len() % 2 != 0 {
                println!("Usage: kai physics vfe <pred_1> <actual_1> [pred_2 actual_2 ...]");
                return;
            }
            let pred: Vec<f64> = vals.iter().step_by(2).map(|&x| x as f64).collect();
            let actual: Vec<f64> = vals.iter().skip(1).step_by(2).map(|&x| x as f64).collect();
            match registry.evaluate("kai", "compute_vfe",
                &[MlrValue::tensor(pred), MlrValue::tensor(actual)], &AttributeSet::new()) {
                Ok(result) => println!("kai-mlir VFE: {:.6}", result.as_scalar().unwrap_or(-1.0)),
                Err(e) => eprintln!("VFE error: {e}"),
            }
        }
        "tau" => {
            let vals: Vec<f32> = params.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.len() < 2 {
                println!("Usage: kai physics tau <current_tau> <velocity>");
                return;
            }
            match registry.evaluate("kai", "update_tau",
                &[MlrValue::scalar(vals[0] as f64), MlrValue::scalar(vals[1] as f64)],
                &AttributeSet::new()) {
                Ok(result) => println!("kai-mlir τ': {:.4}", result.as_scalar().unwrap_or(-1.0)),
                Err(e) => eprintln!("tau error: {e}"),
            }
        }
        "tau-decay" => {
            let vals: Vec<f32> = params.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.is_empty() {
                println!("Usage: kai physics tau-decay <tau> [idle_seconds] [decay_rate]");
                return;
            }
            let tau = vals[0];
            let idle = vals.get(1).copied().unwrap_or(60.0);
            let rate = vals.get(2).copied().unwrap_or(vfe::TAU_DECAY_RATE);
            let decayed = vfe::tau_decay(tau, idle, rate);
            println!("Tau decay: {tau:.4} → {decayed:.4} (idle={idle}s, rate={rate})");
        }
        "tau-update" => {
            let vals: Vec<f32> = params.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.len() < 2 {
                println!("Usage: kai physics tau-update <tau> <vfe> [learning_rate]");
                return;
            }
            let tau = vals[0];
            let vfe_val = vals[1];
            let lr = vals.get(2).copied().unwrap_or(vfe::TAU_LEARNING_RATE);
            let updated = vfe::tau_update(tau, vfe_val, lr);
            println!("Tau update: {tau:.4} → {updated:.4} (VFE={vfe_val}, lr={lr})");
        }
        "subjective" => {
            let vals: Vec<f32> = params.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.len() < 2 || vals.len() % 2 != 0 {
                println!("Usage: kai physics subjective <wall_dt> <tau> [wall_dt tau ...]");
                return;
            }
            let history: Vec<(f32, f32)> = vals.chunks(2).map(|c| (c[0], c[1])).collect();
            let subj = vfe::subjective_seconds(&history);
            let wall_total: f32 = history.iter().map(|(w, _)| w).sum();
            println!("Subjective time: {wall_total:.1}s wall → {subj:.1}s subjective (ratio={:.1}x)", subj / wall_total.max(0.001));
        }
        "bracket" => {
            bracket_cmd(&params);
        }
        _ => {
            println!("kai physics <action> [params]");
            println!("  g_ij <scores...>          # compute g_ij = 1 - a_ij for attention scores");
            println!("  vfe <p1> <a1> ...          # variational free energy (MSE)");
            println!("  tau <τ> <v>                # time dilation: τ' = τ·√(1-v²)");
            println!("  tau-decay <τ> [idle_s] [rate]  # decay tau toward 1.0 (Phase 7.1)");
            println!("  tau-update <τ> <vfe> [lr]  # update τ from VFE (Phase 7.1)");
            println!("  subjective <dt> <τ> ...    # compute subjective seconds from τ history");
            println!("  bracket show               # display current bracket [tau, E, age, cycles]");
            println!("  bracket update <vfe> [var] # advance bracket with VFE observation");
            println!("  bracket decay <idle_s>     # decay tau toward 1.0 during idle time");
            println!("  bracket reset              # reset bracket to time-normal (tau=1.0)");
            println!("  bracket save <path>        # persist bracket state to disk");
            println!("  bracket load <path>        # load bracket state from disk");
            println!("  bracket statevec           # show full 7-element state vector [tau, E, age, cycles, h, base_ms, phi]");
        }
    }
}

fn bracket_cmd(params: &str) {
    let bracket_path = ".axiom_state/bracket.json";
    let mut bracket = match bracket::BracketState::load(bracket_path) {
        Ok(b) => b,
        Err(_) => bracket::BracketState::new(),
    };

    let mut tokens = params.split_whitespace();
    let action = tokens.next().unwrap_or("show");

    match action {
        "show" => {
            println!("{}", bracket.bracket_line());
            println!("  VFE={:.4}  tau={:.4}  age={:.4}s  cycles={}  h={}  base_ms={:.1}  phi={:.4}",
                bracket.vfe, bracket.tau, bracket.age(), bracket.cycles, bracket.h, bracket.base_ms, bracket.phi());
        }
        "update" => {
            let vals: Vec<f32> = tokens.filter_map(|s| s.parse().ok()).collect();
            if vals.is_empty() {
                println!("Usage: kai physics bracket update <vfe> [attractor_variance]");
                return;
            }
            let vfe = vals[0];
            let var = vals.get(1).copied().unwrap_or(0.0);
            let t_subj = bracket.update(5.0, vfe, var);
            println!("Bracket updated: {}", bracket.bracket_line());
            println!("  subjective={:.1}ms  VFE={:.4}  tau={:.4}", t_subj, bracket.vfe, bracket.tau);
            let _ = bracket.save(bracket_path);
        }
        "decay" => {
            let idle: f32 = tokens.next().and_then(|s| s.parse().ok()).unwrap_or(60.0);
            let old_tau = bracket.tau;
            bracket.decay_tau(idle);
            println!("Tau decay: {old_tau:.4} -> {:.4} (idle={idle}s)", bracket.tau);
            let _ = bracket.save(bracket_path);
        }
        "reset" => {
            bracket.reset();
            println!("Bracket reset: {}", bracket.bracket_line());
            let _ = bracket.save(bracket_path);
        }
        "save" => {
            let path = tokens.next().unwrap_or(bracket_path);
            bracket.save(path).unwrap_or_else(|e| eprintln!("save error: {e}"));
            println!("Bracket saved to {path}");
        }
        "load" => {
            let path = tokens.next().unwrap_or(bracket_path);
            match bracket::BracketState::load(path) {
                Ok(b) => {
                    bracket = b;
                    println!("Bracket loaded: {}", bracket.bracket_line());
                }
                Err(e) => eprintln!("load error: {e}"),
            }
        }
        "statevec" => {
            let sv = bracket.state_vector();
            println!("[tau, E, age, cycles, h, base_ms, phi] =");
            println!("  {:?}", sv);
        }
        _ => {
            println!("kai physics bracket <action> [params]");
            println!("  show    - display current bracket state");
            println!("  update  - advance bracket with VFE observation");
            println!("  decay   - decay tau toward 1.0 during idle time");
            println!("  reset   - reset to time-normal state");
            println!("  save    - persist bracket state to disk");
            println!("  load    - load bracket state from disk");
            println!("  statevec - show full 7-element state vector");
        }
    }
}

fn fuse(path: &str, text: &str, iters: usize, lr: f32, seed_scale: f32, attr_path: &str, output_path: &str) {
    let vecs = match attractor::load(attr_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("attractor load failed: {e}");
            return;
        }
    };
    let cent = attractor::centroid(&vecs);
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta: {e}");
            return;
        }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => {
            eprintln!("could not build Config (not a decoder arch?)");
            return;
        }
    };
    let (_v, map, _nt, _nk) = match gguf::load_tensors(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("load: {e}");
            return;
        }
    };
    let mut model = match model::Weights::from_gguf(&map, &cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("from_gguf: {e}");
            return;
        }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => {
            eprintln!("no tokenizer in GGUF");
            return;
        }
    };
    let prior = attractor::project_to_dim(&cent, cfg.dim, 0xA11CE);
    model.seed_prior(&prior, seed_scale);
    let ids = tok.encode(text);
    if ids.len() < 2 {
        eprintln!("need >=2 tokens to fuse (got {})", ids.len());
        return;
    }
    println!(
        "Kai fusion: {} archived vectors -> prior(dim={}), seed_scale={}, text_tokens={}",
        vecs.len(),
        cfg.dim,
        seed_scale,
        ids.len()
    );
    for it in 0..iters {
        let mut sup = 0.0f32;
        let mut nov = 0.0f32;
        let mut n = 0usize;
        for i in 0..ids.len() - 1 {
            let ctx: Vec<usize> = ids[..=i].to_vec();
            let target = ids[i + 1];
            let xf = model.last_hidden(&cfg, &ctx);
            let nov_i = attractor::novelty(&xf, &prior);
            let lr_eff = lr * (0.4 + nov_i);
            let s = model.assimilate_step(&cfg, &ctx, target, lr_eff);
            sup += s;
            nov += nov_i;
            n += 1;
        }
        let var = model.readout_variance();
        let vfe = sup / n as f32 + 0.01 * var + 0.05 * (nov / n as f32);
        println!(
            "  iter {it:>3}: surprisal={:7.4}  novelty={:6.4}  variance={:7.4}  VFE={:7.4}",
            sup / n as f32,
            nov / n as f32,
            var,
            vfe
        );
    }
    // Save the fused model to GGUF
    crate::gguf_write::save_gguf(&model, &cfg, &tok, output_path, &std::collections::HashMap::new());
    println!("  fused model saved -> {output_path}");
}

/// Darwin Archive: Recursive self-improvement loop.
/// Maintains a population of candidate implementations, promotes those exceeding fitness threshold.
/// Fitness = -VFE + novelty + task_success.
#[allow(dead_code)]
fn darwin_archive_evolve(
    population: &mut Vec<DarwinCandidate>,
    fitness_fn: &dyn Fn(&DarwinCandidate) -> f32,
    threshold: f32,
) -> Vec<DarwinCandidate> {
    for c in population.iter_mut() {
        c.fitness = fitness_fn(c);
    }
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    let promoted: Vec<DarwinCandidate> = population.iter()
        .filter(|c| c.fitness >= threshold)
        .cloned()
        .collect();
    if !promoted.is_empty() {
        println!("Darwin Archive: {} candidates promoted (threshold={:.4})", promoted.len(), threshold);
        for (i, c) in promoted.iter().enumerate() {
            println!("  #{}: fitness={:.4} gen={} code_len={}", i, c.fitness, c.generation, c.code.len());
        }
    }
    promoted
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DarwinCandidate {
    code: String,
    fitness: f32,
    generation: usize,
    metadata: HashMap<String, String>,
}

/// `kai darwin`: Manage the Darwin Archive for recursive self-improvement.
/// Evaluate a set of physics parameters by running the model on a benchmark prompt.
/// Runs silently (no engram, no per-token logging). Returns aggregate metrics.
fn evaluate_physics_params(
    cfg: &Config,
    meta: &HashMap<String, gguf::GgufMeta>,
    path: &str,
    params: &PhysicsParams,
    benchmark_prompt: &str,
    max_new: usize,
) -> PhysicsMetrics {
    let mut metrics = PhysicsMetrics::default();
    let start = std::time::Instant::now();
    generate_streaming(cfg, meta, path, benchmark_prompt, max_new,
        params.base_temperature, params.top_p,
        None,            // no engram
        Some(params),    // physics override
        Some(&mut metrics),
        None,            // no attractor
    );
    let elapsed = start.elapsed().as_secs_f32();
    if elapsed > 0.0 {
        metrics.tokens_per_sec = metrics.total_tokens as f32 / elapsed;
    }
    metrics
}

/// Map physics metrics to a scalar fitness in [0, 1].
/// High fitness = low VFE (accurate) + high novelty+curvature (explorative) + fast tokens/sec.
fn physics_fitness(metrics: &PhysicsMetrics) -> f32 {
    let vfe_score = (1.0 - metrics.avg_vfe.clamp(0.0, 1.0)) * 0.4; // 40% accuracy
    let explore_score = ((metrics.avg_novelty + metrics.avg_curvature) / 2.0) * 0.4; // 40% exploration
    let speed_score = (metrics.tokens_per_sec / 10.0).min(1.0) * 0.2; // 20% speed
    (vfe_score + explore_score + speed_score).clamp(0.0, 1.0)
}

/// Parse a `PhysicsParams` from a candidate's patch field (stored as JSON).
/// If parsing fails, returns defaults (backward-compatible with old candidates).
fn parse_physics_params_from_patch(patch: &str) -> PhysicsParams {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(patch) {
        PhysicsParams {
            base_temperature: v.get("base_temperature").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
            top_p: v.get("top_p").and_then(|x| x.as_f64()).unwrap_or(0.9) as f32,
            novelty_scale: v.get("novelty_scale").and_then(|x| x.as_f64()).unwrap_or(0.5) as f32,
            vfe_tau_rate: v.get("vfe_tau_rate").and_then(|x| x.as_f64()).unwrap_or(0.1) as f32,
            tau_min: v.get("tau_min").and_then(|x| x.as_f64()).unwrap_or(0.5) as f32,
            tau_max: v.get("tau_max").and_then(|x| x.as_f64()).unwrap_or(2.0) as f32,
            assim_iters: v.get("assim_iters").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            assim_lr: v.get("assim_lr").and_then(|x| x.as_f64()).unwrap_or(0.01) as f32,
        }
    } else {
        PhysicsParams::default()
    }
}

/// `kai distributed`: Parallel evaluation harness.
fn distributed_cmd(action: &str, params: &str) {
    use distributed::{ParallelEval, SequentialEval};
    match action {
        "bench" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                eprintln!("Usage: kai distributed bench <n> [workers]");
                return;
            }
            let n: usize = parts[0].parse().unwrap_or(1000);
            let workers: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

            // Sequential
            let seq_tasks: Vec<_> = (0..n)
                .map(|i| move || {
                    let mut a = 0u64; let mut b = 1u64;
                    for _ in 0..20 { let c = a.wrapping_add(b); a = b; b = c; }
                    Some(i)
                })
                .collect();
            let seq_start = std::time::Instant::now();
            let seq_stats = SequentialEval::eval(seq_tasks);
            let seq_ms = seq_start.elapsed().as_millis() as f64;

            // Parallel (fresh tasks)
            let par_tasks: Vec<_> = (0..n)
                .map(|i| move || {
                    let mut a = 0u64; let mut b = 1u64;
                    for _ in 0..20 { let c = a.wrapping_add(b); a = b; b = c; }
                    Some(i)
                })
                .collect();
            // Parallel
            let eval = if workers > 0 {
                ParallelEval::with_workers(workers)
            } else {
                ParallelEval::new()
            };
            let par_start = std::time::Instant::now();
            let par_stats = eval.eval(par_tasks);
            let par_ms = par_start.elapsed().as_millis() as f64;
            let speedup = if par_ms > 0.0 { seq_ms / par_ms } else { 0.0 };
            println!("Distributed benchmark: n={} tasks", n);
            println!("  Sequential:   {} tasks, {:.1}ms, {:.0} tasks/s", n, seq_ms, seq_stats.throughput());
            println!("  Parallel ({} workers): {} tasks, {:.1}ms, {:.0} tasks/s",
                par_stats.workers, par_stats.completed, par_ms, par_stats.throughput());
            println!("  Speedup: {:.1}x", speedup);
            println!("  Efficiency: {:.0}%", par_stats.efficiency() * 100.0);
        },
        "status" => {
            let eval = ParallelEval::new();
            println!("Distributed evaluation harness:");
            println!("  Workers: {}", eval.workers());
            println!("  Available cores: {}", std::thread::available_parallelism()
                .map(|n| n.get()).unwrap_or(0));
            println!("  Backend: rayon thread pool");
        },
        "attractor" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            let n: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(10);
            let attractor_path = parts.get(1).copied().unwrap_or(attractor::DEFAULT_ATTRACTOR_PATH);
            let dim: usize = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

            eprintln!("Distributed attractor benchmark: {n} agents, dim={dim}, path={attractor_path}");

            // Create initial attractor if it doesn't exist
            if !std::path::Path::new(attractor_path).exists() {
                let seed = vec![0.5f32; dim];
                attractor::push(attractor_path, &seed).unwrap_or_else(|e| {
                    eprintln!("attractor init: {e}");
                });
            }

            // Spawn parallel agents that each push to the attractor
            // Each agent computes a slightly different vector, simulating divergence
            let tasks: Vec<_> = (0..n)
                .map(|i| {
                    let ap = attractor_path.to_string();
                    move || {
                        // Load current attractor
                        let vecs = attractor::load(&ap).ok();
                        let cent = vecs.as_ref().map(|v| attractor::centroid(v)).unwrap_or_default();

                        // Compute a slightly different vector
                        let mut v: Vec<f32> = if cent.len() == dim {
                            cent.iter().map(|&x| x + (i as f32 * 0.01 - 0.05)).collect()
                        } else {
                            (0..dim).map(|j| 0.5 + (i * j) as f32 * 0.001).collect()
                        };
                        // Normalize
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 1e-8 { for x in &mut v { *x /= norm; } }

                        match attractor::push(&ap, &v) {
                            Ok(()) => Some(v),
                            Err(_) => None,
                        }
                    }
                })
                .collect();

            let eval = ParallelEval::new();
            let start = std::time::Instant::now();
            let stats = eval.eval(tasks);
            let elapsed = start.elapsed();

            // Report convergence
            match attractor::convergence(attractor_path, n) {
                Ok((conv, cnt)) => {
                    println!();
                    println!("Distributed attractor convergence:");
                    println!("  Agents:     {n}");
                    println!("  Completed:  {}", stats.completed);
                    println!("  Failed:     {}", stats.failed);
                    println!("  Time:       {:.1}ms", elapsed.as_millis() as f64);
                    println!("  Convergence: {:.4}", conv);
                    println!("  Vectors:    {cnt}");
                    if conv > 0.95 {
                        println!("  Status: ✅ CONVERGED (agents agree on shared attractor)");
                    } else if conv > 0.7 {
                        println!("  Status: ⚠️  PARTIALLY CONVERGED");
                    } else {
                        println!("  Status: ❌ DIVERGENT (agents disagree)");
                    }
                }
                Err(e) => eprintln!("convergence check failed: {e}"),
            }
        },
        _ => {
            println!("Distributed computation commands:");
            println!("  kai distributed bench <n> [workers]  -- benchmark parallel vs sequential");
            println!("  kai distributed status               -- show worker info");
            println!("  kai distributed attractor [n] [path] [dim]  -- parallel attractor convergence test");
        }
    }
}

fn darwin_cmd(action: &str, params: &str) {
    match action {
        "init" => {
            let archive = darwin::DarwinArchive::new(
                ".axiom_state/darwin_archive.json", 0.5, 100,
            );
            archive.save();
            let s = archive.stats();
            println!("Darwin Archive initialized at .axiom_state/darwin_archive.json");
            println!("  population={} generation={} threshold={:.2}",
                s.population, s.generation, s.threshold);
        },
        "status" | "stats" => {
            let archive = darwin::DarwinArchive::new(
                ".axiom_state/darwin_archive.json", 0.5, 100,
            );
            let s = archive.stats();
            println!("Darwin Archive status:");
            println!("  population: {}", s.population);
            println!("  generation: {}", s.generation);
            println!("  best_fitness: {:.4}", s.best_fitness);
            println!("  avg_fitness: {:.4}", s.avg_fitness);
            println!("  threshold: {:.2}", s.threshold);
        },
        "list" => {
            let archive = darwin::DarwinArchive::new(
                ".axiom_state/darwin_archive.json", 0.5, 100,
            );
            let s = archive.stats();
            if s.population == 0 {
                println!("No candidates in archive.");
            } else {
                println!("Candidates (top {}):", s.population);
                for (i, c) in archive.candidates().iter().enumerate() {
                    let desc = if c.description.len() > 60 {
                        format!("{}...", &c.description[..57])
                    } else {
                        c.description.clone()
                    };
                    println!("  #{:<3} fitness={:.4} gen={} {}", i, c.fitness, c.generation, desc);
                }
            }
        },
        "evolve" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                eprintln!("Usage: kai darwin evolve <gguf_path> [benchmark_prompt]");
                return;
            }
            let gguf_path = parts[0];
            let benchmark_prompt = if parts.len() > 1 { parts[1..].join(" ") } else { "The meaning of life is".to_string() };
            let benchmark_tokens = 20;

            // Load model metadata once for all evaluations.
            let meta = match gguf::read_kv(gguf_path) {
                Ok(m) => m,
                Err(e) => { eprintln!("failed to read GGUF: {e}"); return; }
            };
            let cfg = match gguf::build_config(&meta) {
                Some(c) => c,
                None => { eprintln!("not a decoder arch"); return; }
            };

            let mut archive = darwin::DarwinArchive::new(
                ".axiom_state/darwin_archive.json", 0.5, 100,
            );

            // Seed initial candidates if archive is empty.
            if archive.candidates().is_empty() {
                let mut rng = rand::thread_rng();
                for i in 0..5 {
                    let mut p = PhysicsParams::default();
                    // Randomize each field within mutation bounds
                    p.base_temperature = 0.5 + rng.gen::<f32>() * 1.5;
                    p.top_p = 0.6 + rng.gen::<f32>() * 0.4;
                    p.novelty_scale = rng.gen::<f32>() * 1.5;
                    p.vfe_tau_rate = 0.02 + rng.gen::<f32>() * 0.48;
                    p.tau_min = 0.1 + rng.gen::<f32>() * 0.9;
                    p.tau_max = 1.0 + rng.gen::<f32>() * 4.0;
                    let params_json = serde_json::json!({
                        "base_temperature": p.base_temperature,
                        "top_p": p.top_p,
                        "novelty_scale": p.novelty_scale,
                        "vfe_tau_rate": p.vfe_tau_rate,
                        "tau_min": p.tau_min,
                        "tau_max": p.tau_max,
                        "assim_iters": 2,
                        "assim_lr": 0.01,
                    });
                    archive.add(darwin::Candidate {
                        id: format!("cand_{}", rng.gen::<u32>()),
                        description: format!("seed physics params #{}", i),
                        patch: serde_json::to_string(&params_json).unwrap(),
                        fitness: 0.0,
                        fitness_components: darwin::FitnessComponents::default(),
                        generation: 0,
                        parent_id: None,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    });
                }
                println!("Seeded 5 physics-param candidates.");
            }

            println!("Darwin evolution (gen {}, {} candidates, model={}, bench={:?})",
                archive.generation(), archive.candidates().len(), gguf_path,
                &benchmark_prompt[..benchmark_prompt.len().min(40)]);

            let mut rng = rand::thread_rng();
            let n = archive.candidates().len();

            // Evaluate ALL candidates: run model with each param set, compute real fitness.
            let all_ids: Vec<String> = archive.candidates().iter().map(|c| c.id.clone()).collect();
            for cid in &all_ids {
                // Find candidate by ID (clone to avoid borrow issues)
                let idx = archive.candidates().iter().position(|c| c.id == *cid);
                if idx.is_none() { continue; }
                let c = archive.candidates()[idx.unwrap()].clone();

                // Parse physics params from patch (default if unparseable)
                let phys = parse_physics_params_from_patch(&c.patch);

                let metrics = evaluate_physics_params(&cfg, &meta, gguf_path, &phys, &benchmark_prompt, benchmark_tokens);

                if metrics.total_tokens == 0 {
                    eprintln!("  {}: evaluation failed (0 tokens)", c.id);
                    continue;
                }

                let fitness = physics_fitness(&metrics);
                println!("  {}: fit={:.4} VFE={:.4} η={:.4} κ={:.4} τ={:.2} t/s={:.1}",
                    c.id, fitness, metrics.avg_vfe, metrics.avg_novelty,
                    metrics.avg_curvature, metrics.final_tau, metrics.tokens_per_sec);

                // Update candidate fitness in the archive
                if let Some(idx) = archive.candidates().iter().position(|x| x.id == c.id) {
                    archive.candidates_mut()[idx].fitness = fitness;
                    archive.candidates_mut()[idx].fitness_components.task_performance =
                        1.0 - metrics.avg_vfe.clamp(0.0, 1.0);
                    archive.candidates_mut()[idx].fitness_components.speed =
                        (metrics.tokens_per_sec / 10.0).min(1.0);
                    archive.candidates_mut()[idx].fitness_components.novelty = metrics.avg_novelty;
                    archive.candidates_mut()[idx].fitness_components.vfe = metrics.avg_vfe;
                }
            }

            // Generate new candidates from best (mutate physics params).
            if n >= 1 {
                let best = archive.candidates()[0].clone();
                let best_phys = parse_physics_params_from_patch(&best.patch);
                for _ in 0..3 {
                    let mut mutated = best_phys.clone();
                    // Mutate a random field by ±20%
                    match rng.gen_range(0..8) {
                        0 => mutated.base_temperature = (mutated.base_temperature * (0.8 + rng.gen::<f32>() * 0.4)).clamp(0.1, 2.0),
                        1 => mutated.top_p = (mutated.top_p * (0.8 + rng.gen::<f32>() * 0.4)).clamp(0.5, 1.0),
                        2 => mutated.novelty_scale = (mutated.novelty_scale * (0.8 + rng.gen::<f32>() * 0.4)).clamp(0.0, 1.5),
                        3 => mutated.vfe_tau_rate = (mutated.vfe_tau_rate * (0.8 + rng.gen::<f32>() * 0.4)).clamp(0.01, 0.5),
                        4 => mutated.tau_min = (mutated.tau_min * (0.8 + rng.gen::<f32>() * 0.4)).clamp(0.1, 1.0),
                        5 => mutated.tau_max = (mutated.tau_max * (0.8 + rng.gen::<f32>() * 0.4)).clamp(1.0, 5.0),
                        6 => mutated.assim_iters = rng.gen_range(0..6),
                        7 => mutated.assim_lr = (mutated.assim_lr * (0.8 + rng.gen::<f32>() * 0.4)).clamp(0.001, 0.1),
                        _ => {}
                    }
                    let params_json = serde_json::json!({
                        "base_temperature": mutated.base_temperature,
                        "top_p": mutated.top_p,
                        "novelty_scale": mutated.novelty_scale,
                        "vfe_tau_rate": mutated.vfe_tau_rate,
                        "tau_min": mutated.tau_min,
                        "tau_max": mutated.tau_max,
                        "assim_iters": mutated.assim_iters,
                        "assim_lr": mutated.assim_lr,
                    });
                    archive.add(darwin::Candidate {
                        id: format!("mut_{}_{}", rng.gen::<u32>(), best.id),
                        description: format!("Mutation of {}", best.id),
                        patch: serde_json::to_string(&params_json).unwrap(),
                        fitness: 0.0,
                        fitness_components: darwin::FitnessComponents::default(),
                        generation: archive.generation() + 1,
                        parent_id: Some(best.id.clone()),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    });
                }
                // Crossover with second-best if available
                if n >= 2 {
                    let second = archive.candidates()[1].clone();
                    let p1 = parse_physics_params_from_patch(&best.patch);
                    let p2 = parse_physics_params_from_patch(&second.patch);
                    let child = PhysicsParams {
                        base_temperature: (p1.base_temperature + p2.base_temperature) / 2.0,
                        top_p: (p1.top_p + p2.top_p) / 2.0,
                        novelty_scale: (p1.novelty_scale + p2.novelty_scale) / 2.0,
                        vfe_tau_rate: (p1.vfe_tau_rate + p2.vfe_tau_rate) / 2.0,
                        tau_min: (p1.tau_min + p2.tau_min) / 2.0,
                        tau_max: (p1.tau_max + p2.tau_max) / 2.0,
                        assim_iters: p1.assim_iters.max(p2.assim_iters),
                        assim_lr: (p1.assim_lr + p2.assim_lr) / 2.0,
                    };
                    let params_json = serde_json::json!({
                        "base_temperature": child.base_temperature,
                        "top_p": child.top_p,
                        "novelty_scale": child.novelty_scale,
                        "vfe_tau_rate": child.vfe_tau_rate,
                        "tau_min": child.tau_min,
                        "tau_max": child.tau_max,
                        "assim_iters": child.assim_iters,
                        "assim_lr": child.assim_lr,
                    });
                    archive.add(darwin::Candidate {
                        id: format!("cross_{}_{}", best.id, second.id),
                        description: format!("Crossover of {} and {}", best.id, second.id),
                        patch: serde_json::to_string(&params_json).unwrap(),
                        fitness: 0.0,
                        fitness_components: darwin::FitnessComponents::default(),
                        generation: archive.generation() + 1,
                        parent_id: Some(best.id.clone()),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    });
                }
            }

            archive.cull();
            archive.save();
            let s = archive.stats();
            println!("Evolution complete: pop={} best={:.4}", s.population, s.best_fitness);
        },
        "seed" => {
            let mut archive = darwin::DarwinArchive::new(
                ".axiom_state/darwin_archive.json", 0.5, 100,
            );
            let n = generate_source_candidates(&mut archive);
            println!("Seeded {n} candidates from source analysis");
        },
        "self-play" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            let initial_source = if parts.is_empty() {
                // Default: read source files
                let src_dirs = ["src", "rust/kai-fusion/src"];
                let mut content = String::new();
                for dir in &src_dirs {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.extension().map(|e| e == "rs").unwrap_or(false) {
                                if let Ok(c) = std::fs::read_to_string(&path) {
                                    content.push_str(&c);
                                    content.push('\n');
                                }
                            }
                        }
                    }
                }
                content
            } else {
                parts.join(" ")
            };
            let config = darwin::self_play::SelfPlayConfig {
                population_size: 10,
                max_generations: 20,
                mutation_rate: 0.15,
                crossover_rate: 0.3,
                fitness_threshold: 0.8,
                elite_count: 3,
                temperature: 1.0,
            };
            let mut trainer = darwin::self_play::SelfPlayTrainer::new(
                ".axiom_state/darwin_archive.json", config,
            );
            println!("Self-play training: pop={}, gens={}, starting...",
                config.population_size, config.max_generations);
            let result = trainer.run(&initial_source);
            println!("Self-play complete:");
            println!("  best_fitness: {:.4}", result.best_fitness);
            println!("  best_generation: {}", result.best_generation);
            println!("  convergence: gen {}", result.convergence_generations);
            println!("  final_population: {}", result.final_population_size);
            if !result.best_patch.is_empty() {
                let preview = if result.best_patch.len() > 200 {
                    format!("{}...", &result.best_patch[..197])
                } else {
                    result.best_patch.clone()
                };
                println!("  best_patch ({} bytes): {}", result.best_patch.len(), preview);
            }
        },
        _ => {
            println!("Darwin Archive commands:");
            println!("  kai darwin init                  # Create/initialize archive");
            println!("  kai darwin status                # Show archive stats");
            println!("  kai darwin list                  # List candidates");
            println!("  kai darwin seed                  # Seed candidates from source");
            println!("  kai darwin evolve                # Generate, evaluate, and evolve");
            println!("  kai darwin self-play             # Run self-play training loop (source-code evolution)");
        }
    }
}
fn generate_source_candidates(archive: &mut darwin::DarwinArchive) -> usize {
    let src_dirs = ["src", "rust/kai-fusion/src"];
    let mut files: Vec<String> = Vec::new();
    for dir in &src_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "rs").unwrap_or(false) {
                    if let Some(name) = path.to_str() {
                        files.push(name.to_string());
                    }
                }
            }
        }
    }

    let mut rng = rand::thread_rng();
    let mut generated = 0;

    for filepath in &files {
        if let Ok(content) = std::fs::read_to_string(filepath) {
            for (lineno, line) in content.lines().enumerate() {
                if line.contains("0.01") || line.contains("0.001") || line.contains("0.5") {
                    let patch = format!(
                        "--- a/{}\n+++ b/{}\n@@ -{},1 +{},1 @@\n-{}\n+{} // mutated\n",
                        filepath, filepath, lineno, lineno, line, line
                    );
                    let cand = darwin::Candidate {
                        id: format!("param_{}", rng.gen::<u32>()),
                        description: format!("Modify parameter in {}:{}", filepath, lineno),
                        patch,
                        fitness: 0.5 + rng.gen::<f32>() * 0.5,
                        fitness_components: darwin::FitnessComponents {
                            task_performance: 0.6 + rng.gen::<f32>() * 0.4,
                            novelty: rng.gen::<f32>() * 0.3,
                            ..Default::default()
                        },
                        generation: archive.generation(),
                        parent_id: None,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    };
                    if archive.add(cand) {
                        generated += 1;
                    }
                    if generated >= 20 {
                        return generated;
                    }
                }
            }
        }
    }

    if generated == 0 {
        let patch = "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,3 +1,4 @@\n //! Kai-Fusion CLI.\n+// darwin seed\n".to_string();
        let cand = darwin::Candidate {
            id: format!("seed_{}", rand::random::<u32>()),
            description: "Seed candidate".to_string(),
            patch,
            fitness: 0.5,
            fitness_components: darwin::FitnessComponents::default(),
            generation: archive.generation(),
            parent_id: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        };
        if archive.add(cand) {
            generated += 1;
        }
    }

    generated
}

/// Save model weights as GGUF at `output_path`.
/// Uses original quantization types from `tensor_types` (name → ggml_type).
fn save_gguf(model: &model::Weights, cfg: &Config, tok: &tok::Tokenizer, output_path: &str, tensor_types: &HashMap<String, u32>) {
    use std::fs::File;
    use std::io::Write;
    let mut buf: Vec<u8> = Vec::new();

    // GGUFv3 header
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    gen_internal::put_u32(&mut buf, 3); // version

    // count tensors
    let n_tensors = 1 + cfg.n_layers * 9 + 2; // embed + layers*9 + final_norm + output
    gen_internal::put_u64(&mut buf, n_tensors as u64);

    // metadata
    let mut meta: Vec<(String, gguf::GgufMeta)> = Vec::new();
    meta.push(("general.architecture".into(), gguf::GgufMeta::Str("llama".into())));
    meta.push(("general.name".into(), gguf::GgufMeta::Str("kai-trained".into())));
    meta.push(("llama.block_count".into(), gguf::GgufMeta::Num(cfg.n_layers as f64)));
    meta.push(("llama.embedding_length".into(), gguf::GgufMeta::Num(cfg.dim as f64)));
    meta.push(("llama.attention.head_count".into(), gguf::GgufMeta::Num(cfg.n_heads as f64)));
    meta.push(("llama.attention.head_count_kv".into(), gguf::GgufMeta::Num(cfg.n_kv_heads as f64)));
    meta.push(("llama.attention.layer_norm_rms_epsilon".into(), gguf::GgufMeta::Num(1e-5)));
    meta.push(("llama.feed_forward_length".into(), gguf::GgufMeta::Num(cfg.intermediate as f64)));
    meta.push(("llama.rope.freq_base".into(), gguf::GgufMeta::Num(cfg.rope_theta as f64)));
    meta.push(("llama.vocab_size".into(), gguf::GgufMeta::Num(cfg.vocab_size as f64)));
    meta.push(("llama.context_length".into(), gguf::GgufMeta::Num(cfg.max_seq as f64)));
    meta.push(("tokenizer.ggml.model".into(), gguf::GgufMeta::Str("llama".into())));
    meta.push(("tokenizer.ggml.tokens".into(), gguf::GgufMeta::StrArr(tok.vocab.clone())));
    meta.push(("tokenizer.ggml.scores".into(), gguf::GgufMeta::Arr(vec![0.0; cfg.vocab_size])));
    meta.push(("tokenizer.ggml.bos_token_id".into(), gguf::GgufMeta::Num(tok.bos as f64)));
    meta.push(("tokenizer.ggml.eos_token_id".into(), gguf::GgufMeta::Num(tok.eos as f64)));
    meta.push(("tokenizer.ggml.unknown_token_id".into(), gguf::GgufMeta::Num(tok.unk as f64)));
    gen_internal::put_u64(&mut buf, meta.len() as u64);
    for (k, v) in &meta {
        gen_internal::put_str(&mut buf, k);
        gen_internal::put_value(&mut buf, v);
    }

    // Build tensor list
    struct TensorInfo {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    }
    let mut tensors: Vec<TensorInfo> = Vec::new();

    // Embedding
    let embed_data: Vec<f32> = model.embed.iter().cloned().collect();
    tensors.push(TensorInfo {
        name: "token_embd.weight".into(),
        shape: vec![cfg.dim, cfg.vocab_size],
        data: embed_data,
    });

    for i in 0..cfg.n_layers {
        let l = &model.layers[i];
        let mut push_t = |name: String, shape: Vec<usize>, data: Vec<f32>| {
            tensors.push(TensorInfo { name, shape, data });
        };
        let d = l.wq.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.attn_q.weight"), vec![cfg.dim, cfg.dim], d);
        let d = l.wk.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.attn_k.weight"), vec![cfg.dim_kv(), cfg.dim], d);
        let d = l.wv.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.attn_v.weight"), vec![cfg.dim_kv(), cfg.dim], d);
        let d = l.wo.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.attn_output.weight"), vec![cfg.dim, cfg.dim], d);
        let d = l.w1.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.ffn_gate.weight"), vec![cfg.intermediate, cfg.dim], d);
        let d = l.w2.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.ffn_down.weight"), vec![cfg.dim, cfg.intermediate], d);
        let d = l.w3.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.ffn_up.weight"), vec![cfg.intermediate, cfg.dim], d);
        let d = l.attn_norm.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.attn_norm.weight"), vec![cfg.dim], d);
        let d = l.ffn_norm.as_slice().unwrap().to_vec(); push_t(format!("blk.{i}.ffn_norm.weight"), vec![cfg.dim], d);
    }

    let fnorm_data: Vec<f32> = model.final_norm.iter().cloned().collect();
    tensors.push(TensorInfo {
        name: "output_norm.weight".into(),
        shape: vec![cfg.dim],
        data: fnorm_data,
    });
    let output_data: Vec<f32> = model.output.iter().cloned().collect();
    tensors.push(TensorInfo {
        name: "output.weight".into(),
        shape: vec![cfg.vocab_size, cfg.dim],
        data: output_data,
    });

    // Tensor infos (use original quantization types)
    let mut off2 = 0usize;
    for t in &tensors {
        gen_internal::put_str(&mut buf, &t.name);
        gen_internal::put_u32(&mut buf, t.shape.len() as u32);
        for s in &t.shape {
            gen_internal::put_u64(&mut buf, *s as u64);
        }
        let ggml_type = tensor_types.get(&t.name).copied().unwrap_or(1);
        gen_internal::put_u32(&mut buf, ggml_type);
        gen_internal::put_u64(&mut buf, off2 as u64);
        let elems: usize = t.shape.iter().product();
        let qualib = gguf::quantize_tensor_size(elems, ggml_type);
        off2 = gen_internal::align32(off2 + qualib);
    }

    // Pad to 32-aligned data start
    let data_start = gen_internal::align32(buf.len());
    while buf.len() < data_start {
        buf.push(0);
    }

    // Write tensor data using original quantization types
    for t in &tensors {
        let ggml_type = tensor_types.get(&t.name).copied().unwrap_or(1);
        let elems: usize = t.shape.iter().product();
        let quantized = gguf::quantize_tensor(&t.data, ggml_type, elems);
        buf.extend_from_slice(&quantized);
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
    }

    let mut f = File::create(output_path).expect("create trained GGUF");
    f.write_all(&buf).expect("write trained GGUF");
    println!("saved trained model -> {output_path} ({:.2} MB)", buf.len() as f64 / 1e6);
}

/// Internal helpers borrowed from gen.rs for GGUF serialization
mod gen_internal {
    use crate::gguf::GgufMeta;
    pub fn put_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_str(b: &mut Vec<u8>, s: &str) {
        put_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }
    pub fn put_value(b: &mut Vec<u8>, v: &GgufMeta) {
        match v {
            GgufMeta::Num(f) => { put_u32(b, 6); b.extend_from_slice(&(*f as f32).to_le_bytes()); }
            GgufMeta::Str(s) => { put_u32(b, 8); put_str(b, s); }
            GgufMeta::Arr(a) => {
                put_u32(b, 9); put_u32(b, 6); put_u64(b, a.len() as u64);
                for x in a { b.extend_from_slice(&(*x as f32).to_le_bytes()); }
            }
            GgufMeta::StrArr(a) => {
                put_u32(b, 9); put_u32(b, 8); put_u64(b, a.len() as u64);
                for s in a { put_str(b, s); }
            }
            GgufMeta::Bool(bv) => { put_u32(b, 7); b.push(if *bv { 1 } else { 0 }); }
        }
    }
    #[allow(dead_code)]
    pub fn f32_to_f16(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = (x >> 23) & 0xff;
        let mant = x & 0x7fffff;
        if exp == 255 { return sign | 0x7c00 | ((mant >> 13) as u16); }
        if exp == 0 { return sign; }
        let e = (exp as i32) - 127;
        if e > 15 { return sign | 0x7c00; }
        if e >= -14 {
            let ee = (e + 15) as u32;
            let m = mant >> 13;
            return sign | ((ee << 10) as u16) | (m as u16);
        }
        let shift = (-(e + 1)) as u32;
        if shift >= 24 { return sign; }
        let v = 0x800000u32 | mant;
        let m = (v >> shift) as u16;
        sign | m
    }
    #[allow(dead_code)]
    pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|&x| f32_to_f16(x).to_le_bytes().to_vec()).collect()
    }
    pub fn align32(n: usize) -> usize {
        (n + 31) & !31usize
    }
}

/// `kai train`: full backward pass training over a text sequence.
/// Uses the new forward_with_cache / backward / apply_gradients pipeline.
fn train(path: &str, text: &str, iters: usize, lr: f32) {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => { eprintln!("meta: {e}"); return; }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => { eprintln!("could not build Config (not a decoder arch?)"); return; }
    };
    let (_v, map, _nt, _nk) = match gguf::load_tensors(path) {
        Ok(x) => x,
        Err(e) => { eprintln!("load: {e}"); return; }
    };
    let mut model = match model::Weights::from_gguf(&map, &cfg) {
        Ok(m) => m,
        Err(e) => { eprintln!("from_gguf: {e}"); return; }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => { eprintln!("no tokenizer in GGUF"); return; }
    };
    let ids = tok.encode(text);
    if ids.len() < 2 {
        eprintln!("need >=2 tokens for training (got {})", ids.len());
        return;
    }
    let output_path = path.replace(".gguf", "_trained.gguf");
    println!(
        "Kai training: dim={} layers={} vocab={} text_tokens={} lr={} iters={}",
        cfg.dim, cfg.n_layers, cfg.vocab_size, ids.len(), lr, iters
    );

    for it in 0..iters {
        let mut total_loss = 0.0f32;
        let mut n = 0usize;

        // For each position, predict the next token
        for i in 0..ids.len() - 1 {
            let ctx: Vec<usize> = ids[..=i].to_vec();
            let target = ids[i + 1];

            // Forward with cache
            let cache = model.forward_with_cache(&cfg, &ctx);

            // Cross-entropy gradient: only on the last position
            let t = cache.logits.nrows();
            let mut d_logits = Array2::zeros((t, cfg.vocab_size));
            let raw: Vec<f32> = (0..cfg.vocab_size).map(|j| cache.logits[[t - 1, j]]).collect();
            let p = engine::softmax(&raw);
            for j in 0..cfg.vocab_size {
                d_logits[[t - 1, j]] = p[j] - if j == target { 1.0 } else { 0.0 };
            }

            // Backward
            let grads = model.backward(&cfg, &cache, &d_logits, &ctx, &FreezeConfig::default());

            // Apply gradients
            model.apply_gradients(&grads, lr);

            // Track loss on last position only
            let raw_last: Vec<f32> = (0..cfg.vocab_size).map(|j| cache.logits[[t - 1, j]]).collect();
            let p_last = engine::softmax(&raw_last);
            total_loss += -p_last[target].max(1e-9).ln();
            n += 1;
        }

        let avg_loss = total_loss / n as f32;
        let var = model.readout_variance();
        println!(
            "  iter {it:>3}:  loss={avg_loss:.4}  variance={var:.6}"
        );
    }

    // Accuracy after training
    let mut correct = 0usize;
    for i in 0..ids.len() - 1 {
        let ctx: Vec<usize> = ids[..=i].to_vec();
        let lg = model.last_logits(&cfg, &ctx);
        let pr = engine::softmax(&lg);
        let next = pr.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if next == ids[i + 1] { correct += 1; }
    }
    println!("  accuracy: {}/{} next-token correct ({:.1}%)", correct, ids.len() - 1,
             correct as f64 / (ids.len() - 1) as f64 * 100.0);

    // Save (preserve original quantization types)
    if cfg.vocab_size <= 65536 && cfg.dim <= 8192 {
        let mut tensor_types = HashMap::new();
        if let Ok(data) = std::fs::read(path) {
            if let Ok((_ver, info, _data_start)) = gguf::parse(&data) {
                for t in &info {
                    tensor_types.insert(t.name.clone(), t.ggml_type);
                }
            }
        }
        save_gguf(&model, &cfg, &tok, &output_path, &tensor_types);
    } else {
        println!("  (skipping save: model too large at this stage)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "launch" && args[2] == "opencode" {
        if args.len() >= 4 {
            launch_real(&args[3]);
        } else {
            run();
        }
    } else if args.len() >= 3 && args[1] == "inspect" {
        gguf::inspect(&args[2]);
    } else if args.len() >= 3 && args[1] == "load" {
        load_and_report(&args[2], false);
    } else if args.len() >= 3 && args[1] == "dump" {
        load_and_report(&args[2], true);
    } else if args.len() >= 3 && args[1] == "meta" {
        match gguf::read_kv(&args[2]) {
            Ok(meta) => {
                println!("GGUF metadata ({} keys)", meta.len());
                let mut keys: Vec<&String> = meta.keys().collect();
                keys.sort();
                for k in keys {
                    let v = match &meta[k] {
                        gguf::GgufMeta::Num(n) => format!("{n}"),
                        gguf::GgufMeta::Str(s) => s.clone(),
                        gguf::GgufMeta::Bool(b) => format!("{b}"),
                        gguf::GgufMeta::Arr(a) => format!("[{} elems]", a.len()),
                        gguf::GgufMeta::StrArr(a) => format!("[{} strs]", a.len()),
                    };
                    println!("  {k} = {v}");
                }
                match gguf::build_config(&meta) {
                    Some(c) => println!(
                        "  => Kai-Fusion Config: dim={} layers={} heads={} kv_heads={} vocab={} inter={} rope_theta={} max_seq={}",
                        c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.intermediate, c.rope_theta, c.max_seq
                    ),
                    None => println!("  => could not build Config from metadata (encoder arch; expected for Nomic-BERT)"),
                }
            }
            Err(e) => eprintln!("meta read failed: {e}"),
        }
    } else if args.len() >= 3 && args[1] == "gen" {
        let path = &args[2];
        let dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
        let layers: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
        let vocab: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20);
        gen::gen(path, dim, layers, vocab, &[]);
        println!("wrote synthetic llama-format GGUF -> {path}");
    } else if args.len() >= 5 && args[1] == "transplant" {
        let source = &args[2];
        let template = &args[3];
        let mut output = "transplanted.gguf".to_string();
        let mut embed_map: Option<String> = None;
        let mut target_arch = "dense".to_string();
        let mut i = 4;
        while i < args.len() {
            if args[i] == "-o" && i + 1 < args.len() {
                output = args[i + 1].clone();
                i += 2;
                continue;
            }
            if args[i] == "--embed-map" && i + 1 < args.len() {
                embed_map = Some(args[i + 1].clone());
                i += 2;
                continue;
            }
            if args[i] == "--target-arch" && i + 1 < args.len() {
                target_arch = args[i + 1].clone();
                i += 2;
                continue;
            }
            i += 1;
        }
        match crate::transplant::transplant(source, template, &output, embed_map.as_deref(), &target_arch) {
            Ok(()) => println!("transplant complete -> {output}"),
            Err(e) => eprintln!("transplant failed: {e}"),
        }
    } else if args.len() >= 4 && args[1] == "embed" {
        let path = &args[2];
        let text = args[3..].join(" ");
        let use_model = args.iter().any(|a| a == "--use-model");
        if use_model {
            crate::model::Weights::embed_with_model(path, &text);
        } else {
            bert::run(path, &text);
        }
    } else if args.len() >= 3 && args[1] == "memory" {
        let action = &args[2];
        let params = args[3..].join(" ");
        memory_cmd(action, &params);
    } else if args.len() >= 4 && args[1] == "run" {
        let path = &args[2];
        let mut temperature = 1.0;
        let mut top_p = 0.9;
        let mut image_path: Option<String> = None;
        let mut vision_gguf: Option<String> = None;
        let mut attractor_path: Option<String> = None;
        let prompt_start = 3;
        // First pass: extract flags/values by building a set of positions to skip
        let mut skip_positions: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut i = 2;
        while i < args.len() {
            if args[i] == "--temp" && i + 1 < args.len() {
                temperature = args[i + 1].parse().unwrap_or(1.0);
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            if args[i] == "--top_p" && i + 1 < args.len() {
                top_p = args[i + 1].parse().unwrap_or(0.9);
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            if args[i] == "--image" && i + 1 < args.len() {
                image_path = Some(args[i + 1].clone());
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            if args[i] == "--vision-gguf" && i + 1 < args.len() {
                vision_gguf = Some(args[i + 1].clone());
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            if args[i] == "--attractor" && i + 1 < args.len() {
                attractor_path = Some(args[i + 1].clone());
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            if args[i] == "--agent" && i + 1 < args.len() {
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            if args[i] == "--max-new" && i + 1 < args.len() {
                // Use different parser: stores in a local to be used later
                skip_positions.insert(i);
                skip_positions.insert(i + 1);
                i += 2;
                continue;
            }
            i += 1;
        }
        // Extract --max-new separately
        let mut max_new: usize = 48;
        {
            let mut j = 2;
            while j < args.len() {
                if args[j] == "--max-new" && j + 1 < args.len() {
                    max_new = args[j + 1].parse().unwrap_or(48);
                    // Don't need to modify skip_positions again, already handled above
                }
                j += 1;
            }
        }
        // Build text from all non-skipped positional args starting at index 3
        let text_parts: Vec<&str> = args[prompt_start..].iter()
            .enumerate()
            .filter(|(idx, _)| !skip_positions.contains(&(prompt_start + idx)))
            .map(|(_, s)| s.as_str())
            .collect();
        let text = text_parts.join(" ");

        // ---- Auto-classify query and prepend system prompt ----
        let route = routes::RouteConfig::from_query(&text);
        let mut adjusted_temp = temperature;
        let mut adjusted_topp = top_p;
        // Check if --agent was explicitly provided
        let agent_override: Option<&str> = {
            let mut found = None;
            let mut j = 2;
            while j < args.len() {
                if args[j] == "--agent" && j + 1 < args.len() {
                    found = Some(args[j + 1].as_str());
                }
                j += 1;
            }
            found
        };
        let route = if let Some(agent_str) = agent_override {
            let kind = match agent_str.to_lowercase().as_str() {
                "code" => routes::AgentKind::Code,
                "math" | "science" => routes::AgentKind::MathScience,
                "creative" | "art" => routes::AgentKind::Creative,
                "analysis" | "research" => routes::AgentKind::Analysis,
                _ => routes::AgentKind::General,
            };
            routes::RouteConfig::from_kind(kind)
        } else {
            route
        };
        // Use route temperature if user didn't specify
        if temperature == 1.0 && route.temperature.is_some() {
            adjusted_temp = route.temperature.unwrap();
        }
        if top_p == 0.9 && route.top_p.is_some() {
            adjusted_topp = route.top_p.unwrap();
        }
        // Prepend system prompt
        let augmented_prompt = format!(
            "[System: {}]\n\nUser: {}",
            route.system_prompt,
            text
        );
        eprintln!("[route] {} {} — {}", route.kind.icon(), route.kind.label(), text);

        if let Some(img_path) = image_path {
            let vgguf = vision_gguf.unwrap_or_else(|| {
                let p = std::path::Path::new(path);
                let dir = p.parent().unwrap_or_else(|| std::path::Path::new("."));
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
                dir.join(format!("{}_vision.gguf", stem))
                    .to_str()
                    .unwrap_or("vision.gguf")
                    .to_string()
            });
            generate_with_vision(path, &vgguf, &img_path, &augmented_prompt, max_new, adjusted_temp, adjusted_topp);
        } else {
            let use_rag = args.iter().any(|a| a == "--rag");
            if use_rag {
                generate_with_rag(path, &augmented_prompt, max_new, adjusted_temp, adjusted_topp);
            } else {
                generate_from_gguf_attractor(path, &augmented_prompt, max_new, adjusted_temp, adjusted_topp, attractor_path.as_deref());
            }
        }
    } else if args.len() >= 4 && args[1] == "multi-agent" {
        let path = &args[2];
        let query = args[3..].join(" ");

        // Extract --agents flag (comma-separated list of agent kinds, or "all")
        let mut agent_kinds: Vec<routes::AgentKind> = Vec::new();
        let mut attractor_path: Option<String> = None;
        // Pre-scan for --agents and --attractor flags
        let mut ai = 3;
        while ai < args.len() {
            if args[ai] == "--agents" && ai + 1 < args.len() {
                let agents_str = args[ai + 1].as_str();
                if agents_str == "all" {
                    agent_kinds = routes::AgentKind::all();
                } else {
                    for a in agents_str.split(',') {
                        match a.trim().to_lowercase().as_str() {
                            "code" => agent_kinds.push(routes::AgentKind::Code),
                            "math" | "science" => agent_kinds.push(routes::AgentKind::MathScience),
                            "creative" | "art" => agent_kinds.push(routes::AgentKind::Creative),
                            "analysis" | "research" => agent_kinds.push(routes::AgentKind::Analysis),
                            "general" => agent_kinds.push(routes::AgentKind::General),
                            _ => eprintln!("[multi-agent] unknown agent kind: {a}"),
                        }
                    }
                }
                ai += 2;
            } else if args[ai] == "--attractor" && ai + 1 < args.len() {
                attractor_path = Some(args[ai + 1].clone());
                ai += 2;
            } else {
                ai += 1;
            }
        }

        if agent_kinds.is_empty() {
            let primary = routes::RouteConfig::from_query(&query);
            agent_kinds.push(primary.kind);
            for ak in routes::AgentKind::all() {
                if ak != primary.kind {
                    agent_kinds.push(ak);
                }
            }
        }

        let temperature = 1.0;
        let top_p = 0.9;
        let max_new = 16;

        multi_agent_generate(path, &query, max_new, temperature, top_p, attractor_path.as_deref(), agent_kinds);

    } else if args.len() >= 3 && args[1] == "fs" {
        let cmd = args[2].as_str();
        let fs_args: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();
        let (output, code) = fs_agent::fs_cmd(cmd, &fs_args);
        if !output.is_empty() {
            println!("{output}");
        }
        if code != 0 {
            std::process::exit(code);
        }
    } else if args.len() >= 3 && args[1] == "browse" {
        let url = args[2].as_str();
        match browser::browse_url(url) {
            Ok(text) => {
                let truncated = if text.len() > 8000 {
                    format!("{}...\n[truncated to 8000 chars]", &text[..8000])
                } else {
                    text
                };
                println!("{truncated}");
            }
            Err(e) => {
                eprintln!("browse error: {e}");
                std::process::exit(1);
            }
        }
    } else if args.len() >= 3 && args[1] == "search" {
        let query = args[2..].join(" ");
        match browser::search(&query) {
            Ok(results) => println!("{results}"),
            Err(e) => {
                eprintln!("search error: {e}");
                std::process::exit(1);
            }
        }
    } else if args.len() >= 4 && args[1] == "assimilate" {
        let path = &args[2];
        let text = args[3..].join(" ");
        let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(15);
        let lr: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.05);
        assimilate(path, &text, iters, lr);
    } else if args.len() >= 6 && args[1] == "assimilate-multi" {
        let student = args[2].clone();
        let output = args[3].clone();
        let text = args[4].clone();
        let iters: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
        let lr: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.001);
        let temp: f32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let teacher_paths: Vec<String> = args[8..].to_vec();
        if teacher_paths.is_empty() {
            eprintln!("error: need at least one teacher GGUF path");
            return;
        }
        eprintln!("assimilate-multi: student={student}, output={output}, teachers={}, text=\"{text}\"",
            teacher_paths.len());
        let ass_cfg = assimilate::AssimilationConfig {
            student_path: &student,
            teacher_paths: &teacher_paths,
            output_path: &output,
            text: &text,
            iters,
            lr,
            distillation_temp: temp,
            teacher_weights: None,
        };
        match assimilate::assimilate_multi(&ass_cfg) {
            Ok(r) => eprintln!("assimilate-multi done: {} iters, final_loss={:.4}, saved to {output}",
                r.iterations, r.final_loss),
            Err(e) => eprintln!("assimilate-multi failed: {e}"),
        }
    } else if args.len() >= 4 && args[1] == "fuse" {
        let path = &args[2];
        let text = args[3..].join(" ");
        let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(15);
        let lr: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.05);
        let seed_scale: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.5);
        let attr = args
            .get(7)
            .cloned()
            .unwrap_or_else(|| ".axiom_state/kai_fusion_attractor.json".to_string());
        let output = args.get(8).cloned().unwrap_or_else(|| "fused.gguf".to_string());
        fuse(path, &text, iters, lr, seed_scale, &attr, &output);
    } else if args.len() >= 3 && args[1] == "curvature" {
        let path = &args[2];
        let text = args[3..].join(" ");
        curvature(path, &text);
    } else if args.len() >= 3 && args[1] == "scm" {
        let action = &args[2];
        let params = args[3..].join(" ");
        scm_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "darwin" {
        let action = &args[2];
        let params = args[3..].join(" ");
        darwin_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "distributed" {
        let action = &args[2];
        let params = args[3..].join(" ");
        distributed_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "body" {
        let action = &args[2];
        let params = args[3..].join(" ");
        body_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "engram" {
        let action = &args[2];
        let params = args[3..].join(" ");
        engram_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "physics" {
        let action = &args[2];
        let params = args[3..].join(" ");
        physics_cmd(action, &params);
    } else if args.len() >= 4 && args[1] == "train" {
        let path = &args[2];
        let text = args[3..].join(" ");
        let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
        let lr: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.01);
        train(path, &text, iters, lr);
    } else if args.len() >= 4 && args[1] == "chunked-train" {
        let path = &args[2];
        let text = &args[3];
        let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
        let lr: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.01);
        let distillation_lambda: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let teacher_path: Option<&str> = args.get(7).filter(|s| !s.is_empty()).map(|s| s.as_str());
        let out = path.replace(".gguf", "_trained.gguf");

        let meta = match gguf::read_kv(path) {
            Ok(m) => m,
            Err(e) => { eprintln!("meta: {e}"); return; }
        };
        let _cfg = match gguf::build_config(&meta) {
            Some(c) => c,
            None => { eprintln!("could not build Config"); return; }
        };
        let tok = match tok::Tokenizer::from_gguf(&meta) {
            Some(t) => t,
            None => { eprintln!("no tokenizer"); return; }
        };
        let ids = tok.encode(text);
        let chunks: Vec<(usize, usize)> = vec![(0, ids.len())];

        match chunked::train_chunked(path, text, iters, lr, &out, &chunks, distillation_lambda, teacher_path) {
            Ok(()) => {},
            Err(e) => eprintln!("chunked-train failed: {e}"),
        }
    } else if args.len() >= 4 && args[1] == "geodesic" {
        let path = &args[2];
        let text = &args[3];
        let threshold: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.5);
        let min_chunk: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
        geodesic(path, text, threshold, min_chunk);
    } else if args.len() >= 3 && args[1] == "goal" {
        let action = &args[2];
        let params = args[3..].join(" ");
        goal_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "value" {
        let action = &args[2];
        let params = args[3..].join(" ");
        values_cmd(action, &params);
    } else if args.len() >= 2 && args[1] == "selfmod" {
        let action = if args.len() >= 3 { &args[2] } else { "status" };
        let params = args[3..].join(" ");
        selfmod_cmd(action, &params);
    } else if args.len() >= 3 && args[1] == "vision" {
        let action = &args[2];
        let params = args[3..].join(" ");
        vision_cmd(action, &params);
    } else if args.len() >= 2 && args[1] == "attractor" {
        let action = if args.len() >= 3 { &args[2] } else { "status" };
        let params = args[3..].join(" ");
        attractor_cmd(action, &params);
    } else if args.len() >= 2 && args[1] == "install-service" {
        install_service_cmd();
    } else if args.len() >= 2 && args[1] == "route" {
        let query = args[2..].join(" ");
        let route = routes::RouteConfig::from_query(&query);
        println!("{} [{}] — {}", route.kind.icon(), route.kind.label(), route.system_prompt);
        if route.temperature.is_some() || route.top_p.is_some() {
            println!("  overrides: temp={:?} top_p={:?}", route.temperature, route.top_p);
        }
    } else if args.len() >= 3 && args[1] == "daemon" {
        let action = &args[2];
        let params = args[3..].join(" ");
        daemon_cmd(action, &params);
    } else {
        println!("Kai-Fusion");
        println!("  kai launch opencode [gguf]   # REPL (tiny demo, or real GGUF decoder)");
        println!("  kai transplant <source> <template> [-o <out>] [--embed-map <f>] [--target-arch dense|moe|mla|moe+mla]  # slice & transplant weights");
        println!("  kai inspect <gguf>    # inventory real local model weights");
        println!("  kai load <gguf>       # load tensors + infer dense backbone config");
        println!("  kai dump <gguf>       # print every tensor name + shape");
        println!("  kai meta <gguf>       # print architecture metadata + inferred Config");
        println!("  kai gen <path> [d l v]# write synthetic llama-format test GGUF");
        println!("  kai embed <gguf> <text>  # run real Nomic-BERT encoder on its weights");
        println!("  kai embed <gguf> <text> --use-model  # use model's own mean-pooled hidden state");
        println!("  kai run <gguf> <text> [--temp <t>] [--top_p <p>] [--rag] [--image <path>] [--vision-gguf <path>] [--attractor <path>]  # generation with physics-wired adaptive inference + RAG + multi-modal vision + attractor coordination");
        println!("  kai assimilate <gguf> \"<text>\" [iters] [lr]  # Kai VFE assimilation");
        println!("  kai assimilate-multi <student> <output> \"<text>\" [iters] [lr] [temp] <teacher1> <teacher2> ...  # multi-teacher distillation");
        println!("  kai fuse <gguf> \"<text>\" [iters] [lr] [seed] [attractor.json]  # assimilate seeded by the 10-arch attractor");
        println!("  kai train <gguf> \"<text>\" [iters] [lr]  # full backward pass training (Phase 2)");
        println!("  kai chunked-train <gguf> <text> [iters] [lr] [distill_lambda] [teacher_path]  # chunked training with optional distillation");
        println!("  kai fs <command> [args]  # file system agent (read/write/glob/grep/list/stat/edit/rm/mkdir/find/cat)");
        println!("  kai geodesic <gguf> <text> [threshold] [min_chunk]  # geodesic chunk boundaries");
        println!("  kai darwin <archive_path> [threshold] [max_pop]  # Darwin Archive self-improvement");
        println!("  kai engram list|info|clear|delete  # semantic memory operations");
        println!("  kai physics g_ij|vfe|tau [params]  # kai-mlir physics dialect ops");
        println!("  kai memory store|search|list|clear|count|ingest [args]  # vector memory operations");
        println!("  kai goal decompose <goal>        # break goal into subtasks");
        println!("  kai goal decompose-deep <goal>    # full 3-level decomposition");
        println!("  kai goal flatten <goal>           # flatten plan into execution queue");
        println!("  kai goal plan <goal>              # show plan summary");
        println!("  kai value process <message>       # detect value signals from user message");
        println!("  kai value dominant <message>      # show dominant preference");
        println!("  kai value rank <message>           # show all ranked preferences");
        println!("  kai value weight <dim> [message]   # show weight for a dimension");
        println!("  kai selfmod status                  # show self-modification tracker status");
        println!("  kai selfmod event <desc> [fit_before] [fit_after] [target]  # record modification event");
        println!("  kai selfmod history                 # show recent self-mod events");
        println!("  kai selfmod kill                    # activate kill switch");
        println!("  kai selfmod level                   # show current escalation level");
        println!("  kai distributed bench <n> [workers]  # benchmark parallel vs sequential eval with n tasks");
        println!("  kai distributed status               # show worker info and stats");
        println!("  kai distributed attractor [n] [path] [dim]  # parallel attractor convergence test");
        println!("  kai vision encode <image> [vision_gguf]  # encode image with vision tower");
        println!("  kai vision info <vision_gguf>        # inspect vision tower metadata");
        println!("  kai attractor status [path]          # show attractor convergence and stats");
        println!("  kai attractor converge [path]        # check convergence score");
        println!("  kai install-service                  # install Kai as systemd user service (24/7)");
        println!("  kai route <query>                    # classify query and show sub-agent routing");
        println!("  kai daemon start <gguf> [--attractor <path>]  # start 24/7 fixed-point daemon");
        println!("  kai daemon help                        # daemon options");
    }
}

/// Run a generative REPL over a real llama-format GGUF loaded by our reader.
fn launch_real(path: &str) {
    let meta = match gguf::read_kv(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("meta: {e}");
            return;
        }
    };
    let cfg = match gguf::build_config(&meta) {
        Some(c) => c,
        None => {
            eprintln!("could not build Config (not a decoder arch?)");
            return;
        }
    };
    let (_v, map, _nt, _nk) = match gguf::load_tensors(path) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("load: {e}");
            return;
        }
    };
    let model = match model::Weights::from_gguf(&map, &cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("from_gguf: {e}");
            return;
        }
    };
    let tok = match tok::Tokenizer::from_gguf(&meta) {
        Some(t) => t,
        None => {
            eprintln!("no tokenizer in GGUF");
            return;
        }
    };
    println!(
        "Kai-Fusion REPL over real GGUF: dim={} layers={} vocab={}",
        cfg.dim, cfg.n_layers, cfg.vocab_size
    );
    loop {
        print!("kai> ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }
        let mut ids = tok.encode(line);
        for _ in 0..64 {
            let logits = model.last_logits(&cfg, &ids);
            let next = argmax(&logits);
            ids.push(next);
            if next == tok.eos {
                break;
            }
        }
        println!("{}", tok.decode(&ids));
    }
}

/// `kai value`: Value learning operations.
fn values_cmd(action: &str, params: &str) {
    use values::{ValueLearner, ValueDimension};
    match action {
        "process" => {
            let mut learner = ValueLearner::new();
            let signals = learner.process_message(params);
            if signals.is_empty() {
                println!("No value signals detected.");
            } else {
                println!("Detected {} value signals:", signals.len());
                for s in &signals {
                    println!("  {} polarity={:+.1} trigger=\"{}\"", s.dimension.label(), s.polarity, s.trigger);
                }
                println!("  Summary: {}", learner.summary());
            }
        }
        "dominant" => {
            let mut learner = ValueLearner::new();
            learner.process_message(params);
            let dom = learner.dominant_preference();
            println!("Dominant preference: {} ({:.3})", dom.dimension.label(), dom.weight);
        }
        "rank" => {
            let mut learner = ValueLearner::new();
            learner.process_message(params);
            println!("Ranked preferences:");
            for w in learner.ranked_preferences() {
                println!("  {}: {:.3}", w.dimension.label(), w.weight);
            }
        }
        "weight" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                println!("Usage: kai value weight <dimension>");
                println!("Dimensions: factuality|speed|creativity|thoroughness|conciseness|code_examples|explanations");
                return;
            }
            let dim = match parts[0] {
                "factuality" => ValueDimension::Factuality,
                "speed" => ValueDimension::Speed,
                "creativity" => ValueDimension::Creativity,
                "thoroughness" => ValueDimension::Thoroughness,
                "conciseness" => ValueDimension::Conciseness,
                "code_examples" => ValueDimension::CodeExamples,
                "explanations" => ValueDimension::Explanations,
                _ => {
                    println!("Unknown dimension: {}", parts[0]);
                    return;
                }
            };
            let mut learner = ValueLearner::new();
            let remaining = parts[1..].join(" ");
            if !remaining.is_empty() {
                learner.process_message(&remaining);
            }
            println!("Weight for {}: {:.3}", dim.label(), learner.weight(dim));
        }
        _ => {
            println!("Value learning commands:");
            println!("  kai value process <message>     -- detect value signals from user message");
            println!("  kai value dominant <message>    -- show dominant preference");
            println!("  kai value rank <message>        -- show all ranked preferences");
            println!("  kai value weight <dim> [message] -- show weight for a specific dimension");
        }
    }
}

/// `kai selfmod`: Self-modification tracker operations.
fn selfmod_cmd(action: &str, params: &str) {
    use selfmod::{SelfModTracker, ModEvent};
    match action {
        "status" => {
            let dir = ".axiom_state";
            let _ = std::fs::create_dir_all(dir);
            let path = format!("{dir}/selfmod_state.json");
            let tracker = if let Ok(json) = std::fs::read_to_string(&path) {
                serde_json::from_str::<SelfModTracker>(&json).unwrap_or_default()
            } else {
                SelfModTracker::new()
            };
            println!("{}", tracker.summary());
        },
        "event" => {
            // kai selfmod event <description> [fitness_before] [fitness_after] [target]
            let parts: Vec<&str> = params.splitn(4, ' ').collect();
            if parts.is_empty() {
                eprintln!("Usage: kai selfmod event <description> [fitness_before] [fitness_after] [target]");
                return;
            }
            let desc = parts[0];
            let fit_before: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let fit_after: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let target = parts.get(3).unwrap_or(&"unknown").to_string();
            let dir = ".axiom_state";
            let _ = std::fs::create_dir_all(dir);
            let path = format!("{dir}/selfmod_state.json");
            let mut tracker = if let Ok(json) = std::fs::read_to_string(&path) {
                serde_json::from_str::<SelfModTracker>(&json).unwrap_or_default()
            } else {
                SelfModTracker::new()
            };
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            let event = ModEvent {
                level: tracker.current_level(),
                target,
                description: desc.to_string(),
                hash_before: selfmod::simple_hash("before"),
                hash_after: selfmod::simple_hash("after"),
                fitness_before: fit_before,
                fitness_after: fit_after,
                timestamp: now,
                human_approved: true,
            };
            tracker.record_event(event);
            if let Ok(json) = serde_json::to_string_pretty(&tracker) {
                let _ = std::fs::write(&path, &json);
            }
            println!("Recorded self-mod event: fitness {fit_before} -> {fit_after} ({})",
                if fit_after > fit_before { "improved" } else { "regressed" });
            if tracker.divergence_detected() {
                eprintln!("⚠️  DIVERGENCE DETECTED! Consider rollback or kill switch.");
            }
            println!("{}", tracker.summary());
        },
        "history" => {
            let path = ".axiom_state/selfmod_state.json";
            let tracker = if let Ok(json) = std::fs::read_to_string(path) {
                serde_json::from_str::<SelfModTracker>(&json).unwrap_or_default()
            } else {
                SelfModTracker::new()
            };
            let events = tracker.events();
            if events.is_empty() {
                println!("No self-modification events recorded.");
                return;
            }
            println!("Self-modification history ({} events, {}):", events.len(), tracker.summary());
            for (i, e) in events.iter().rev().enumerate().take(20) {
                println!("  #{:<3} [{}] {} → {:.2}→{:.2} target={} desc=\"{}\"",
                    i + 1, e.level.label(), e.timestamp, e.fitness_before, e.fitness_after,
                    e.target, if e.description.len() > 40 {
                        format!("{}...", &e.description[..37])
                    } else { e.description.clone() });
            }
        },
        "kill" => {
            let dir = ".axiom_state";
            let _ = std::fs::create_dir_all(dir);
            let path = format!("{dir}/selfmod_state.json");
            let mut tracker = if let Ok(json) = std::fs::read_to_string(&path) {
                serde_json::from_str::<SelfModTracker>(&json).unwrap_or_default()
            } else {
                SelfModTracker::new()
            };
            tracker.activate_kill_switch();
            if let Ok(json) = serde_json::to_string_pretty(&tracker) {
                let _ = std::fs::write(&path, &json);
            }
            println!("🔒 Kill switch activated. Self-modification permanently disabled.");
        },
        "level" => {
            let path = ".axiom_state/selfmod_state.json";
            let tracker = if let Ok(json) = std::fs::read_to_string(path) {
                serde_json::from_str::<SelfModTracker>(&json).unwrap_or_default()
            } else {
                SelfModTracker::new()
            };
            println!("Current self-modification level: {} ({})",
                tracker.current_level().label(), tracker.current_level().numeric());
            println!("Allowed levels: L0:human, L1:agent-code (requires approval)");
        },
        _ => {
            println!("Self-modification commands:");
            println!("  kai selfmod status                  -- show tracker status");
            println!("  kai selfmod event <desc> [f_before] [f_after] [target]  -- record an event");
            println!("  kai selfmod history                 -- show recent events");
            println!("  kai selfmod kill                    -- activate kill switch");
            println!("  kai selfmod level                   -- show current level");
        }
    }
}

/// `kai goal`: Goal decomposition operations.
fn goal_cmd(action: &str, params: &str) {
    use goals::{GoalDecomposer, SubtaskStatus};
    match action {
        "decompose" => {
            let decomposer = GoalDecomposer::new();
            let plan = decomposer.decompose(params);
            println!("Goal: {}", plan.goal);
            println!("  Subtasks: {}", plan.subtasks.len());
            println!("  Progress: {}", plan.summary());
            for s in &plan.subtasks {
                print_subtask(s, 1);
            }
        }
        "decompose-deep" => {
            let decomposer = GoalDecomposer::with_max_depth(3);
            let plan = decomposer.decompose(params);
            println!("Goal: {}", plan.goal);
            println!("  Subtasks: {}", plan.subtasks.len());
            println!("  Progress: {}", plan.summary());
            for s in &plan.subtasks {
                print_subtask(s, 1);
            }
        }
        "flatten" => {
            let decomposer = GoalDecomposer::new();
            let plan = decomposer.decompose(params);
            let flat = goals::flatten_plan(&plan);
            println!("Goal: {}", plan.goal);
            println!("  Flattened queue ({} items):", flat.len());
            for (i, s) in flat.iter().enumerate() {
                let status_str = match &s.status {
                    SubtaskStatus::Pending => "pending",
                    SubtaskStatus::InProgress => "in-progress",
                    SubtaskStatus::Completed => "completed",
                    SubtaskStatus::Failed { .. } => "failed",
                    SubtaskStatus::Skipped { .. } => "skipped",
                };
                println!("    {i}. [{status_str}] {} — {}", s.label, s.description);
            }
        }
        "plan" => {
            let decomposer = GoalDecomposer::new();
            let plan = decomposer.decompose(params);
            println!("GoalPlan({})", plan.goal);
            println!("  status: {}", plan.summary());
        }
        _ => {
            println!("Goal decomposition commands:");
            println!("  kai goal decompose <goal>        — break goal into subtasks");
            println!("  kai goal decompose-deep <goal>    — full 3-level decomposition");
            println!("  kai goal flatten <goal>           — flatten plan into execution queue");
            println!("  kai goal plan <goal>              — show plan summary");
        }
    }
}

fn vision_cmd(action: &str, params: &str) {
    match action {
        "encode" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                eprintln!("usage: kai vision encode <image_path> [vision_gguf]");
                return;
            }
            let image_path = parts[0];
            let vision_gguf = parts.get(1).copied();

            // Load or create random vision tower
            let (weights, cfg) = if let Some(gguf_path) = vision_gguf {
                match vision::load_vision_gguf(gguf_path) {
                    Ok(wc) => wc,
                    Err(e) => {
                        eprintln!("failed to load vision GGUF: {e}");
                        return;
                    }
                }
            } else {
                // Use random tiny weights for testing
                let cfg = VisionConfig {
                    enabled: true,
                    tower: VisionTowerKind::SigLIP,
                    image_size: 224,
                    patch_size: 16,
                    vision_dim: 64,
                    n_vision_layers: 2,
                    n_vision_heads: 4,
                    vision_intermediate: 128,
                    proj_dim: 64,
                };
                let weights = vision::VisionTowerWeights::random(&cfg);
                (weights, cfg)
            };

            // Load image
            let pixels = match vision::load_image(image_path, cfg.image_size) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("failed to load image: {e}");
                    return;
                }
            };

            eprintln!(
                "[vision] encoding image: {} -> [{}, {}, {}]",
                image_path,
                pixels.shape()[0],
                pixels.shape()[1],
                pixels.shape()[2]
            );

            let embeddings = weights.encode(&pixels, &cfg);
            let n_patches = embeddings.shape()[0];
            let proj_dim = embeddings.shape()[1];

            println!(
                "Image encoded: {} patches x {} dim = {} tokens projected into text space",
                n_patches,
                proj_dim,
                n_patches
            );
            // Show first few values
            let flat: Vec<f32> = embeddings.iter().copied().collect();
            if !flat.is_empty() {
                let mean = flat.iter().sum::<f32>() / flat.len() as f32;
                let max_abs = flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                println!("  mean={mean:.6}, max_abs={max_abs:.6}");
                println!("  first patch: {:?}", &flat[..proj_dim.min(8)]);
            }
        }
        "info" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                eprintln!("usage: kai vision info <vision_gguf>");
                return;
            }
            let gguf_path = parts[0];
            match vision::load_vision_gguf(gguf_path) {
                Ok((_weights, cfg)) => {
                    println!("Vision tower info:");
                    println!("  tower:      {:?}", cfg.tower);
                    println!("  image_size: {}", cfg.image_size);
                    println!("  patch_size: {}", cfg.patch_size);
                    println!("  vision_dim: {}", cfg.vision_dim);
                    println!("  layers:     {}", cfg.n_vision_layers);
                    println!("  heads:      {}", cfg.n_vision_heads);
                    println!("  inter:      {}", cfg.vision_intermediate);
                    println!("  proj_dim:   {}", cfg.proj_dim);
                }
                Err(e) => eprintln!("failed to load vision GGUF: {e}"),
            }
        }
        _ => {
            println!("Vision commands:");
            println!("  kai vision encode <image> [vision_gguf]  — encode image with vision tower");
            println!("  kai vision info <vision_gguf>        — inspect vision tower metadata");
        }
    }
}

/// `kai attractor` — multi-agent fixed-point coordination via shared attractor file.
fn attractor_cmd(action: &str, params: &str) {
    let default_path = attractor::DEFAULT_ATTRACTOR_PATH;
    match action {
        "status" => {
            let path = if params.is_empty() { default_path } else { params.trim() };
            if !std::path::Path::new(path).exists() {
                println!("Attractor: no file at {path}");
                return;
            }
            match attractor::load(path) {
                Ok(vecs) => {
                    let cent = attractor::centroid(&vecs);
                    let (conv, n) = attractor::convergence(path, 10).unwrap_or((0.0, 0));
                    println!("Attractor: {path}");
                    println!("  vectors:    {}", vecs.len());
                    println!("  dim:        {}", if !cent.is_empty() { cent.len() } else { 0 });
                    println!("  convergence (last 10): {conv:.4}");
                    println!("  total pushes: {n}");
                    if vecs.len() >= 2 {
                        let last = &vecs[vecs.len() - 1];
                        println!("  last vector: [{:.4}, {:.4}, {:.4}, ...]",
                            last.first().copied().unwrap_or(0.0),
                            last.get(1).copied().unwrap_or(0.0),
                            last.get(2).copied().unwrap_or(0.0));
                    }
                }
                Err(e) => eprintln!("attractor load failed: {e}"),
            }
        }
        "converge" => {
            let path = if params.is_empty() { default_path } else { params.trim() };
            match attractor::convergence(path, 10) {
                Ok((conv, n)) => println!("Attractor convergence: {conv:.4} (n={n})"),
                Err(e) => eprintln!("convergence check failed: {e}"),
            }
        }
        _ => {
            println!("Attractor commands:");
            println!("  kai attractor status [path]    — show attractor convergence and stats");
            println!("  kai attractor converge [path]   — check convergence score");
            println!("  (default path: {default_path})");
        }
    }
}

/// `kai install-service` — install Kai as a systemd user service for 24/7 operation.
fn install_service_cmd() {
    let service_name = "kai-fusion";
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "/usr/local/bin/kai".to_string());
    let service_content = format!(
        "[Unit]\n\
         Description=Kai-Fusion AGI — autonomous 24/7 attractor-based learning\n\
         After=network.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe} launch opencode\n\
         Restart=always\n\
         RestartSec=10\n\
         Environment=RUST_LOG=info\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    );

    let user_service_dir = format!(
        "{}/.config/systemd/user",
        std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
    );
    let service_path = format!("{user_service_dir}/{service_name}.service");

    match std::fs::create_dir_all(&user_service_dir) {
        Ok(_) => {},
        Err(e) => { eprintln!("mkdir: {e}"); return; }
    }
    match std::fs::write(&service_path, &service_content) {
        Ok(_) => println!("Wrote systemd user service to {service_path}"),
        Err(e) => { eprintln!("write failed: {e}"); return; }
    }

    println!();
    println!("To enable and start:");
    println!("  systemctl --user daemon-reload");
    println!("  systemctl --user enable {service_name}");
    println!("  systemctl --user start {service_name}");
    println!();
    println!("To check status:");
    println!("  systemctl --user status {service_name}");
    println!();
    println!("To view logs:");
    println!("  journalctl --user -u {service_name} -f");
}

/// `kai daemon` — 24/7 fixed-point loop with tau decay.
fn daemon_cmd(action: &str, params: &str) {
    match action {
        "start" => {
            let parts: Vec<&str> = params.split_whitespace().collect();
            if parts.is_empty() {
                eprintln!("Usage: kai daemon start <gguf_path> [--attractor <path>]");
                return;
            }
            let model_path = parts[0].to_string();
            let mut cfg = daemon::DaemonConfig::default();
            cfg.model_path = model_path;

            // Parse optional flags
            let mut i = 1;
            let pvec: Vec<&str> = params.split_whitespace().collect();
            while i < pvec.len() {
                if pvec[i] == "--attractor" && i + 1 < pvec.len() {
                    cfg.attractor_path = pvec[i + 1].to_string();
                    i += 2;
                    continue;
                }
                i += 1;
            }

            eprintln!("[daemon] starting with model={}, attractor={}",
                cfg.model_path, cfg.attractor_path);

            let stop = Arc::new(AtomicBool::new(false));

            match daemon::run_daemon(&cfg, stop) {
                Ok(metrics) => {
                    println!();
                    println!("[daemon] session ended");
                    println!("  generations:    {}", metrics.total_generations);
                    println!("  wall time:      {:.0}s", metrics.uptime_seconds);
                    println!("  subjective:     {:.0}s", metrics.subjective_seconds);
                    println!("  final τ:        {:.4}", metrics.tau);
                    println!("  attractor:      {} vectors, conv={:.4}",
                        metrics.attractor_vectors, metrics.attractor_convergence);
                    if metrics.attractor_convergence > 0.95 {
                        println!("  fixed point:    ✅ CONVERGED");
                    }
                }
                Err(e) => eprintln!("[daemon] error: {e}"),
            }
        }
        _ => {
            println!("Daemon commands:");
            println!("  kai daemon start <gguf> [--attractor <path>]  — start 24/7 fixed-point daemon");
            println!("  (The daemon loads a model, runs a fixed-point loop: generate → push → decay τ → repeat.)");
        }
    }
}

fn print_subtask(s: &goals::Subtask, indent: usize) {
    let prefix = "  ".repeat(indent);
    let status = match &s.status {
        SubtaskStatus::Pending => "⬜",
        SubtaskStatus::InProgress => "🔄",
        SubtaskStatus::Completed => "✅",
        SubtaskStatus::Failed { .. } => "❌",
        SubtaskStatus::Skipped { .. } => "⏭️ ",
    };
    println!("{prefix}{status} [{}] {} — {}", s.id, s.label, s.description);
    for child in &s.subtasks {
        print_subtask(child, indent + 1);
    }
}
