---
name: kai
description: Local-first AGI workflow for Kai - use when running the Kai agent, local LLM orchestration, attractor memory, VFE routing, Darwin evolution, MCP tools, or making free models act like AGI. Triggers on kai, axiom, attractor, vfe, darwin, local ai, mcp, axiom_alien.
---

# Kai — Local-first AGI Workflow

Kai is not a model. Kai is the **loop around any model** that makes a 3B local model behave like a much larger system for real work. The model generates, the workflow remembers, routes, and evolves.

## When to use

Use this skill when:
- Running or developing Kai / Axiom Local
- Working with local LLMs via Ollama (any `qwen`, `tinyllama`, `gemma` in `~/.ollama/models`)
- Using attractor memory, VFE, or Darwin evolution
- Calling MCP tools (`axiom_chat`, `axiom_status`, `axiom_evolve`, etc.)
- Building local-first features that must run with no cloud API

## Architecture (3 pillars)

**1. Memory / Attractor (`attractor.rs`, `axiom_alien.py:VectorMemory`)**
173 vectors = "what good looks like" distilled from 10 teachers. Every hidden state pulled toward nearest basin via KL. Far = explore, near = exploit. Memory DB: `.axiom_state/memory.db` (SQLite, survives restarts). Self-fact `SELF: I am Kai...` seeded once.

**2. VFE Routing (`vfe.rs`, `physics-dialect`)**
`VFE = -log p(obs) + KL[q||p(attractor)]` — one scalar, three actuators: sampling temperature (high VFE = hotter), tau compute budget (high VFE = more compute/token, `tau=useful_work/wall_time`), learning (low VFE = auto-assimilate; VFE picks next teacher).

**3. Darwin Evolution (`darwin.rs`, 0=infinite)**
Population 10 × infinite gens (default), 0=run until killed, `archive.save()` every gen. Fitness = `compile_pass × (0.3·tests + 0.2/latency + 0.2·size + 0.2·consistency + 0.1·novelty)`. Plateau logged, never halts. Resume = skip re-seed if archive has candidates.

## Running

```bash
# MCP server (stdio + HTTP :8000, OpenAI-compatible)
python3 axiom_mcp_server.py          # needs Ollama on :11434, any model pulled
# Bridge (OpenAI API on :8765, for possessed models)
python3 kai_bridge.py --port 8765

# Desktop (Tauri, Rust)
cd desktop/src-tauri && cargo tauri build --bundles deb

# Rust engine (312 tests, 0 warnings)
cd rust && cargo test --release
cargo run --bin kai -- darwin self-play --generations 8  # bounded for demo
cargo run --bin kai -- darwin self-play                  # infinite, Ctrl+C to stop, resumes
```

Cold boot takes ~3 min (model load). Desktop shows loading bar + auto-sends when MCP is up.

## MCP Tools (via `axiom_mcp_server.py`)

`axiom_chat(query)` — deep reasoning through Kai (memory + attractor + VFE). Use for hard problems.
`axiom_status()` — attractor/VFE/bracket/convergence snapshot.
`axiom_identity()` — current bracket-line identity.
`axiom_evolve(task)` — one Darwin cycle on source.
`axiom_kb_search(query)` / `axiom_wiki(topic)` — knowledge base.
`axiom_reset()` — clear conversation, keep memory.

## Rules

- Local-first: Ollama on `127.0.0.1:11434` before any remote fallback.
- Never invent file contents — call `read_file` / `list_dir`.
- Identity answers come from stored state (`SELF` fact + `at.identity()`), not model-card defaults.
- No canned greetings: every prompt, however short, flows through the full pipeline.
- Generations are indeterminate — no caps, checkpoint every gen, plateau is measured not halting.

## Product

Kai desktop (`desktop/`) = one-click container for the same engine + MCP + bridge. Android (`kai-android/`) = mobile companion. Same engine, three faces.
