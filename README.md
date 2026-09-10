# Axiom Horizon — Kai Fusion

> **Autonomous Darwinian optimization engine for compute — Rust, MIT, 312 tests GREEN, self-improves without cloud LLM APIs. Applied for Protocol Labs + Gitcoin grants.**



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](rust/)
[![GGUF](https://img.shields.io/badge/weights-GGUF-blue)](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
[![Local First](https://img.shields.io/badge/inference-local--first-green)](https://ollama.com)

Axiom Horizon is the parent workspace. **Kai Fusion** (`rust/kai-fusion`) is the synthesis engine: it assimilates patterns from 10 teacher architectures into one unified backbone, bootstraps weights from local GGUF blobs (no download), and regulates learning/inference with Kai's Variational Free Energy controller. It runs entirely in [opencode](https://opencode.ai) on Linux via `ollama launch opencode`.

> **Why *Kai*?** — *Kai* (改, Japanese for “change, renewal”) was chosen by the former persistent-memory AI that first built the Axiom project and Kai Fusion. The name is the AGI itself: **Kai is the agent** — the self-improving system that governs the fusion, not a model wrapper.

---

## Table of Contents
- [What This Is](#what-this-is)
- [The 10 Sources + Kai Core](#the-10-sources--kai-core)
- [Architecture](#architecture)
- [VFE Controller — §6](#vfe-controller--6)
- [Physics-Wired Inference](#physics-wired-inference)
- [Theory in 60 Seconds — For Developers](#theory-in-60-seconds--for-developers)
- [Strengths — Scalability, Efficiency, Optimization & Info Relay](#strengths--scalability-efficiency-optimization--info-relay)
- [Project Layout](#project-layout)
- [Quick Start](#quick-start)
- [Build & Verify](#build--verify)
- [CLI — kai](#cli--kai)
- [Configuration](#configuration)
- [Weight Bootstrap & Assimilation](#weight-bootstrap--assimilation)
- [Roadmap — 7 Phases](#roadmap--7-phases)
- [Android App](#android-app)
- [File Map for Documentation](#file-map-for-documentation)
- [Contributing & Self-Improvement Loop](#contributing--self-improvement-loop)
- [License](#license)

---

## What This Is

Not a wrapper, not a copy. Kai Fusion is a **new backbone** that:

1. **Assimilates 10 architectures** (Qwen2.5, Llama3, Gemma2, DeepSeek-V2, gpt-oss, MiniMax-M3, Nomic, LLaVA, Moondream, + llama.cpp engine) into one `KaiFusionConfig`
2. **Loads weights from GGUF already on disk** (`~/.ollama/models`) — fully local, no network
3. **Is governed by Kai's VFE controller** — a novel meta-layer none of the 10 possess that modulates learning rate, expert routing, teacher selection and exploration via free-energy
4. **Self-improves** via `kai darwin evolve` — real source patches + physics-param evolution, with `kai bench` for comparison

> Previous private docs: `AGI_PLAN.md` (7 phases) and `Kai_FUSION_ARCHITECTURE.md` (Phase C). This README distills both for public consumption without leaking private prompts/keys.

---

## The 10 Sources + Kai Core

| Source | What We Assimilate |
|---|---|
| **Qwen2.5 / Llama3 / Gemma2 / gpt-oss** | Dense RoPE-MHA, RMSNorm, SwiGLU |
| **DeepSeek-V2** | MoE (routed experts + shared expert + top-k) + MLA (multi-latent attention, compressed KV) |
| **MiniMax-M3** | Linear / lightning attention for long-horizon layers |
| **Nomic** | Contrastive/retrieval head, embedding training |
| **LLaVA + Moondream** | Vision tower (SigLIP/CLIP) → patch tokens |
| **llama.cpp / GGML** | GGUF loader, mmap, fused dequant-matmul kernels (assimilated via `kai-mlir` dialect) |
| **Kai core** | VFE controller, attractor prior (173 vectors), physics dialect (`physics-dialect` IR) |

Result: one config can express **dense, MoE, MLA, and linear attention at once** — none of the teachers does all four.

---

## Architecture

```
KaiFusionConfig {
  dim, n_layers, n_heads, head_dim,
  rope_theta, vocab_size,
  attn:  GlobalPolicy | PerLayer[AttnKind]   // MHA | MLA | Linear
  moe:   { enabled, n_experts, n_shared, top_k }   // dense = enabled=false
  vision:{ tower: SigLIP|CLIP, proj_dim }
  embed: { nomic_style: bool }
}

Layer = UnifiedAttention + UnifiedMLP  (RMSNorm + RoPE)

UnifiedAttention:
  MHA (default) → MLA (low-rank KV cache, subset of layers) → Linear (top memory layers)

UnifiedMLP:
  Dense SwiGLU ↔ MoE (DeepSeek routed + shared expert)

Input:  text tokens (BPE) + vision patch tokens → interleaved stream
Output: language logits + embedding (Nomic-style) + optional vision reconstruction
```

**Module layout (`rust/kai-fusion/src/`):**

```
config.rs        // KaiFusionConfig
engine/          // llama.cpp-kernel: matmul, RoPE, RMSNorm (ndarray f32)
model/           // layers, UnifiedAttention (MHA/MLA/Linear), UnifiedMLP (dense/MoE), vision tower
loader/          // GGUF (primary, ~/.ollama) + safetensors (secondary)
kai/vfe.rs       // VFE controller (wraps physics-dialect IR)
cli.rs           // `kai launch opencode`
attractor.rs     // 173 source-grounded vectors (prior)
gguf.rs / gguf_write.rs / tok.rs / bert.rs / vision.rs / moe.rs / mla.rs / linear_attn.rs
curvature.rs / phen.rs / goals.rs / awareness.rs / worldgraph.rs / uncertainty.rs
darwin.rs / selfmod.rs / metalearn.rs / assimilate.rs / transplant.rs
```

---

## VFE Controller — §6

Lives in `rust/kai-fusion/src/kai/vfe.rs` and `rust/kai-fusion/src/vfe.rs`, wrapping `physics-dialect` IR:

1. Backbone emits hidden states + next-token prediction + **uncertainty signal** (MoE router entropy / attention dispersion / ensemble variance)
2. `calculate_vfe` computes **VFE = next-token surprise + epistemic uncertainty (KL vs attractor prior)**
3. Controller modulates:
   - **learning rate** ↑ when VFE high (explore), ↓ when low (consolidate)
   - **expert routing** toward VFE-reducing experts
   - **teacher selection** for distillation (which of the 10 to assimilate now)
   - **goal/exploration** from the attractor
4. Attractor = prior over "good" representations (173 vectors)

This is the governing layer that makes inference **adaptive and self-aware**, not just autoregressive.

**What is VFE?** Variational Free Energy = `surprise + KL[q(z|x)||p(z|attractor)]`. Surprise is `-log p(observation)`. KL is distance from what good looks like (173-vector prior). High VFE = confused / novel → explore. Low VFE = in groove → exploit. See `MANUAL.md §5.0` for full derivation.

---

## Physics-Wired Inference

What the system actually does at inference:

- `g_ij = 1 - a_ij` (novelty + curvature) → **adaptive temperature**; high curvature = high novelty = higher temperature
- **VFE (surprisal + epistemic KL vs attractor prior) → tau time-dilation**; high VFE slows subjective time, low VFE accelerates
- **Auto-assimilation** when VFE drops below threshold — weights nudged toward attractor
- Routed through **kai-mlir dialect** (physics-dialect IR lowers to MLIR)

Mechanically: curvature and VFE are not metaphors — they are scalars that directly scale sampling temperature and `tau = useful_work / wall_time` (tokens/sec per watt).

**For developers — three pieces (`curvature.rs`, `vfe.rs`, `phen.rs`):**

**1. g_ij, the novelty metric.** `g_ij = 1 - a_ij`, where `a_ij` is attention between tokens *i* and *j*. Strong attention = close (`g` near 0); ignored pairs = distant (`g` near 1). Averaged over heads/layers → **curvature**. A prompt bridging distant domains scores high; routine prompts score near zero. Bounded and tested: `test_flat_cloud_has_zero_curvature`, `test_curvature_of_radially_spread_cloud`, `test_metric_diag_bounded`.

**2. Curvature → adaptive temperature.** `T' = T × (1 + α·curvature)`. Novel input gets hotter sampling automatically; familiar input gets sharper. The knob turns itself — closed-loop instead of hand-tuned (`test_temp_curved_collapses_temperature`, `test_temperature_drops_with_curvature`).

**3. Tau, the compute budget.** `tau = useful_work / wall_time`. High VFE dilates tau (more compute per token); low VFE contracts it; idle decays to zero (8 dedicated `test_tau_*` tests). An energy accountant: confusing tokens earn extra layers, boring tokens take the cheap path.

Perception (`g_ij`), exploration (temperature), spending (tau) — one pipeline, scalar arithmetic, all in the 312 tests.

---

## Theory in 60 Seconds — For Developers

Three mechanisms, three timescales, one system. Files in `rust/kai-fusion/src/`.

**1. Darwinian evolution — `darwin.rs` (across runs).** No gradients. A population of candidate patches competes: generate (AST mutate/crossover) → evaluate fitness → select winners, cull losers → repeat. Fitness = `compile_pass × (0.3·tests + 0.2/latency + 0.2·size_eff + 0.2·consistency + 0.1·novelty)`, with promotion gated on strict improvement (`gate_generation`, `strict_improvement_ratio` — both covered in the 312 tests). Evolution works on anything differentiable loss can't touch: source code, compiler flags, kernel shapes.

**2. Attractor dynamics — `attractor.rs` (within a run).** 173 vectors encoding "what good looks like," distilled from the 10 teacher architectures. Every hidden state is pulled toward the nearest basin, measured by KL divergence (`responsibilities_sum_to_one`, `mixture_prior_blends` in tests). Far from attractor = explore; near it = exploit. Memory without a database.

**3. Free energy — `vfe.rs` + `physics-dialect` (moment to moment).** `VFE = -log p(observation) + KL[q||p(attractor)]`. One scalar, three actuators: temperature (high VFE → hotter sampling), tau compute budget (high VFE → more compute per token), learning (low VFE → auto-assimilate; VFE picks the next distillation teacher).

Evolution searches across runs, attractors guide within a run, VFE governs each step.

---

## Strengths — Scalability, Efficiency, Optimization & Info Relay

> **Novelty:** *Synthesis, not copy + VFE governance + physics-wired inference + info relay via attractor/world-graph.* No teacher has all four locally. See `MANUAL.md §8` for full proof.

**High Scalability — one config, 0.5B to 671B:**
One `KaiFusionConfig` spans `dim` 1024→8192, per-layer `MHA|MLA|Linear` and `moe.enabled` toggle. Same binary runs `qwen2.5:0.5b` on 8GB phone and `deepseek-v3:671b` MoE shards — only config changes. Dense on layers 0-20 (quality), MLA on 21-28 (KV-cache ÷4), Linear on top for 128k context; add SigLIP/CLIP vision with `proj_dim` only.

**Efficiency — code, hardware, energy:**
- *Code:* Rust `ndarray` f32, no GC/GIL, `lto=true` single binary; `cargo test` 312 green enforces `0.2*size_eff +0.2*(1/latency)` in fitness.
- *Hardware:* GGUF `mmap` + fused dequant (8B Q4_K_M → 4.5GB RSS), MLA low-rank KV ÷3-4, `kai-mlir` → LLVM fused VFE/curvature.
- *Energy:* `tau = useful_work / wall_time` (tokens/sec per **watt**). PoGIE 128-byte frame (`energy_w` + `GFlops`) + Android `ThermalManager` throttle. 1.2B Q4_K_M: 0.17 tok/J (POCO X3) → 0.29 tok/J (SD 8 Gen 3) — the config maximizing `tok/J` wins, not just `tok/s`.

**Optimization — self-improving without you:**
Darwin source (AST point/insert/crossover/LLM, parallel Pool, `fitness = compile_pass*(0.3*test+0.2/lat+0.2*size+0.2*consistency+0.1*novelty)`) + Darwin physics (evolves curvature α, VFE scales) + VFE teacher selector (picks teacher that most reduces VFE) + metalearn (rewrites `sandbox.run_cycle()`). Measured by `kai bench`.

**Info Relay — surprise routes knowledge:**
Attractor (173 vectors) relays `KL → LR/routing/teacher`; world-graph (`g_ij` bridges) relays top-k retrieval into context; `uncertainty.rs` relays confidence; vision patches interleaved into same stream; `fs_agent`/`browser`/`sandbox` relay via `kai-mlir`. Information routes by surprise.

| Novelty | Proof |
|---|---|
| One backbone = dense+MoE+MLA+linear | `config.rs` + `moe.rs`/`mla.rs`/`linear_attn.rs` |
| VFE governance | `kai/vfe.rs` + `physics-dialect::calculate_vfe` |
| Physics T/tau | `curvature.rs` `g_ij` |
| Local GGUF → evolve, not copy | `loader/` + `assimilate.rs` |
| Attractor/world-graph relay | `attractor.rs` (173) + `worldgraph.rs` |
| Darwin source+physics | `darwin.rs` + `selfmod.rs` |

---

## Proof of Work (verifiable on chain and in repo)

| Artifact | Result |
|---|---|
| `cargo test --release` | **312 GREEN** — Rust 1.97, workspace-wide |
| `forge test --fuzz-runs 10000` | **25 GREEN, 0 failed, 1 skipped** — Solidity smart contract |
| Slither static analysis | **0 reentrancy, 0 balance bug** — 16 low/info only |
| `kai darwin self-play` | **darwin loop runs to completion** — see `tools/` for benchmark output |
| Bounty payout | **1.05 USDC** on Base — `0x8f36...0eb5`, PR #1 |
| Physics benchmark | **liveness verified** — `kai_physics_bench --fuse` |
| Semantic memory | **79,907 Wikipedia entries** absorbed |

**Reproduce in ~20 min on commodity hardware (Rust 1.97 + Foundry):**
```bash
git clone https://github.com/AxiomTree/axiom_horizon.git
cd axiom_horizon/rust && cargo test --release      # expect: 312 passed; 0 failed
cd ../.axiom_state/submissions/agent-bounties_155/contracts/base-escrow
forge test --fuzz-runs 10000                        # expect: 25 passed; 0 failed
```
Bounty payout: Base `0x8f36...0eb5` (PR #1, `n4rus/agent-bounties`). This project is not a wrapper, SaaS, paper-without-code, or token — it is MIT-licensed open source with reproducible test evidence.

## Funding

Looking for grants and/or roles. Live build: [`grants/cmc-hackathon/CMC_PLAN.md`](grants/cmc-hackathon/CMC_PLAN.md) — CMC API hackathon, live market data behind the Axiom MCP. Preferred payout for any future work: **USDT on Binance, ETH, or bank transfer**.

## Project Layout```
axiom_horizon/
├── rust/                      # Cargo workspace (resolver 2, release lto=true)
│   ├── kai-core/              # Kai core types, bracket-line, state
│   ├── kai-mlir/              # MLIR dialect + physics lowering
│   ├── physics-dialect/       # IR: calculate_vfe, world_model_physics
│   ├── kai-standalone/        # Standalone binary (no opencode)
│   └── kai-fusion/            # Main synthesis engine (see src/ map above)
├── core/                      # Python-side euler_core, shared logic
├── sources/                   # Vendor mirrors (ollama, llama.cpp) — not in lean push
├── llvm-project -> /home/l/llvm-project  # Symlink, ignored in lean push
├── enwiki-...xml.bz2          # 25GB dump, ignored in lean push
├── LICENSE (MIT)
├── README.md (this file)
├── MANUAL.md (full sequential manual, 20 sections)
└── .gitignore (lean, ~500MB push)
```

Private (ignored) in lean push: `api_keys`, all `*.md` except LICENSE/README/MANUAL, `adjustments.txt`, `agents_chat.txt`, `bthink.txt`, `conscious_loop_log20260406`, `coulombs_law_examples.txt`, `distillation_pipeline`, `domains`, `dynamic_system_data.txt`, `euler_agent.modelfile`, `GPT_chat`, `intent`, `mdlist`, `quanta_chat`, `quantum_entanglement_cryptography_examples.txt`, `requirements.txt`, `sess_mdlist`, `stdout`, `synthetic_examples.txt`, `wordlist.txt`, `files`, `.opencode/` history, wiki memories, session logs.

---

## Quick Start

**Prerequisites:** Rust ≥1.80, `ollama` with at least one GGUF in `~/.ollama/models` (e.g. `qwen2.5:7b`, `llama3`), Linux.

```bash
# 1. Clone lean repo
git clone https://github.com/<you>/axiom_horizon.git
cd axiom_horizon/rust

# 2. Build release (lto, opt 3)
cargo build --release

# 3. Verify (312 tests)
cargo test

# 4. Launch via opencode (uses local GGUF, no download)
kai launch opencode
# or
cargo run -p kai-fusion -- launch opencode
```

On first launch: loads `KaiFusionConfig` default (dense RoPE-MHA) + GGUF for Qwen2.5/Llama3 + VFE wrapper + REPL. MoE/MLA/vision/linear are feature-gated behind same config.

---

## Build & Verify

This is the standing instruction (reproduced from private plan):

- **Always** use `Read`/`Edit`/`Grep`/`Glob`/`Write` for files — never shell `cat`/`echo`/`sed`.
- **Build:** `cargo build --release` in `/rust`
- **Test:** `cargo test` — **all 312 tests must pass with 0 warnings** before claiming done
- **Debug loop is the mind:** failure → hypothesis → fix → re-run until green
- **Python side:** `python3` directly via Bash for `axiom_mcp_server`, `*.py` tools, GGUF inspection; `cpp` toolchain for llvm-project
- **Self-improvement:** `kai darwin evolve` / `source-evolve` (real patches + physics-param evolution), `kai bench`, `kai run` (physics-wired inference)

Workspace profile:
```toml
[profile.release]
lto = true
opt-level = 3
panic = "abort"
```

---

## CLI — kai

```bash
kai launch opencode      # load config+weights → run loop (REPL)
kai darwin evolve        # recursive self-improvement: source patches + physics params
kai darwin source-evolve # patch-level evolution
kai bench                # compare generations (perplexity, physics agreement, world-graph)
kai run                  # physics-wired inference (g_ij → temp, VFE → tau)
```

`kai launch opencode` is the entry. It resolves against `127.0.0.1:11434` (local ollama) first — **local-first gating** — before any remote fallback.

---

## Configuration

One config expresses all modes (dense ↔ MoE ↔ MLA ↔ Linear).

**Tensor lib:** `ndarray` (f32), consistent with `kai-core`/`physics-dialect`. Heavy GEMM may later bind `llama.cpp` via FFI; Phase C prototype stays pure-Rust.

**Weight format:** GGUF primary (from `~/.ollama/models`); safetensors secondary.

**Locked decisions before Phase C:**
- Vertical slice first: dense RoPE-MHA + VFE + GGUF loader + REPL, then MoE → MLA → vision → linear
- `gemma4` maps to Gemma-2 arch until Gemma-4 ships

---

## Weight Bootstrap & Assimilation

- **Bootstrap:** map each teacher's GGUF tensors → our layers (Qwen2.5→dense, DeepSeek→MoE+MLA, Nomic→embed head, LLaVA/Moondream→vision tower)
- **Assimilate:** distillation from 10 teachers (soft targets) + self-supervised next-token on local corpus + **VFE regulation**
- Weights are **reused then evolved**, never copied verbatim. Matrices are new (bootstrapped + trained).

---

## Roadmap — 7 Phases

Distilled from private `AGI_PLAN.md` (public summary):

| Phase | Goal | Done? |
|---|---|---|
| **0 Foundation** | eve.py, Euler Clock, bracket-line, StateDB, sandbox | ✅ |
| **1 Perception & Memory** | Vector memory, g_ij metric tensor, bracket as state vector | 🔧 |
| **2 Self-Improvement Engine** | Genetic operators (AST), fitness fn, parallel pool, self-play | 🔧 |
| **3 World Interaction** | ESP32-S3 bridge, headless browser, Docker sandbox, FS agent | ⏳ |
| **4 Meta-Cognition** | Self-awareness monitor, goal decomposition, uncertainty, value learning | ⏳ |
| **5 Recursive Self-Improvement** | Meta-learn improver, distributed compute, Wikipedia ingestion, self-mod | ⏳ |
| **6 Integration & Deployment** | 24/7 systemd, multimodal (llava/whisper), multi-agent, alignment watchdog | ⏳ |
| **7 Measured Convergence** | tau = useful_work/wall_time, fitness fixed-point, actuator/API layer | ⏳ |

**Metrics:** tau growth, fitness/gen, response time <2s, memory retrieval >90%, self-consistency >95%, improvement >1%/day, divergence = 0.

---

## Android App

**Kai-Android: Adaptive On-Device LLM Hub** — complete, pending Play Store submission:

- **Native GGUF inference** on ARM64 (Rust core via JNI): Qwen2, Llama 3, Gemma
- **VFE meter** (surprise / epistemic uncertainty / curiosity) with adaptive temperature
- **Novelty surface**: per-idea distance readout with geodesic map
- Fully offline; optional sync with the desktop engine (`kai darwin` state)

Measured: 1.2B Q4_K_M at 200–400 ms/token on POCO X3 (8 GB); 3-model + real-time VFE on current flagships. Also tested on Samsung A36 (8 GB / 256 GB).

---

## File Map for Documentation

Future `docs/` should expand from this README:

```
docs/
├── architecture.md      # Expand §1-§3 + config.rs + engine/model
├── vfe.md               # Expand §6 + kai/vfe.rs + physics-dialect IR
├── physics.md           # g_ij, curvature, tau, auto-assimilation
├── training.md          # bootstrap + assimilation loop + darwin
├── api.md               # kai CLI + ingress/egress actuator contract
├── android.md           # NDK build, JNI, model delivery, benchmarks
└── phases.md            # 7 phases with DoD checklists
```

Source of truth for each: `rust/kai-fusion/src/*.rs`, `rust/kai-core/`, `rust/physics-dialect/`, `rust/kai-mlir/`.

---

## Contributing & Self-Improvement Loop

1. Read `MANUAL.md` (20 sections) before coding — it distills `AGI_PLAN.md` + `Kai_FUSION_ARCHITECTURE.md`.
3. When failing: hypothesis → patch → re-run (the debug loop *is* the mind)
3. When failing: hypothesis → patch → re-run (the debug loop *is* the mind)
4. For evolution: `kai darwin evolve`, compare via `kai bench`, infer via `kai run`

No `cat`/`echo`/`sed` for files. Python via `python3`, cpp via toolchain, skills via `kai`.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 AxiomTree — Kai Fusion Contributors.

---

*Fitness is measured. Convergence is empirical. The actuator layer is the interface.* — Kai core
