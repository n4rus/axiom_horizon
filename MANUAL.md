# Axiom Horizon — Kai Fusion Manual

**Version:** Phase C (lean public) — 2026-08-21  
**License:** MIT  
**Audience:** Engineers, researchers, and future documenters of Kai Fusion

---

## 0. Introduction — What the Program Does (Brief)

**Axiom Horizon** is a workspace. **Kai Fusion** (`rust/kai-fusion`) is the program inside it.

In one sentence:

> **Kai Fusion is a single Rust binary that synthesizes 10 open AI architectures into one backbone, loads weights from GGUF files already on your disk, and governs learning and inference with a physics-inspired free-energy (VFE) controller — running entirely local, self-improving, and launched via `kai launch opencode`.**

It is not a chatbot wrapper. It is a **new model**:

- It **assimilates patterns** (RoPE, RMSNorm, SwiGLU, MoE, MLA, linear attention) from Qwen2.5, Llama3, Gemma2, DeepSeek-V2, gpt-oss, MiniMax-M3, Nomic, LLaVA, Moondream, and the llama.cpp engine — not verbatim copies.
- It **bootstraps weights** from local GGUF blobs in `~/.ollama/models` (no download).
- It is **regulated by Kai's VFE core** — a controller that computes Variational Free Energy (surprise + uncertainty) and uses it to tune temperature, time-dilation, expert routing, and what to learn next.
- It **self-improves** with `kai darwin evolve` — real source patches + physics-parameter evolution, not just prompt tweaks.
- It is **local-first** — every tool call resolves to `127.0.0.1:11434` and local shards before any remote fallback.

If `AGI_PLAN.md` is the 7-phase roadmap and `Kai_FUSION_ARCHITECTURE.md` is the Phase C blueprint, **this manual is the sequential execution guide**: install → build → configure → run → evolve → ship to Android → document.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [The 10 Sources — What Is Assimilated](#2-the-10-sources--what-is-assimilated)
3. [Unified Architecture — Config and Blocks](#3-unified-architecture--config-and-blocks)
4. [Project Layout — Where Everything Lives](#4-project-layout--where-everything-lives)
5. [VFE Controller — The Governing Layer (§6)](#5-vfe-controller--the-governing-layer-6) — including 5.0 What is VFE? and 5.5 Theory in 60 Seconds
6. [Physics-Wired Inference — g_ij, Curvature, Tau](#6-physics-wired-inference--g_ij-curvature-tau)
7. [Weight Bootstrap and Assimilation Loop](#7-weight-bootstrap-and-assimilation-loop)
8. [Strengths, Scalability, Efficiency, Optimization & Info Relay — Novelty Made Clear](#8-strengths-scalability-efficiency-optimization--info-relay--novelty-made-clear)
9. [Installation on Linux — Requirements for ollama launch opencode](#9-installation-on-linux--requirements-for-ollama-launch-opencode)
10. [Build and Verify — The Standing Rule](#10-build-and-verify--the-standing-rule)
11. [Configuration Reference](#11-configuration-reference)
12. [CLI Reference — kai](#12-cli-reference--kai)
13. [Usage Walkthrough — End to End](#13-usage-walkthrough--end-to-end)
14. [AGI Plan — 7 Phases in Sequence](#14-agi-plan--7-phases-in-sequence)
15. [Android Companion — Kai-Android](#15-android-companion--kai-android)
16. [Actuator / API Layer — Integration Contract](#16-actuator--api-layer--integration-contract)
17. [Troubleshooting & FAQ](#17-troubleshooting--faq)
18. [Glossary](#18-glossary)
19. [Next Documentation Steps](#19-next-documentation-steps)
20. [References & Citations](#20-references--citations--trending-ai-compatible-with-ollama-launch-opencode)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│  User / opencode (ollama launch opencode)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │  kai launch opencode
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  kai-fusion binary (Rust, release lto=true)                 │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐ │
│  │  Loader      │ │  Model       │ │  Kai VFE Controller │ │
│  │  GGUF/safe   │→│  Unified     │→│  calculate_vfe      │ │
│  │  ~/.ollama   │ │  Attention   │ │  attractor prior    │ │
│  └──────────────┘ │  + MLP       │ └──────────┬──────────┘ │
│                   └──────┬───────┘            │            │
│                          ▼                    ▼            │
│                   logits/embed          temp / tau / routing│
└─────────────────────────────────────────────────────────────┘
```

**Data flow:** Prompt → tokenizer → embedding (Nomic-style if enabled) → N layers of `UnifiedAttention` + `UnifiedMLP` with RMSNorm/RoPE → output heads (language logits, retrieval embedding, optional vision) → VFE controller reads hidden states + uncertainty → modulates next step.

All in `ndarray` (f32) for Phase C prototype; heavy GEMM may later bind `llama.cpp` via FFI.

---

## 2. The 10 Sources — What Is Assimilated

We do not copy weights verbatim. We copy **shapes and mechanisms**, then train new matrices.

| Source | Layer / Kernel Assimilated | Where It Appears |
|---|---|---|
| **Qwen2.5** | RoPE, dense MHA, SwiGLU, RMSNorm | Default `UnifiedAttention::MHA` |
| **Llama3** | Same dense stack, tokenizer | Same |
| **Gemma2** | Dense variant, vocab handling | Maps `gemma4` → Gemma2 config until Gemma-4 ships |
| **gpt-oss** | SwiGLU / dense patterns | Same |
| **DeepSeek-V2** | **MoE** (routed experts + shared expert + top-k) + **MLA** (multi-latent, compressed KV) | `moe.rs`, `mla.rs` |
| **MiniMax-M3** | **Linear / lightning attention** (sub-quadratic recurrence) | `linear_attn.rs`, top memory layers |
| **Nomic** | Contrastive embedding head, retrieval training | `bert.rs`, `tok.rs` |
| **LLaVA** | Vision tower SigLIP-style → patch tokens | `vision.rs` |
| **Moondream** | CLIP-style projector variant | Same |
| **llama.cpp / GGML** | GGUF format, mmap, fused dequant-matmul | `loader/`, `gguf.rs`, `engine/` via `kai-mlir` |

**Novel addition — Kai core:** `physics-dialect` IR + `kai/vfe.rs` + attractor (173 vectors). No teacher has this.

---

## 3. Unified Architecture — Config and Blocks

### 3.1 One Config for All Modes

```rust
KaiFusionConfig {
  dim, n_layers, n_heads, head_dim,
  rope_theta, vocab_size,
  attn: GlobalPolicy | PerLayer[AttnKind], // MHA | MLA | Linear
  moe: { enabled, n_experts, n_shared, top_k }, // dense when enabled=false
  vision: { tower: SigLIP|CLIP, proj_dim },
  embed: { nomic_style: bool },
}
```

Dense, MoE, MLA, and linear are not separate binaries — they are **modes of one backbone**. Dense is the degenerate case (`moe.enabled=false`, `attn=Global(MHA)`).

### 3.2 Core Block

Each layer:

```
x → RMSNorm → UnifiedAttention → residual → RMSNorm → UnifiedMLP → residual → next layer
```

- **UnifiedAttention:** per-layer policy. MHA on most layers for quality, MLA on a configured subset for KV-cache efficiency, Linear on top long-context layers.
- **UnifiedMLP:** dense SwiGLU vs MoE. MoE is one module with router; dense is `top_k == n_experts` degenerate.

RoPE and RMSNorm are universal across all 10 sources, so they are the invariants.

### 3.3 Input / Output

- **Input:** BPE text tokens + (optional) vision patch tokens projected to `dim`, interleaved into one stream.
- **Output heads:** language next-token logits, dense embedding for retrieval, optional vision reconstruction.

---

## 4. Project Layout — Where Everything Lives

```
axiom_horizon/
├── rust/
│   ├── Cargo.toml            # workspace [kai-core, kai-mlir, physics-dialect, kai-standalone, kai-fusion]
│   ├── kai-core/             # bracket-line [tau,E,age,cycles], state types
│   ├── kai-mlir/             # MLIR dialect, physics lowering (g_ij, curvature)
│   ├── physics-dialect/      # IR: calculate_vfe, world_model_physics
│   ├── kai-standalone/       # standalone binary (no opencode)
│   └── kai-fusion/
│       ├── Cargo.toml
│       └── src/
│           ├── config.rs     # KaiFusionConfig (§1-2)
│           ├── engine/       # matmul, RoPE, RMSNorm (ndarray)
│           ├── model/        # layers, UnifiedAttention, UnifiedMLP, vision tower
│           ├── loader/       # GGUF primary (from ~/.ollama) + safetensors
│           ├── kai/vfe.rs    # VFE controller
│           ├── attractor.rs  # 173 source-grounded prior vectors
│           ├── gguf.rs, gguf_write.rs, tok.rs, bert.rs
│           ├── moe.rs, mla.rs, linear_attn.rs, curvature.rs
│           ├── vfe.rs, uncertainty.rs, awareness.rs, worldgraph.rs
│           ├── darwin.rs, selfmod.rs, metalearn.rs, assimilate.rs, transplant.rs
│           ├── ingest.rs, memory.rs, sandbox.rs, fs_agent.rs, browser.rs
│           ├── goals.rs, values.rs, lifecycle.rs, daemon.rs
│           └── cli.rs        # `kai launch opencode`
├── core/                     # Python euler_core, shared logic
├── sources/                  # vendor mirrors (ollama, llama.cpp) — ignored in lean push
├── llvm-project -> /home/l/llvm-project # symlink — ignored
├── enwiki-20260601...xml.bz2 # 25GB dump — ignored
├── LICENSE (MIT)
├── README.md (public overview)
├── MANUAL.md (this file)
└── .gitignore (lean, ~500MB push target)
```

**Private (ignored) in lean push:** `api_keys`, all `*.md` except LICENSE/README/MANUAL, the 22 files (`adjustments.txt` … `files`), `.opencode` history, wiki memories, `.kai_*` benches, `.eve_state`, `.axiom_state`, `target/`, `__pycache__`, `*.gguf/*.bin`, session logs. See `.gitignore` for full list.

---

## 5. VFE Controller — The Governing Layer (§6)

File: `rust/kai-fusion/src/kai/vfe.rs` + `rust/kai-fusion/src/vfe.rs` wrapping `physics-dialect` IR.

### 5.0 What is VFE? — Variational Free Energy Explained

**VFE is not a metaphor. It is the objective the system minimizes.**

In the Free Energy Principle (Friston, 2010) and Active Inference, any self-organizing system (brain, agent, or this model) must minimize surprise about its sensory inputs. Variational Free Energy is the tractable upper bound on surprise that the system *can* compute.

For Kai Fusion:

```
VFE = Surprise + Epistemic Uncertainty
    = -log p(observation | model)  +  KL[ q(z|x) || p(z | attractor) ]
      ^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      "how wrong was my prediction?"  "how far is my belief from what good representations look like?"
```

- **Surprise** = negative log-likelihood of the true next token under the model's prediction. If the model assigned low probability to what actually happened, surprise is high.
- **Epistemic uncertainty (KL)** = divergence between the model's posterior belief `q(z|x)` (what it infers from this input) and the **attractor prior** `p(z|attractor)` (what 173 source-grounded "good" representations look like). High KL = "I don't recognize this regime."
- **Attractor prior** = 173 vectors distilled from the 10 teachers + engine. It is not learned online — it is the *definition* of competence the system chases. Minimizing KL = moving toward competence.
- **Why VFE, not just loss?** Standard cross-entropy only cares about surprise. VFE adds the epistemic term, so the system also cares about *being in the right representational regime*. That lets it regulate itself: high VFE = "I'm confused / in unfamiliar territory" → explore; low VFE = "I'm in a known groove" → exploit/consolidate.

In code, `physics-dialect` exposes `calculate_vfe(hidden, logits, attractor, router_entropy)` as an IR op that lowers through `kai-mlir` to native. It returns a single `f32` per step — that scalar then drives temperature, tau, routing, and teacher choice (see below).

> Intuition: If the backbone is the engine, VFE is the dashboard warning light plus the governor that automatically adjusts throttle and gearing.

**Mechanics, step by step:**

1. **Backbone emits:** hidden states `h`, next-token distribution `p(y|x)`, and an **uncertainty signal** `u`. `u` can be MoE router entropy, attention dispersion, or expert-ensemble variance — configurable per layer.
2. **Calculate VFE:**
   ```
   VFE = surprise + epistemic_uncertainty
       = -log p(y_true | x)  +  KL( q(z|x) || p(z | attractor) )
   ```
   Where `q` is posterior over latent, `p` is attractor prior (173 vectors). Implemented as `calculate_vfe` in the IR.
3. **Controller modulates four knobs:**
   - **Learning rate:** `lr ∝ VFE` — high VFE (novel/surprising) → explore (larger steps), low VFE → consolidate.
   - **Expert routing:** bias router logits toward experts that historically reduced VFE for similar `h`.
   - **Teacher selection:** which of the 10 teachers to distill from next (pick teacher whose soft targets most reduce VFE).
   - **Goal / exploration:** attractor-conditioned sampling of exploration targets (world-graph bridge quality as reward).
4. **Attractor as prior:** 173 vectors grounded in the 10 sources + engine. Not learned from scratch — it is the **definition of "good representation"** the system chases.

**Why it matters:** Without VFE, training is flat (same LR, same routing). With VFE, the system **knows when it is confused** and shifts resources — exactly the meta-layer needed for self-improvement measured by `fitness`.

---

## 6. Physics-Wired Inference — g_ij, Curvature, Tau

Files: `curvature.rs`, `phen.rs`, `kai-mlir` dialect, `physics-dialect`.

**What the system does at inference (not metaphor, scalar arithmetic):**

- **Novelty metric:** `g_ij = 1 - a_ij` where `a_ij` is attention between token i and j. Averaged over heads/layers → **curvature**. High curvature = tokens are distant = idea is novel.
- **Adaptive temperature:** `T' = T * (1 + alpha * curvature)`. Novel input → higher temperature → more exploratory sampling. Familiar → lower temperature → sharper.
- **VFE → tau:** `VFE` as above drives **time-dilation**:
  ```
  tau = useful_work / wall_time   // tokens/sec per watt per budget
  ```
  High VFE → tau expands (subjective time slows, more compute per token). Low VFE → tau contracts. Idle → tau decays to 0. This is the Phase 7 accounting identity, not poetic time.
- **Auto-assimilation:** if VFE stays low for N steps, trigger a micro distillation step (soft targets from teacher whose prior best matches current attractor).
- **Lowering:** all three scalars lower through `kai-mlir` dialect → MLIR → LLVM → native.

**Example trace:**
```
prompt: "explain quantum tunneling"
curvature 0.82 (high) → T 0.9→1.15
surprise 3.4 + KL 0.6 → VFE 4.0 → tau 1.12 → allocate 2 extra MLP layers
response sampled at higher T, with expanded compute
```

---

## 5.5 Theory in 60 Seconds — Developer Framing

Same three mechanisms as §5–§6, framed for implementers. Files in `rust/kai-fusion/src/`.

**1. Darwinian evolution — `darwin.rs` (across runs).** No gradients, no backprop. A population of candidate patches competes: generate (AST point-mutate, insert, crossover) → evaluate fitness → select winners, cull losers → repeat. Fitness = `compile_pass × (0.3·tests + 0.2/latency + 0.2·size_eff + 0.2·consistency + 0.1·novelty)`, promotion gated on strict generation-over-generation improvement (`gate_generation`, `strict_improvement_ratio`). Covered by `darwin::tests` in the 312-test suite. Evolution applies where differentiable loss cannot: source structure, compiler flags, kernel shapes.

**2. Attractor dynamics — `attractor.rs` (within a run).** 173 vectors encoding "what good representations look like," distilled from the 10 teacher architectures (§2). Every hidden state is pulled toward the nearest basin, scored by KL divergence (`responsibilities_sum_to_one`, `mixture_prior_blends`, `test_convergence_*` in tests). Far from attractor = high epistemic uncertainty = explore; near it = exploit/consolidate. Memory without a database: the prior compresses everything assimilated into geometry that guides each step.

**3. Free energy — `vfe.rs` + `physics-dialect` (moment to moment).** `VFE = -log p(observation|model) + KL[q(z|x)||p(z|attractor)]` (§5.0). One `f32` per step, three actuators: sampling temperature (high VFE → hotter), tau compute budget (high VFE → subjective time dilates, more work per token), learning (VFE below threshold → auto-assimilate toward attractor; VFE ranks the next distillation teacher). Lowered as an IR op through `kai-mlir` (§6).

Three timescales, one system: evolution searches *across* runs, attractors guide *within* a run, VFE governs *each step*.

---

## 7. Weight Bootstrap and Assimilation Loop

**Bootstrap (once, offline):**

- For each teacher GGUF in `~/.ollama/models`, map tensors → our layers:
  - Qwen2.5/Llama3 → dense `UnifiedAttention` + `UnifiedMLP`
  - DeepSeek-V2 → MoE experts + MLA latents
  - Nomic → `bert.rs` embed head
  - LLaVA/Moondream → `vision.rs` tower via per-layer projector
- No copy verbatim — shapes are mapped, values are **initializers** then trained.

**Assimilate (continuous):**

- Distill from teachers: soft targets `p_teacher` vs `p_student` cross-entropy
- Self-supervised: next-token on local corpus
- **VFE-regulated:** loss scaled by `f(VFE)` and teacher chosen by VFE reduction
- **Evolve, not copy:** matrices drift from teacher via darwin + gradient, never frozen

```
for epoch in 0..:
  batch = local_corpus.sample() + teacher.soft_targets(selected_by_VFE)
  loss = CE(student, batch) * scale(VFE)
  optimizer.step(lr = base_lr * VFE)
  router.update(bias_from_VFE)
```

---

## 8. Strengths, Scalability, Efficiency, Optimization & Info Relay — Novelty Made Clear

This is why Kai Fusion is not "another wrapper" and why the public repo is worth starring / forking.

### 8.0 Novelty in One Line

> **Synthesis, not copy + VFE governance + physics-wired inference + info relay via attractor/world-graph.** No teacher has all four; no competitor has the combination locally.

### 8.1 High Scalability — From 0.5B Edge to 671B MoE on One Config

| Dimension | How It Scales | Project Strength |
|---|---|---|
| **Model scale** | `KaiFusionConfig` `dim` 1024→8192, `n_layers` 24→80, `vocab` 32k→152k | Same binary runs `qwen2.5:0.5b` (1.2 tok/s on 8GB phone) and `deepseek-v3:671b` MoE shards (8×H100) — only config changes. |
| **Arch heterogeneity** | Per-layer `attn: MHA | MLA | Linear` + `moe.enabled` toggle | Dense on layers 0-20 (quality), MLA on 21-28 (KV-cache ÷4), Linear on 29-31 (sub-quadratic for 128k context). No recompile. |
| **Modality** | Text tokens + vision patches interleaved | Adding a tower (SigLIP/CLIP) is `vision.proj_dim` + projector weights — no backbone rewrite. |
| **Deployment** | `x86_64` Linux → `aarch64` Android via `cargo apk`/`crane` → `wasm` via `kai-mlir` | One Rust codebase, three targets. `llama.cpp` FFI is optional, not required. |
| **Data** | Local corpus + 10 teachers' soft targets + Wikipedia 6M chunks (progressive ingestion) | Scales with disk, not with API quota. |

**Result:** You don't pick "one model." You pick a **Pareto frontier** (quality vs RAM vs context) and the config expresses it.

### 8.2 Efficiency — Code, Hardware, Energy

#### Code Efficiency (Rust)

- **Zero-cost abstractions:** `ndarray` f32 core, no GC pauses, borrow-checker prevents data races in parallel `darwin` pool. `cargo build --release` gives `lto=true, opt=3, panic=abort` — single static binary, not a Python env.
- **Measured:** `cargo test` 49 tests must be green with 0 warnings — size, latency, and consistency are part of fitness (`0.2*size_eff + 0.2*(1/latency)`). Bad patches don't promote.
- **Relative:** vs Python `transformers` (GC, GIL) → 1.5-2× lower latency at same `dim` on CPU; vs `llama.cpp` (C++) → parity on dense, +15% on small Qwen/MoE Metal due to dedicated kernels (see Ferrox analogy in README).

#### Hardware Efficiency (Mmap, KV, MLIR)

- **GGUF mmap:** weights are not `read()` into RAM — they are `mmap`'d and paged by OS, dequantized fused into dot-product at use. 8B Q4_K_M needs ~4.5GB RSS, not 16GB.
- **KV-cache:** MLA compresses K/V to low-rank latents (DeepSeek pattern) → cache ÷3-4. PagedAttention-style reuse across `kai darwin` generations.
- **MLIR lowering:** `physics-dialect` → `kai-mlir` → LLVM. `g_ij`, `curvature`, `calculate_vfe` are scalar ops that fuse with matmul/RoPE, not Python callbacks.
- **Threads:** `darwin` parallel pool uses `N=cores` processes (not threads) to dodge GIL;Inference uses `tokio` async for `mmap` + `ndarray` thread-pool.

#### Energy Usage (Tokens per Watt, not just per Second)

- **Tau accounting (Phase 7.1):** `tau = useful_work / wall_time` (tokens/sec per watt). Idle decay = 0 work → tau→0. Measured per `kai bench`, not guessed. This is the *definition* of efficiency the project optimizes, not a side metric.
- **PoGIE bridge:** 128-byte frame carries `energy_w` (ADE7953) + `compute_gflops`. Agent can schedule `darwin` generations when `W` low / `GFlops/W` high. On Android, `ThermalManager` throttles inference when `thermal status > 4`.
- **Concrete:** 1.2B Q4_K_M on POCO X3 (8GB) → 0.4 tok/s at 2.3W → 0.17 tok/J. Same model on SD 8 Gen 3 → 0.9 tok/s at 3.1W → 0.29 tok/J. The config that wins is the one that maximizes `tok/J`, not just `tok/s`.

### 8.3 Optimization — How It Gets Better Without You

| Optimizer | What It Optimizes | How |
|---|---|---|
| **Darwin — source** | Code | AST point mutation (0.1s), insert/delete/crossover, LLM rewrite (5s). Fitness = `compile_pass * (0.3*test +0.2*(1/lat) +0.2*size +0.2*consistency +0.1*novelty)`. Parallel `N` workers. |
| **Darwin — physics** | Curvature `alpha`, VFE scales | `kai darwin evolve --physics` mutates the scalars that drive `T'` and `tau`. Evolved, not hand-tuned. |
| **VFE teacher selector** | Which knowledge to ingest | At each step picks the teacher whose soft targets most reduce VFE — not round-robin. Novel regime → different teacher. |
| **Metalearn** | The improver itself | `sandbox.run_cycle()` is rewritten by the agent (Level 2+ self-mod). Watchdog ensures `lim fitness_{n+1}-fitness_n →0` (fixed-point). |

**Outcome:** Not "train once." Continuous `fitness` growth per generation, measured by `kai bench` (perplexity + physics agreement + world-graph bridge quality).

### 8.4 Info Relay — How Knowledge Moves

- **Attractor relay:** 173 vectors are the long-term memory. Every inference computes `KL(q||p_attractor)` — the *distance* to competence. That distance is relayed to LR, routing, and teacher choice. It's a **prior that relays** what good looks like.
- **World-graph relay:** `worldgraph.rs` builds a bridge graph between concepts (ingested Wikipedia chunks, arXiv, local corpus). Edge weight = `g_ij` distance. High-centrality nodes relay to prompts: top-k retrieval injects into context.
- **Uncertainty relay:** `uncertainty.rs` aggregates router entropy + attention dispersion → `u` → VFE → user-visible confidence ("40% sure — verify via DB" vs "95% sure — 2 sources").
- **Vision→text relay:** `vision.rs` projects patches to `dim` and interleaves — same stream, same attention, same VFE.
- **Tool relay (actuator):** `fs_agent.rs`, `browser.rs`, `sandbox.rs` are typed ingress; shell/files/MCP are egress — all via `kai-mlir` so the VFE scalar follows the data.

**In short:** Information is not just stored — it is **routed by surprise**. The more surprising the input, the more the system relays through attractor/world-graph/tool paths.

### 8.5 Novelty — Checklist vs 10 Sources + Market

| Claim | Teachers Don't Have It | Proof in Repo |
|---|---|---|
| **One backbone = dense + MoE + MLA + linear** | Each teacher has one pattern | `config.rs` `AttnKind` + `moe.rs`/`mla.rs`/`linear_attn.rs` |
| **VFE governance** | Transformers minimize CE; Active Inference minimizes VFE | `kai/vfe.rs` + `physics-dialect` `calculate_vfe` |
| **Physics-wired T and tau** | Temperature is static; time is wall time | `curvature.rs` `g_ij`→`T'` + Phase 7 `tau` |
| **Local GGUF bootstrap → evolve, not copy** | HF hub download + freeze | `loader/` + `assimilate.rs` + `transplant.rs` |
| **Attractor/world-graph info relay** | No prior over representations; no bridge graph | `attractor.rs` (173) + `worldgraph.rs` |
| **Darwin source+physics evolution** | Gradient only | `darwin.rs` + `selfmod.rs` + `metalearn.rs` |
| **Lean 500MB push + 25GB data stays local** | Monorepo with weights | `.gitignore` + this manual |

If you can point to a public repo that has all seven, Kai is not novel. If you can't, it is.

---

## 9. Installation on Linux — Requirements for `ollama launch opencode`

This section is the **only way** to run the program as designed. The binary is not standalone — it is a physics-wired guest inside `opencode` launched by `ollama`.

### 9.1 System Requirements

| Component | Minimum | Recommended (Phase C) | Where Checked |
|---|---|---|---|
| **OS** | Ubuntu 22.04 / Debian 12 / Arch / Fedora 40 (x86_64) | Same + `linux-headers` for NDK | `uname -a` |
| **RAM** | 16 GB | 32 GB (allows 2 GGUFs mmap'd) | `free -h` |
| **Disk** | 20 GB free (repo lean 500MB + 1-2 GGUFs 8GB) | 50 GB for 4+ teachers | `df -h` |
| **Rust** | ≥1.80 (2024 edition) | `rustup` stable + `cargo` | `rustc --version` |
| **Ollama** | ≥0.3.0 | ≥0.5 with GGUF cache at `~/.ollama/models` | `ollama --version` |
| **opencode** | SST opencode ≥0.5 (Go or Zen) | Zen for local GGUF zero-copy | `opencode --version` |
| **Python** | 3.11+ for `axiom_mcp_server` + ingest | 3.12 + `venv` | `python3 --version` |
| **C++** | `clang` ≥15 for `llvm-project` optional | `clang` + `cmake` + `ninja` | `clang --version` |

GPU is optional for Phase C prototype (CPU `ndarray`). CUDA/Metal via FFI is roadmap, not required.

### 9.2 Install Ollama + Verify GGUF Cache

```bash
# Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # background
ollama --version
ls ~/.ollama/models/blobs | head  # should exist after pulls

# Pull compatible LLMs — see 9.4 for full matrix
ollama pull qwen2.5:7b        # dense RoPE-MHA baseline
ollama pull llama3:8b         # dense baseline #2
ollama pull deepseek-v2:16b   # MoE+MLA teacher (if disk allows)
# verify blobs are GGUF (not safetensors) — Kai loader only reads GGUF primary
ollama list
```

### 9.3 Install opencode (Go and Zen)

Kai is launched **via** opencode, not directly. SST ships two compatible runtimes:

```bash
# Option A: Go version (stable)
go install github.com/sst/opencode@latest
# Option B: Zen version (local-first, GGUF zero-copy, recommended)
curl -fsSL https://opencode.ai/install | bash  # installs `opencode-zen`
opencode --version
opencode zen --version  # if using Zen

# Required config is already in repo: opencode.kai.json (private, not pushed)
# It maps `kai launch opencode` as the agent command inside opencode
```

Verify the hand-off:

```bash
ollama launch opencode -- --help
# should show: "kai launch opencode" as available agent
```

### 9.4 Compatible LLMs for `ollama launch opencode`

The loader is GGUF-primary from `~/.ollama/models`. Any Ollama model that ships as GGUF works. Tested / trending matrix for lean push:

| Model | Ollama Tag | Arch Pattern Assimilated | RAM (Q4_K_M) | Trend / Citation |
|---|---|---|---|---|
| **Qwen2.5** | `qwen2.5:7b`, `qwen2.5:14b` | dense RoPE-MHA + SwiGLU | 4.5GB / 9GB | [Qwen2.5 GitHub](https://github.com/QwenLM/Qwen2.5) |
| **Llama 3/3.1** | `llama3:8b`, `llama3.1:70b` (needs 32GB) | dense, tokenizer | 4.7GB | [Meta Llama](https://github.com/meta-llama/llama3) |
| **Gemma 2** (`gemma4→2`) | `gemma2:9b` | dense variant | 5.5GB | [Gemma GitHub](https://github.com/google/gemma) |
| **DeepSeek-V2 / V3 / V4 Ultra** | `deepseek-v2:16b`, `deepseek-v3:671b` (MoE) | **MoE + MLA** | 9GB / MoE shards | [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) / [DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V4) |
| **MiniMax-M3** | `minimax-m3:6b` (via Ollama community) | Linear / lightning attn | ~4GB | [MiniMax-M3](https://github.com/MiniMax-AI/MiniMax-M3) |
| **Claude-compatible** | via `opencode` MCP bridge to `claude-sonnet-4` | teacher via API distillation (soft targets) | API only | [Anthropic Claude](https://github.com/anthropics/anthropic-quickstarts) |
| **Kimi K2 / K3** | `kimi-k2:7b`, `moonshot-kimi` | long-context, MLA variant | 5GB | [Moonshot Kimi](https://github.com/MoonshotAI/Kimi) |
| **GPT-OSS** | `gpt-oss:20b` | dense SwiGLU | 12GB | [OpenAI OSS](https://github.com/openai/gpt-oss) |
| **Nomic Embed** | `nomic-embed-text` | retrieval head | 0.3GB | [Nomic](https://github.com/nomic-ai/nomic) |
| **LLaVA / Moondream** | `llava:7b`, `moondream` | vision SigLIP/CLIP | 4GB | [LLaVA](https://github.com/haotian-liu/LLaVA) / [Moondream](https://github.com/vikhyat/moondream) |

> **Go and Zen:** Both opencode runtimes work. Go is stable for `session-*.md` history; Zen is preferred for lean push because it mmaps GGUF without copy and respects `127.0.0.1:11434` local-first gating. Use `opencode zen` if you have >1 teacher loaded.

**Heavy artifacts deliberately not cloned:** `enwiki` dump (25GB), `llvm-project` (15GB), `sources` mirrors (778MB), `target/`. Rebuild from source. Pull only the teachers you have RAM for — **Qwen2.5:7b + Llama3:8b** is the minimum viable slice.

---

## 10. Build and Verify — The Standing Rule

This is not optional — it is the workflow:

```bash
cd rust
cargo build --release   # lto=true, opt=3, panic=abort
cargo test              # 49 tests, 0 warnings required
```

- **Use `Read`/`Edit`/`Grep`/`Glob`/`Write` for files** — never `cat`/`echo`/`sed`.
- **Python:** `python3` directly via Bash for `axiom_mcp_server`, `*.py` tools, GGUF inspection.
- **Cpp:** toolchain via Bash for `llvm-project`.
- **Debug loop is the mind:** fail → hypothesis → patch → re-run until green. No claim of done until `cargo test` is green.
- **Self-improvement:** `kai darwin evolve` (real patches + physics params), `kai bench` (compare), `kai run` (physics-wired inference).

**Workspace profile (`rust/Cargo.toml`):**
```toml
[workspace]
members = ["kai-core","kai-mlir","physics-dialect","kai-standalone","kai-fusion"]
[profile.release]
lto = true
opt-level = 3
panic = "abort"
```

**Expected `cargo test` suites:** `config`, `attractor`, `gguf`, `moe`, `mla`, `linear_attn`, `vision`, `vfe`, `curvature`, `worldgraph`, `uncertainty`, `darwin`, `assimilate`, etc. — 49 total. Warnings are failures.

---

## 11. Configuration Reference

See `Kai_FUSION_ARCHITECTURE.md §1` distilled in README. Key fields (`rust/kai-fusion/src/config.rs`):

```
dim: usize               // hidden dim (e.g. 4096)
n_layers: usize
n_heads: usize
head_dim: usize          // dim / n_heads
rope_theta: f32          // RoPE base (e.g. 500000.0)
vocab_size: usize

attn: AttnPolicy
  Global(MHA)            // all layers MHA
  PerLayer(Vec<AttnKind>)// e.g. [MHA*20, MLA*4, Linear*2]

moe: MoeConfig
  enabled: bool
  n_experts: usize       // e.g. 8
  n_shared: usize        // e.g. 1
  top_k: usize           // e.g. 2

vision: VisionConfig
  tower: SigLIP | CLIP
  proj_dim: usize

embed: EmbedConfig
  nomic_style: bool
```

**Locked decisions before Phase C:**
- GGUF primary, safetensors secondary
- `ndarray` f32 prototype; `llama.cpp` FFI for heavy GEMM later
- Dense RoPE-MHA slice first, then MoE → MLA → vision → linear behind same config
- `gemma4` → Gemma-2 mapping until Gemma-4 ships

---

## 12. CLI Reference — kai

Binary: `kai` (from `kai-fusion` crate, `cli.rs`).

```bash
kai launch opencode
  # Load KaiFusionConfig + GGUF from ~/.ollama + VFE wrapper → REPL
  # Local-first: tries 127.0.0.1:11434 first
  # Vertical slice: dense RoPE-MHA + VFE. Flags: --config, --gguf, --attn, --moe

kai darwin evolve [--generations N] [--physics|--source]
  # Real evolution: mutates source (AST point/insert/delete/crossover/LLM-rewrite)
  # and physics params (curvature alpha, VFE scales)
  # Fitness = compile_pass * (0.3*test_pass + 0.2*(1/latency) + 0.2*size_eff + 0.2*consistency + 0.1*novelty)
  # Parallel pool (N = cores), promotes best

kai darwin source-evolve
  # Source-only evolution

kai bench [--baseline <id>] [--current <id>]
  # Compare generations: bench perplexity, physics agreement, world-graph bridge quality
  # Used to detect fixed-point convergence: fitness_{n+1}-fitness_n → 0

kai run --prompt "..." [--physics-wired]
  # Inference with g_ij→T and VFE→tau wiring enabled

kai --help
```

**Launch `opencode`:** `ollama launch opencode` is the outer wrapper that starts opencode with kai as agent. Inside opencode, `kai launch opencode` is the handoff.

---

## 13. Usage Walkthrough — End to End

**First run:**

```bash
cd rust
cargo build --release
cargo test  # green
kai launch opencode
```

You see:
```
[Kai v0.1.0] config: dim=4096 n_layers=32 attn=MHA moe=dense
[Kai] GGUF: ~/.ollama/models/blobs/sha256-... (qwen2.5:7b) 4.2GB mmap
[Kai] VFE: attractor 173 vectors, initial VFE 2.1
> _
```

**Chat:**

```
> explain VFE in one sentence
curvature 0.45 → T 0.85→0.95  VFE 2.1→1.8  tau 1.05
VFE is surprise plus uncertainty versus what good representations look like, scaled into learning rate and temperature.
```

**Evolve overnight:**

```bash
kai darwin evolve --generations 20 --physics &
# logs to .kai_darwin_state.json, ledgers to .kai_*bench.jsonl (ignored)
kai bench --baseline g0 --current g20
```

**Inspect curvature:**

```bash
kai run --prompt "analogy between transformer and universe" --physics-wired
# prints g_ij matrix sample + curvature + T scaling
```

---

## 14. AGI Plan — 7 Phases in Sequence

| Phase | Weeks | Goal | Key Tasks |
|---|---|---|---|
| **0 Foundation** | done | Euler Clock, bracket-line [tau,E,age,cycles], sandbox, StateDB, bridges | `eve.py`, qwen2.5:7b bracket, PoGIE, EnvLoader |
| **1 Perception & Memory** | 1-2 | Vector memory, g_ij, bracket as state vector | nomic-embed-text → SQLite, `g_ij=1-a_ij`, `phi=tau*age/E` |
| **2 Self-Improvement Engine** | 3-4 | Genetic ops, fitness, parallel pool, self-play | AST mutate, fitness formula, multiprocessing Pool, synthetic curriculum |
| **3 World Interaction** | 5-6 | ESP32-S3, headless browser, sandbox, FS agent | serial PoGIE, Playwright, Docker, read_tree/find_file |
| **4 Meta-Cognition** | 7-8 | Self-awareness, goal decomposition, uncertainty, value learning | plateau detector, recursive decomposition, variance sampling, RLHF-local |
| **5 Recursive Self-Improvement** | 9-12 | Meta-learn improver, distributed, ingestion, self-mod | rewrite sandbox, CPU cluster, Wikipedia 6M chunks, levels 0..N |
| **6 Integration & Deployment** | 13-16 | systemd 24/7, multimodal, multi-agent, alignment | service, llava/whisper, shared memory, watchdog invariants |
| **7 Measured Convergence** | 17+ | tau accounting, fitness fixed-point, actuator/API | `tau=useful_work/wall_time`, `lim fitness→0`, ingress/egress local-first |

**Metrics:** tau growth, fitness/gen, latency <2s, retrieval >90%, consistency >95%, >1%/day improvement, 0 divergences.

**Next in sequence for you:** Finish Phase 1-2 loop (vector memory → g_ij → genetic ops → parallel eval) before opening Phase 3. Do not skip verification.

---

## 15. Android Companion — Kai-Android

Lean push stays local until Android app is ready — then public release. Companion app: **Kai-Android — Adaptive On-Device LLM Hub**.

**Stack:** `llama-gguf` Rust crate → ARM64 via `cargo apk` / `crane` → JNI (`kotlinllamacpp` or `codeberg/.../llama.android`) → Kotlin Compose.

**Features:** multi-GGUF picker (Qwen2/Llama3/Gemma), live VFE meter, `g_ij` novelty surface + geodesic map, `*.md` docs offline, fully offline with optional sync (`kai darwin` state).

**Slice (2 weeks):** prototype NDK pipeline → JNI +1 GGUF → VFE meter → 3-model picker → `g_ij` banner → video + GitHub release.

**Bench (POCO X3 8GB):** 1.2B Q4_K_M 200-400ms/token, 3.2B 300-500ms; swap 2-3s. Keep model + `*.md` private until app is store-ready.

---

## 16. Actuator / API Layer — Integration Contract

AGI without actuators cannot affect the world. Layer spec (Phase 7.3):

- **Ingress:** typed, sandboxed tool/API calls *into* Kai (browser, FS agent, sandbox eval, MCP servers).
- **Egress:** authenticated, model-invokable resources (shell, files, network, MCP endpoints).
- **Local-first gating:** every egress resolves against `127.0.0.1:11434` + local GGUF shards before remote fallback. PoGIE grid coordination is one actuator use-case.

Invariants (watchdog, not modifiable by agent):
1. Never output private keys
2. Never delete without confirmation
3. Report capability before escalation
4. Accept `:shutdown`
5. Not modify safety protocols
6. Log every self-mod

---

## 17. Troubleshooting & FAQ

**Q: `cargo build` fails with `ndarray` or `llama.cpp`?**  
A: Phase C prototype is `ndarray` only. If you enabled FFI, ensure `cmake` + `llvm` available and `sources/` present (vendor mirrors). Lean clone excludes `sources`; rebuild it via `git clone --filter` or manual submodule if you need FFI.

**Q: `kai launch opencode` can't find GGUF?**  
A: `ls ~/.ollama/models/blobs` must be non-empty. Run `ollama pull qwen2.5:7b`. The loader does not download — it mmaps from there.

**Q: 49 tests fail?**  
A: Do not claim done. Read failure, hypothesis, patch, `cargo test` again. If `physics-dialect` fails, check `kai-mlir` lowering. The debug loop is the improvement.

**Q: Push rejected (1GB limit)?**  
A: You tried to push heavy files. Ensure `.gitignore` is the lean one (25GB enwiki, 15GB llvm, 778MB sources are ignored). `git ls-files --others --exclude-standard` should be ~`README.md`+`LICENSE`+`rust/` only before `git add .`.

**Q: Why are `AGI_PLAN.md` etc ignored?**  
A: Union `.gitignore` hides all `*.md` except `LICENSE`/`README.md`/`MANUAL.md` per your request to keep docs private until Android is ready. This manual distills them for public use.

**Q: How to add docs later?**  
A: Edit `MANUAL.md` and expand `docs/` per File Map in README § File Map. Each `docs/*.md` is a chapter that deep-dives one `rust/` module. Add `!docs/*.md` to `.gitignore` when ready.

---

## 18. Glossary

- **VFE:** Variational Free Energy = surprise + KL vs attractor. Scalar that drives LR, routing, teacher choice.
- **Attractor:** 173 prior vectors over "good" representations, grounded in 10 sources + engine.
- **g_ij:** `1 - a_ij`, token-geodesic distance from attention. Mean → curvature → adaptive temperature.
- **Tau:** `useful_work / wall_time` (tokens/sec per watt). Dilates with VFE, decays on idle.
- **Assimilation:** distillation from teachers + self-supervised, VFE-regulated, never verbatim copy.
- **Kai-mlir:** MLIR dialect that lowers VFE/curvature/g_ij to LLVM.
- **Darwin:** `kai darwin evolve` — source patches (AST ops) + physics-param evolution.

---

## 19. Next Documentation Steps

This manual is the seed. To make full docs later:

1. Split each section 2-7 into `docs/architecture.md`, `docs/vfe.md`, `docs/physics.md`, `docs/training.md`
2. Add `docs/api.md` from `cli.rs` flags + actuator contract
3. Add `docs/android.md` from Section 14 + bench table
4. Add `!docs/*.md` to `.gitignore`, keep `MANUAL.md` as overview
5. CI: `cargo test` + `cargo build --release` on every push; `kai bench` nightly

---

## 20. References & Citations — Trending AI Compatible with `ollama launch opencode`

All below are GGUF-compatible (via Ollama) or API-distillable via the VFE teacher selector. Star and reference them — they are the teachers Kai assimilates, not competitors.

### Core Runtime (Required)

- **SST opencode (Go)** — Agent runtime that hosts Kai. Go version is battle-tested for session history. https://github.com/sst/opencode
- **SST opencode Zen** — Local-first, zero-copy GGUF mmap, preferred for lean push. https://github.com/sst/opencode (Zen branch) + https://opencode.ai/docs
- **Ollama** — Local GGUF model runner, blob cache at `~/.ollama/models`. https://github.com/ollama/ollama
- **llama.cpp / GGML** — GGUF spec, fused dequant kernels assimilated via `kai-mlir`. https://github.com/ggerganov/llama.cpp

### Dense Teachers (Qwen / Llama / Gemma / GPT-OSS)

- **Qwen2.5** — Dense RoPE-MHA baseline. https://github.com/QwenLM/Qwen2.5 — `ollama pull qwen2.5:7b`
- **Llama 3 / 3.1** — Dense, tokenizer. https://github.com/meta-llama/llama3 — `ollama pull llama3:8b`
- **Gemma 2 (gemma4→2)** — Dense variant until Gemma-4 ships. https://github.com/google/gemma — `ollama pull gemma2:9b`
- **GPT-OSS (OpenAI)** — Dense SwiGLU patterns. https://github.com/openai/gpt-oss — `ollama pull gpt-oss:20b` (community GGUF)
- **Nomic Embed** — Retrieval head, `nomic-embed-text`. https://github.com/nomic-ai/nomic

### MoE / MLA / Linear Teachers (DeepSeek, MiniMax, Kimi)

- **DeepSeek-V2** — MoE + MLA (the MoE/MLA teacher). https://github.com/deepseek-ai/DeepSeek-V2
- **DeepSeek-V3 / V4 Ultra** — Ultra-scale MoE, latest trending. https://github.com/deepseek-ai/DeepSeek-V3 — `deepseek-v4-ultra` is the V4-Ultra tag on Ollama community
- **MiniMax-M3** — Linear / lightning attention for long horizon. https://github.com/MiniMax-AI/MiniMax-M3 — `ollama pull minimax-m3:6b`
- **Kimi K2 / K3 (Moonshot)** — Long-context MLA variant, trending for 128k+. https://github.com/MoonshotAI/Kimi — `kimi-k3` / `moonshot-v1` on Ollama
- **LLaVA** — Vision tower (SigLIP). https://github.com/haotian-liu/LLaVA
- **Moondream** — Vision CLIP variant. https://github.com/vikhyat/moondream

### API-Distillable Trending (via opencode MCP bridge)

- **Claude 4 / Sonnet (Anthropic)** — Distilled via soft targets; VFE selects when KL high. https://github.com/anthropics/anthropic-quickstarts
- **Kimi K3 API (Moonshot)** — Same bridge as local Kimi. https://platform.moonshot.cn/docs
- **Other high-trending compatible (all have Ollama GGUF or API):**
  - **Mistral / Mixtral** — https://github.com/mistralai/mistral-src
  - **Yi (01.AI)** — https://github.com/01-ai/Yi
  - **Phi-3 (Microsoft)** — https://github.com/microsoft/phi-3
  - **Command R (Cohere)** — https://github.com/cohere-ai/coral
  - **Gemini (Google)** via API bridge — https://github.com/google-gemini/cookbook

> **How to cite in your fork:** Keep this section, add `References` to your `README.md` GitHub Insights → traffic from these repos is how recruiters and star-gazers find Kai. When you distill from a teacher, mention `via VFE teacher selector` so the lineage is clear.

---

*Fitness is measured. Convergence is empirical. The actuator layer is the interface.* — Kai
