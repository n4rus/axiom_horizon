# Protocol Labs Research Grant — kai-fusion / kai-mlir

**Applicant:** l (AxiomTree) — São José dos Pinhais, Paraná, Brazil
**Project:** kai-fusion + kai-mlir
**Requested amount:** $12,000 USDT (over 6 months)
**Application date:** 2026-09-01
**Repo:** github.com/AxiomTree/axiom_horizon (to be set public upon submission)

---

## 1. Project Summary (200 words)

**kai-fusion** is an autonomous optimization engine written in Rust that combines Darwinian evolution, attractor dynamics, and Variational Free Energy (VFE) minimization to self-improve the structure of computation itself. The current `kai-mlir` LLVM dialect fuses VFE and curvature operations at the compiler level, producing optimized kernels without manual tuning.

The system runs continuous self-play cycles: each generation proposes a mutated strategy, evaluates it on a battery of physics-grounded queries (conservation laws, electrostatics, entropy), and promotes the highest-fitness strategies into the next generation. The result is a code-generating engine that adapts to the shape of the problem rather than the shape of the programmer's assumptions.

This grant funds six months of full-time work to ship:
- A reproducible `kai-fusion` benchmark harness on public infrastructure
- A first-class `kai-mlir` LLVM dialect with documented passes
- Integration with libp2p so that distributed nodes can share attractor state and co-evolve across machines, with the IPFS content-addressed store providing the persistence layer

The intended audience is **Rust/MLIR developers and research engineers** who need autonomous optimization for compute kernels, data-center scheduling, or scientific simulation, without paying for cloud LLM API calls.

---

## 2. Problem Statement (300 words)

Modern compute infrastructure has two structural problems that compound each other:

**Problem A — Optimization is static.** LLVM, GCC, and ML frameworks generate code based on heuristics written years ago. A kernel that performs well on yesterday's hardware underperforms on today's GPU. A scheduler that worked for yesterday's batch sizes thrashes on today's. There is no closed loop between "what the workload actually looks like at runtime" and "how the code that runs it is generated."

**Problem B — Self-improving AI is a cloud-only product.** Every Darwinian / evolutionary / "agent that improves itself" demo in 2026 either requires a hosted LLM API (vendor lock-in, monthly bill, no reproducibility) or runs only on a single box with no networking. There is no open, reproducible, distributed version that researchers can run on commodity hardware and audit the full state of.

**kai-fusion solves both.** It runs entirely on local CPU/GPU. Its state is small (<1GB), inspectable, and checkpointable. The `kai-mlir` dialect makes the optimization auditable at the IR level — you can diff two versions of the same kernel and see exactly what the attractor engine decided. And the design is naturally distributable: each node evolves locally and can share attractor snapshots with peers, with IPFS providing content-addressed persistence so that any node can recover the full evolutionary history from any other node.

For data-center workloads (cooling optimization, GPU kernel selection, memory pressure tuning), this is the missing layer. For scientific computing, it is a way to discover kernels that hand-tuned codegen misses. For AGI research, it is one of the few open systems where "the model improved itself and here is the proof" is reproducible.

---

## 3. Solution + 6-Month Roadmap (500 words)

### What exists today (working, verified, in repo)

- `kai-fusion` core engine — 312 tests GREEN, 0 warnings across `cargo build --release` (Rust 1.97)
- `kai-mlir` — LLVM dialect for fused VFE/curvature operations; first-pass lowering verified
- `kai darwin self-play` — full Darwinian evolve loop; runs to completion, outputs best_fitness at end
- 27 / 10,000 fuzz runs passing on the autonomous-bounty smart contract (Solidity + forge) with **0 reentrancy, 0 balance bug** per slither
- 79,907 Wikipedia entries absorbed into `.kai_wiki_memory.*.json` as semantic attractor memory
- `kai_physics_bench --fuse` nightly harness (3 local teacher models → fused queries → JSONL)

### What this grant funds (Month-by-Month)

**Month 1 — Harden and publish the core**
- Lock the public API of `kai-fusion` and `kai-mlir`
- Write integration tests that any reviewer can `cargo test --release` and see green
- Publish `axiom_horizon` to GitHub with a strong README (proof table included) and MANUAL
- Tag v0.1.0

**Month 2 — libp2p integration**
- Wrap attractor state in a `libp2p` request/response protocol
- Nodes can fetch a peer's `attractor_state.bin` by content hash
- Document the protocol in `PROTOCOL.md`

**Month 3 — IPFS persistence**
- Store attractor snapshots on IPFS via local daemon + public gateway fallback
- Every Darwinian generation is content-addressed and recoverable
- Build a CLI: `kai darwin sync --peer /ip4/.../tcp/...`

**Month 4 — Public benchmark harness**
- Containerized benchmark (Docker + docker-compose) any reviewer can run
- Compare `kai-fusion`-optimized kernel vs hand-tuned baseline on 5 standard workloads
- Publish numbers in `BENCHMARKS.md`

**Month 5 — kai-mlir upstream attempt**
- Draft an RFC for the dialect against the official MLIR project
- Get review from at least one MLIR maintainer
- If accepted, submit the first pass; if not, keep the dialect in-tree as a vendored submodule

**Month 6 — Maintenance + handoff**
- Write `MAINTENANCE.md` for what happens after the grant ends
- Apply for Round 2 of the grant ($25k–$50k) to fund a 2-engineer team
- Publish a final report to the Protocol Labs community

### What this grant does NOT fund
- Hosting fees for closed-source cloud services
- Token launches, NFT mints, or any speculative financial instrument
- Marketing or paid promotion

---

## 4. Budget Breakdown ($12,000 USDT over 6 months)

| Category              | Amount (USDT) | Justification                                                                 |
|-----------------------|---------------|-------------------------------------------------------------------------------|
| Living expenses       | 4,800         | $800/month × 6 — rent, food, utilities in São José dos Pinhais, Brazil        |
| Medical expenses      | 1,200         | $200/month × 6 — copays and medications SUS does not fully cover             |
| Dev machine (one-time)| 2,827         | Permanent rebuild, priced at Brazilian retail (see parts table below)         |
| Home power uplift     |   150         | ~$25/mo extra for compute rig on Copel Paraná residential rates              |
| Cloud relay (tiny)    |    60         | Fly.io / Hetzner relay for libp2p peer discovery + public benchmark URL      |
| KYC + legal           |   500         | International wire / USDT onboarding fees, MEI setup if needed               |
| Reserve / overrun     | 2,463         | BRL price drift, health overruns, month-7 bridge                             |
| **Total**             | **12,000**    |                                                                               |

### Parts list (permanent dev machine, BRT retail Sept 2026, USD @5.5)

| Part | Spec | R$ | $ |
|---|---|---|---|
| Board | MSI Z790 ATX (2× PCIe: compute + display) | 1,600 | 291 |
| CPU | i7-14700F (20c, parallel builds + benches) | 2,000 | 364 |
| RAM | DDR5 64GB 2×32 (dual-channel full speed, 2 slots free) | 4,600 | 836 |
| GPU (compute) | RTX 5060 Ti 16GB (fuse set ≈ 15GB, fits) | 4,600 | 836 |
| GPU (display, kept) | GTX 1660S — drives monitors, keeps 16GB free | 0 | 0 |
| PSU | 850W | 1,000 | 182 |
| Case | ATX | 750 | 136 |
| Storage | Reuse existing NVMe (add from reserve if needed) | 0 | 0 |
| **Total** | | **15,550** | **2,827** |

**Why 5060 Ti 16GB and not 5070 Ti / 5080:** all three are 16GB — identical VRAM ceiling, same max model. The premium buys speed only, not capability. On R$0.75/kWh home power, overnight batch runs make speed the least valuable metric. Savings stay in reserve. The 1660S keeps displays off the compute card so the full 16GB serves inference.

**Note on compensation:** $800/month for full-time work is below any international dev rate. It is a living-expenses grant, not a salary. The applicant is funding the rest of the gap from the existing bounty earnings (1.05 USDC Base) and savings. The grant removes the survival pressure so the work can be done at all.

**Why USDT, not BRL:** Brazilian recipients face 3–8% spread on USD→BRL conversions plus IOF tax. USDT (or USDC) received on Base or Polygon allows the applicant to convert to BRL at market rate with no remittance intermediary. Preference: **USDC on Base** (cheaper gas, no Polygon spam) or **USDT on Polygon** as fallback.

---

## 5. Team + Past Work

### Applicant

- **Handle:** l (AxiomTree)
- **Location:** São José dos Pinhais, Paraná, Brazil
- **Status:** Self-taught, no formal CS degree, chronic back pain + chronic cluster headache (work-impairing but not work-preventing; doctor-prescribed pacing)
- **Languages:** Portuguese (native), English (professional working)
- **Operating since:** 2018 (8 years of independent study and self-directed work after a delivery job ended)

### Track record (verifiable on chain and in git)

- **Bounty payout on Base:** 1.05 USDC at `0x8f36...0eb5` (PR #1, n4rus/agent-bounties)
- **Repo:** github.com/AxiomTree/axiom_horizon — 312 tests GREEN, 25/10000 fuzz GREEN, 0 reentrancy, 0 balance bug
- **Artifacts:**
  - `kai-fusion` — Darwinian attractor engine, self-play runs to completion; `best_fitness: 140.66` recorded Aug 2026 (re-run to get fresh number)
  - `kai-mlir` — LLVM dialect for fused VFE/curvature
  - `kai_physics_bench --fuse` — autonomous optimization loop
  - 79,907 entries absorbed into long-term semantic memory

### Why this applicant delivers despite chronic conditions

The applicant has been working through daily chronic back pain and episodic cluster headaches (often multi-day) for the entire 8-year self-study period. The motivation is not financial optimization but a double aim: (1) **advance the science of autonomous optimization and self-improving systems** as a contribution to human knowledge, and (2) **produce artifacts that are useful today and durable for the future** — code, benchmarks, papers, and runnable demos that outlive any single grant cycle.

This is not a person who disappears when conditions get hard. It is a person who has been getting harder work done, in worse conditions, for longer than most funded teams have existed. The grant removes the survival ceiling so the work can compound.

### Past work — what was published and what is in repo

| Artifact                                         | Status        | Evidence                       |
|--------------------------------------------------|---------------|--------------------------------|
| `cargo test --release` — 312 GREEN, 0 warnings | In repo       | CI log on `axiom_horizon`      |
| `forge test --fuzz-runs 10000` — 25 GREEN; 0 failed; 1 skipped | In repo       | Foundry output archived        |
| `slither .` — 0 reentrancy, 0 balance bug        | In repo       | Slither JSON archived          |
| `kai darwin self-play` — runs to completion (140.66 on Aug 2026 run) | In repo       | log output        |
| 79,907 wiki entries → semantic memory            | In repo       | 96 `.kai_wiki_memory.*.json`   |
| `kai-mlir` LLVM dialect                          | First pass    | In repo, IR-level tests green  |
| Bounty payout                                    | On-chain      | Base 0x8f36...0eb5             |

---

## 6. Maintenance Plan (after 6 months)

If the grant succeeds, the maintenance model is:

1. **Sustainability via Round 2** — apply for the next tier of PL funding ($25k–$50k) to cover a 2-engineer team (1 Rust core, 1 MLIR/devtools)
2. **Bounty revenue** — continue shipping small bounties to keep cash flow while the team grows
3. **Optional Gitcoin / Optimism RPGF** — if `kai-fusion` becomes adopted as public infrastructure, retroactive public-goods funding becomes a non-dilutive option
4. **Independence clause** — the project remains MIT-licensed open source regardless of future funding. No lock-in to any single grantor.

The codebase does not require a company to maintain. A single contributor with a 5-year-old laptop can read it, run it, and ship a patch. That is by design.

---

## 7. Open Source License

MIT — the project has been MIT from day one. The grant does not change the license. Any work produced with grant funds is contributed back to the public MIT-licensed repository.

---

## 8. Why Protocol Labs specifically

Protocol Labs funds the **infrastructure layer of the open web**: IPFS, Filecoin, libp2p. `kai-fusion` is exactly that — an open, reproducible, distributed optimization layer that:

- Uses **Rust** (the language PL championed)
- Uses **libp2p** for the peer-to-peer attractor-sync protocol (Month 2 of the roadmap)
- Uses **IPFS** for content-addressed persistence of evolutionary state (Month 3 of the roadmap)
- Produces **artifacts useful to IPFS / Filecoin users** (faster codecs, better crypto kernels, smarter content-routing heuristics)

The integration is not theoretical — it is in the budget, on the timeline, and in the deliverable list.

---

## 9. Submission checklist (for the applicant)

- [ ] Repo `AxiomTree/axiom_horizon` set to public
- [ ] README.md, MANUAL.md, this `PROTOCOL_LABS_GRANT.md` all in repo under `/grants/`
- [ ] 2-minute Loom video recorded: `cargo test --release` green + `kai darwin self-play` running + completion output on screen
- [ ] KYC information ready (CPF, proof of address, self-employed / autônomo status)
- [ ] Wallet address for USDT/USDC on Base or Polygon confirmed
- [ ] Submit to: https://github.com/protocol/research-grants (or current PL grants intake)

---

*This application is being prepared in the `grants/` directory of the project repo so that the application, evidence, and code are reviewed together. The applicant is ready to respond to any review within 48 hours.*
