# Gitcoin Grants — kai-fusion / kai-mlir

**Applicant:** l (AxiomTree) — São José dos Pinhais, Paraná, Brazil
**Project:** kai-fusion + kai-mlir
**Requested amount:** $12,000 (over 6 months, deployed via Grants Stack direct grant)
**Application date:** 2026-09-05
**Repo:** github.com/n4rus/axiom_horizon (to be set public upon submission)

---

## TL;DR

kai-fusion is a Rust-based autonomous optimization engine that uses Darwinian evolution over attractor dynamics and Variational Free Energy (VFE) to self-improve the structure of compute. It runs locally, ships with 312 green tests, and is designed to be auditable, reproducible, and distributable via libp2p + IPFS. We are requesting $12k in direct funding to allow a self-taught developer with chronic back pain and cluster headaches to build a permanent dev machine, ship the public benchmark harness, libp2p sync, and the first upstream MLIR dialect RFC in 6 months.

This is not a moonshot pitch. The repo is real, the tests are green, the artifacts are inspectable, and the roadmap is a flat list of things that get done one at a time.

---

## Why kai-fusion matters for public goods

Most "AI that improves itself" demos in 2026 require a hosted LLM API. They are:

- **Non-reproducible** — every run depends on a vendor's model version
- **Non-auditable** — the state is opaque and proprietary
- **Non-distributable** — they live in one vendor's cloud
- **Expensive** — the API bill grows linearly with usage

kai-fusion is none of those. It is:

- **Reproducible** — `cargo test --release` produces the same green
- **Auditable** — every attractor state is checkpointed, Darwinian generations content-addressed as of Month 3, every LLVM pass diff visible
- **Distributable** — libp2p sync (Month 2), IPFS persistence (Month 3)
- **Cheap to run** — local CPU/GPU, no API bill

This makes it a genuine public good: any researcher, anywhere, can run it on commodity hardware, fork it, and ship a derivative. Gitcoin's mandate is to fund exactly this kind of infrastructure.

---

## What we ship with the grant

| Month | Deliverable                                                | Status today   |
|-------|------------------------------------------------------------|----------------|
| 1     | `axiom_horizon` published, v0.1.0 tagged, README + MANUAL  | Almost done    |
| 2     | libp2p sync protocol for attractor state                   | Designed       |
| 3     | IPFS persistence of evolutionary snapshots                 | Designed       |
| 4     | Public benchmark harness (Docker, 5 workloads)             | Prototype only |
| 5     | MLIR dialect upstream RFC                                  | First pass     |
| 6     | Sustainability docs + final report                             | In this grant  |

---

## Budget ($12,000 over 6 months)

| Item                       | Amount (USD) | Why                                                |
|----------------------------|--------------|----------------------------------------------------|
| Living expenses            | 4,800        | $800/month — São José dos Pinhais, Brazil          |
| Medical expenses           | 1,200        | $200/month — copays SUS does not cover             |
| Rebuild (one-time)         | 2,827        | Z790 + i7-14700F + 64GB + 5060 Ti 16GB (BRT retail)|
| Home power uplift          |   150        | ~$25/mo extra on Copel residential                 |
| Cloud relay (tiny)         |    60        | Fly.io relay for libp2p + public URL               |
| KYC + legal                |   500        | USDT/USDC onboarding, MEI setup                    |
| Reserve / overrun          | 2,463        | BRL drift, health overruns, month-7 bridge         |
| **Total**                  | **12,000**   |                                                    |

---

## Why this applicant

- 8 years of independent study, self-taught
- Evidence first: 312 green tests, 0 warnings, 25/10000 fuzz green, 0 reentrancy, 0 balance bug
- Bounty payout received (Base 1.05 USDC) — proof of shipping on chain
- Works with chronic back pain and cluster headaches under doctor-prescribed pacing (see medical budget line)
- Aim: advance autonomous optimization as human knowledge and ship durable artifacts. Not a financial optimization, a human-knowledge optimization.

---

## Repo

`github.com/n4rus/axiom_horizon`

Key paths:
- `rust/kai-fusion/` — Darwinian attractor engine
- `rust/kai-fusion/src/daemon.rs` — 20-cycle goal decomposer
- `rust/kai-mlir/` — LLVM dialect
- `tools/kai_physics_bench.py` — physics-grounded optimization benchmark
- `grants/` — this application and the Protocol Labs parallel application

---

## License

MIT. No change with funding. No lock-in.

---

## How to fund

Direct grant via Gitcoin Grants Stack:
- Create a campaign at https://grants.gitcoin.co
- Set initial matching pool = 0 (bootstrap from self if needed)
- Receive matching once the round opens, or accept direct contributions

Preferred payout: USDT on Binance (BEP20 preferred; funder's cheapest supported network acceptable).

---

*Submitted alongside the Protocol Labs IPFS Tooling Grant application (`grants/PROTOCOL_LABS_GRANT.md`). Both applications share the same repo, roadmap, and budget — the applicant is open to either funder.*
