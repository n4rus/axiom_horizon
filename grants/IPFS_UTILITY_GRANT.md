# IPFS Utility Grant — Distributed Attractor Sync for kai-fusion (libp2p + IPFS)

**Applicant:** l (AxiomTree) — São José dos Pinhais, Paraná, Brazil
**Project:** kai-fusion attractor sync (libp2p protocol + IPFS persistence)
**Requested amount:** $8,000 (over 3 months)
**Application date:** 2026-09-05
**Repo:** github.com/n4rus/axiom_horizon (public, MIT)
**Fits:** IPFS Implementations / Data Utilities grants ($5k–25k, 1–3 months, one building block done well)

---

## 1. Problem (why IPFS needs this)

Evolutionary / self-improving compute systems keep their state in one box or one vendor's cloud. There is no open protocol for **peers to share evolving program state** — snapshots that any node can fetch by content hash, verify, and continue evolving from. IPFS has content addressing; libp2p has the transport. What is missing is a worked, shipped example of **evolutionary state as content-addressed data**: small snapshots (<1MB), content-hash addressable, synced peer-to-peer, with a CLI and a benchmark proving two machines co-evolve better than one.

kai-fusion (Rust, MIT, 312 tests green, 0 warnings) is the host system: a Darwinian optimization engine whose attractor state is already small, checkpointed, and serializable. This grant funds the missing layer — the open sync protocol plus IPFS persistence — as reusable building blocks, documented so any IPFS developer can reuse the pattern.

## 2. Approach (3 months, one building block)

**Month 1 — libp2p sync protocol.**
Wrap attractor snapshots in a libp2p request/response protocol: peers advertise snapshot CIDs, fetch by content hash, verify integrity on receipt. Document wire format in `PROTOCOL.md`. Deliverable: two local nodes syncing state, tested in repo.

**Month 2 — IPFS persistence.**
Persist every Darwinian generation snapshot to IPFS via local daemon, with public-gateway fallback for retrieval. Generations become CIDs; full evolutionary history recoverable from any node holding the chain. CLI: `kai darwin sync --peer /ip4/.../tcp/...`. Deliverable: snapshot → CID → fetch → resume evolution, demonstrated end-to-end.

**Month 3 — Distributed proof + report.**
Docker harness running 2+ nodes co-evolving on 5 workloads, vs single-node baseline. Publish numbers in `BENCHMARKS.md` and a public write-up (personal blog + IPFS forum) suitable for CID Congress-style presentation. Deliverable: reproducible benchmark anyone can `docker compose up`.

Out of scope (unfunded, on the public roadmap): MLIR upstream RFC, public API lock, v0.1.0 release mechanics.

## 3. User feedback and adoption plan

- **Primary users:** kai-fusion operators running multi-machine evolution (exists today on LAN; this grant makes it peer-to-peer).
- **Secondary users:** any IPFS/libp2p developer needing the *pattern* — small-state sync over libp2p with IPFS backup is directly reusable for CRDTs, model checkpoints, game state, sensor fusion.
- **Feedback loop:** monthly progress notes on the IPFS forum; benchmark harness doubles as the adoption vehicle (if reviewers can run it, users can adopt it).
- **Maintenance:** 10%-style upkeep continues solo regardless (see sustainability note in repo grants); the protocol doc is the durable artifact.

## 4. Qualifications (prior open-source work)

- **Repo:** github.com/n4rus/axiom_horizon — Rust workspace, `cargo test --release` → 312 passed, 0 warnings; `cargo build --release` clean.
- **Smart contracts:** 25/10,000 fuzz runs green, Slither 0 reentrancy / 0 balance bug; 1.05 USDC bounty paid on Base (PR #1, `n4rus/agent-bounties`).
- **Benchmarks:** `kai_physics_bench --fuse` harness running (liveness verified, consistency 0.926); 12b-tier fuse run records PHYSICS WINS +0.46 — see `wiki_2_done.md` in repo.
- 8 years independent study, self-taught; works with chronic conditions under doctor-prescribed pacing (medical line in budget so treatment never competes with dev time).

## 5. Budget ($8,000 over 3 months)

| Item | Amount (USD) | Why |
|---|---|---|
| Living expenses | 2,400 | $800/month × 3 — São José dos Pinhais, Brazil |
| Medical expenses | 600 | $200/month × 3 — copays SUS does not cover |
| Dev machine share | 2,827 | One-time permanent rebuild (Z790 + i7-14700F + 64GB + RTX 5060 Ti 16GB, BRT retail; full parts in repo Appendix A) — serves this grant and all follow-on work |
| Home power uplift | 75 | ~$25/mo extra on Copel residential |
| Cloud relay (tiny) | 30 | Fly.io relay for libp2p peer discovery during testing |
| KYC + legal | 250 | USDT/USDC onboarding |
| Reserve | 818 | BRL drift, health overruns |
| **Total** | **8,000** | |

$800/month is living expenses, not a salary. The one-time machine cost is shared across this and all future work — it is charged once, here.

Preferred payout: USDT on Binance or ETH (BEP20 preferred; funder's cheapest supported network acceptable).

## 6. License

MIT — all work produced under this grant is contributed back to the public MIT-licensed repository. No IP transfer, no lock-in.

---

*Companion to the repo's 6-month roadmap (`grants/PROTOCOL_LABS_GRANT.md`): this proposal funds the IPFS-relevant slice (sync + persistence + distributed proof) as a standalone 3-month unit. The research-grants portal program is paused, so this utility-grant path is the live door for the same work.*
