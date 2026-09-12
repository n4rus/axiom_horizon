# Variational Hunt — Pre-registered Plan v0.1

**Target:** Variational Omni — 6 assets (Core OLP Vault, Treasury, Settlement Pool Factory, etc.)
**Bounty:** $100k max | $10k critical floor | $5k/$1k High/Medium | PoC + KYC required
**Repo:** `strata-hunt` clone (local-only, Immunefi-gated scope) + `contracts` harness
**Date:** 2026-09-12
**Status:** PRE-REGISTERED — scope/method frozen, logs append-only
**Hash:** git commit (this file) — timestamp

## 1. Scope (Immunefi live)

Core OLP Vault, Protocol Treasury, Settlement Pool Factory + 3 more (see Immunefi /scope). Isolated pools (User <> OLP), bilateral, non-netted. Focus: smart contracts only (web/app out of scope).

## 2. Invariants to fuzz

- **Vault Solvency:** `vault.totalAssets` conserved across deposit/withdraw/liquidate.
- **Pool Isolation:** funds in pool A never affect pool B.
- **No Freeze Without Cause:** no legitimate deposit/withdraw reverts on consistent state.
- **Liquidation Correctness:** under-margin → liquidatable, over-margin → not.
- **Fee/Treasury:** 20% spread to treasury exact, no dust/miscount.

## 3. Method (fuzz harness, free compute)

1. Slither + Aderyn (already clean on Strata, reuse pattern).
2. Foundry harness per asset: `test/hunt/Variational*.t.sol` — stateful, 200k runs/seed.
3. Invariants above as `assertLe`/`assertEq` + `vm.expectRevert` for negative cases.
4. Overnight loop `run-overnight.sh` rotating seeds (like Strata 6k+ seeds, 0 fails prior).

## 4. Payout Target

Critical = $10k min (theft/insolvency/permanent freeze). One critical = month funded. High = $5k. Below $3k not submitted.

## 5. Evidence

- `strata-hunt/overnight.log` — rotating seeds, `exit=0` per seed
- `strata-hunt/contracts/test/hunt/Variational*.t.sol` — harnesses (reproducible: `forge test --match-contract HuntVariational`)
- PoC gate: no submission without locally-executed Foundry PoC (1-asset floor invalid per Immunefi rules — enforce >=10 assets seeded)

## 6. No-Edit Rule

This plan frozen at v0.1 hash. Scope/method changes = `VARIATIONAL_v0.2.md` with diff.

---
*Same timestamp discipline as TRIAL.md — paper trail before payout.*
