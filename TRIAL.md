# Axiom AGI Trial — Pre-registered Protocol v0.1

**Author:** Axiom (n4rus) — Independent research programme
**Date:** 2026-09-12
**Status:** PRE-REGISTERED — no edits after timestamp except via versioned addendum
**Timestamp:** git commit hash (this file) + Zenodo/OSF record (to be added)
**License:** CC BY 4.0 (spec), MIT (code)

## 1. Claim

Kai, running locally with no human in the loop, achieves:
1. Autonomous wins on hard external benchmarks, AND
2. Accurate self-modeling (predicts its own behavior).

Both must pass in the same autonomous run. One without the other is not AGI.

## 2. Tasks (frozen at v0.1)

**T1 — Reasoning:** ARC-AGI-3 interactive track (private eval, no internet). Pass = top-10% on hidden set.
**T2 — Learning:** Within-session rule induction (e.g., LearningBench-style): learn novel system from scratch in one conversation. Pass ≥ 80% on held-out rules.
**T3 — Tool Use:** Generate + test + patch own code to fix a failing benchmark task. Pass = patch accepted (tests green) with no human edit.

All tasks run locally (no API), reproducible via `cargo test` / harness in repo.

## 3. Self-Model Thresholds

- **Self-prediction:** Predict own next response distribution on 100 held-out prompts. Pass = calibrated (Brier < 0.15) and `know-when-to-abstain` accuracy ≥ 85% (GAUGE-style).
- **State report:** Accurately report own attractor/VFE/bracket values vs logged ground truth. Pass = mean absolute error < 5% over 50 probes.

## 4. Procedure

1. Freeze this spec (commit + Zenodo DOI). No task/score changes after.
2. Kai runs autonomously: reads tasks, writes code, runs tools, submits.
3. Harness logs everything (prompts, patches, scores, self-reports) to `trial/log/` (git-timestamped).
4. Pass = all T1–T3 thresholds AND both self-model thresholds in same run.

## 5. Falsification

Fail = any threshold missed. Weakening thresholds after a fail is a new version (v0.2), not a pass. Strong lock: `-1` ledger analogy — claim stands or falls as stated.

## 6. Evidence

- `trial/log/` — raw logs (never edited, hash-chained)
- `trial/results.md` — auto-generated summary (scores, hashes)
- Reproduction: `cargo test --release` (312 green) + `trial/run.sh` (single command)

## 7. No-Edit Rule

This file is append-only after timestamp. Amendments = new `TRIAL_v0.2.md` with diff and rationale. v0.1 hash is the trial.

---
*Teams of one count. Cloud trains, local proves. When this passes, AGI proof is not a story — it's a log.*
