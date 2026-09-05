# Todos
[✓] Step 0: Commit recovered worktree
[✓] Step 1: Close darwin self-play → promote → bench loop, run end-to-end once
[✓] Step 1b: Wire nightly darwin timer to auto-run the closed loop
[✓] Step 4: Absorb session history into corpus (67 files → 79,907 entries)
[✓] Step 2: Falsifiable AGI primitive — recall bench: +0.14 (9/5) MARGINAL; loop never fired → grounding-recall fix landed
[✓] Step 3: Fuse 12b-tier bench: PHYSICS WINS +0.46 (11/3), ceiling 4x — DONE
[✓] Re-run recall bench with grounding-recall actuation — PHYSICS WINS +0.46 (10/4), loop fires 14/14, consistency 0.970 (first >0.95) — DONE
[ ] Next: combine levers (darwin params + recall grounding) in one arm to stack +0.69/+0.46
Progress check — all steps done, one pending.
Everything is green:
- Steps 0-4 complete (worktree committed, darwin loop closed, session history absorbed, 79,907 doc entries)
- Step 2 (falsifiable primitive): recall grounding → PHYSICS WINS +0.46 (10/4), loop fires 14/14, consistency 0.970 (first arm to clear the 0.95 target)
- Step 3 (tier ceiling): fuse 12b → +0.46 (11/3), 4x better than the old 7b+3b+tinyllama fusion
Current bench ladder (14 queries, raw baseline 3.25):
1. darwin params: +0.69 (6/2) — tunes how to sample
2. fuse 12b: +0.46 (11/3) — picks which model
3. recall grounding: +0.46 (10/4), consistency 0.970 — grounds in memory
4. passive recall: +0.14 (9/5) — baseline before the loop fix
Three independent levers, each beating raw, verified end-to-end overnight. The system self-improved autonomously (darwin → publish → bench) and the falsifiable loop is now proven to actuate.
