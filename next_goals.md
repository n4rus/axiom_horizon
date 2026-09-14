# next_goals.md — private AGI track (local only, never public)

**Doctrine:** generations are indeterminate like real life. No caps, no
plateaus-as-stops. Plateau is *measured and logged*, never a halt.
Every generation checkpoints so a kill/restart loses at most one gen and
resumes from the last saved generation.

## 1. Infinite generations + resume (this session)
- `max_generations: 0` = infinite. `--generations N` bounds a run when needed
  (video demos, timed benches).
- Skip re-seeding when the archive already holds candidates (resume, not restart).
- `archive.save()` every generation, not just at the end.
- Per-gen heartbeat to stderr: gen number, best fitness, streak state.
- Plateau (5 stagnant gens) logs a notice and keeps evolving — mutation-rate
  adaptation already widens search on plateau; the system kicks itself.
- Video impact: Scene 3 must pass `--generations 8` or it never completes.

## 2. Long run + measure (i3, electricity only — start anytime)
- Lift caps, run overnight/multiday on the current rig.
- Collect fitness/gen curves; read convergence via `awareness.summary`.
- First question answered: does it converge (Aₙ₊₁≈Aₙ), diverge, or wander?
- Cost: ~250W home power. No new hardware needed.

## 3. Evolve fitness itself (i3, code + CPU)
- Point `metalearn` at fitness *components* (novelty + usefulness judges),
  not just strategies. Who judges the judges is the wall; this is the drill.
- No new hardware needed. Pure code + overnight runs.

## 4. L2 demo (i3, code + CPU)
- Agent patches its own darwin operators under existing safety gates.
- Measured delta, gates-held log. Headline moment: "Level 2, gates held,
  fitness +x" — one measured experiment, not one breakthrough.
- No new hardware needed.

## 5. Bigger local reasoner (REQUIRES new rig)
- 12b-tier teachers already proved bigger brains move the needle
  (fuse 12b PHYSICS WINS +0.46).
- Needs the 5060 Ti 16GB box (grant-dependent). Blocked until funded.
- Until then: 7b-tier overnight runs on the i3.

**Order:** 1 (today) → 2 (tonight, overnight) → 3 → 4 → 5 (post-grant).
