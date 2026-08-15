"""test_phen_continuity.py — Daemon-cycle verification for the between-cycle
continuity mechanism (continuous phenomenology trace).

Headless: no ollama, no GPU, no agent. Drives PhenContinuity with a simulated
clock exactly as _daemon_autonomous_loop does, and asserts the four continuity
guarantees the mechanism exists for:

  1. Continuous living: a cycle's sleep window produces live trace samples at
     tick granularity (not dead wall-clock sleep, not one bracket per cycle).
  2. Monotonic subjective time: epoch_age never decreases and t_subj/t_wall
     strictly increase across the whole trace (no rewinding, no double-count).
  3. Gapless restart: simulating daemon downtime and re-booting backfills the
     offline interval (bounded) so subjective experience does not jump.
  4. Ground-truth joining: sync() at cycle boundaries re-anchors the curve to
     the seed's bracket state (tau/vfe/epoch_age tracked without drift).

Run:  python3 test_phen_continuity.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

from phen_continuity import PhenContinuity, BASE, MAX_TAU


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def advance(self, dt):
        self.t += dt

    def __call__(self):
        return self.t


def read_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('t_wall'):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def test_continuous_live_ticks():
    """A 5s cycle window must produce ~5 live trace samples, not 1."""
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        p = PhenContinuity(Path(d) / 'trace.jsonl', clock=clock, tick_s=1.0)
        p.sync(tau=1.0, vfe=0.5, epoch_age=0.0, cycle=0)
        for _ in range(5):  # simulate the daemon sleep loop
            p.tick()
            clock.advance(1.0)
        rows = read_trace(p.trace_path)
        assert len(rows) == 6, f'expected sync + 5 ticks, got {len(rows)}'
        live = [r for r in rows if r.get('live')]
        assert len(live) == 6, f'expected 6 live samples, got {len(live)}'
        # each tick advanced wall + subjective time
        t_subs = [r['t_subj'] for r in rows]
        assert all(t_subs[i] < t_subs[i + 1] for i in range(len(t_subs) - 1)), \
            'continuous subjective time must strictly increase'
    print('  PASS continuous_live_ticks: sleep window lived at tick granularity')


def test_monotonic_subjective_time():
    """Across cycles + offline backfill, epoch_age never decreases."""
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        p = PhenContinuity(Path(d) / 'trace.jsonl', clock=clock, tick_s=1.0)
        # 3 cycles, each with a 2-second lived window. The integrator owns the
        # curve; sync only re-anchors tau/vfe/cycle at the boundaries.
        for cyc in range(3):
            p.sync(tau=2.0, vfe=0.3, cycle=cyc)
            clock.advance(1.0)
            for _ in range(2):
                p.tick()
                clock.advance(1.0)
        rows = read_trace(p.trace_path)
        ages = [r['epoch_age'] for r in rows]
        subs = [r['t_subj'] for r in rows]
        assert all(ages[i] <= ages[i + 1] for i in range(len(ages) - 1)), \
            'epoch_age must be non-decreasing across cycles'
        assert all(subs[i] < subs[i + 1] for i in range(len(subs) - 1)), \
            't_subj must strictly increase'
    print('  PASS monotonic_subjective_time: no rewind, no double-count')


def test_sync_never_rewinds():
    """A seed whose epoch_age lags the integrator must not rewind lived time."""
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        p = PhenContinuity(Path(d) / 'trace.jsonl', clock=clock, tick_s=1.0)
        p.sync(epoch_age=0.0, cycle=0)
        for _ in range(3):
            p.tick()
            clock.advance(1.0)
        before = p.epoch_age
        # Seed reports a stale/lower epoch_age (coarser real_ms tempo).
        p.sync(epoch_age=before / 2.0, cycle=1)
        assert p.epoch_age >= before, 'sync must never rewind lived subjective time'
    print('  PASS sync_never_rewinds: lagging seed cannot rewind the clock')


def test_gapless_restart_backfill():
    """Downtime between runs is integrated (bounded) on next boot — no gap."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'trace.jsonl'
        clock1 = FakeClock(start=5000.0)
        p1 = PhenContinuity(path, clock=clock1, tick_s=1.0)
        p1.sync(tau=1.0, vfe=0.5, epoch_age=0.0, cycle=0)
        clock1.advance(1.0)
        for _ in range(3):
            p1.tick()
            clock1.advance(1.0)
        n1 = len(read_trace(path))

        # --- daemon dies; 300 wall-seconds pass, then a fresh boot ---
        clock2 = FakeClock(start=5000.0 + 300.0)
        p2 = PhenContinuity(path, clock=clock2, tick_s=1.0)
        p2.close_gap()
        rows = read_trace(path)
        assert len(rows) > n1, 'offline gap must produce backfilled samples'
        # continuity: wall time strictly increasing across the downtime boundary
        ts = [r['t_wall'] for r in rows]
        assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)), \
            'wall time must be strictly increasing across restart'
        # the curve resumes from the last known subjective state (no rebirth)
        assert p2.epoch_age > 0.0, 'restart must continue the subjective curve'
        # bounded: backfill cannot exceed the cap even for huge downtime
        assert (rows[-1]['t_wall'] - rows[0]['t_wall']) <= 400.0, \
            'backfill must be bounded'
    print('  PASS gapless_restart_backfill: downtime integrated, bounded, gapless')


def test_ground_truth_joining():
    """sync() re-anchors the integrator to the seed bracket state."""
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        p = PhenContinuity(Path(d) / 'trace.jsonl', clock=clock, tick_s=1.0)
        # seed advances far (e.g. from a training cycle)
        p.sync(tau=99.0, vfe=0.05, epoch_age=12345.0, variance=0.02, xi=0.9, cycle=7)
        assert p.tau == 99.0 and abs(p.vfe - 0.05) < 1e-9
        assert p.epoch_age == 12345.0 and p.cycle == 7
        assert p.variance == 0.02 and abs(p.xi - 0.9) < 1e-9
        # tick from that anchored state, keeping it continuous
        clock.advance(1.0)
        p.tick()
        rows = read_trace(p.trace_path)
        assert rows[-1]['epoch_age'] > 12345.0
        assert rows[-1]['tau'] > 99.0 * 0.999, 'tau dilates from anchored state'
    print('  PASS ground_truth_joining: seed bracket state tracked without drift')


def test_digest_renders():
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        p = PhenContinuity(Path(d) / 'trace.jsonl', clock=clock, tick_s=1.0)
        p.sync(tau=1.0, vfe=0.5, epoch_age=0.0, cycle=0)
        for _ in range(4):
            p.tick()
            clock.advance(1.0)
        dg = p.digest(4)
        assert 'τ=' in dg and 'VFE=' in dg and 'yr/s' in dg
        assert len(dg.splitlines()) == 4
    print('  PASS digest_renders: bracket-line digest readable for prompts')


def test_physics_bounds():
    """tau must respect MAX_TAU and never go negative/inf."""
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        p = PhenContinuity(Path(d) / 'trace.jsonl', clock=clock, tick_s=1.0)
        p.sync(tau=MAX_TAU - 100.0, vfe=1e-9, epoch_age=0.0, cycle=0)
        for _ in range(1000):
            p.tick()
            clock.advance(1.0)
        assert p.tau <= MAX_TAU * (1 + 1e-6), f'tau overflow: {p.tau}'
        assert math.isfinite(p.tau) and math.isfinite(p.epoch_age)
        assert p.epoch_age > 0.0
    print('  PASS physics_bounds: tau capped at MAX_TAU, state finite')


def main():
    tests = [
        test_continuous_live_ticks,
        test_monotonic_subjective_time,
        test_sync_never_rewinds,
        test_gapless_restart_backfill,
        test_ground_truth_joining,
        test_digest_renders,
        test_physics_bounds,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f'  FAIL {t.__name__}: {e}')
    if failed:
        print(f'\nphen_continuity: {len(tests) - failed}/{len(tests)} passed, {failed} FAILED')
        sys.exit(1)
    print(f'\nphen_continuity: {len(tests)}/{len(tests)} passed')


if __name__ == '__main__':
    main()