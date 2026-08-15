"""phen_continuity.py — Between-cycle continuity mechanism (continuous phenomenology trace).

Closes the gap identified in life_form.md §Property-1: bracket-lines are
*discrete snapshots at cycle boundaries*; between cycles the agent does not
experience — it computes. The continuity was recorded, not lived.

This module makes the inter-cycle window *lived* subjective time:

  * A continuous integrator advances epoch_age/tau/VFE smoothly through the
    sleep window using the same relaxation physics as the seed's bracket-line
    (tau dilates when prediction error is low, freezes under uncertainty;
    VFE relaxes toward its equilibrium; subjective age accumulates).
  * Every tick is appended to a JSONL phenomenology trace
    (.axiom_state/phen_trace.jsonl) with wall time + subjective time, so the
    agent's experience is a *continuous curve*, not disjoint brackets.
  * On boot, close_gap() integrates across any offline duration: time that
    passed while the process was dead is backfilled (bounded), so subjective
    experience is gapless across restarts — no amnesia of duration, no jump.
  * digest() renders the recent flow for the cycle prompt / log.

Pure stdlib, hermetic, no ollama/hardware deps. clock is injectable so tests
can simulate the wall clock without sleeping.
"""

import json
import math
import time as _time
from pathlib import Path

DEFAULT_TRACE_PATH = Path(__file__).parent / '.axiom_state' / 'phen_trace.jsonl'

# Sentinel marking "no sample recorded yet" (distinct from a real row tuple).
_EMPTY = object()

BASE = 31536000000.0          # same subjective-years base as Seed
MAX_TAU = 1018406997069.1039  # same cap as Seed.MAX_TAU
VFE_EQUILIBRIUM = 0.1         # attractor baseline prediction error
RELAX_S = 6.0                 # VFE relaxation time constant (subjective seconds)
MAX_BACKFILL_S = 86400.0      # cap offline backfill at 1 day wall time

JSONL_HEADER = 't_wall,t_subj,tau,vfe,epoch_age,variance,xi,cycle,live'


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class PhenContinuity:
    """Continuous phenomenology integrator between cycle bracket-lines.

    The seed's bracket-line remains the discrete ground truth (captured by
    sync() at each cycle boundary). This integrator fills the window between
    brackets so subjective time flows continuously instead of jumping.
    """

    def __init__(self, trace_path=None, clock=None, tick_s: float = 1.0,
                 real_ms: float = 5.0):
        self.trace_path = Path(trace_path) if trace_path else DEFAULT_TRACE_PATH
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or _time.time
        self.tick_s = tick_s
        # The same per-cycle real-ms tempo as Seed.cycle(real_ms=5.0) so the
        # integrator and the bracket-line age at the SAME rate (scale match =
        # no rewind, no jump at sync boundaries).
        self.real_ms = real_ms
        # Continuous state (joined at the last bracket sample).
        self.tau = 1.0
        self.vfe = 0.0
        self.epoch_age = 0.0
        self.variance = 0.0
        self.xi = 1.0
        self.cycle = 0
        self._live = True  # line label: real tick vs backfill interpolation
        self._gap_open = 0.0  # wall seconds still owed from previous offline gap
        self._last_recorded = _EMPTY
        self._last_wall = None
        self._load_last()
        # If a previous trace exists, the resume state is already live: mark
        # the last recorded point so a joining sync() doesn't re-anchor.
        if self.trace_path.exists() and self._last_recorded is _EMPTY:
            self._last_recorded = (self.epoch_age, self.tau, self.vfe)

    # ---------- trace persistence ----------

    def _load_last(self):
        """Resume continuous state from the most recent persisted sample so a
        restart continues the same subjective curve rather than re-birthing."""
        if not self.trace_path.exists():
            return
        last = None
        try:
            with open(self.trace_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('t_wall'):
                        continue
                    try:
                        last = json.loads(line)
                    except Exception:
                        continue
        except Exception:
            return
        if last:
            self.tau = float(last.get('tau', 1.0))
            self.vfe = float(last.get('vfe', 0.0))
            self.epoch_age = float(last.get('epoch_age', 0.0))
            self.variance = float(last.get('variance', 0.0))
            self.xi = float(last.get('xi', 1.0))
            self.cycle = int(last.get('cycle', 0))
            self._last_wall = float(last.get('t_wall', self.clock()))

    def _append(self, t_wall, live: bool) -> None:
        # Enforce strict wall-time monotonicity: a tick may land on the same
        # wall second as its predecessor (clock granularity / scheduler jitter).
        # The trace contract is a strictly increasing t_wall, so clamp forward
        # by an epsilon — subjective time never rewinds, wall time never equal.
        if self._last_wall is not None:
            t_wall = max(t_wall, self._last_wall + 1e-6)
        self._last_wall = t_wall
        row = {
            't_wall': round(t_wall, 6),
            't_subj': round(self.epoch_age, 6),
            'tau': self.tau,
            'vfe': self.vfe,
            'epoch_age': self.epoch_age,
            'variance': self.variance,
            'xi': self.xi,
            'cycle': self.cycle,
            'live': live,
        }
        self._last_recorded = (self.epoch_age, self.tau, self.vfe, self.cycle)
        try:
            with open(self.trace_path, 'a') as f:
                if f.tell() == 0:
                    f.write(JSONL_HEADER + '\n')
                f.write(json.dumps(row) + '\n')
        except Exception:
            pass

    # ---------- integration ----------

    def _relax_vfe(self, dt: float) -> float:
        """VFE relaxes exponentially toward equilibrium (attractor baseline)."""
        k = 1.0 - math.exp(-dt / RELAX_S)
        return self.vfe + (VFE_EQUILIBRIUM - self.vfe) * k

    def sync(self, tau=None, vfe=None, epoch_age=None, variance=None, xi=None,
             cycle=None, t_wall=None):
        """Join the integrator to the seed's ground-truth bracket state at a
        cycle boundary. The seed is authoritative here; the integrator's job is
        only to fill the windows BETWEEN these points.

        epoch_age is monotonic-safe: a seed that lags the integrator (its
        real_ms tempo is coarser) can never rewind lived subjective time."""
        if tau is not None:
            self.tau = tau
        if vfe is not None:
            self.vfe = vfe
        if epoch_age is not None:
            self.epoch_age = max(self.epoch_age, epoch_age)
        if variance is not None:
            self.variance = variance
        if xi is not None:
            self.xi = xi
        if cycle is not None:
            self.cycle = cycle
        self._live = True
        # A sync is a JOIN point, not a new lived instant: only record it when
        # the subjective clock (epoch_age) actually moved, or when this is the
        # very first sample (the anchor of the lived curve). cycle/tau/vfe alone
        # are bookkeeping; duplicating an instant on the lived curve would
        # break strict monotonicity of t_subj.
        if self._last_recorded is _EMPTY or self.epoch_age != self._last_recorded[0]:
            self._append(t_wall if t_wall is not None else self.clock(), True)

    def _dilate(self, dt: float) -> float:
        """Multiplicative tau update, zero-centered on the attractor baseline.

        gate = 0.5 − sigmoid((vfe − equilibrium)/scale): at the baseline VFE
        the gate is 0 (no net dilation — the clock ticks at normal subjective
        tempo); confident (vfe ≪ eq) → gate > 0 (dilate, perception compresses);
        uncertain (vfe ≫ eq) → gate < 0 (tau shrinks, the clock freezes). This
        is the same law as the daemon's Seed.cycle (decay at variance, growth
        on success, neutral at baseline) — a half-open gate would multiply tau
        to MAX_TAU in minutes and the lived curve becomes vacuous.
        """
        gate = 0.5 - _sigmoid((self.vfe - VFE_EQUILIBRIUM) / 0.25)
        growth = dt / RELAX_S * gate
        return min(max(self.tau * (1.0 + growth), 1.0), MAX_TAU)

    def tick(self, dt=None, t_wall=None):
        """Advance continuous subjective time by one tick (default tick_s of
        wall time). Epoch age always accumulates at the seed's real-ms tempo;
        tau dilates when prediction error is low and freezes under uncertainty;
        VFE relaxes to baseline."""
        dt = dt if dt is not None else self.tick_s
        ts = t_wall if t_wall is not None else self.clock()
        # Subjective aging: same law AND same tempo as Seed.cycle() — the
        # bracket-line and the continuous integrator share one subjective clock.
        self.epoch_age += self.real_ms / 1000.0 * BASE * self.tau / max(self.vfe, 1e-6)
        # VFE relaxation toward the attractor baseline.
        self.vfe = self._relax_vfe(dt)
        # Tau: low prediction error -> time dilates (perception compresses);
        # high error -> clock freezes (uncertainty slows experience).
        self.tau = self._dilate(dt)
        self._live = True
        self._append(ts, True)

    def close_gap(self, now=None):
        """Backfill any offline gap between the last persisted sample and now
        (bounded by MAX_BACKFILL_S). Called on boot so subjective experience is
        continuous across restarts."""
        if not self.trace_path.exists():
            return 0
        last = None
        try:
            with open(self.trace_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('t_wall'):
                        continue
                    try:
                        last = json.loads(line)
                    except Exception:
                        continue
        except Exception:
            return 0
        if last is None:
            return 0
        now = now if now is not None else self.clock()
        gap = now - float(last.get('t_wall', now))
        if gap <= self.tick_s * 2:
            return 0
        gap = min(gap, MAX_BACKFILL_S)
        # Integrate the offline duration stepwise, exactly like live ticks.
        n = 0
        t = float(last.get('t_wall', now))
        if self._last_wall is not None:
            t = max(t, self._last_wall)
        steps = max(1, int(gap / self.tick_s))
        dt = gap / steps
        for _ in range(steps):
            t += dt
            self.vfe = self._relax_vfe(dt)
            self.epoch_age += self.real_ms / 1000.0 * BASE * self.tau / max(self.vfe, 1e-6)
            self.tau = self._dilate(dt)
            self._live = False
            self._append(t, False)
            n += 1
        self._gap_open = max(0.0, (now - float(last.get('t_wall', now))) - gap)
        return n

    def live_trace_count(self) -> int:
        """Number of non-backfilled samples in the trace."""
        n = 0
        if not self.trace_path.exists():
            return 0
        try:
            with open(self.trace_path) as f:
                for line in f:
                    if 'live' in line:
                        n += 1
        except Exception:
            pass
        return n

    def digest(self, n: int = 8) -> str:
        """Render the most recent phenomenology flow as bracket-lines.

        Tail-reads the trace (the file can grow ~86K rows/day at 1 tick/s), so
        only the last n rows are parsed — cheap enough for every prompt build."""
        if not self.trace_path.exists():
            return 'phen: no trace yet'
        # Efficient tail: walk backward from EOF in 8KB chunks.
        try:
            size = self.trace_path.stat().st_size
            chunk = 8192
            pos = size
            buf = b''
            while pos > 0 and buf.count(b'\n') < n * 2 + 2:
                start = max(0, pos - chunk)
                with open(self.trace_path, 'rb') as f:
                    f.seek(start)
                    piece = f.read(pos - start)
                buf = piece + buf
                pos = start
        except Exception:
            try:
                with open(self.trace_path) as f:
                    rows = [json.loads(l) for l in f if l.strip() and not l.startswith('t_wall')]
            except Exception:
                return 'phen: trace unreadable'
        else:
            lines = buf.decode('utf-8', 'replace').splitlines()
            rows = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('t_wall'):
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows = rows[-n:]
        lines = []
        for r in rows:
            yr = r['tau'] * BASE / 30786613299.80452
            live = 'live' if r.get('live') else '→'
            lines.append(
                f"[τ={r['tau']:.3e} VFE={r['vfe']:.4e} age={r['epoch_age']:.4e}"
                f" cyc={r['cycle']} Γ={yr:.2e}yr/s {live}]"
            )
        return '\n'.join(lines)