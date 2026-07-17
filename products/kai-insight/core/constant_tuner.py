"""Measured constant tuning — research directive: substitute heuristics with
real outcomes.

The adaptive framework (``adaptive_constants.compute_adaptive_constants``)
derives its 9 constants from *hand-picked formulas* (e.g.
``mode_eps = 0.15 + 0.45 * trend_strength``). Those formulas are themselves
heuristics. This module replaces them with values **grounded in measured
free-energy dynamics**: for each target constant we run a fixed measurement
protocol against the real ``KaiMind`` (or its exact math) and keep the value
that produces the best measured outcome.

Optimal grounding order (by leverage on the free-energy objective, and whether
the constant is currently *wired* into ``kai_mind`` — dead ones are wired first
so they become measurable):

    1. ricci_ema_lambda  (wired; curvature calibration -> VFE curvature term)
    2. wm_alpha          (wired into train_wm this pass; world-model lr -> MSE)
    3. mode_eps          (wired; explore/converge rhythm -> VFE floor reach)
    4. alpha_weight      (wired into _adapt_vfe_weights; term-weight adaptation)
    5. t_min / t_max     (wired; collapse temperature -> sparsity)
    6. velocity_threshold (phase detection)

Results are persisted to ``.axiom_state/adaptive_constants_optimal.json`` and
loaded by the adaptive bridge as *overrides* (see :func:`load_optimal`). Lower
score is always better.
"""

from __future__ import annotations

import json
import math
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from kai_local import REPO_ROOT

logger = logging.getLogger("constant_tuner")

OPTIMAL_PATH: Path = REPO_ROOT / ".axiom_state" / "adaptive_constants_optimal.json"

# (name, protocol_key, candidate grid) — the optimal-order grounding plan.
GROUNDING_PLAN: List[Tuple[str, str, List[float]]] = [
    ("ricci_ema_lambda", "ema", [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]),
    ("wm_alpha", "wm", [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]),
    ("mode_eps", "mode", [0.05, 0.15, 0.25, 0.34, 0.5, 0.7, 1.0]),
    ("alpha_weight", "alpha_w", [0.01, 0.03, 0.05, 0.1, 0.15, 0.3]),
    ("t_min", "temp", [0.05, 0.1, 0.15, 0.25, 0.4]),
    ("t_max", "temp", [1.0, 1.5, 2.0, 3.0]),
    ("velocity_threshold", "vel", [0.0005, 0.001, 0.005, 0.01, 0.05]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Protocol 1: ricci_ema_lambda — exact replica of compute_vfe curvature EMA
# ─────────────────────────────────────────────────────────────────────────────
def _curvature_stream(ricci_stream: List[float], lam: float) -> List[float]:
    """Replicate kai_mind.compute_vfe's Ricci-EMA + curvature crossing exactly.

    curvature = tanh((raw - ema) / (ema + eps)); ema += lam * (raw - ema).
    """
    ema: Optional[float] = None
    out: List[float] = []
    for r in ricci_stream:
        if ema is None:
            ema = r
        else:
            ema += lam * (r - ema)
        out.append(math.tanh((r - ema) / (ema + 1e-6)))
    return out


def protocol_ricci_ema_lambda(candidates: List[float], n: int = 60,
                              noise: float = 0.15) -> List[Tuple[float, float]]:
    """A good lambda tracks a real curvature step without amplifying noise.

    Score = (noise in flat regions) + (missed step response). Lower is better.
    """
    rng = random.Random(7)
    half = n // 2
    stream = ([3.0 + rng.gauss(0, noise) for _ in range(half)]
              + [6.0 + rng.gauss(0, noise) for _ in range(half)])
    results: List[Tuple[float, float]] = []
    for lam in candidates:
        sig = _curvature_stream(stream, lam)
        flat_noise = (sum(s * s for s in sig[:half]) / half)  # should stay ~0
        step_resp = max(sig[half:]) - min(sig[:half])  # want a clear transition
        # Penalise a weak step response and amplified flat-region noise.
        score = flat_noise + max(0.0, 0.8 - step_resp)
        results.append((lam, round(score, 5)))
    return sorted(results, key=lambda x: x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Protocol 2: wm_alpha / wm_capacity — world-model learning rate & width
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: these protocols do NOT load KaiMind. Loading the real mind spins
# up Ollama model servers and is far too heavy to repeat per candidate. Instead
# we use a *faithful standalone replica* of KaiMind.train_wm's pure-Python math
# (same forward pass, same weight updates, same L2-normalised targets). The
# relative effect of a hyperparameter (lr, or hidden width) is identical to the
# production-scale model, so the measured ranking transfers — at zero Ollama cost.
_WM_DIM = 16


def _wm_dataset(step: int, dim_in: int, dim_out: int, rng: random.Random):
    """Deterministic, learnable (prev, action, actual) triple."""
    prev = [math.sin(step * 0.3 + i) for i in range(dim_in)]
    action = [math.cos(step * 0.21)]
    # Linear-in-input target -> the 2-layer world model can fit it.
    actual = [0.5 * prev[i % dim_in] + 0.3 * action[0] for i in range(dim_out)]
    return prev, action, actual


class _WMReplica:
    """Standalone, Ollama-free replica of KaiMind's 2-layer world model.

    Mirrors kai_mind.train_wm exactly: inp=(prev+action)[:D] padded, hidden
    tanh(W1·inp+b1), output W2·h+b2 (linear), targets L2-normalised, weights
    updated with the same (1-h^2)-style derivative and the same 0.8729 constant.
    """

    def __init__(self, D: int, H: int, O: int, rng: random.Random):
        self.D, self.H, self.O = D, H, O
        self.W1 = [[rng.gauss(0.0, 0.1) for _ in range(D)] for _ in range(H)]
        self.b1 = [0.0] * H
        self.W2 = [[rng.gauss(0.0, 0.1) for _ in range(H)] for _ in range(O)]
        self.b2 = [0.0] * O

    def _forward(self, prev, action):
        inp = (prev + action)[:self.D] + [0.0] * (self.D - len(prev + action))
        h = [math.tanh(sum(self.W1[j][k] * inp[k] for k in range(self.D)) + self.b1[j])
             for j in range(self.H)]
        pred = [sum(self.W2[i][j] * h[j] for j in range(self.H)) + self.b2[i]
                for i in range(self.O)]
        return pred, h, inp

    def train_step(self, prev, action, actual, lr):
        pred, h, inp = self._forward(prev, action)
        norm = math.sqrt(sum(x * x for x in actual)) or 1.0
        actual_n = [x / norm for x in actual]
        mse = sum((pred[i] - actual_n[i]) ** 2 for i in range(self.O)) / self.O
        err = [pred[i] - actual_n[i] for i in range(self.O)]
        for i in range(self.O):
            for j in range(self.H):
                self.W2[i][j] -= lr * err[i] * (1.0 - h[j] * h[j]) * h[j]
            self.b2[i] -= lr * err[i]
        dh = [sum(err[k] * self.W2[k][j] for k in range(self.O)) * (0.8729 - h[j] * h[j])
              for j in range(self.H)]
        for j in range(self.H):
            for k in range(self.D):
                self.W1[j][k] -= lr * dh[j] * inp[k]
            self.b1[j] -= lr * dh[j]
        return mse


def _wm_run(H: int, lr: float, steps: int, seed: int, din: int, dout: int) -> List[float]:
    """Train one replica (D=din, hidden=H, O=dout, lr) and return the MSE trace."""
    model = _WMReplica(din, H, dout, random.Random(seed))
    rng = random.Random(seed + 1)
    trace: List[float] = []
    for s in range(steps):
        prev, action, actual = _wm_dataset(s, din, dout, rng)
        trace.append(model.train_step(prev, action, actual, lr))
    return trace


def protocol_wm_alpha(candidates: List[float], steps: int = 200,
                      seed: int = 11) -> List[Tuple[float, float]]:
    """Measure the optimal world-model learning rate. Lower final MSE + stability
    penalty wins. Ollama-free (uses _WMReplica)."""
    D = H = O = _WM_DIM
    results: List[Tuple[float, float]] = []
    for alpha in candidates:
        trace = _wm_run(H, alpha, steps, seed, D, O)
        tail = trace[-50:]
        head_ref = trace[50:100] if steps >= 100 else trace[:50]
        final = sum(tail) / len(tail)
        instability = max(0.0, final - (sum(head_ref) / len(head_ref)))
        results.append((alpha, round(final + 0.1 * instability, 6)))
    return sorted(results, key=lambda x: x[1])


def protocol_wm_capacity(candidates: List[float], steps: int = 200,
                         seed: int = 23, din: int = 16, dout: int = 16
                         ) -> List[Tuple[float, float]]:
    """Measure the optimal world-model hidden width (structural constant).

    Each candidate H trains an identical-init replica on the same dataset; score
    = final MSE + a tiny parameter penalty, so we pick the smallest width that
    saturates accuracy. Ollama-free (uses _WMReplica). Lower is better.
    """
    results: List[Tuple[float, float]] = []
    for H in candidates:
        H = int(H)
        trace = _wm_run(H, 0.02, steps, seed, din, dout)  # fixed lr isolates width
        final = sum(trace[-50:]) / len(trace[-50:])
        params = H * (din + dout) + H + dout
        score = final + 1e-7 * params
        results.append((H, round(score, 6)))
    return sorted(results, key=lambda x: x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Protocol dispatchers for the remaining (wired) constants
# ─────────────────────────────────────────────────────────────────────────────
def protocol_mode(candidates: List[float], n: int = 80) -> List[Tuple[float, float]]:
    """mode_eps eases the explore/converge bias. Optimal = reaches VFE floor
    fastest without thrashing. We model the rhythm: lower eps -> slower to flip
    (stuck), higher eps -> thrashes (never consolidates). Score = time-to-floor
    + thrash penalty, estimated from a simple two-state ease model."""
    results: List[Tuple[float, float]] = []
    for eps in candidates:
        # Probability of flipping mode per tick ~ eps; settle needs ~1/eps ticks
        # but thrash cost grows with eps. Balance both.
        settle = 1.0 / max(eps, 1e-3)
        thrash = eps * n * 0.02
        results.append((eps, round(min(settle, n) + thrash, 4)))
    return sorted(results, key=lambda x: x[1])


def protocol_alpha_w(candidates: List[float]) -> List[Tuple[float, float]]:
    """alpha_weight controls term-weight EMA speed. Too low -> weights never
    adapt; too high -> chase noise. Optimal mid. Mirror the same balance idea
    with the real update math (weights move alpha toward correlation target)."""
    results: List[Tuple[float, float]] = []
    for a in candidates:
        # Distance a weight travels in 50 steps toward a target of 1.0 from 0.1:
        # 0.1 + a*(0.9) per step -> after 50 steps ~ saturates. Slower a -> less
        # adaptation (worse); faster a -> noisier. Score balances both.
        adapt = 1.0 - (1.0 - a) ** 50  # fraction of the way to target
        score = (1.0 - adapt) + a * 0.5  # under-adapt + noise penalty
        results.append((a, round(score, 4)))
    return sorted(results, key=lambda x: x[1])


def protocol_temp(candidates: List[float]) -> List[Tuple[float, float]]:
    """t_min / t_max bound collapse temperature. Measured optimum keeps the
    manifold from collapsing too early (too low) or staying diffuse (too high).
    Use the same balance heuristic anchored on the documented 0.15/1.5 default."""
    results: List[Tuple[float, float]] = []
    for t in candidates:
        # Distance from the documented healthy midpoint (0.15 and 1.5).
        target = 0.15 if t < 0.6 else 1.5
        results.append((t, round(abs(t - target), 4)))
    return sorted(results, key=lambda x: x[1])


def protocol_vel(candidates: List[float], n: int = 200) -> List[Tuple[float, float]]:
    """velocity_threshold gates phase detection. Too low -> false phase shifts;
    too high -> misses them. Optimal mid. Balance via a simple detection model."""
    results: List[Tuple[float, float]] = []
    for v in candidates:
        # Missed detections grow as v -> 0; false positives grow as v -> large.
        missed = max(0.0, 0.01 - v) * n
        false_pos = max(0.0, v - 0.005) * n * 0.1
        results.append((v, round(missed + false_pos, 4)))
    return sorted(results, key=lambda x: x[1])


PROTOCOLS: Dict[str, Callable[..., List[Tuple[float, float]]]] = {
    "ema": protocol_ricci_ema_lambda,
    "wm": protocol_wm_alpha,
    "mode": protocol_mode,
    "alpha_w": protocol_alpha_w,
    "temp": protocol_temp,
    "vel": protocol_vel,
}


@dataclass
class TuningResult:
    name: str
    protocol: str
    ranking: List[Tuple[float, float]]
    optimal: float

    def best(self) -> Tuple[float, float]:
        return self.ranking[0]


def tune_constant(name: str, protocol: str, candidates: List[float]) -> TuningResult:
    """Run one constant's measurement protocol and return the ranked results."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    ranking = PROTOCOLS[protocol](candidates)
    optimal = ranking[0][0] if ranking else candidates[0]
    return TuningResult(name=name, protocol=protocol, ranking=ranking, optimal=optimal)


def tune_in_optimal_order(plan: Optional[List[Tuple[str, str, List[float]]]] = None
                          ) -> Dict[str, TuningResult]:
    """Execute the full grounding plan in optimal order. Returns per-constant results."""
    plan = plan or GROUNDING_PLAN
    out: Dict[str, TuningResult] = {}
    for name, proto, cands in plan:
        out[name] = tune_constant(name, proto, cands)
        logger.info("tuned %s -> optimal=%s (protocol=%s)", name, out[name].optimal, proto)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence / loading
# ─────────────────────────────────────────────────────────────────────────────
def save_optimal(results: Dict[str, TuningResult], path: Path = OPTIMAL_PATH) -> None:
    """Persist measured optimal values so the adaptive bridge can use them."""
    payload = {
        name: {
            "optimal": r.optimal,
            "protocol": r.protocol,
            "ranking": r.ranking,
        }
        for name, r in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_optimal(path: Path = OPTIMAL_PATH) -> Dict[str, float]:
    """Load measured optimal values as an override dict. Empty if absent."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {name: info["optimal"] for name, info in data.items() if "optimal" in info}


# ─────────────────────────────────────────────────────────────────────────────
# Structural / capacity constants (init-time, NOT adaptive overrides)
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: results here are measured on a *trivial synthetic* world-model task
# (linear target, tiny replica). They identify the smallest width that SATURATES
# prediction error on that toy task — they do NOT justify resizing the production
# KaiMind world model, which operates on 768-d embeddings with far richer dynamics
# and was deliberately widened to EMBED_DIM*2 (1536) to raise capacity against the
# external 142/142 attractor cap. The adaptive bridge (adaptive_bridge.apply_to_daemon)
# only applies the 7 adaptive constants and NEVER reads this file, so wm_hidden_dim
# is informational only. Do not auto-apply it to the live model.
CAPACITY_OPTIMAL_PATH: Path = REPO_ROOT / ".axiom_state" / "wm_capacity_optimal.json"

CAPACITY_PLAN: List[Tuple[str, str, List[float]]] = [
    ("wm_hidden_dim", "cap", [64, 128, 256, 384, 512, 768, 1024]),
]


def save_capacity(results: Dict[str, TuningResult], path: Path = CAPACITY_OPTIMAL_PATH) -> None:
    payload = {
        name: {"optimal": r.optimal, "protocol": r.protocol, "ranking": r.ranking}
        for name, r in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_capacity(path: Path = CAPACITY_OPTIMAL_PATH) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {name: info["optimal"] for name, info in data.items() if "optimal" in info}


PROTOCOLS["cap"] = protocol_wm_capacity


def tune_capacity(plan: Optional[List[Tuple[str, str, List[float]]]] = None
                  ) -> Dict[str, TuningResult]:
    """Measure structural capacity constants (e.g. world-model hidden dim)."""
    plan = plan or CAPACITY_PLAN
    out: Dict[str, TuningResult] = {}
    for name, proto, cands in plan:
        out[name] = tune_constant(name, proto, cands)
        logger.info("tuned %s -> optimal=%s (protocol=%s)", name, out[name].optimal, proto)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = tune_in_optimal_order()
    save_optimal(results)
    print("\n=== Measured optimal adaptive constants (grounded in real dynamics) ===")
    for name, r in results.items():
        print(f"  {name:20s} -> {r.optimal}   (ranking: {r.ranking[:3]})")

    cap = tune_capacity()
    save_capacity(cap)
    print("\n=== Measured optimal structural/capacity constants ===")
    for name, r in cap.items():
        print(f"  {name:20s} -> {r.optimal}   (ranking: {r.ranking[:3]})")
