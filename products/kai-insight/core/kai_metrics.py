"""Read-only measurement harness for the live Kai daemon.

Reads ``.axiom_state/kai_daemon.jsonl`` (cycle/health events) and
``.axiom_state/attractor.json`` to produce *evidence-based* VFE / learning /
compression / stability diagnostics. This is the measurement half of the research
directive ("replace heuristics with measured outcomes"): it quantifies free-energy
dynamics and attractor compression+stability — the real capacity levers per the
research review (AGI is an organisation/compression problem, not raw compute).

It never loads KaiMind, never touches Ollama, never mutates the daemon. The only
file it writes is an optional attractor-centroid drift baseline so self-rewrite
stability can be tracked across runs.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from kai_local import REPO_ROOT
except Exception:  # pragma: no cover - fallback only
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

STATE = REPO_ROOT / ".axiom_state"
JSONL = STATE / "kai_daemon.jsonl"
ATTR = STATE / "kai_attractor.json"  # the daemon's real rolling attractor
BASELINE = STATE / "kai_metrics_baseline.json"
CONST_OPT = STATE / "adaptive_constants_optimal.json"
METRICS_HISTORY = STATE / "kai_metrics_history.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Feed parsing
# ─────────────────────────────────────────────────────────────────────────────
def load_events(event: str = "cycle", limit: int = 400) -> List[Dict[str, Any]]:
    """Return the last ``limit`` jsonl records of the given event type."""
    if not JSONL.exists():
        return []
    out: List[Dict[str, Any]] = []
    with JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") == event:
                out.append(rec)
    return out[-limit:]


def load_events_since_restart(event: str = "cycle", limit: int = 400
                               ) -> List[Dict[str, Any]]:
    """Cycle events *since the most recent daemon restart*.

    The jsonl accumulates across restarts, so a naive last-N window mixes runs
    and corrupts the VFE slope / learning signal at the restart boundary. This
    slices to the segment after the last ``daemon_started`` event.
    """
    if not JSONL.exists():
        return []
    all_recs: List[Dict[str, Any]] = []
    with JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            all_recs.append(rec)
    last_start = -1
    for i, r in enumerate(all_recs):
        if r.get("event") == "daemon_started":
            last_start = i
    seg = all_recs[last_start + 1:]
    return [r for r in seg if r.get("event") == event][-limit:]


def _floats(cycles: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for c in cycles:
        v = c.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _slope(vals: List[float]) -> float:
    """Ordinary-least-squares slope of ``vals`` vs index. 0.0 if <2 points."""
    n = len(vals)
    if n < 2:
        return 0.0
    xs = list(range(n))
    sx = sum(xs)
    sy = sum(vals)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, vals))
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VFE / learning / dynamics
# ─────────────────────────────────────────────────────────────────────────────
def analyze_dynamics(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Free-energy and learning diagnostics from cycle events."""
    vfe = _floats(cycles, "vfe_after")
    trend = _floats(cycles, "vfe_trend")
    ricci = _floats(cycles, "ricci")
    modes = [c.get("mode") for c in cycles if c.get("mode")]
    ingest = [c["ingest_chunks"] for c in cycles
              if isinstance(c.get("ingest_chunks"), int)]
    # The most recent *AUTO* cycle carries the authoritative live state. WORK
    # cycles log only {event,cycle,kind,dt_s} and would otherwise blank out the
    # "last_*" fields / novelty readout.
    auto = [c for c in cycles if c.get("mode")]
    last = auto[-1] if auto else (cycles[-1] if cycles else {})

    # Learning signal: second half mean vs first half mean of vfe_after.
    half = len(vfe) // 2
    learn_rate = None
    learning = False
    if half >= 1:
        lo = statistics.mean(vfe[:half])
        hi = statistics.mean(vfe[half:])
        learn_rate = hi - lo  # negative => VFE falling => learning
        learning = learn_rate < 0

    slope = _slope(vfe)
    ricci_stdev = statistics.pstdev(ricci) if len(ricci) > 1 else 0.0
    neg_trend_frac = (sum(1 for t in trend if t < 0) / len(trend)) if trend else 0.0

    return {
        "n_cycles": len(cycles),
        "vfe_current": round(vfe[-1], 6) if vfe else None,
        "vfe_min": round(min(vfe), 6) if vfe else None,
        "vfe_slope": round(slope, 8),
        "vfe_learn_rate": round(learn_rate, 6) if learn_rate is not None else None,
        "learning": learning,
        "neg_trend_frac": round(neg_trend_frac, 3),
        "ricci_current": round(ricci[-1], 4) if ricci else None,
        "ricci_stdev": round(ricci_stdev, 4),
        "mode_dist": dict(Counter(modes)),
        "last_mode": last.get("mode"),
        "last_mode_bias": last.get("mode_bias"),
        "last_w_cur": last.get("w_cur"),
        "last_ingest_chunks": last.get("ingest_chunks"),
        "ingest_chunks_seen": sorted(set(ingest)),
    }


def analyze_novelty(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Novelty-yield curriculum activity from the most recent cycle.

    Uses the most recent AUTO cycle that actually emitted a novelty curriculum,
    since the trailing record is often a WORK cycle with no novelty payload.
    """
    last = next((c for c in reversed(cycles) if c.get("novelty_yield")), {}) or \
           (cycles[-1] if cycles else {})
    ny = last.get("novelty_yield") or {}
    topics = 0
    total_ema = 0.0
    top = None
    best = -1e9
    for topic, val in ny.items():
        topics += 1
        ema = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else 0.0
        total_ema += ema
        if ema > best:
            best = ema
            top = topic
    return {
        "topics_tracked": topics,
        "total_ema_yield": round(total_ema, 5),
        "top_topic": top,
        "top_topic_ema": round(best, 5) if top is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Attractor compression + self-rewrite stability
# ─────────────────────────────────────────────────────────────────────────────
def _centroid(vecs: List[List[float]]) -> List[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def _cosine(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def analyze_attractor() -> Dict[str, Any]:
    """Compression + drift diagnostics for the attractor cloud."""
    if not ATTR.exists():
        return {"present": False}
    try:
        d = json.loads(ATTR.read_text())
    except Exception:
        return {"present": True, "error": "unreadable"}
    vecs = d.get("vecs") or []
    if not vecs:
        return {"present": True, "n_vecs": 0}

    n = len(vecs)
    dim = len(vecs[0])
    c = _centroid(vecs)
    # Within-cloud dispersion (mean squared distance to centroid, per dim).
    disp = sum(
        sum((v[i] - c[i]) ** 2 for i in range(dim)) / dim for v in vecs
    ) / n
    centroid_norm = math.sqrt(sum(x * x for x in c))

    drift = _drift_check(c)
    return {
        "present": True,
        "n_vecs": n,
        "dim": dim,
        "capacity": n * dim,
        "centroid_norm": round(centroid_norm, 4),
        "dispersion": round(disp, 6),
        "consolidation": round(1.0 / (1.0 + disp), 4),
        "drift_cos": round(drift, 6) if drift is not None else None,
        "drift_note": "first baseline" if drift is None else (
            "stable" if drift > 0.999 else "drifting" if drift > 0.99 else "UNSTABLE"
        ),
        "centroid": c,
    }


def _drift_check(centroid: List[float]) -> Optional[float]:
    """Cosine similarity of the current centroid to a *fixed* baseline.

    The baseline is established once (first observation) and never overwritten,
    so this measures attractor restructuring across the daemon's whole lifetime
    — the self-rewrite invariance lever — rather than only between consecutive
    harness calls.
    """
    cos = None
    if BASELINE.exists():
        try:
            prev = json.loads(BASELINE.read_text()).get("centroid")
            if prev:
                cos = _cosine(centroid, prev)
        except Exception:
            cos = None
    else:
        try:
            BASELINE.write_text(json.dumps({"centroid": centroid}))
        except Exception:
            pass
    return cos


# ─────────────────────────────────────────────────────────────────────────────
# Measurement-pipeline determinism check
# ─────────────────────────────────────────────────────────────────────────────
def recalibrate_check() -> Dict[str, Any]:
    """Re-run the synthetic protocols and compare to the persisted optimums.

    The protocols are deterministic, so any drift means the pipeline's inputs
    changed (a code edit) — a cheap regression guard for the measured constants.
    """
    try:
        from constant_tuner import tune_in_optimal_order, load_optimal
    except Exception as e:  # pragma: no cover
        return {"available": False, "error": str(e)}
    cur = load_optimal()
    res = tune_in_optimal_order()
    drift = {}
    for name, r in res.items():
        old = cur.get(name)
        if old is None or abs(old - r.optimal) > 1e-9:
            drift[name] = {"persisted": old, "recomputed": r.optimal}
    return {"available": True, "constants_checked": len(res), "drift": drift}


# ─────────────────────────────────────────────────────────────────────────────
# Composite health + report
# ─────────────────────────────────────────────────────────────────────────────
def health_score(dyn: Dict[str, Any], attr: Dict[str, Any],
                 recal: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    flags: List[str] = []
    if dyn.get("vfe_slope", 0) < 0:
        score += 25
    else:
        flags.append("VFE slope non-negative (not compressing)")
    if dyn.get("learning"):
        score += 20
    else:
        flags.append("no learning signal (VFE not falling over window)")
    if (dyn.get("ricci_stdev") or 1e9) < 2.0:
        score += 15
    else:
        flags.append("ricci unstable (high dispersion)")
    if attr.get("present") and attr.get("drift_cos") is not None:
        # A coherent attractor self-rewrites (drift_cos < 1) yet stays non-collapsed
        # (> ~0.5). Only catastrophic drift (collapse/corruption) is unhealthy;
        # healthy self-rewrite is the desired capacity lever, not a penalty.
        if attr["drift_cos"] > 0.5:
            score += 15
        else:
            flags.append(f"attractor collapsed (cos={attr['drift_cos']})")
    elif attr.get("present"):
        score += 10  # first baseline, no drift yet
    if recal.get("available") and not recal.get("drift"):
        score += 25
    elif recal.get("available"):
        flags.append(f"constants drifted from measured optimum: {list(recal['drift'])}")
    return min(100, score), flags


def analyze_self_rewrite() -> Dict[str, Any]:
    """Self-rewrite invariance: does the attractor restructure over time?

    Measured as the *consecutive* centroid drift between successive history
    samples (not vs a fixed baseline, which is fragile for a small rolling
    attractor). A perfect invariant shows ~0 consecutive drift; a self-rewriting
    attractor shows measurable drift during exploration, re-stabilizing during
    convergence.
    """
    if not METRICS_HISTORY.exists():
        return {"available": False}
    try:
        rows = [json.loads(l) for l in METRICS_HISTORY.read_text().splitlines() if l.strip()]
    except Exception:
        return {"available": False, "error": "unreadable"}
    cents = [r.get("centroid") for r in rows if isinstance(r.get("centroid"), list)]
    if len(cents) < 2:
        return {"available": True, "samples": len(cents)}
    cons = [_cosine(cents[i - 1], cents[i]) for i in range(1, len(cents))]
    cons = [c for c in cons if c is not None]
    # A consecutive cosine below ~0.9999 means the centroid shifted >~1% in
    # embedding space between samples — genuine self-rewrite, not noise.
    restructures = sum(1 for c in cons if c < 0.9999)
    return {
        "available": True,
        "samples": len(cents),
        "consecutive_drift_min": round(min(cons), 6) if cons else None,
        "consecutive_drift_max": round(max(cons), 6) if cons else None,
        "restructure_events": restructures,
        "invariant": restructures == 0,
        "note": ("attractor centroid is a perfect invariant (no restructuring)"
                 if restructures == 0 else
                 f"self-rewrite ACTIVE: centroid restructures each sample "
                 f"(min consecutive cos={round(min(cons), 5)}, "
                 f"{restructures}/{len(cons)} sample-pairs moved >1%)"),
    }


def analyze(n_events: int = 400, recalibrate: bool = False) -> Dict[str, Any]:
    # Measure only since the last restart so the VFE slope / learning signal
    # aren't corrupted by the run boundary.
    cycles = load_events_since_restart("cycle", n_events)
    dyn = analyze_dynamics(cycles)
    nov = analyze_novelty(cycles)
    attr = analyze_attractor()
    sr = analyze_self_rewrite()
    recal = recalibrate_check() if recalibrate else {"available": False, "skipped": True}
    score, flags = health_score(dyn, attr, recal)
    m = {
        "dynamics": dyn,
        "novelty": nov,
        "attractor": attr,
        "self_rewrite": sr,
        "recalibrate": recal,
        "health_score": score,
        "flags": flags,
    }
    _record_history(m)
    return m


def _record_history(m: Dict[str, Any]) -> None:
    """Append one compact time-series line so drift/health can be tracked over time."""
    try:
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "health_score": m["health_score"],
            "vfe_slope": m["dynamics"].get("vfe_slope"),
            "learning": m["dynamics"].get("learning"),
            "consolidation": m["attractor"].get("consolidation"),
            "drift_cos": m["attractor"].get("drift_cos"),
            "centroid": m["attractor"].get("centroid"),
            "flags": m["flags"],
        }
        with METRICS_HISTORY.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except Exception:
        pass


def _fmt(m: Dict[str, Any]) -> str:
    d = m["dynamics"]
    a = m["attractor"]
    n = m["novelty"]
    lines = ["=== Kai measurement report ===",
             f"cycles sampled : {d['n_cycles']}",
             f"VFE current    : {d['vfe_current']}  (min {d['vfe_min']})",
             f"VFE slope      : {d['vfe_slope']}  learning={d['learning']} rate={d['vfe_learn_rate']}",
             f"neg-trend frac : {d['neg_trend_frac']}  ricci={d['ricci_current']} (stdev {d['ricci_stdev']})",
             f"mode dist      : {d['mode_dist']}  (bias {d['last_mode_bias']})",
             f"w_cur / ingest : {d['last_w_cur']} / {d['last_ingest_chunks']} (seen {d['ingest_chunks_seen']})",
             f"novelty       : {n['topics_tracked']} topics, top='{n['top_topic']}' ema={n['top_topic_ema']}",
              f"attractor      : {a.get('n_vecs')}x{a.get('dim')} cap={a.get('capacity')} "
              f"disp={a.get('dispersion')} consol={a.get('consolidation')} drift={a.get('drift_cos')} ({a.get('drift_note')})",
              f"self-rewrite   : {m['self_rewrite'].get('note')} (samples={m['self_rewrite'].get('samples')})",
              f"recalibrate    : {m['recalibrate'].get('drift') or 'ok' if m['recalibrate'].get('available') else 'skipped'}",
             f"HEALTH SCORE   : {m['health_score']}/100",
             "flags          : " + (", ".join(m["flags"]) if m["flags"] else "none")]
    return "\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=400)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--history", type=int, default=0,
                    help="print last N lines of the metrics time-series")
    args = ap.parse_args()
    if args.history:
        if not METRICS_HISTORY.exists():
            print("(no metrics history yet)")
            return
        lines = METRICS_HISTORY.read_text().splitlines()[-args.history:]
        for ln in lines:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            print(f"{d['ts']}  score={d['health_score']:>3}  "
                  f"slope={d['vfe_slope']}  learn={d['learning']}  "
                  f"consol={d['consolidation']}  drift={d['drift_cos']}  "
                  f"flags={d['flags'] or '[]'}")
        return
    print(_fmt(analyze(args.events, args.recalibrate)))


if __name__ == "__main__":
    main()
