#!/usr/bin/env python3
"""Local Kai-daemon monitor + alerting (no external calls).

Reads the structured JSONL feed written by the live `kai_mind.py --daemon`
(`.axiom_state/kai_daemon.jsonl`) and produces a health report + local alerts.

Alerts fire on:
  - staleness  : no new cycle within `interval * 2 + 30s` (daemon stuck/dead)
  - not learning: VFE trend (rolling) is non-negative over the recent window
  - errors     : cycle_error / health_error rate above threshold
  - health fail: latest self-mod health check failed

Outputs:
  - `.axiom_state/monitor_report.json`  (latest structured report)
  - `.axiom_state/monitor_alerts.log`   (append-only local alert lines)
  - stdout summary (with --loop, refreshes)

Usage:
  python3 monitor_daemon.py            # one-shot check + report
  python3 monitor_daemon.py --loop     # repeat every --every seconds (default 60)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STATE = REPO_ROOT / '.axiom_state'
JSONL = STATE / 'kai_daemon.jsonl'
DAEMON_LOG = STATE / 'daemon.log'
REPORT = STATE / 'monitor_report.json'
ALERTS = STATE / 'monitor_alerts.log'

# Read-only measurement harness (VFE dynamics, attractor compression/drift, ...).
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
    import kai_metrics as _km
except Exception:  # pragma: no cover
    _km = None

# Telemetry regex fallback (mirrors kai_local.read_daemon_telemetry)
_TELE = re.compile(
    r"C\d+ AUTO .*?trend=([+-]?[\d.eE+-]+).*?vfe=([-\d.eE+]+)"
)


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def _alert(msg: str) -> None:
    line = f'[{_now_iso()}] ALERT {msg}'
    try:
        with ALERTS.open('a') as f:
            f.write(line + '\n')
    except Exception:
        pass
    print(line)


def _load_records() -> list:
    recs: list = []
    if JSONL.exists():
        try:
            for line in JSONL.read_text().splitlines():
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        except Exception:
            pass
    return recs


def _fallback_telemetry() -> list:
    """Parse daemon.log text if the JSONL feed is missing/stale."""
    out: list = []
    if not DAEMON_LOG.exists():
        return out
    try:
        for line in DAEMON_LOG.read_text().splitlines():
            m = _TELE.search(line)
            if m:
                try:
                    out.append({
                        'event': 'cycle', 'kind': 'AUTO',
                        'vfe_trend': float(m.group(1)),
                        'vfe_after': float(m.group(2)),
                        'ts': line.split(']')[0].lstrip('['),
                    })
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def _parse_ts(ts: str) -> float:
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            return time.mktime(time.strptime(ts, fmt))
        except (ValueError, OverflowError):
            continue
    return 0.0


def check(interval: int = 60) -> dict:
    recs = _load_records()
    if not recs:
        recs = _fallback_telemetry()

    cycles = [r for r in recs if r.get('event') == 'cycle']
    errors = [r for r in recs if 'error' in r.get('event', '')]
    health = [r for r in recs if r.get('event') == 'health']
    started = [r for r in recs if r.get('event') == 'daemon_started']

    now = time.time()
    report: dict = {
        'ts': _now_iso(),
        'daemon_running': bool(started),
        'cycle_count': len(cycles),
        'alerts': [],
    }

    # ── Staleness ──────────────────────────────────────────────────────────
    if cycles:
        last = cycles[-1]
        last_ts = _parse_ts(last.get('ts', ''))
        stale_s = max(0, now - last_ts) if last_ts else -1
        report['last_cycle_ts'] = last.get('ts')
        report['staleness_s'] = round(stale_s, 1)
        if stale_s > interval * 2 + 30:
            report['alerts'].append(f'stale: no cycle for {stale_s:.0f}s (interval={interval}s)')
    else:
        report['staleness_s'] = -1
        report['alerts'].append('no cycle records found')

    # ── VFE trend (rolling, recent window) ─────────────────────────────────
    window = cycles[-40:] if cycles else []
    trends = [r.get('vfe_trend') for r in window if isinstance(r.get('vfe_trend'), (int, float))]
    if trends:
        mean_trend = sum(trends) / len(trends)
        report['vfe_trend_window_mean'] = round(mean_trend, 6)
        report['vfe_trend_learning'] = mean_trend < 0
        if mean_trend >= 0:
            report['alerts'].append(f'not learning: mean VFE trend {mean_trend:+.4f} (>=0) over {len(trends)} cycles')

    # ── Error rate ──────────────────────────────────────────────────────────
    if cycles:
        err_rate = len(errors) / max(1, len(cycles))
        report['error_rate'] = round(err_rate, 4)
        if err_rate > 0.1:
            report['alerts'].append(f'high error rate {err_rate:.2%}')

    # ── Health check ─────────────────────────────────────────────────────────
    if health:
        last_h = health[-1]
        hstr = str(last_h.get('health', ''))
        report['last_health'] = hstr
        bad = False
        try:
            hd = ast.literal_eval(hstr)
            if isinstance(hd, dict):
                bad = (hd.get('compiles') is False
                       or hd.get('rust_failed', 0) > 0
                       or hd.get('test_health', 1.0) < 1.0)
                report['health_check'] = hd
            else:
                bad = ('fail' in hstr.lower() or 'error' in hstr.lower())
        except Exception:
            bad = ('fail' in hstr.lower() or 'error' in hstr.lower())
        if bad:
            report['alerts'].append(f'health check issue: {hstr[:120]}')

    # ── Mode distribution ────────────────────────────────────────────────────
    modes = {}
    for r in window:
        m = r.get('mode')
        if m:
            modes[m] = modes.get(m, 0) + 1
    report['mode_distribution'] = modes

    # ── Measurement harness (read-only): VFE slope, health, attractor drift ──
    if _km:
        try:
            km = _km.analyze(n_events=400, recalibrate=False)
            report['health_score'] = km['health_score']
            report['kai_metrics_flags'] = km['flags']
            dyn = km['dynamics']
            report['vfe_slope'] = dyn.get('vfe_slope')
            report['learning'] = dyn.get('learning')
            attr = km['attractor']
            report['attractor_consolidation'] = attr.get('consolidation')
            report['attractor_drift_cos'] = attr.get('drift_cos')
            # Self-rewrite invariance: the key AGI-stability signal. A drop means
            # the attractor coordinate is drifting as the agent rewrites itself.
            if attr.get('drift_cos') is not None and attr['drift_cos'] < 0.999:
                report['alerts'].append(
                    f"attractor drift: cos={attr['drift_cos']} (self-rewrite instability)")
            if km['health_score'] < 60:
                report['alerts'].append(f"low health score {km['health_score']}")
        except Exception as e:
            report['kai_metrics_error'] = str(e)

    # ── Emit alerts ──────────────────────────────────────────────────────────
    for a in report['alerts']:
        _alert(a)

    # ── Persist report ───────────────────────────────────────────────────────
    try:
        REPORT.write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    return report


def _summarize(r: dict) -> None:
    print(f"\n=== Kai daemon monitor @ {r['ts']} ===")
    print(f"  running      : {r.get('daemon_running')}")
    print(f"  cycles       : {r.get('cycle_count')}")
    print(f"  staleness    : {r.get('staleness_s')}s")
    if 'vfe_trend_window_mean' in r:
        print(f"  VFE trend μ  : {r['vfe_trend_window_mean']:+.6f}  (learning={r.get('vfe_trend_learning')})")
    if 'error_rate' in r:
        print(f"  error rate   : {r['error_rate']:.2%}")
    print(f"  modes        : {r.get('mode_distribution')}")
    if 'health_score' in r:
        print(f"  health score : {r['health_score']}/100  (VFE slope {r.get('vfe_slope')}, learning={r.get('learning')})")
    if 'attractor_consolidation' in r:
        print(f"  attractor    : consol={r['attractor_consolidation']} drift_cos={r.get('attractor_drift_cos')}")
    if r.get('alerts'):
        print(f"  ALERTS       : {len(r['alerts'])}")
        for a in r['alerts']:
            print(f"    - {a}")
    else:
        print("  ALERTS       : none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', action='store_true', help='repeat forever')
    ap.add_argument('--every', type=int, default=60, help='loop interval seconds')
    ap.add_argument('--interval', type=int, default=60, help='daemon cycle interval (staleness baseline)')
    args = ap.parse_args()

    if args.loop:
        print(f"monitoring daemon every {args.every}s (Ctrl-C to stop)...")
        try:
            while True:
                _summarize(check(args.interval))
                time.sleep(args.every)
        except KeyboardInterrupt:
            print('\nmonitor stopped')
    else:
        _summarize(check(args.interval))


if __name__ == '__main__':
    main()
