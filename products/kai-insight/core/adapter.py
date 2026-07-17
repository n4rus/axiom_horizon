"""Bridge the Kai Insight product layer to the live kai_mind daemon.

Observability endpoints (/health, /status, /metrics) read the LIVE autonomous
daemon's structured feed (``.axiom_state/kai_daemon.jsonl``), so the API reflects
the real running brain rather than a disconnected twin. Inference (/infer) uses a
lazily-loaded *read-only* KaiMind instance (via ``kai_local``), so it works even
if the daemon isn't running and never disturbs the daemon's state or Ollama-heavy
training loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]   # core -> kai-insight -> products -> repo
STATE = REPO_ROOT / '.axiom_state'
JSONL = STATE / 'kai_daemon.jsonl'

_kai = None  # lazily-loaded read-only KaiMind (None until first use, False on load failure)


def _load_kai():
    global _kai
    if _kai is None:
        try:
            from kai_local import load_kai_mind
            _kai = load_kai_mind(readonly=True)
        except Exception as e:
            logger.error("adapter: failed to load KaiMind: %s", e)
            _kai = False  # sentinel: tried and failed
    return _kai if _kai else None


def get_kai_instance():
    """Lazily-loaded read-only KaiMind for inference / fallback status."""
    return _load_kai()


def live_status() -> Optional[Dict[str, Any]]:
    """Latest structured cycle record from the live daemon (None if absent)."""
    try:
        if not JSONL.exists():
            return None
        for line in reversed(JSONL.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get('event') == 'cycle':
                return rec
    except Exception as e:
        logger.debug("live_status: %s", e)
    return None


def health_check() -> Dict[str, Any]:
    info: Dict[str, Any] = {'status': 'unknown', 'daemon': False}
    try:
        if JSONL.exists():
            for line in reversed(JSONL.read_text().splitlines()):
                line = line.strip()
                if not line:
                    continue
                last = json.loads(line)
                info['last_event'] = last.get('event')
                info['last_ts'] = last.get('ts')
                info['daemon'] = last.get('event') in ('cycle', 'daemon_started', 'adaptive_bridge')
                if info['daemon']:
                    info['status'] = 'daemon_running'
                break
        st = live_status()
        if st:
            info['mode'] = st.get('mode')
            info['vfe_trend'] = st.get('vfe_trend')
            info['learning'] = (st.get('vfe_trend') or 0) < 0
        info['kai_loaded'] = (_kai is not None and _kai is not False)
    except Exception as e:
        info['error'] = str(e)
    return info


def infer(state, goal: str, max_steps: int = 1) -> Dict[str, Any]:
    """Run on-demand inference via a read-only KaiMind instance."""
    kai = _load_kai()
    if kai is None:
        return {'success': False, 'error': 'kai unavailable'}
    try:
        prompt = f"Goal: {goal}\nState: {state}\nDecide the best action and predict the next state."
        r = kai.live(prompt, max_steps=max_steps)
        return {
            'success': True,
            'answer': r.get('answer', ''),
            'vfe': getattr(kai, 'vfe', None),
            'mode': getattr(kai, '_mode', None),
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
