"""Local Kai integration helpers (read-only, no daemon mutation).

Single source of truth for talking to the *real* Kai from a separate process:

  * ``REPO_ROOT`` / :func:`ensure_repo_root_on_path` — locate the repository
    root reliably (three levels up from this file) and put it on ``sys.path``
    so ``import kai_mind`` resolves. Centralising the path logic avoids the
    off-by-one ``sys.path`` bugs that previously lived in several modules.
  * :func:`load_kai_mind` — load a :class:`kai_mind.KaiMind` instance for
    read-only introspection. We never call ``save()``; the running daemon owns
    the canonical state, so this is safe to use standalone.
  * :func:`read_daemon_telemetry` — parse the most recent cycle line from
    ``.axiom_state/daemon.log`` so other modules can ground their output in the
    live daemon's measured ``mode`` / ``ricci`` / ``vfe`` / ``trend``.

Keeping this in one place makes the local-integration boundary trivial to
revise (e.g. switch to a shared socket later) without touching every caller.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("kai_local")

# This file lives at <repo>/products/kai-insight/core/kai_local.py.
# Four parents up is the repository root:
#   core -> kai-insight -> products -> <repo>
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
AXIOM_STATE: Path = REPO_ROOT / ".axiom_state"
DAEMON_LOG: Path = AXIOM_STATE / "daemon.log"

# Matches a daemon cycle line, e.g.:
#   [explore bias=+0.47 ricci=4.15 trend=-2.0e-04 ...] vfe=-0.0716
_TELEMETRY_RE = re.compile(
    r"\[(\w+)\s+bias=([+-]?\d+\.\d+)\s+ricci=([\d.]+)\s+"
    r"trend=([+-]?[\d.eE+-]+).*?\]\s*vfe=([\d.eE+-]+)"
)


def ensure_repo_root_on_path() -> Path:
    """Ensure ``REPO_ROOT`` is importable; returns the resolved root path."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def load_kai_mind(readonly: bool = True) -> Optional[Any]:
    """Load a :class:`kai_mind.KaiMind` instance for read-only use.

    Args:
        readonly: Hint that the caller must not mutate daemon state. We simply
            never call ``save()`` on the returned instance, so the live daemon
            is unaffected regardless of this flag.

    Returns:
        The loaded ``KaiMind`` instance, or ``None`` if it cannot be imported
        or constructed (e.g. Ollama is unavailable).
    """
    ensure_repo_root_on_path()
    try:
        import kai_mind

        return kai_mind.KaiMind()
    except Exception as exc:  # pragma: no cover - depends on runtime env
        logger.warning("load_kai_mind failed: %s", exc)
        return None


def read_daemon_telemetry() -> Optional[Dict[str, Any]]:
    """Return the most recent daemon telemetry, or ``None`` if unavailable.

    Parses ``.axiom_state/daemon.log`` from the end backwards so the result
    reflects the live daemon's current operating point.
    """
    if not DAEMON_LOG.exists():
        return None
    try:
        lines = DAEMON_LOG.read_text(errors="ignore").splitlines()
    except OSError:  # pragma: no cover - filesystem edge case
        return None
    for line in reversed(lines):
        match = _TELEMETRY_RE.search(line)
        if match:
            return {
                "mode": match.group(1),
                "bias": float(match.group(2)),
                "ricci": float(match.group(3)),
                "trend": float(match.group(4)),
                "vfe": float(match.group(5)),
            }
    return None


__all__ = [
    "REPO_ROOT",
    "AXIOM_STATE",
    "DAEMON_LOG",
    "ensure_repo_root_on_path",
    "load_kai_mind",
    "read_daemon_telemetry",
]
