"""Bridge between Kai's Python mind and the high-performance Rust core.

The Rust extension (``kai_core``) provides SIMD-accelerated world-model
prediction and free-energy kernels. This module locates the compiled
extension, loads it, and exposes drop-in replacements for the hot numerical
loops used by :class:`KaiMind` in ``kai_mind.py``.

If the extension cannot be found or fails at runtime, every function falls
back to an equivalent, pure-Python implementation so Kai never breaks.

Loading strategy
----------------
1. Try a direct ``import kai_core`` (extension already on ``sys.path``).
2. Otherwise discover the compiled cdylib under ``rust/target/{release,debug}``
   (``libkai_core.so``) or a prebuilt ``kai_core.so``, copy it to an
   importable ``kai_core.so`` underneath ``.axiom_state/rust_ext/``, and import.
"""

from __future__ import annotations

import importlib
import math
import shutil
import sys
from pathlib import Path
from typing import List

# numpy is required by the Rust extension's C-API at call time.
try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a hard repo dependency
    _np = None

_RUST_MOD = None
_RUST_AVAILABLE = False


def _discover_and_load() -> bool:
    """Locate and import the ``kai_core`` extension. Idempotent."""
    global _RUST_MOD, _RUST_AVAILABLE
    if _RUST_MOD is not None:
        return _RUST_AVAILABLE
    if _np is None:
        return False

    # 1. Already importable.
    try:
        import kai_core as _mod  # type: ignore

        _RUST_MOD = _mod
        _RUST_AVAILABLE = True
        return True
    except Exception:
        pass

    # 2. Discover a compiled cdylib and expose it as ``kai_core.so``.
    here = Path(__file__).resolve().parent
    candidates: List[Path] = []
    candidates += sorted(here.glob("rust/target/release/libkai_core.so"))
    candidates += sorted(here.glob("rust/target/debug/libkai_core.so"))
    candidates += sorted(here.glob("kai_core.so"))
    candidates += sorted(here.glob("rust/kai_core.so"))

    for src in candidates:
        target = src
        if src.name.startswith("lib"):
            cache_dir = here / ".axiom_state" / "rust_ext"
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / "kai_core.so"
            if (
                not target.exists()
                or target.stat().st_mtime < src.stat().st_mtime
            ):
                shutil.copy(src, target)
        load_dir = str(target.parent)
        if load_dir not in sys.path:
            sys.path.insert(0, load_dir)
        try:
            import kai_core as _mod  # type: ignore

            _RUST_MOD = _mod
            _RUST_AVAILABLE = True
            return True
        except Exception:
            continue
    return False


# Resolve availability once at import time.
RUST_AVAILABLE: bool = _discover_and_load()


def _as_f32(arr) -> "_np.ndarray":
    return _np.asarray(arr, dtype=_np.float32)


def predict(state: List[float], action: List[float],
            W1: List[List[float]], b1: List[float],
            W2: List[List[float]], b2: List[float]) -> List[float]:
    """World-model forward pass. Returns a ``list[float]`` (matches the
    pure-Python implementation in :meth:`KaiMind.predict` exactly)."""
    if _RUST_AVAILABLE and _RUST_MOD is not None:
        try:
            out = _RUST_MOD.predict_world_model(
                _as_f32(state), _as_f32(action),
                _as_f32(W1), _as_f32(b1), _as_f32(W2), _as_f32(b2),
            )
            return out.tolist()
        except Exception:
            pass
    # Pure-Python fallback — identical mathematics.
    inp = list(state) + list(action)
    n_cols = len(inp)
    hidden = [
        math.tanh(sum(W1[i][j] * inp[j] for j in range(n_cols)) + b1[i])
        for i in range(len(W1))
    ]
    return [
        sum(W2[i][k] * hidden[k] for k in range(len(hidden))) + b2[i]
        for i in range(len(W2))
    ]


def train_step(state: List[float], action: List[float], actual: List[float],
               W1: List[List[float]], b1: List[float],
               W2: List[List[float]], b2: List[float], lr: float):
    """One in-place SGD training step of the world model, Rust-accelerated.

    Ports the pure-Python backward pass in :meth:`KaiMind.train_wm` (the
    ``hidden * input + output * hidden`` MAC hotspot) to the ``kai_core``
    kernel, which replicates the exact update rule and statement ordering.

    Returns ``(mse, W1, b1, W2, b2)`` with weights updated, or ``None`` when the
    Rust extension is unavailable (caller then runs its pure-Python fallback).
    The learning-rate schedule and ``actual`` normalization stay in Python, so
    the learning dynamics are unchanged.
    """
    if _RUST_AVAILABLE and _RUST_MOD is not None:
        try:
            w1 = _as_f32(W1)
            bb1 = _as_f32(b1)
            w2 = _as_f32(W2)
            bb2 = _as_f32(b2)
            m = float(_RUST_MOD.train_step_world_model(
                _as_f32(state), _as_f32(action), _as_f32(actual),
                w1, bb1, w2, bb2, float(lr),
            ))
            return m, w1.tolist(), bb1.tolist(), w2.tolist(), bb2.tolist()
        except Exception:
            pass
    return None


def mse(pred: List[float], actual: List[float]) -> float:
    """Mean squared error between prediction and observation.

    Uses the Rust ``compute_vfe`` kernel with zero regularization (variance=0,
    novelty=0) when available; otherwise a pure-Python equivalent.
    """
    if _RUST_AVAILABLE and _RUST_MOD is not None:
        try:
            return float(_RUST_MOD.compute_vfe(_as_f32(pred), _as_f32(actual), 0.0, 0.0))
        except Exception:
            pass
    n = max(len(pred), 1)
    return sum((p - a) ** 2 for p, a in zip(pred, actual)) / n


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Normalized cosine similarity (Rust-backed when available)."""
    if _RUST_AVAILABLE and _RUST_MOD is not None:
        try:
            return float(_RUST_MOD.cosine_similarity(_as_f32(a), _as_f32(b)))
        except Exception:
            pass
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
