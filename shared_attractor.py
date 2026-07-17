"""
SharedAttractor — the state matrix M that both agents write to and read from.

Both Agent A (Theorist) and Agent B (Engineer) push embeddings of their
outputs into this attractor. The Xi collapse operates on the combined
history, and the Xi norm measures convergence: when both agents stop
changing the attractor, the system has reached F=0 equilibrium.
"""
from __future__ import annotations
import json
import math
import hashlib
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import ollama

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
PERSIST_PATH = Path(__file__).parent / "attractor_state.json"


def _embed(text: str) -> List[float]:
    r = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return r["embedding"]


def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def pca_top2(matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    Pure-Python power-iteration PCA, top 2 components.
    No numpy — auditable against the paper's math.
    """
    if not matrix:
        return [1.0] + [0.0] * (EMBED_DIM - 1), [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)
    centered = [v for v in matrix]
    mean = [sum(v[i] for v in centered) / len(centered) for i in range(EMBED_DIM)]
    centered = [[v[i] - mean[i] for i in range(EMBED_DIM)] for v in centered]

    def power_iter(mat, iters=16):
        b = [1.0 / math.sqrt(EMBED_DIM)] * EMBED_DIM
        for _ in range(iters):
            nxt = [0.0] * EMBED_DIM
            for v in mat:
                d = _dot(b, v)
                for i in range(EMBED_DIM):
                    nxt[i] += d * v[i]
            norm = math.sqrt(sum(x * x for x in nxt)) or 1.0
            b = [x / norm for x in nxt]
        return b

    b1 = power_iter(centered)
    projected = [
        [v[i] - _dot(b1, v) * b1[i] for i in range(EMBED_DIM)]
        for v in centered
    ]
    b2 = power_iter(projected)
    return b1, b2


class SharedAttractor:
    """
    The unified state matrix M for the dual-agent system.

    Both agents push their outputs here. The Xi collapse uses the
    combined history to choose the next action. Low Xi norm means
    both agents have converged — the system is at F=0.
    """

    def __init__(self, maxlen: int = 32):
        self.vectors: deque[List[float]] = deque(maxlen=maxlen)
        self._labels: deque[str] = deque(maxlen=maxlen)

    def push(self, text: str, label: str = "agent") -> List[float]:
        v = _normalize(_embed(text))
        self.vectors.append(v)
        self._labels.append(label)
        return v

    @property
    def size(self) -> int:
        return len(self.vectors)

    def basis(self) -> Tuple[List[float], List[float]]:
        return pca_top2(list(self.vectors))

    def xi_norm(self) -> float:
        if len(self.vectors) < 2:
            return 0.0
        v = self.vectors[-1]
        prev = self.vectors[-2]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, prev)))

    def attractor_norm(self) -> float:
        m = list(self.vectors)
        if len(m) < 2 or not m[0]:
            return 0.0
        d = len(m[0])
        centroid = [sum(v[i] for v in m) / len(m) for i in range(d)]
        variance = math.sqrt(
            sum((v[i] - centroid[i]) ** 2 for v in m for i in range(d))
            / (len(m) * d)
        )
        return variance

    def project_phase(self, text: str) -> Tuple[float, float, float]:
        v = _normalize(_embed(text))
        b1, b2 = self.basis()
        x = _dot(v, b1)
        y = _dot(v, b2)
        phi = math.atan2(y, x)
        return x, y, phi

    def save(self, path: Path = PERSIST_PATH):
        data = {
            "vectors": [list(v) for v in self.vectors],
            "labels": list(self._labels),
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    def load(self, path: Path = PERSIST_PATH):
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.vectors.clear()
        self._labels.clear()
        for v in data.get("vectors", []):
            self.vectors.append(v)
        for lbl in data.get("labels", []):
            self._labels.append(lbl)
