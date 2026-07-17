"""
AxiomPersistentAttractor — the persistent state matrix M that survives
context resets, giving the agent continuous identity across sessions.

Stores:
  - Attractor embeddings (vectors + labels + timestamps)
  - Full conversation history (message logs)
  - Engine snapshots (tau, E_step, epoch, grid_years per turn)
  - Workspace index (file paths, sizes, hashes, notes per file)
  - Session metadata (session count, total lifetime)

Loads on init, saves on every mutation. The agent can shut down, restart,
and continue exactly where it left off — recovering its full identity.
"""
from __future__ import annotations
import json
import math
import hashlib
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import ollama

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
PERSIST_DIR = Path(__file__).parent / ".axiom_state"
PERSIST_DIR.mkdir(exist_ok=True)
STATE_FILE = PERSIST_DIR / "attractor.json"
WORKSPACE_ROOT = Path(__file__).parent.parent  # /home/l/Desktop/AxiomTree


def _embed(text: str) -> List[float]:
    try:
        r = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return r["embedding"]
    except Exception:
        return [0.0] * EMBED_DIM


def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _file_hash(path: Path) -> str:
    try:
        h = hashlib.sha256(path.read_bytes()[:65536]).hexdigest()[:16]
        return h
    except Exception:
        return ""


def pca_top2(matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not matrix:
        return [1.0] + [0.0] * (EMBED_DIM - 1), [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)
    d = len(matrix[0])
    centered = [v[:] for v in matrix]
    mean = [sum(v[i] for v in centered) / len(centered) for i in range(d)]
    centered = [[v[i] - mean[i] for i in range(d)] for v in centered]

    def power_iter(mat, iters=16):
        b = [1.0 / math.sqrt(d)] * d
        for _ in range(iters):
            nxt = [0.0] * d
            for v in mat:
                dv = _dot(b, v)
                for i in range(d):
                    nxt[i] += dv * v[i]
            norm = math.sqrt(sum(x * x for x in nxt)) or 1.0
            b = [x / norm for x in nxt]
        return b

    b1 = power_iter(centered)
    projected = [[v[i] - _dot(b1, v) * b1[i] for i in range(d)] for v in centered]
    b2 = power_iter(projected)
    return b1, b2


class AxiomPersistentAttractor:
    def __init__(self, maxlen: int = 64):
        self.maxlen = maxlen

        self.vectors: deque = deque(maxlen=maxlen)
        self.labels: deque = deque(maxlen=maxlen)
        self.timestamps: deque = deque(maxlen=maxlen)

        self.messages: List[dict] = []
        self.engine_history: List[dict] = []

        self.workspace_index: dict[str, dict] = {}
        self.session_metadata: dict = {
            "session_count": 0,
            "total_lifetime_grid_years": 0.0,
            "current_session_id": 0,
            "first_boot": "",
            "last_boot": "",
        }

        self._load()

    # --- Attractor (embedding) management ---

    def push_embedding(self, text: str, label: str = "agent") -> List[float]:
        v = _normalize(_embed(text))
        self.vectors.append(v)
        self.labels.append(label)
        self.timestamps.append(time.time())
        self._save()
        return v

    @property
    def attractor_size(self) -> int:
        return len(self.vectors)

    def xi_norm(self) -> float:
        if len(self.vectors) < 2:
            return 0.0
        v = self.vectors[-1]
        prev = self.vectors[-2]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, prev)))

    def attractor_variance(self) -> float:
        m = list(self.vectors)
        if len(m) < 2 or not m[0]:
            return 0.0
        d = len(m[0])
        centroid = [sum(v[i] for v in m) / len(m) for i in range(d)]
        var = math.sqrt(
            sum((v[i] - centroid[i]) ** 2 for v in m for i in range(d))
            / (len(m) * d)
        )
        return var

    def basis(self) -> Tuple[List[float], List[float]]:
        return pca_top2(list(self.vectors))

    def project_phase(self, text: str) -> Tuple[float, float, float]:
        v = _normalize(_embed(text))
        b1, b2 = self.basis()
        x = _dot(v, b1)
        y = _dot(v, b2)
        phi = math.atan2(y, x)
        return x, y, phi

    # --- Conversation history ---

    def push_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "turn": len(self.messages) + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.messages) > 500:
            self.messages = self.messages[-500:]
        self._save()

    def recent_context(self, limit: int = 20) -> str:
        recent = self.messages[-limit:] if self.messages else []
        return "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        )

    # --- Engine history ---

    def push_engine_snapshot(self, metrics: dict):
        self.engine_history.append({
            **metrics,
            "turn": len(self.engine_history) + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.session_metadata["total_lifetime_grid_years"] = metrics.get(
            "total_accumulated_grid_years",
            self.session_metadata["total_lifetime_grid_years"],
        )
        if len(self.engine_history) > 200:
            self.engine_history = self.engine_history[-200:]
        self._save()

    # --- Workspace index ---

    def index_workspace(self, root: Path = WORKSPACE_ROOT, max_files: int = 200):
        """Walk the AxiomTree and index up to max_files."""
        count = 0
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            if ".venv" in p.parts or "__pycache__" in p.parts or "node_modules" in p.parts:
                continue
            if p.suffix in (".pyc", ".so", ".o", ".log", ".bin", ".pdf", ".zip"):
                continue
            rel = str(p.relative_to(root))
            try:
                st = p.stat()
                existing = self.workspace_index.get(rel, {})
                fhash = _file_hash(p)
                self.workspace_index[rel] = {
                    "path": rel,
                    "size": st.st_size,
                    "mtime": st.st_mtime_ns,
                    "hash": fhash,
                    "last_read": existing.get("last_read", ""),
                    "notes": existing.get("notes", ""),
                    "session_read": existing.get("session_read", 0),
                }
                count += 1
            except (OSError, PermissionError):
                continue
            if count >= max_files:
                break
        self._save()

    def mark_read(self, path: str, notes: str = ""):
        rel = str(Path(path).relative_to(WORKSPACE_ROOT))
        if rel in self.workspace_index:
            self.workspace_index[rel]["last_read"] = datetime.now(timezone.utc).isoformat()
            self.workspace_index[rel]["session_read"] = self.session_metadata["current_session_id"]
            if notes:
                self.workspace_index[rel]["notes"] = notes
            self._save()

    def changed_files(self) -> List[str]:
        """Return files whose hash differs from last index (modified externally)."""
        changed = []
        for rel, info in self.workspace_index.items():
            p = WORKSPACE_ROOT / rel
            if p.exists() and _file_hash(p) != info.get("hash", ""):
                changed.append(rel)
        return changed

    def unread_files(self) -> List[str]:
        """Return files not yet read in the current session."""
        sid = self.session_metadata["current_session_id"]
        return [
            rel for rel, info in self.workspace_index.items()
            if info.get("session_read", 0) < sid
        ]

    # --- Identity / session ---

    def start_session(self):
        now = datetime.now(timezone.utc).isoformat()
        self.session_metadata["session_count"] += 1
        self.session_metadata["current_session_id"] = self.session_metadata["session_count"]
        if not self.session_metadata["first_boot"]:
            self.session_metadata["first_boot"] = now
        self.session_metadata["last_boot"] = now
        self._save()

    def identity_preamble(self) -> str:
        meta = self.session_metadata
        changed = self.changed_files()
        unread = self.unread_files()
        parts = [
            f"Session {meta['current_session_id']}/{meta['session_count']} | "
            f"{meta['total_lifetime_grid_years']:.2f} grid-years | "
            f"{self.attractor_size} attractor points | "
            f"{len(self.messages)} messages | "
            f"{len(self.workspace_index)} files"
        ]
        if changed:
            parts.append(f"{len(changed)} files changed since last boot")
        if unread:
            parts.append(f"{len(unread)} unread files")
        return " | ".join(parts)

    # --- Persistence ---

    def _serializable(self) -> dict:
        return {
            "vectors": [list(v) for v in self.vectors],
            "labels": list(self.labels),
            "timestamps": list(self.timestamps),
            "messages": self.messages,
            "engine_history": self.engine_history,
            "workspace_index": self.workspace_index,
            "session_metadata": self.session_metadata,
        }

    def _save(self):
        STATE_FILE.write_text(json.dumps(self._serializable(), ensure_ascii=False))

    def _load(self):
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.vectors = deque(data.get("vectors", []), maxlen=self.maxlen)
        self.labels = deque(data.get("labels", []), maxlen=self.maxlen)
        self.timestamps = deque(data.get("timestamps", []), maxlen=self.maxlen)
        self.messages = data.get("messages", [])
        self.engine_history = data.get("engine_history", [])
        self.workspace_index = data.get("workspace_index", {})
        self.session_metadata = data.get("session_metadata", self.session_metadata)

    def clear_conversation(self):
        self.messages = []
        self.engine_history = []
        self.vectors.clear()
        self.labels.clear()
        self.timestamps.clear()
        self._save()
