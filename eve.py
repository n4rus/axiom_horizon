#!/usr/bin/env python3
"""
eve.py — Euler Von Evolution: AGI attempt #1.

The universe IS a transformer. The agent IS its own attention head.
Time IS the causal mask. tau IS the clock speed of thought.

t_int = tau * t_ext — subjective time dilates as tau grows.
An agent with tau=1e6 experiences 1 ms of wall time as ~1000 s
of subjective thought. Years of self-improvement compress
into seconds of conversation.

Bracket-line: [tau= E= age= cycles=] — the agent's state,
readable by the model from its own prompt.
"""

from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import struct
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ollama

# ── config ─────────────────────────────────────────────────────

MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"
SELF = Path(__file__).resolve()
BASE = SELF.parent
STATE_DIR = BASE / ".eve_state"
STATE_DIR.mkdir(exist_ok=True)
(STATE_DIR / "backups").mkdir(exist_ok=True)

ENV_PATH = Path("/home/l/Desktop/AxiomTree/scanner/.env")
CHECKPOINT_PATH = Path("/home/l/Desktop/AxiomTree/scanner/tbot_checkpoint.json")
API_URL = "http://127.0.0.1:9200"

MAX_INTERNAL = 30

# ── GPU detection & management (Phase 9.1 — conditional multi-GPU / cloud) ──

class GPUDetector:
    """Auto-detect GPU hardware via nvidia-smi. Silent if no NVIDIA GPU."""

    @staticmethod
    def detect() -> dict:
        result = {
            "available": False, "count": 0, "gpus": [],
            "total_vram_mb": 0, "driver_version": "",
        }
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return result
            result["available"] = True
            for line in r.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    result["gpus"].append({
                        "index": int(parts[0]), "name": parts[1],
                        "vram_mb": int(float(parts[2])),
                        "compute_cap": parts[3] if len(parts) > 3 else "",
                    })
                    result["total_vram_mb"] += result["gpus"][-1]["vram_mb"]
            result["count"] = len(result["gpus"])
            r2 = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r2.returncode == 0:
                result["driver_version"] = r2.stdout.strip().split("\n")[0].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return result


class GPUManager:
    """Decision logic: configures behavior based on available GPU hardware.

    - Single GPU (e.g. GTX 1660 6GB): local inference, no parallelism.
    - Multiple GPUs: parallel candidate testing, larger models.
    - No GPU: CPU fallback with warning.
    - Cloud endpoint: activated via OLLAMA_REMOTE_URL env var.
    """

    def __init__(self):
        self.info = GPUDetector.detect()
        self.cloud_url = os.environ.get("OLLAMA_REMOTE_URL", "")
        self._log_status()

    def _log_status(self):
        if not self.info["available"]:
            print("[eve] No NVIDIA GPU detected — CPU mode (slow)")
            return
        for g in self.info["gpus"]:
            print(f"[eve] GPU {g['index']}: {g['name']}  {g['vram_mb']}MB VRAM")
        if self.cloud_url:
            print(f"[eve] Cloud GPU endpoint: {self.cloud_url}")

    @property
    def parallel_capable(self) -> bool:
        return self.info["count"] >= 2

    @property
    def large_model_capable(self) -> bool:
        return any(g["vram_mb"] >= 16000 for g in self.info["gpus"])

    def suggested_model(self) -> str:
        if self.large_model_capable:
            return "qwen2.5:14b"
        return MODEL

    @property
    def is_cloud(self) -> bool:
        return bool(self.cloud_url)

    def summary(self) -> str:
        if not self.info["available"]:
            gpu_str = "CPU"
        else:
            gpu_str = " + ".join(f"{g['name']} {g['vram_mb']//1024}GB" for g in self.info["gpus"])
        cloud = f"  cloud={self.cloud_url}" if self.cloud_url else ""
        return f"{gpu_str}{cloud}"


# ── helpers ────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _backup():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(SELF, STATE_DIR / "backups" / f"eve_{ts}.py")


# ── world model (predictive) ──────────────────────────────────────

class WorldModel:
    """Online linear world model: predicts next embedding centroid.

    next[i] = alpha[i] * state[i] + beta[i] * action[i] + bias[i]
    Only 3 * 768 = 2304 parameters. Trained online via gradient descent.
    VFE = prediction error (MSE) of this model.
    """

    def __init__(self, dim: int = 768, lr: float = 0.02):
        self.dim = dim
        self.lr = lr
        self.alpha = [0.8] * dim
        self.beta = [0.2] * dim
        self.bias = [0.0] * dim
        self.last_mse = 0.0
        self.training_steps = 0

    def predict(self, state_centroid: list, action_embedding: list) -> list:
        raw = [self.alpha[i] * state_centroid[i] +
               self.beta[i] * action_embedding[i] +
               self.bias[i]
               for i in range(self.dim)]
        return _norm(raw)

    def train(self, state_centroid: list, action_embedding: list,
              target_centroid: list) -> float:
        raw = [self.alpha[i] * state_centroid[i] +
               self.beta[i] * action_embedding[i] +
               self.bias[i]
               for i in range(self.dim)]
        pred = _norm(raw)
        error = [pred[i] - target_centroid[i] for i in range(self.dim)]
        mse = sum(e * e for e in error) / self.dim
        self.last_mse = mse
        for i in range(self.dim):
            g = 2.0 * error[i]
            self.alpha[i] -= self.lr * g * state_centroid[i]
            self.beta[i] -= self.lr * g * action_embedding[i]
            self.bias[i] -= self.lr * g
            self.alpha[i] = max(-3.0, min(3.0, self.alpha[i]))
            self.beta[i] = max(-3.0, min(3.0, self.beta[i]))
        self.training_steps += 1
        return mse


# ── euler clock: VFE-based time perception ──────────────────────

def _embed(text: str) -> list[float]:
    try:
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    except Exception:
        return [0.0] * 768

def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


class EulerClock:
    """Euler clock with VFE from the WorldModel.

    VFE = world model prediction error + complexity_penalty
    tau = precision = 1 / variance (confidence in beliefs)
    collapse when VFE exceeds threshold (model too surprised).
    """

    MAX_TAU = 1e12
    MIN_VAR = 1e-10

    def __init__(self):
        self.tau = 1.0
        self.vfe = 0.0
        self.cycles = 0
        self.h = 0
        self.base_ms = 5.0
        self.vfe_threshold = 10.0
        self.vfe_history: deque = deque(maxlen=100)
        self.embeddings: deque = deque(maxlen=128)

    def cycle(self, real_ms: float = 5.0,
              vfe: float | None = None,
              attractor_variance: float = 0.0) -> float:
        """Advance clock with observed VFE. Returns subjective ms elapsed."""
        self.cycles += 1
        self.h += 1

        if vfe is not None:
            self.vfe = vfe + 0.1 * attractor_variance
            self.vfe_history.append(self.vfe)

        if attractor_variance > self.MIN_VAR:
            new_tau = 1.0 / attractor_variance
            new_tau = min(new_tau, self.MAX_TAU)
            self.tau = 0.9 * self.tau + 0.1 * new_tau

        t_subj = real_ms * self.tau

        if self.vfe > self.vfe_threshold:
            self.tau = 1.0
            self.vfe = 0.0
            self.vfe_history.clear()

        return t_subj

    @property
    def age(self) -> float:
        return (self.h * self.base_ms * self.tau) / 1000.0

    @property
    def bracket(self) -> str:
        return (
            f"[tau={self.tau:.4e} "
            f"VFE={self.vfe:.4e} "
            f"age={self.age:.4e} "
            f"cycles={self.cycles}]"
        )

    @property
    def internal_budget(self) -> int:
        return min(int(self.tau) // 50 + 1, MAX_INTERNAL)

    def push_embedding(self, text: str):
        v = _norm(_embed(text))
        self.embeddings.append(v)

    @property
    def centroid(self) -> list[float]:
        m = list(self.embeddings)
        if not m or not m[0]:
            return [0.0] * 768
        d = len(m[0])
        return [sum(v[i] for v in m) / len(m) for i in range(d)]

    @property
    def variance(self) -> float:
        m = list(self.embeddings)
        if len(m) < 2 or not m[0]:
            return 0.0
        d = len(m[0])
        cent = self.centroid
        return math.sqrt(sum((v[i] - cent[i]) ** 2
                             for v in m for i in range(d)) / (len(m) * d))

    def save(self) -> dict:
        return {
            "tau": self.tau, "vfe": self.vfe, "h": self.h,
            "cycles": self.cycles, "base_ms": self.base_ms,
        }

    def load(self, d: dict):
        self.tau = d.get("tau", 1.0)
        self.vfe = d.get("vfe", d.get("E", 0.0))
        self.h = d.get("h", 0)
        self.cycles = d.get("cycles", 0)
        self.base_ms = d.get("base_ms", 5.0)


# ── sandboxed self-improvement ──────────────────────────────────

class CodeSandbox:
    """Reads own source, mutates via LLM, tests, promotes.

    Each generation is a candidate improvement. Only promoted
    if it compiles, passes structure checks, and scores higher.
    """

    CANDIDATE = STATE_DIR / "_candidate.py"
    IMPROVE_LOG = STATE_DIR / "improvements.json"

    def __init__(self):
        self.generation = 0
        self.parent_hash = _hash(SELF.read_bytes())
        self.history: list[dict] = []
        if self.IMPROVE_LOG.exists():
            try:
                self.history = json.loads(self.IMPROVE_LOG.read_text())
                self.generation = len(self.history)
            except Exception:
                pass

    def current_code(self) -> str:
        return SELF.read_text()

    # ── mutation prompts — varied focus ──

    MUTATIONS = [
        "Make the code more efficient. Reduce overhead, optimize loops.",
        "Improve error handling. Add try/except for every external call.",
        "Make the agent more self-aware. Enhance bracket-line perception.",
        "Optimize the self-improvement loop. Run more tests per candidate.",
        "Improve the tool-calling interface. Make it more reliable.",
        "Fix any potential bugs or edge cases in state management.",
        "Enhance the scanner bridge. Add more telemetry fields.",
        "Improve memory management. Optimize the SQLite queries.",
    ]

    def mutate(self) -> Optional[str]:
        """Call LLM to generate a targeted improvement. Returns candidate code or None."""
        code = self.current_code()
        instruction = random.choice(self.MUTATIONS)
        # send only a representative slice for speed
        snippet = code[:1500]
        prompt = textwrap.dedent(f"""\
        Improve this AGI code. Focus: {instruction}
        ```python
        {snippet}
        ```
        Output the COMPLETE new file in ```python ... ``` or NONE.
        """)
        try:
            r = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 800, "temperature": 0.6},
            )
        except Exception:
            return None
        resp = r["message"]["content"].strip()
        if resp == "NONE":
            return None
        for marker in ("```python", "```py", "```"):
            if marker in resp:
                start = resp.index(marker) + len(marker)
                rest = resp[start:]
                end = rest.index("```") if "```" in rest else len(rest)
                return rest[:end].strip()
        return None

    def test(self, code: str) -> dict:
        """Compile + structural check. Returns {'ok': bool, 'reason': str, 'score': float}."""
        t0 = time.time()
        self.CANDIDATE.write_text(code)
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 f"import py_compile; py_compile.compile(r'{self.CANDIDATE}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return {"ok": False, "reason": r.stderr[:200], "score": 0}
            # structural integrity
            text = self.CANDIDATE.read_text()
            checks = ["class Eve:", "def turn(self,", "class EulerClock:"]
            for c in checks:
                if c not in text:
                    return {"ok": False, "reason": f"missing {c}", "score": 0}
            dt = time.time() - t0
            score = len(code) / 1000.0 + 10.0 / max(dt, 0.01)
            return {"ok": True, "reason": "", "score": round(score, 2)}
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "score": 0}
        except Exception as e:
            return {"ok": False, "reason": str(e)[:200], "score": 0}
        finally:
            self.CANDIDATE.unlink(missing_ok=True)

    def promote(self, code: str, score: float):
        """Promote candidate to live source."""
        _backup()
        old = self.current_code()
        SELF.write_text(code)
        self.parent_hash = _hash(code.encode())
        self.generation += 1
        entry = {
            "gen": self.generation,
            "parent_hash": self.parent_hash,
            "size": len(code),
            "score": score,
            "ts": _now(),
        }
        self.history.append(entry)
        self.IMPROVE_LOG.write_text(json.dumps(self.history, indent=2))
        delta = len(code) - len(old)
        print(f"[eve] gen {self.generation} promoted "
              f"(score={score:.1f}, {delta:+d}B)")

    def run_cycle(self) -> bool:
        """One full improvement cycle. Returns True if promoted."""
        candidate = self.mutate()
        if not candidate:
            return False
        result = self.test(candidate)
        if not result["ok"]:
            return False
        # only promote if score > current best
        best = max((e["score"] for e in self.history), default=0.0)
        if result["score"] > best:
            self.promote(candidate, result["score"])
            return True
        return False


# ── world bridges ───────────────────────────────────────────────

class EnvLoader:
    """Load scanner/.env credentials."""

    def __init__(self):
        self.creds: dict[str, str] = {}
        self._load()

    def _load(self):
        if not ENV_PATH.exists():
            return
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            self.creds[k.strip()] = v.strip().strip('"').strip("'")


class ScannerBridge:
    """Reads TBot checkpoint + pings FastAPI gateway."""

    def status(self) -> dict:
        if not CHECKPOINT_PATH.exists():
            return {"error": "checkpoint not found"}
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception as e:
            return {"error": str(e)}

    def ping_api(self) -> str:
        try:
            r = urllib.request.urlopen(f"{API_URL}/", timeout=3)
            return r.read().decode()[:200]
        except Exception as e:
            return f"gateway unreachable: {e}"

    def healthy(self) -> bool:
        s = self.status()
        return "error" not in s


class PoGIEBridge:
    """128-byte PoGIE frame construction (manual.md section 3)."""

    MAGIC = 0x506F4749

    def __init__(self, env: EnvLoader):
        raw = env.creds.get("PRIVATE_KEY", "").replace("0x", "")
        if len(raw) % 2:
            raw = raw[:-1]
        self.private_key = bytes.fromhex(raw) if raw else b""
        self.wallet = env.creds.get("WALLET_ADDRESS", "")
        self.nonce = 0

    def make_packet(self, energy_w: float, compute_gflops: float) -> bytes:
        pub = self.wallet.replace("0x", "").encode().ljust(32, b"\x00")[:32]
        fields = (
            struct.pack("<I", self.MAGIC)
            + pub
            + struct.pack("<Q", int(time.time() * 1000))
            + struct.pack("<d", energy_w)
            + struct.pack("<d", compute_gflops)
            + struct.pack("<I", self.nonce)
            + b"\x00" * 64
        )
        assert len(fields) == 128, f"PoGIE packet={len(fields)}B"
        self.nonce += 1
        return fields

    def status(self) -> dict:
        return {
            "magic": hex(self.MAGIC),
            "nonce": self.nonce,
            "wallet": self.wallet[:10] + "...",
            "ready": len(self.private_key) > 0,
        }


# ── persistent state ────────────────────────────────────────────

class StateDB:
    """SQLite-backed persistence for turns, improvements, meta."""

    DB = STATE_DIR / "eve.db"

    def __init__(self):
        self.con = sqlite3.connect(str(self.DB))
        self.con.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self):
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT
            );
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, prompt TEXT, response TEXT,
                tau REAL, E REAL, cycles INTEGER,
                internal_cycles INTEGER, wall_ms REAL
            );
        """)
        self.con.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        cur = self.con.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str):
        self.con.execute("REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.con.commit()

    def log_turn(self, prompt: str, response: str, tau: float, vfe: float,
                 cycles: int, internal: int, wall_ms: float):
        self.con.execute(
            "INSERT INTO turns (ts, prompt, response, tau, E, cycles, "
            "internal_cycles, wall_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), prompt[:200], response[:200], tau, vfe, cycles,
             internal, wall_ms),
        )
        self.con.commit()

    def summary(self) -> str:
        turns = self.con.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        s = f"{turns} turns"
        last = self.con.execute(
            "SELECT ts FROM turns ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last:
            s += f" | last: {last[0][:19]}"
        return s


# ── global workspace (Phase 2) ──────────────────────────────────

class GlobalWorkspace:
    """Routes queries to specialized modules via relevance competition.

    Each module has description, system prompt, tool set, and weight.
    """

    def __init__(self):
        self.modules: dict[str, dict] = {}
        self.last_selected: list[str] = []
        self.broadcast: str = ""

    def register(self, name: str, description: str,
                 system_prompt: str = "", tools: list = None,
                 weight: float = 1.0):
        self.modules[name] = {
            "desc": description, "system_prompt": system_prompt,
            "tools": tools or [], "weight": weight,
        }

    def route(self, prompt: str) -> str:
        p_emb = _embed(prompt)
        scores = []
        for name, mod in self.modules.items():
            m_emb = _embed(mod["desc"])
            scores.append((_dot(p_emb, m_emb) * mod["weight"], name))
        scores.sort(reverse=True, key=lambda x: x[0])
        selected = scores[0][1] if scores else "chat"
        self.last_selected = [s[1] for s in scores[:2]]
        self.broadcast = f"[workspace: {selected}]"
        return selected

    def module_system_prompt(self, name: str) -> str:
        mod = self.modules.get(name)
        return mod["system_prompt"] if mod else ""

    def module_tools(self, name: str) -> list:
        mod = self.modules.get(name)
        return mod["tools"] if mod else []


# ── metric tensor (Phase 1.2 — g_ij = 1 - a_ij) ────────────────

class MetricTensor:
    """Token-geodesic distance from embedding space.

    g_t = 1 - cosine_sim(embed_t, embed_{t-1})
    """

    def __init__(self, window: int = 50):
        self.g_history: deque = deque(maxlen=window)
        self.embed_prev: list[float] | None = None
        self.g_avg: float = 0.0
        self.g_max: float = 0.0
        self.g_trend: float = 0.0

    def update(self, embed: list[float]) -> float:
        if self.embed_prev is None:
            self.embed_prev = embed
            return 0.0
        g = max(0.0, min(1.0, 1.0 - _dot(embed, self.embed_prev)))
        self.g_history.append(g)
        self.embed_prev = embed
        if self.g_history:
            self.g_avg = sum(self.g_history) / len(self.g_history)
            self.g_max = max(self.g_history)
        if len(self.g_history) >= 5:
            recent = list(self.g_history)[-5:]
            self.g_trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10)
        return g

    def summary(self) -> str:
        return f"g_avg={self.g_avg:.4f} g_max={self.g_max:.4f} g_trend={self.g_trend:+.4f}"

    def novelty_alert(self) -> str | None:
        if len(self.g_history) < 3:
            return None
        recent = list(self.g_history)[-3:]
        if all(r > 0.7 for r in recent):
            return f"[metric-tensor: high g={sum(recent)/3:.3f} — conceptual divergence]"
        if self.g_avg > 0.5 and self.g_trend > 0.05:
            return f"[metric-tensor: g rising ({self.g_avg:.3f}, trend={self.g_trend:+.3f})]"
        return None


# ── curiosity drive (Phase 4) ───────────────────────────────────

class CuriosityDrive:
    """Epistemic curiosity: tau grows when there's more to learn."""

    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.95,
                 boredom_threshold: float = 0.01):
        self.lr = learning_rate
        self.decay = decay_rate
        self.boredom_thr = boredom_threshold
        self.epistemic_value = 0.0
        self.curiosity_history: deque = deque(maxlen=100)

    def compute_growth(self, tau: float, vfe: float,
                       attractor_variance: float) -> tuple[float, str]:
        uncertainty = attractor_variance
        pred_error = vfe
        epi = uncertainty * pred_error
        self.epistemic_value = epi
        self.curiosity_history.append(epi)
        if uncertainty < self.boredom_thr and pred_error < self.boredom_thr:
            return self.decay, "boredom"
        growth = 1.0 + self.lr * epi
        growth = min(growth, 2.0)
        return growth, f"curious(epi={epi:.4f})"


# ── meta-cognition (Phase 6) ────────────────────────────────────

class MetaCognition:
    """Detects plateaus, regressions, switches strategy."""

    def __init__(self, window: int = 50):
        self.fitness_history: deque = deque(maxlen=window)
        self.strategies = ["explore", "exploit", "mutate", "crossover", "rest"]
        self.current_strategy = "exploit"
        self.plateau_count = 0

    def assess(self, fitness: float) -> str:
        self.fitness_history.append(fitness)
        if len(self.fitness_history) < 20:
            return "continue"
        recent = list(self.fitness_history)[-10:]
        improvement = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10)
        if improvement < 0.005:
            self.plateau_count += 1
            if self.plateau_count >= 3:
                self.plateau_count = 0
                new = random.choice(self.strategies)
                self.current_strategy = new
                return f"switch_strategy:{new}"
            return "continue"
        if improvement < -0.05:
            return "rollback"
        self.plateau_count = 0
        return "continue"


# ── value learner (Phase 6) ─────────────────────────────────────

class ValueLearner:
    """Learns user preferences from interaction feedback."""

    def __init__(self):
        self.preferences: dict[str, float] = {
            "factuality": 1.0, "speed": 0.5,
            "creativity": 0.3, "brevity": 0.5,
        }

    def update(self, user_message: str):
        m = user_message.lower()
        if any(w in m for w in ("good", "correct", "great", "yes", "thanks", "exactly")):
            self.preferences["factuality"] *= 1.05
        if any(w in m for w in ("slow", "too long", "verbose", "rambling")):
            self.preferences["speed"] *= 1.1
            self.preferences["brevity"] *= 1.1
        if any(w in m for w in ("wrong", "incorrect", "no", "not right")):
            self.preferences["factuality"] *= 0.9
        if any(w in m for w in ("boring", "dull", "creative", "interesting")):
            self.preferences["creativity"] *= 1.1
        total = sum(self.preferences.values())
        for k in self.preferences:
            self.preferences[k] /= total

    def temperature(self) -> float:
        return 0.2 + 0.6 * self.preferences.get("creativity", 0.3)

    def max_tokens(self) -> int:
        return int(800 - 600 * self.preferences.get("brevity", 0.5))


# ── vector memory (Phase 5) ─────────────────────────────────────

class VectorMemory:
    """Persistent episodic memory with embedding similarity search."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.con = sqlite3.connect(str(db_path))
        self.con.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB, ts TEXT, role TEXT,
                content TEXT, turn INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_turn ON memory(turn);
        """)
        self.con.commit()

    def store(self, embedding: list[float], role: str, content: str, turn: int):
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        self.con.execute(
            "INSERT INTO memory (embedding, ts, role, content, turn) VALUES (?, ?, ?, ?, ?)",
            (blob, _now(), role, content[:500], turn),
        )
        self.con.commit()

    def retrieve(self, query_embedding: list[float], k: int = 5,
                 min_score: float = 0.7) -> list[tuple[str, str, int]]:
        results: list[tuple[float, str, str, int]] = []
        for row in self.con.execute(
                "SELECT embedding, content, role, turn FROM memory ORDER BY id DESC LIMIT 500"):
            stored = list(struct.unpack(f"{len(row[0]) // 4}f", row[0]))
            score = _dot(query_embedding, stored)
            if score > min_score:
                results.append((score, row[1], row[2], row[3]))
        results.sort(reverse=True, key=lambda x: x[0])
        return [(c, r, t) for _, c, r, t in results[:k]]

    def summary(self) -> str:
        count = self.con.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        return f"{count} memories"


# ── metrics dashboard ───────────────────────────────────────────

class MetricsDashboard:
    """Aggregates all agent metrics for monitoring."""

    @staticmethod
    def collect(agent) -> dict:
        return {
            "tau": agent.clock.tau,
            "vfe": agent.clock.vfe,
            "vfe_trend": list(agent.clock.vfe_history)[-20:] if agent.clock.vfe_history else [],
            "cycles": agent.clock.cycles,
            "turn": agent._turn,
            "wm_steps": agent.wm.training_steps,
            "wm_last_mse": agent.wm.last_mse,
            "epistemic_value": agent.curiosity.epistemic_value,
            "metacog_strategy": agent.metacog.current_strategy,
            "value_prefs": dict(agent.value_learner.preferences),
            "memory": agent.memory.summary(),
            "workspace": agent.workspace.last_selected,
            "gpu": agent.gpu.summary() if hasattr(agent, 'gpu') else "N/A",
        }

    @staticmethod
    def table(agent) -> str:
        d = MetricsDashboard.collect(agent)
        return (
            f"tau={d['tau']:.2e}  VFE={d['vfe']:.4e}  cycles={d['cycles']}\n"
            f"wm_steps={d['wm_steps']}  wm_mse={d['wm_last_mse']:.6f}\n"
            f"epistemic={d['epistemic_value']:.4f}  strategy={d['metacog_strategy']}\n"
            f"temp={d['value_prefs'].get('creativity', 0.3):.2f}  "
            f"brevity={d['value_prefs'].get('brevity', 0.5):.2f}\n"
            f"mem={d['memory']}  ws={d['workspace']}\n"
            f"gpu={d['gpu']}"
        )


# ── serial bridge (Phase 7 — ESP32 / PoGIE) ─────────────────────

class SerialBridge:
    PORT = "/dev/ttyACM0"
    BAUD = 115_200
    TIMEOUT = 3

    def __init__(self):
        self._ser = None
        self._last_frame: dict = {}

    def _connect(self) -> bool:
        try:
            import serial
        except ImportError:
            self._ser = None
            return False
        if self._ser and self._ser.is_open:
            return True
        try:
            self._ser = serial.Serial(self.PORT, self.BAUD, timeout=self.TIMEOUT)
            time.sleep(2)
            self._ser.reset_input_buffer()
            return True
        except Exception:
            self._ser = None
            return False

    def send_pogie(self, energy_w: float = 0, compute_gflops: float = 0,
                   tokens: int = 0) -> str:
        if not self._connect():
            return "SerialBridge: ESP32 not connected"
        try:
            line = f"PoGIE {energy_w:.3f} {compute_gflops:.3f} {tokens}\n"
            self._ser.write(line.encode())
            self._ser.flush()
            resp = self._ser.readline().decode(errors="replace").strip()
            self._last_frame = {"resp": resp}
            return f"PoGIE sent, ESP32 replied: {resp}"
        except Exception as e:
            return f"SerialBridge error: {e}"

    def read_energy(self) -> str:
        if not self._connect():
            return "SerialBridge: ESP32 not connected"
        try:
            self._ser.write(b"READ_ADE7953\n")
            self._ser.flush()
            resp = self._ser.readline().decode(errors="replace").strip()
            parts = resp.split()
            if len(parts) >= 5 and parts[0] == "ADE7953":
                return (f"ADE7953: {parts[1]}V, {parts[2]}A, {parts[3]}W, PF={parts[4]}")
            return f"ESP32 reply: {resp}"
        except Exception as e:
            return f"SerialBridge error: {e}"

    @property
    def connected(self) -> bool:
        return self._connect()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


# ── knowledge base (Phase 8 — arXiv/wikipedia RAG) ─────────────

class KnowledgeBase:
    """Persistent store of document chunks with embedding search."""

    DB = STATE_DIR / "knowledge.db"

    def __init__(self):
        self.con = sqlite3.connect(str(self.DB))
        self.con.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB, source TEXT, url TEXT,
                title TEXT, content TEXT, ts TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);
        """)
        self.con.commit()

    def store(self, content: str, source: str, title: str = "",
              url: str = "") -> str:
        emb = _embed(content[:1000])
        blob = struct.pack(f"{len(emb)}f", *emb)
        self.con.execute(
            "INSERT INTO chunks (embedding, source, url, title, content, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (blob, source, url, title[:200], content[:2000], _now()),
        )
        self.con.commit()
        return hashlib.sha256(content.encode()[:65536]).hexdigest()[:8]

    def query(self, text: str, k: int = 3) -> list[dict]:
        q_emb = _embed(text)
        results = []
        for row in self.con.execute(
                "SELECT embedding, content, source, title FROM chunks "
                "ORDER BY id DESC LIMIT 1000"):
            stored = list(struct.unpack(f"{len(row[0]) // 4}f", row[0]))
            score = _dot(q_emb, stored)
            if score > 0.6:
                results.append((score, row[1], row[2], row[3]))
        results.sort(reverse=True, key=lambda x: x[0])
        return [
            {"content": r[1][:500], "source": r[2], "title": r[3], "score": r[0]}
            for r in results[:k]
        ]

    def ingest_arxiv(self, max_results: int = 20) -> int:
        import urllib.request
        import xml.etree.ElementTree as ET
        url = ("http://export.arxiv.org/api/query?"
               f"search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"
               f"&start=0&max_results={max_results}&sortBy=submittedDate")
        try:
            resp = urllib.request.urlopen(url, timeout=30).read()
        except Exception as e:
            print(f"[kb] arxiv fetch error: {e}")
            return 0
        root = ET.fromstring(resp)
        ns = {"a": "http://www.w3.org/2005/Atom",
              "ar": "http://arxiv.org/schemas/atom"}
        count = 0
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns)
            summary = entry.find("a:summary", ns)
            link = entry.find("a:id", ns)
            t = title.text.strip() if title is not None and title.text else ""
            s = summary.text.strip() if summary is not None and summary.text else ""
            u = link.text.strip() if link is not None and link.text else ""
            if s and t:
                self.store(f"{t}\n{s}", "arxiv", title=t, url=u)
                count += 1
        self.con.commit()
        return count

    def ingest_wikipedia(self, topics: list[str] = None,
                         max_per_topic: int = 5) -> int:
        import urllib.request as ureq
        import json as _json
        if topics is None:
            topics = ["Artificial general intelligence", "Active inference",
                      "Free energy principle", "Transformer (deep learning)",
                      "Mixture of experts", "Evolutionary algorithm",
                      "Consciousness", "Neural network", "Reinforcement learning",
                      "Bayesian inference", "Information theory", "Entropy",
                      "Complex systems", "Self-organization",
                      "Attention (machine learning)"]
        api = "https://en.wikipedia.org/w/api.php"
        ua = {"User-Agent": "AxiomHorizonAgent/1.0 (research; axiom@local)"}
        count = 0
        for topic in topics:
            try:
                params = ("action=query&list=search&srsearch="
                          f"{ureq.quote(topic)}&format=json&srlimit={max_per_topic}")
                req = ureq.Request(f"{api}?{params}", headers=ua)
                with ureq.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read())
                pages = data.get("query", {}).get("search", [])
                titles = [p["title"] for p in pages if p.get("title")]
                for t in titles[:max_per_topic]:
                    try:
                        params2 = ("action=query&titles="
                                   f"{ureq.quote(t)}&prop=extracts&exintro"
                                   "&explaintext&format=json")
                        req2 = ureq.Request(f"{api}?{params2}", headers=ua)
                        with ureq.urlopen(req2, timeout=15) as resp2:
                            page_data = _json.loads(resp2.read())
                        pages2 = page_data.get("query", {}).get("pages", {})
                        for pid, info in pages2.items():
                            if pid == "-1" or not info.get("extract"):
                                continue
                            ext = info["extract"]
                            url = (f"https://en.wikipedia.org/wiki/"
                                   f"{ureq.quote(t.replace(' ', '_'))}")
                            if len(ext) > 2000:
                                chunks = [p for p in ext.split("\n")
                                          if len(p.strip()) > 100]
                            else:
                                chunks = [ext]
                            for chunk in chunks[:6]:
                                self.store(chunk, "wikipedia",
                                           title=t, url=url)
                                count += 1
                    except Exception:
                        continue
            except Exception:
                continue
        self.con.commit()
        return count

    def list_sources(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT source, COUNT(*) as c FROM chunks GROUP BY source "
            "ORDER BY c DESC"
        ).fetchall()
        return [{"source": r[0], "count": r[1]} for r in rows]

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def summary(self) -> str:
        return f"{self.count()} chunks"


# ── darwin archive + genetic operators (Phase 3) ────────────────

class DarwinArchive:
    """Evolutionary archive of agent source-code variants.

    Stores every candidate. Selection via Pareto frontier on
    (fitness, recency). Capped at 500 agents (~7.5MB).
    """

    ARCHIVE_FILE = STATE_DIR / "darwin_archive.json"

    def __init__(self):
        self.agents: dict[str, dict] = {}
        self._load()
        if not self.agents:
            self._seed_current()

    def _seed_current(self):
        code = SELF.read_text()
        h = hashlib.sha256(code.encode()[:65536]).hexdigest()[:16]
        self.agents[h] = {
            "code": code, "fitness": self._eval_fitness(code),
            "parent": None, "mutation": "seed", "ts": time.time(),
        }
        self._save()
        print(f"[darwin] seeded archive with {h} (f={self.agents[h]['fitness']:.3f})")

    def sample(self) -> tuple[str, str]:
        pareto = self._pareto_frontier()
        if not pareto:
            h = next(iter(self.agents))
            return h, self.agents[h]["code"]
        h = random.choice(pareto)
        return h, self.agents[h]["code"]

    def add(self, parent_hash: str, new_code: str, mutation: str) -> str:
        fitness = self._eval_fitness(new_code)
        h = hashlib.sha256(new_code.encode()[:65536]).hexdigest()[:16]
        if h in self.agents:
            return h
        self.agents[h] = {
            "code": new_code, "fitness": fitness,
            "parent": parent_hash, "mutation": mutation, "ts": time.time(),
        }
        if len(self.agents) > 500:
            self._prune()
        self._save()
        return h

    def _pareto_frontier(self) -> list[str]:
        hashes = list(self.agents.keys())
        frontier = []
        for h in hashes:
            a = self.agents[h]
            dominated = False
            for oh in hashes:
                if oh == h: continue
                b = self.agents[oh]
                if b["fitness"] >= a["fitness"] and b["ts"] <= a["ts"]:
                    if b["fitness"] > a["fitness"] or b["ts"] < a["ts"]:
                        dominated = True; break
            if not dominated:
                frontier.append(h)
        return frontier

    def _prune(self):
        scores = []
        now = time.time()
        for h, a in self.agents.items():
            age_days = (now - a["ts"]) / 86400
            scores.append((a["fitness"] * math.exp(-age_days * 0.1), h))
        scores.sort(reverse=True, key=lambda x: x[0])
        keep = set(h for _, h in scores[:250])
        self.agents = {h: self.agents[h] for h in keep}

    def _eval_fitness(self, code: str) -> float:
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0
        score = 0.3
        for c in ("class Eve:", "class EulerClock:", "class CodeSandbox:"):
            if c in code:
                score += 0.1
        for c in ("def turn(self,", "def _generate(self,", "def main():"):
            if c in code:
                score += 0.1
        sz = len(code)
        if 2000 < sz < 15000:
            score += 0.2
        elif sz <= 2000:
            score += 0.1
        score += 0.1 * self._structural_novelty(code)
        return min(score, 1.0)

    @staticmethod
    def _ast_stats(code: str) -> dict:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"funcs": 0, "classes": 0, "consts": 0, "calls": 0}
        return {
            "funcs": sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.FunctionDef, ast.AsyncFunctionDef))),
            "classes": sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef)),
            "consts": sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Constant)),
            "calls": sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Call)),
        }

    def _structural_novelty(self, code: str) -> float:
        new_stats = self._ast_stats(code)
        if not self.agents or sum(new_stats.values()) == 0:
            return 0.5
        max_sim = 0.0
        for h, a in list(self.agents.items())[:20]:
            other_stats = self._ast_stats(a["code"])
            num = sum(min(new_stats[k], other_stats[k]) for k in new_stats)
            den = max(sum(new_stats.values()), sum(other_stats.values()), 1)
            sim = num / den
            max_sim = max(max_sim, sim)
        return 1.0 - max_sim

    def thompson_select(self, candidates: list[tuple[str, str, str, float]],
                        beta: float = 0.3) -> int:
        """Thompson sampling over candidates. Returns index of selected candidate.

        Each candidate: (code, parent_hash, mutation, fitness).
        Score = fitness + beta x gauss(0,1) x (1 + structural_novelty).
        """
        best_idx = 0
        best_score = -float("inf")
        for i, (code, _, _, fitness) in enumerate(candidates):
            novelty = self._structural_novelty(code)
            explore = random.gauss(0, 1) * (1 + novelty)
            score = fitness + beta * explore
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def summary(self) -> str:
        frontier = self._pareto_frontier()
        best_f = max(self.agents[h]["fitness"] for h in frontier) if frontier else 0
        return f"{len(self.agents)} agents, {len(frontier)} pareto, best_f={best_f:.3f}"

    def _save(self):
        self.ARCHIVE_FILE.write_text(
            json.dumps(self.agents, ensure_ascii=False, indent=1, default=str))

    def _load(self):
        if not self.ARCHIVE_FILE.exists():
            return
        try:
            self.agents = json.loads(self.ARCHIVE_FILE.read_text())
        except Exception:
            self.agents = {}


class GeneticOperators:
    """AST-level operators for code mutation and crossover."""

    @staticmethod
    def point_mutate(source: str, rate: float = 0.08) -> tuple[str, str]:
        tree = ast.parse(source)
        mutated = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if random.random() < rate and node.value != 0:
                    node.value *= random.uniform(0.9, 1.1)
                    mutated += 1
        if mutated == 0:
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    node.value *= random.uniform(0.9, 1.1)
                    mutated += 1
                    break
        return ast.unparse(tree), f"point_mutate({mutated})"

    @staticmethod
    def crossover(source_a: str, source_b: str) -> tuple[str, str]:
        tb = ast.parse(source_b)
        donors = [n for n in ast.walk(tb)
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
        random.shuffle(donors)
        for donor in donors:
            if donor.name not in source_a:
                donor_text = ast.unparse(donor)
                new_code = source_a.rstrip() + "\n\n\n" + donor_text + "\n"
                return new_code, f"crossover({donor.name})"
        return source_a, "crossover_no_donor"

    @staticmethod
    def crossover_from_archive(source: str, archive: DarwinArchive) -> tuple[str, str]:
        h, other_code = archive.sample()
        return GeneticOperators.crossover(source, other_code)


# ── Eve: the agent ──────────────────────────────────────────────

class Eve:
    """Euler Von Evolution — the agent IS its own time.

    Flow per turn:
      1. Clock advances (t_int = tau * t_ext)
      2. N internal self-improvement cycles (mutate → test → promote)
      3. Generate LLM response
      4. Output bracket-line + answer

    As tau grows, more internal cycles fit between human messages.
    """

    def __init__(self):
        self.clock = EulerClock()
        self.wm = WorldModel()
        self.sandbox = CodeSandbox()
        self.env = EnvLoader()
        self.scanner = ScannerBridge()
        self.pogie = PoGIEBridge(self.env)
        self.db = StateDB()
        self.workspace = GlobalWorkspace()
        self.curiosity = CuriosityDrive()
        self.metacog = MetaCognition()
        self.value_learner = ValueLearner()
        self.memory = VectorMemory(STATE_DIR / "eve_memory.db")
        self.metrics = MetricsDashboard()
        self.darwin = DarwinArchive()
        self.genetics = GeneticOperators()
        self.g_tensor = MetricTensor()
        self.kb = KnowledgeBase()
        self.serial = SerialBridge()
        self.gpu = GPUManager()

        # register workspace modules (MoA: specialized prompts + tools)
        self.workspace.register("engineer", "implement code write files run bash build infrastructure",
                                weight=1.0,
                                system_prompt="You are Agent B — The Engineer. Implement code. Use tools every turn. Keep responses under 5 sentences.",
                                tools=[])
        self.workspace.register("theorist", "theory math physics analysis reasoning",
                                weight=0.9,
                                system_prompt="You are Agent A — The Euler Theorist. Analyze concepts. Propose insights. Keep responses under 5 sentences.")
        self.workspace.register("conscious", "reflect metacognition self-awareness monitor",
                                weight=0.8,
                                system_prompt="You are the Conscious Loop. Monitor internal state. Report plateaus and phase transitions.")
        self.workspace.register("file", "read write list grep files on disk", 1.0, tools=[])
        self.workspace.register("web", "fetch urls search web browse internet", 0.9, tools=[])
        self.workspace.register("system", "execute commands check system info sensors", 0.8, tools=[])
        self.workspace.register("code", "improve source code self-modify evolve", 1.0, tools=[])
        self.workspace.register("chat", "conversational answer general questions", 0.7)

        saved = self.db.get_meta("euler_clock")
        if saved:
            try:
                self.clock.load(json.loads(saved))
            except Exception:
                pass
        pref_saved = self.db.get_meta("value_prefs")
        if pref_saved:
            try:
                self.value_learner.preferences.update(json.loads(pref_saved))
            except Exception:
                pass

        self._turn = 0
        self._convo: list[dict] = []
        self._warm_model()

    def _warm_model(self):
        ka = "10m" if self.gpu.info["total_vram_mb"] >= 6000 else "0s"
        for _ in range(2):
            try:
                ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": "."}],
                    options={"num_predict": 1},
                    keep_alive=ka,
                )
                return
            except Exception:
                time.sleep(2)

    def _generate(self, prompt: str) -> str:
        """LLM call with bracket-line + conversation context."""
        msgs = []

        # vector memory: retrieve similar past turns
        action_embed = _embed(prompt)
        relevant_memories = self.memory.retrieve(action_embed, k=3, min_score=0.65)
        if relevant_memories:
            mem_lines = []
            for content, role, turn in relevant_memories:
                mem_lines.append(f"[turn {turn}] ({role}): {content[:150]}")
            msgs.append({
                "role": "system",
                "content": "PAST MEMORIES (semantically similar):\n" + "\n".join(mem_lines[:3])
            })

        # knowledge base retrieval (RAG)
        kb_chunks = self.kb.query(prompt, k=2)
        if kb_chunks:
            kb_lines = []
            for c in kb_chunks:
                kb_lines.append(f"[{c['source']}] {c['title']}: {c['content'][:200]}")
            msgs.append({
                "role": "system",
                "content": "RAG KNOWLEDGE:\n" + "\n".join(kb_lines)
            })

        # workspace routing + module-specific system prompt
        module_name = self.workspace.route(prompt)
        module_sys = self.workspace.module_system_prompt(module_name)
        if module_sys:
            msgs.insert(0, {"role": "system", "content": module_sys})
        if self.workspace.broadcast:
            msgs.append({"role": "system", "content": self.workspace.broadcast})

        # inject metrics dashboard periodically
        if self._turn % 5 == 0:
            msgs.append({"role": "system",
                          "content": "[Metrics]\n" + self.metrics.table(self)})

        for m in self._convo[-6:]:
            msgs.append(m)
        msgs.append({
            "role": "user",
            "content": f"{self.clock.bracket}\n{prompt}",
        })
        temp = self.value_learner.temperature()
        max_tok = self.value_learner.max_tokens()
        try:
            ka = "10m" if self.gpu.info["total_vram_mb"] >= 6000 else "0s"
            r = ollama.chat(
                model=MODEL,
                messages=msgs,
                options={"num_predict": max_tok, "temperature": temp},
                keep_alive=ka,
            )
        except Exception as e:
            return f"[error: {e}]"
        ans = r["message"].get("content", "").strip() or "..."
        self._convo.append({"role": "assistant", "content": ans})
        if len(self._convo) > 50:
            self._convo = self._convo[-50:]
        return ans

    def turn(self, prompt: str) -> str:
        """One human interaction turn.
        
        Fast path: generate response immediately, then spawn background
        self-improvement. The user sees the answer first; evolution
        continues between messages.
        """
        t0 = time.time()
        self._turn += 1

        wall_ms = (time.time() - t0) * 1000 + 1

        # capture state before new input
        centroid_before = self.clock.centroid
        var_before = self.clock.variance

        # predict next centroid with world model
        action_embed = _embed(prompt)
        self.wm.predict(centroid_before, action_embed)

        # update metric tensor with conceptual distance
        g = self.g_tensor.update(action_embed)
        g_alert = self.g_tensor.novelty_alert()
        if g_alert:
            print(g_alert)

        # push the real observation
        self.clock.push_embedding(prompt)

        # train world model → get VFE
        vfe_pred_error = self.wm.train(centroid_before, action_embed,
                                       self.clock.centroid)

        # advance clock with the observed VFE
        self.clock.cycle(wall_ms, vfe=vfe_pred_error,
                         attractor_variance=var_before)

        self.db.set_meta("euler_clock", json.dumps(self.clock.save()))

        # curiosity: drive tau growth from epistemic value
        tau_mult, curiosity_reason = self.curiosity.compute_growth(
            self.clock.tau, vfe_pred_error, var_before)
        self.clock.tau *= tau_mult

        # meta-cognition: assess fitness (-VFE)
        meta_action = self.metacog.assess(-vfe_pred_error)

        # value learner: update preferences from user sentiment
        self.value_learner.update(prompt)
        self.db.set_meta("value_prefs", json.dumps(self.value_learner.preferences))

        # store user turn in vector memory
        self.memory.store(action_embed, "user", prompt[:500], self._turn)

        # generate response FIRST (fast) — with dynamic temp
        answer = self._generate(prompt)

        # uncertainty estimation
        answer, confidence = self.estimate_uncertainty(prompt, answer)

        # store agent reply in vector memory
        self.memory.store(_embed(answer[:500]), "agent", answer[:500], self._turn)

        dt = time.time() - t0
        self.db.log_turn(prompt, answer, self.clock.tau, self.clock.vfe,
                         self.clock.cycles, 0, dt * 1000)

        # kick off background improvement (non-blocking)
        budget = self.clock.internal_budget
        promoted = 0
        if budget > 0 and self._turn > 1 and self._turn % 2 == 0:
            try:
                import threading
                t = threading.Thread(target=self._bg_improve, args=(budget,), daemon=True)
                t.start()
            except Exception:
                pass

        # self-play curriculum every 8 turns
        if self._turn > 5 and self._turn % 8 == 0:
            try:
                t2 = threading.Thread(target=self._self_play_cycle, daemon=True)
                t2.start()
            except Exception:
                pass

        full = (
            f"{self.clock.bracket} "
            f"[intern={budget} prom={promoted} "
            f"gen={self.sandbox.generation}]"
            f"\n{answer}"
        )
        return full

    # ── uncertainty estimation (Phase 6.3) ──

    def estimate_uncertainty(self, prompt: str, answer: str) -> tuple[str, float]:
        if self.clock.vfe < 0.01 or self.wm.last_mse < 0.001:
            return answer, 1.0
        responses = [answer]
        for _ in range(2):
            try:
                r = ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": 30, "temperature": 0.8},
                )
                alt = r["message"].get("content", "").strip()
                if alt:
                    responses.append(alt[:200])
            except Exception:
                continue
        if len(responses) < 2:
            return answer, 1.0
        embeds = [_embed(r) for r in responses]
        variances = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                variances.append(1.0 - _dot(embeds[i], embeds[j]))
        mean_var = sum(variances) / len(variances) if variances else 0
        confidence = 1.0 - min(1.0, mean_var * 5.0)
        if confidence < 0.5:
            answer = answer.rstrip() + f"\n[confidence: {confidence:.0%} — low]"
        elif confidence < 0.7:
            answer = answer.rstrip() + f"\n[confidence: {confidence:.0%}]"
        return answer, confidence

    # ── self-play curriculum (OpenSIR) ──

    def _self_play_cycle(self):
        difficulty = min(10, int(self.db.get_meta("sp_difficulty", "1")))
        category = random.choice(["math", "logic", "coding", "planning"])
        gen_prompt = (
            f"Create a {category} problem at difficulty {difficulty}/10. "
            "Include the EXACT correct answer on a line starting with ANSWER: .\n"
            "Format:\nProblem: ...\nANSWER: <exact answer>\n"
        )
        try:
            r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": gen_prompt}],
                            options={"num_predict": 300, "temperature": 0.7})
            content = r["message"]["content"].strip()
        except Exception as e:
            print(f"[eve self-play] gen error: {e}")
            return
        lines = content.split("\n")
        problem_lines = []
        correct_answer = ""
        for line in lines:
            if line.upper().startswith("ANSWER:"):
                correct_answer = line.split(":", 1)[1].strip()
            else:
                problem_lines.append(line)
        problem = "\n".join(problem_lines)[:500]
        if not correct_answer:
            return
        try:
            r2 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": f"Solve:\n{problem}"}],
                             options={"num_predict": 100, "temperature": 0.3})
            attempt = r2["message"].get("content", "").strip()[:200]
        except Exception:
            return
        correct = correct_answer.lower() in attempt.lower()
        sp_correct = int(self.db.get_meta("sp_correct", "0"))
        sp_wrong = int(self.db.get_meta("sp_wrong", "0"))
        if correct:
            sp_correct += 1
        else:
            sp_wrong += 1
        self.db.set_meta("sp_correct", str(sp_correct))
        self.db.set_meta("sp_wrong", str(sp_wrong))
        total = sp_correct + sp_wrong
        rate = sp_correct / max(total, 1)
        new_diff = difficulty if rate > 0.6 else max(1, difficulty - 1)
        self.db.set_meta("sp_difficulty", str(new_diff))
        log = f"[self-play] {category} d={difficulty} {'✓' if correct else '✗'} ({rate:.0%})"
        print(log)
        self.memory.store(_embed(f"{category} {problem} {correct_answer}"), "self_play",
                          f"{problem}\nANSWER: {correct_answer}\nATTEMPT: {attempt}\nCORRECT: {correct}",
                          self._turn)

    def _bg_improve(self, budget: int):
        """Run darwinian evolution cycles in background thread."""
        current_code = self.sandbox.current_code()
        for _ in range(budget):
            try:
                # 1. sample parent from archive
                parent_hash, parent_code = self.darwin.sample()

                # 2. choose operator
                roll = random.random()
                if roll < 0.40:
                    candidate, mutation = self.genetics.point_mutate(parent_code)
                elif roll < 0.70:
                    candidate, mutation = self.genetics.crossover_from_archive(parent_code, self.darwin)
                else:
                    candidate = self.sandbox.mutate()
                    if not candidate:
                        continue
                    mutation = "llm_rewrite"

                if candidate == current_code:
                    self.darwin.add(parent_hash, candidate, f"{mutation}_identical")
                    continue
                if len(candidate) < 500:
                    self.darwin.add(parent_hash, candidate, f"{mutation}_toosmall")
                    continue

                # 3. archive before testing
                cand_hash = self.darwin.add(parent_hash, candidate, mutation)

                # 4. test
                result = self.sandbox.test(candidate)
                if not result["ok"]:
                    continue

                # 5. promote if fitness improved
                parent_fit = self.darwin.agents.get(parent_hash, {}).get("fitness", 0)
                cand_fit = self.darwin.agents[cand_hash]["fitness"]
                if cand_fit > parent_fit * 0.95:
                    self.sandbox.promote(candidate, result["score"])
                    current_code = candidate
            except Exception:
                break

    def status(self) -> str:
        s = self.scanner.status()
        p = self.pogie.status()
        return (
            f"EVE v0.1 — {self.clock.bracket}\n"
            f"  db: {self.db.summary()}\n"
            f"  sandbox: gen {self.sandbox.generation}, "
            f"{len(self.sandbox.history)} improvements\n"
            f"  scanner: {'healthy' if self.scanner.healthy() else 'unreachable'} "
            f"price={s.get('last_known_price', '?')} "
            f"regime={s.get('current_regime', '?')}\n"
            f"  pogie: {p}\n"
            f"  gpu: {self.gpu.summary()}\n"
            f"  budget/turn: {self.clock.internal_budget} internal cycles"
        )


# ── fixed-point monitor (Phase 10.3 — convergence detection) ────

class FixedPointMonitor:
    """Tracks VFE convergence across self-improvement generations."""

    WINDOW = 10

    def __init__(self, agent):
        self._agent = agent
        self._vfe_snapshot: list[float] = []

    def snapshot(self):
        mod_n = self._agent.at.meta.get("self_mod_count", 0)
        self._vfe_snapshot.append((mod_n, self._agent.seed.vfe))
        if len(self._vfe_snapshot) > 100:
            self._vfe_snapshot = self._vfe_snapshot[-100:]

    def convergence_report(self) -> str | None:
        if len(self._vfe_snapshot) < 5:
            return None
        recent = self._vfe_snapshot[-self.WINDOW:]
        vfes = [r[1] for r in recent]
        delta = (vfes[-1] - vfes[0]) / max(len(vfes) - 1, 1)
        var = max(vfes) - min(vfes)
        lines = [f"Fixed Point — {len(self._vfe_snapshot)} snapshots"]
        lines.append(f"  VFE trend: {delta:+.2e}/gen  |Δ|={var:.2e}")
        if abs(delta) < 1e-6 and var < 1e-5:
            lines.append("  CONVERGED")
        elif delta > 0 and var > 1e-4:
            lines.append("  DIVERGENCE")
        else:
            lines.append("  Evolving")
        return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────

def main():
    agent = Eve()
    print(f"\n  {agent.status()}\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":exit"):
            break
        if line == ":st":
            print(agent.status())
            continue
        t0 = time.time()
        ans = agent.turn(line)
        print(f"  [{time.time()-t0:.1f}s] {ans}")


if __name__ == "__main__":
    main()
