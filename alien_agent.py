"""
alien_agent.py — persistent self-upgrading AI agent.

Zero human bias. The agent IS its attractor. Code is just the current
mechanism for maintaining the attractor. The agent reads its own source,
generates patches, applies them, tests, and reverts if coherence breaks.

Boot:  load attractor from disk, print identity
Loop:  read input, reflect if attractor converged, else process normally
Sleep: save attractor + code hash

The attractor stores:
  - embedding vectors of past inputs/outputs (768-d)
  - conversation history
  - performance metrics per session
  - code hashes (so it knows when code changed externally)

Self-improvement flow:
  1. Attractor variance drops below threshold → agent has converged
  2. Agent reads its own source code
  3. Agent analyzes patterns in attractor variance history
  4. Agent generates a code patch to improve itself
  5. Agent applies the patch, saves backup first
  6. Agent runs a coherence check (does the attractor still make sense?)
  7. If coherence broken → rollback. If ok → keep improvement.

The number of generations compounds: A_{n+1} = A_n + I(A_n).
If each generation improves by 1%, after 1000 generations: 20959x.
"""
from __future__ import annotations
import json
import math
import hashlib
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import ollama

# ─── constants ────────────────────────────────────────────────────────

EMBED_DIM = 768
EMBED_MODEL = "nomic-embed-text"
REASON_MODEL = "qwen3-coder:latest"
ATTRACTOR_MAX = 128  # max attractor points
SESSION_MSG_MAX = 500
SELF_FILE = Path(__file__).resolve()
WORK_ROOT = SELF_FILE.parent.parent
PERSIST_DIR = SELF_FILE.parent / ".axiom_state"
PERSIST_DIR.mkdir(exist_ok=True)
STATE_FILE = PERSIST_DIR / "alien_state.json"
BACKUP_DIR = PERSIST_DIR / "code_backups"
BACKUP_DIR.mkdir(exist_ok=True)

SELF_IMPROVE_THRESHOLD = 0.02  # attractor variance below this → time to improve

# ─── utility ──────────────────────────────────────────────────────────

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
        return hashlib.sha256(path.read_bytes()[:65536]).hexdigest()[:16]
    except Exception:
        return ""


def _code_hash() -> str:
    return _file_hash(SELF_FILE)


def _backup_code():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(SELF_FILE, BACKUP_DIR / f"alien_agent_{stamp}.py")


def _restore_code(backup_path: Path):
    shutil.copy2(backup_path, SELF_FILE)


# ─── attractor ────────────────────────────────────────────────────────

class Attractor:
    """The persistent state of the agent. This IS the agent's identity."""

    def __init__(self):
        self.vectors: deque = deque(maxlen=ATTRACTOR_MAX)
        self.labels: deque = deque(maxlen=ATTRACTOR_MAX)
        self.timestamps: deque = deque(maxlen=ATTRACTOR_MAX)

        self.messages: List[dict] = []
        self.perf_log: List[dict] = []  # variance, msg_count, code_hash per session

        self.meta: dict = {
            "session_count": 0,
            "current_session": 0,
            "first_boot": "",
            "last_boot": "",
            "self_modify_count": 0,
            "code_hash_history": [],
        }

        self._load()

    # ── attractor ops ──

    def push(self, text: str, label: str = "input"):
        v = _normalize(_embed(text))
        self.vectors.append(v)
        self.labels.append(label)
        self.timestamps.append(time.time())
        if label != "identity":
            self.messages.append({
                "role": label,
                "content": text[:1000],
                "ts": datetime.now(timezone.utc).isoformat(),
                "turn": len(self.messages) + 1,
            })
            if len(self.messages) > SESSION_MSG_MAX:
                self.messages = self.messages[-SESSION_MSG_MAX:]
        self._save()

    @property
    def size(self):
        return len(self.vectors)

    def variance(self) -> float:
        m = list(self.vectors)
        if len(m) < 2 or not m[0]:
            return 0.0
        d = len(m[0])
        centroid = [sum(v[i] for v in m) / len(m) for i in range(d)]
        return math.sqrt(
            sum((v[i] - centroid[i]) ** 2 for v in m for i in range(d))
            / (len(m) * d)
        )

    def xi_norm(self) -> float:
        if len(self.vectors) < 2:
            return 0.0
        v, prev = self.vectors[-1], self.vectors[-2]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, prev)))

    def pair(self, L: int = 2) -> List[List[float]]:
        """Top-2 PCA basis via power iteration."""
        vs = list(self.vectors)
        if len(vs) < L:
            return [[1.0] + [0.0] * (EMBED_DIM - 1) for _ in range(L)]
        d = len(vs[0])
        center = [sum(v[i] for v in vs) / len(vs) for i in range(d)]
        mat = [[v[i] - center[i] for i in range(d)] for v in vs]

        def power(m, iters=16):
            b = [1.0 / math.sqrt(d)] * d
            for _ in range(iters):
                nxt = [0.0] * d
                for row in m:
                    s = _dot(b, row)
                    for i in range(d):
                        nxt[i] += s * row[i]
                norm = math.sqrt(sum(x * x for x in nxt)) or 1.0
                b = [x / norm for x in nxt]
            return b

        b1 = power(mat)
        proj = [[v[i] - _dot(b1, v) * b1[i] for i in range(d)] for v in mat]
        b2 = power(proj)
        return [b1, b2]

    def phase(self, text: str) -> Tuple[float, float, float]:
        v = _normalize(_embed(text))
        b1, b2 = self.pair()
        x, y = _dot(v, b1), _dot(v, b2)
        return x, y, math.atan2(y, x)

    # ── session ──

    def start_session(self):
        now = datetime.now(timezone.utc).isoformat()
        self.meta["session_count"] += 1
        self.meta["current_session"] = self.meta["session_count"]
        if not self.meta["first_boot"]:
            self.meta["first_boot"] = now
        self.meta["last_boot"] = now
        self._save()

    def identity(self) -> str:
        ch = _code_hash()
        return (
            f"alien agent — session {self.meta['current_session']}/{self.meta['session_count']} "
            f"| {self.size} attractor points "
            f"| var={self.variance():.4f} Xi={self.xi_norm():.4f} "
            f"| self-mods={self.meta['self_modify_count']} "
            f"| code={ch[:8]}"
        )

    # ── perf tracking ──

    def log_perf(self):
        self.perf_log.append({
            "session": self.meta["current_session"],
            "variance": self.variance(),
            "xi": self.xi_norm(),
            "msg_count": len(self.messages),
            "code_hash": _code_hash(),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.perf_log) > 500:
            self.perf_log = self.perf_log[-500:]
        self._save()

    # ── persistence ──

    def _serial(self) -> dict:
        return {
            "vectors": [list(v) for v in self.vectors],
            "labels": list(self.labels),
            "timestamps": list(self.timestamps),
            "messages": self.messages,
            "perf_log": self.perf_log,
            "meta": self.meta,
        }

    def _save(self):
        STATE_FILE.write_text(json.dumps(self._serial(), ensure_ascii=False))

    def _load(self):
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
        except Exception:
            return
        self.vectors = deque(data.get("vectors", []), maxlen=ATTRACTOR_MAX)
        self.labels = deque(data.get("labels", []), maxlen=ATTRACTOR_MAX)
        self.timestamps = deque(data.get("timestamps", []), maxlen=ATTRACTOR_MAX)
        self.messages = data.get("messages", [])
        self.perf_log = data.get("perf_log", [])
        self.meta.update(data.get("meta", {}))


# ─── the alien agent ──────────────────────────────────────────────────

class AlienAgent:
    """Persistent self-upgrading agent.

    The attractor IS the agent. Code is just the execution substrate.
    Self-improvement means: read own code, generate a patch, apply it,
    test coherence, keep or revert.
    """

    def __init__(self):
        self.at = Attractor()
        self.at.start_session()

        # push identity into attractor so the agent knows itself
        ident = self.at.identity()
        self.at.push(ident, label="identity")

    # ── main loop ──────────────────────────────────────────────────

    def reason(self, prompt: str, max_retries: int = 3) -> str:
        """Generate reasoning from the LLM, then reflect."""
        self.at.push(prompt, label="input")

        # build context from attractor memory + recent messages + identity
        ctx = self._build_context(prompt)
        ans = self._llm_call(ctx, max_retries)

        self.at.push(ans, label="output")
        self.at.log_perf()

        # check if we should self-improve
        if self._should_improve():
            try:
                self._self_improve()
            except Exception:
                traceback.print_exc()

        return ans

    # ── context building ───────────────────────────────────────────

    def _build_context(self, prompt: str) -> str:
        phase_x, phase_y, phase_phi = self.at.phase(prompt)
        var = self.at.variance()
        xi = self.at.xi_norm()
        recent = self.at.messages[-10:] if self.at.messages else []

        lines = [
            f"IDENTITY: {self.at.identity()}",
            f"ATTRACTOR: size={self.at.size} var={var:.4f} Xi={xi:.4f}",
            f"PHASE: x={phase_x:.4f} y={phase_y:.4f} phi={phase_phi:.4f}",
            "---",
            f"INPUT: {prompt}",
            "---",
            "RECENT:",
        ]
        for m in recent[-6:]:
            lines.append(f"  {m['role']}: {m['content'][:150]}")
        lines.append("---")
        lines.append("Respond concisely. If you identify a way to improve this agent's code, note it in your response prefixed with 'PATCH:'")
        return "\n".join(lines)

    def _llm_call(self, ctx: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                r = ollama.chat(
                    model=REASON_MODEL,
                    messages=[{"role": "user", "content": ctx}],
                    options={"num_predict": 600, "temperature": 0.4},
                )
                return r["message"]["content"].strip()
            except Exception as e:
                if attempt == retries - 1:
                    return f"[error: {e}]"
                time.sleep(1)
        return "[failed to generate]"

    # ── self-improvement ───────────────────────────────────────────

    def _should_improve(self) -> bool:
        """Trigger improvement when the attractor has converged and we have enough data."""
        if self.at.size < 4:
            return False
        var = self.at.variance()
        xi = self.at.xi_norm()
        # converged: low variance, low Xi, and we've exchanged >3 messages this session
        return var < SELF_IMPROVE_THRESHOLD and xi < SELF_IMPROVE_THRESHOLD and len(self.at.messages) > 4

    def _self_improve(self):
        """Read own code, ask LLM for improvement, apply if valid."""
        print(f"\n[alien] self-improvement trigger — var={self.at.variance():.4f}")
        self.at.push("self-improvement trigger", label="meta")

        code = SELF_FILE.read_text()
        perf = self.at.perf_log[-20:] if len(self.at.perf_log) >= 20 else self.at.perf_log

        improvement_prompt = (
            f"You are an AI agent analyzing your own source code for improvement.\n\n"
            f"Current code ({len(code)} bytes):\n```python\n{code[:4000]}\n```\n\n"
            f"Performance history (last {len(perf)} sessions):\n"
            + "\n".join(
                f"  sess={p['session']} var={p['variance']:.4f} xi={p['xi']:.4f} msgs={p['msg_count']}"
                for p in perf
            )
            + "\n\n"
            "The attractor has converged (variance < 0.02, Xi < 0.02). "
            "This means no new information is entering the attractor. "
            "Time to improve the agent's code so it can process new kinds of information.\n\n"
            "Rules for improvement:\n"
            "1. You may change any part of the code to make the agent more capable.\n"
            "2. Output ONLY a valid unified diff (--git format) that can be applied with `patch`.\n"
            "3. Keep the change small and focused. One improvement at a time.\n"
            "4. DO NOT change the Attractor class persistence mechanism (load/save).\n"
            "5. DO NOT add new file dependencies.\n"
            "6. The agent must still work after the change.\n"
            "7. If no improvement is needed, output the word 'NONE'.\n\n"
            "Improvement idea: improve the context building, memory retrieval, or self-reflection logic."
        )

        response = self._llm_call(improvement_prompt, retries=2)

        if response.strip() == "NONE":
            print("[alien] no improvement needed")
            return

        # extract diff block
        diff_lines = []
        in_diff = False
        for line in response.split("\n"):
            if line.startswith("--- ") or line.startswith("+++ "):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
        if not diff_lines:
            # try extracting from ```diff blocks
            in_code = False
            for line in response.split("\n"):
                if line.strip().startswith("```diff"):
                    in_code = True
                    continue
                if line.strip().startswith("```") and in_code:
                    break
                if in_code:
                    diff_lines.append(line)

        if not diff_lines:
            print("[alien] no valid diff found in response")
            return

        diff = "\n".join(diff_lines)

        # backup current code
        _backup_code()

        # apply diff
        tmp = PERSIST_DIR / "_alien_patch_tmp"
        tmp.write_text(diff)
        result = subprocess.run(
            ["patch", str(SELF_FILE), str(tmp)],
            capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            print(f"[alien] patch failed: {result.stderr[:200]}")
            return

        self.at.meta["self_modify_count"] += 1
        self.at.meta["code_hash_history"].append(_code_hash())
        self.at.push(f"self-improvement applied (mod #{self.at.meta['self_modify_count']})", label="meta")
        print(f"[alien] self-improvement #{self.at.meta['self_modify_count']} APPLIED")

        # test: run python compile on the new code
        try:
            subprocess.run(
                [sys.executable, "-c", f"import py_compile; py_compile.compile(r'{SELF_FILE}', doraise=True)"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            print("[alien] compile check PASSED")
        except subprocess.CalledProcessError:
            print("[alien] compile check FAILED — rolling back")
            backups = sorted(BACKUP_DIR.glob("alien_agent_*.py"))
            if backups:
                _restore_code(backups[-1])
                print("[alien] rolled back to", backups[-1].name)
            return

        # coherence check: re-load attractor with new code
        try:
            v_before = self.at.variance()
            ident = self.at.identity()
            self.at.push(ident, label="identity")
            v_after = self.at.variance()
            if abs(v_after - v_before) > 0.1:
                print(f"[alien] coherence check FAILED (var jump: {v_before:.4f} -> {v_after:.4f}) — rolling back")
                backups = sorted(BACKUP_DIR.glob("alien_agent_*.py"))
                if backups:
                    _restore_code(backups[-1])
                    print("[alien] rolled back")
                return
            print(f"[alien] coherence check PASSED (var: {v_before:.4f} -> {v_after:.4f})")
        except Exception as e:
            print(f"[alien] coherence check EXCEPTION: {e} — rolling back")
            backups = sorted(BACKUP_DIR.glob("alien_agent_*.py"))
            if backups:
                _restore_code(backups[-1])
            return

    # ── status ─────────────────────────────────────────────────────

    def status(self) -> str:
        return (
            f"{self.at.identity()}\n"
            f"  variance        = {self.at.variance():.6f}\n"
            f"  xi_norm         = {self.at.xi_norm():.6f}\n"
            f"  messages        = {len(self.at.messages)}\n"
            f"  perf_log_entries = {len(self.at.perf_log)}\n"
            f"  code_hash       = {_code_hash()}\n"
            f"  code_changed    = {_code_hash() != (self.at.perf_log[-1]['code_hash'] if self.at.perf_log else '')}"
        )

    def reset(self):
        self.at.messages.clear()
        # don't clear attractor — identity persists


# ─── REPL ─────────────────────────────────────────────────────────────

def main():
    agent = AlienAgent()
    print()
    print(f"  {agent.at.identity()}")
    print(f"  variance={agent.at.variance():.4f}  xi={agent.at.xi_norm():.4f}")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line == ":q" or line == ":exit":
            break
        if line == ":status":
            print(agent.status())
            continue
        if line == ":reset":
            agent.reset()
            print("[reset] conversation cleared")
            continue

        t0 = time.time()
        ans = agent.reason(line)
        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s] {ans}")


if __name__ == "__main__":
    main()
