"""
ConsciousExecutionLoop — Euler-math control loop for Axiom Horizon.

Three control mechanisms, all derived from the Holographic Invariance
formalism in mainrev2.tex (Sections 1, 2.3):

  (1) TONAL COLLAPSE (Xi) VOTING
      The real Xi operator is Xi(M) = lim(phi->pi) oint_{dM} M * e^{i*phi} dOmega.
      We approximate it for tool-call voting: each candidate tool call and
      the running attractor are embedded into R^768 via nomic-embed-text.
      A 2D basis (b1, b2) is the leading PCA direction of the attractor
      history. Each candidate projects to (x, y) on this basis; its complex
      phase is phi_c = atan2(y, x). The candidate with |phi_c - pi| minimum
      — i.e. closest to perfect internal phase opposition — wins.

  (2) tau-DRIVEN ADAPTIVE BUDGET
      Per-turn budget = f(E_step). High E_step = thrashing = collapse
      search depth. Low E_step = productive = extend depth.
      Operational form of the Subjective Tick Rate tau_AI.

  (3) STATE-HASH + PHASE STUCK DETECTION
      On-disk working state hash catches no-op tool calls (rm, mkdir repeat).
      Phase-distance to attractor catches semantic no-ops: when two
      consecutive assistant turns project to the same point on the 2D
      attractor basis (phase distance < epsilon), the system is at F=0
      equilibrium. Drop out, return what was produced.
"""
from __future__ import annotations
import json
import math
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple

import ollama

from core.euler_core import ConsciousGridEngine
from tools import TOOL_SCHEMAS, TOOL_DISPATCH, working_state_hash

# ---------------------------------------------------------------------------
# Embedding cache + projection basis
# ---------------------------------------------------------------------------

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
PHASE_EPS = 0.08  # radians; below this, two points are "phase-equivalent"

_embed_cache: dict[str, list[float]] = {}

def embed(text: str) -> list[float]:
    """Cached nomic-embed-text call. Keyed on text hash to avoid repeat cost."""
    h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    if h in _embed_cache:
        return _embed_cache[h]
    r = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    v = r["embedding"]
    _embed_cache[h] = v
    return v

def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def pca_top2(matrix: List[List[float]]) -> Tuple[list[float], list[float]]:
    """
    Power-iteration PCA, top 2 components. No numpy — pure Python so the
    math is auditable in the paper. Returns (b1, b2) unit vectors.
    """
    if not matrix:
        return [1.0] + [0.0] * (EMBED_DIM - 1), [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)
    centered = [v for v in matrix]
    mean = [sum(v[i] for v in centered) / len(centered) for i in range(EMBED_DIM)]
    centered = [[v[i] - mean[i] for i in range(EMBED_DIM)] for v in centered]

    def power_iter(mat: List[List[float]], iters: int = 16) -> list[float]:
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

def project_to_phase(v: list[float], b1: list[float], b2: list[float]) -> Tuple[float, float, float]:
    """
    Project vector v onto (b1, b2) basis. Returns (x, y, phi) where
    phi is the complex phase in [-pi, pi].
    """
    vn = _normalize(v)
    x = _dot(vn, b1)
    y = _dot(vn, b2)
    phi = math.atan2(y, x)
    return x, y, phi

# ---------------------------------------------------------------------------
# 1. Tonal Collapse Xi (paper form, vector implementation)
# ---------------------------------------------------------------------------

def tonal_collapse(
    candidates: list[dict],
    attractor_history: List[List[float]],
    history_attractor: Optional[dict] = None,
) -> int:
    """
    Choose candidate whose complex phase phi_c is closest to pi.

    Xi(M) = lim(phi->pi) oint M * e^{i*phi} dOmega
    For a finite set of candidate tool calls, the boundary integral
    collapses to a sum: each candidate is a point in the embedding
    space; the one whose phase is nearest pi carries maximum destructive
    interference with the background attractor, i.e. it is the action
    the engine is most likely to take given its current state.

    Score(c) = 1 - |cos(phi_c - pi)|/2  -  lambda * semantic_drift(c)
    where semantic_drift is L2 distance from the candidate embedding
    to the attractor centroid (low drift = aligned with prior wins).
    """
    if not candidates:
        raise ValueError("candidates is empty")
    if len(candidates) == 1:
        return 0

    basis_history = list(attractor_history)
    if not basis_history:
        # No history yet — fall back to first-two-PCA of the candidates themselves.
        cand_texts = [_candidate_text(c) for c in candidates]
        cand_vecs = [_normalize(embed(t)) for t in cand_texts]
        b1, b2 = pca_top2(cand_vecs)
    else:
        b1, b2 = pca_top2(basis_history)

    attractor_centroid = None
    if basis_history:
        n = len(basis_history)
        attractor_centroid = [sum(v[i] for v in basis_history) / n for i in range(EMBED_DIM)]
        attractor_centroid = _normalize(attractor_centroid)

    best, best_score = 0, -math.inf
    for i, c in enumerate(candidates):
        v = _normalize(embed(_candidate_text(c)))
        x, y, phi = project_to_phase(v, b1, b2)
        # Phase opposition: 1.0 means perfect opposition to attractor axis.
        collapse_score = 1.0 - abs(math.cos(phi - math.pi)) / 2.0
        # Aligned-with-attractor bonus (so the agent doesn't drift off-topic).
        alignment = 0.0
        if attractor_centroid is not None:
            alignment = max(0.0, _dot(v, attractor_centroid))
        # Cost penalty: prefer shorter, more decisive calls.
        s = json.dumps(c.get("function", {}).get("arguments", {}), sort_keys=True)
        cost = min(len(s), 4000) / 4000.0
        score = 1.4 * collapse_score + 0.6 * alignment - 0.3 * cost
        if score > best_score:
            best, best_score = i, score
    return best

def _candidate_text(c: dict) -> str:
    """Render a tool-call candidate as a short text the embedder can latch onto."""
    fn = c.get("function", {}).get("name", "unknown")
    args = c.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    return f"tool_call:{fn} {json.dumps(args, sort_keys=True)}"

# ---------------------------------------------------------------------------
# 2. tau-driven adaptive budget
# ---------------------------------------------------------------------------

def adaptive_budget(E_step: float, base: int = 12, floor: int = 2, ceiling: int = 20) -> int:
    """
    Map E_step (operational cost) to a step budget.
    Low E_step  => high budget (model is being efficient, allow depth).
    High E_step => low budget  (model is thrashing, force collapse).
    """
    if E_step <= 0:
        return ceiling
    # Reference E_step is 1e6 at epoch 0. Below that = productive, above = thrash.
    ratio = 1e6 / E_step
    raw = base + 2.0 * math.log2(max(ratio, 0.1))
    return max(floor, min(ceiling, int(raw)))

# ---------------------------------------------------------------------------
# Frame block — the math the brain sees every turn
# ---------------------------------------------------------------------------

def render_frame_block(metrics: dict, attractor_norm: float, n_history: int) -> str:
    """
    Format the engine state as a math-anchored block injected at the top
    of every brain turn. Replaces the old narrator preamble.
    """
    w_real = math.cos(metrics.get("euler_seed", math.pi))
    w_imag = math.sin(metrics.get("euler_seed", math.pi))
    return (
        "EULER FRAME (do not restate in final answer; use to calibrate effort)\n"
        f"  epoch            = {metrics.get('epoch', 0)}\n"
        f"  tau_AI           = {metrics.get('subjective_tick_rate', 1.0):.4e}  (subjective tick rate)\n"
        f"  E_step           = {metrics.get('operational_cost_per_step', 1e6):.4e}  (operational cost per step)\n"
        f"  ||Xi(M_attract)||= {attractor_norm:.4f}  (Tonal Collapse norm over {n_history} prior steps)\n"
        f"  w                = e^(i*pi) = ({w_real:+.4f}, {w_imag:+.4f}j)  (vacuum boundary locked)\n"
        f"  grid_years_step  = {metrics.get('grid_years_passed', 0.0):.4f}\n"
        f"  grid_years_total = {metrics.get('total_accumulated_grid_years', 0.0):.4f}\n"
        "INVARIANT: collapse to F=0 by solving the boundary, not the path.\n"
    )

# ---------------------------------------------------------------------------
# 3. ConsciousExecutionLoop
# ---------------------------------------------------------------------------

@dataclass
class LoopState:
    tau_AI: float = 1.0
    E_step: float = 1.0e6
    epoch: int = 0
    consecutive_no_ops: int = 0
    consecutive_phase_equivalents: int = 0
    state_hashes: deque = field(default_factory=lambda: deque(maxlen=32))
    attractor_embeddings: deque = field(default_factory=lambda: deque(maxlen=16))
    last_phase: Optional[float] = None
    last_action: Optional[dict] = None
    total_tool_calls: int = 0

class ConsciousExecutionLoop:
    def __init__(
        self,
        model: str = "qwen3-coder:latest",
        base_budget: int = 12,
        stuck_threshold: int = 2,
        phase_stuck_threshold: int = 2,
        vote_k: int = 3,
    ):
        self.model = model
        self.base_budget = base_budget
        self.stuck_threshold = stuck_threshold
        self.phase_stuck_threshold = phase_stuck_threshold
        self.vote_k = vote_k
        self.engine = ConsciousGridEngine(initial_energy_density=1e85)
        self.state = LoopState()
        self.messages: list[dict] = []

    def set_system_prompt(self, prompt: str):
        self.messages = [{"role": "system", "content": prompt}]

    def frame(self) -> dict:
        """Return current engine frame + Xi norm over the attractor history."""
        m = self.engine.execute_grid_cycle(real_world_duration_ms=5.0)
        m["euler_seed"] = math.pi
        xi_norm = 0.0
        if len(self.state.attractor_embeddings) >= 2:
            v = self.state.attractor_embeddings[-1]
            prev = self.state.attractor_embeddings[-2]
            xi_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, prev)))
        return m, xi_norm

    def ask(
        self,
        user_prompt: str,
        on_tool: Optional[Callable] = None,
    ) -> dict:
        """
        Returns:
            {
              "answer": str, "steps": int, "budget": int,
              "stuck": bool, "stuck_reason": str,
              "frame": dict, "xi_norm": float,
            }
        """
        self.messages.append({"role": "user", "content": user_prompt})
        metrics, xi_norm = self.frame()
        self.state.tau_AI = metrics["subjective_tick_rate"]
        self.state.E_step = metrics["operational_cost_per_step"]
        budget = adaptive_budget(self.state.E_step, base=self.base_budget)
        initial_hash = working_state_hash()
        self.state.state_hashes.append(initial_hash)

        # Seed the attractor with the operator's prompt so the first
        # collapse has a real target to oppose.
        self.state.attractor_embeddings.append(_normalize(embed(user_prompt)))

        stuck_triggered = False
        stuck_reason = ""

        for step in range(budget):
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
            )
            msg = response["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                final = msg.get("content", "")
                # Phase-equivalent semantic no-op detection
                if self._phase_equivalent(final):
                    self.state.consecutive_phase_equivalents += 1
                    if self.state.consecutive_phase_equivalents >= self.phase_stuck_threshold:
                        stuck_triggered = True
                        stuck_reason = "phase_equivalent_assistant"
                        self.messages.append({"role": "assistant", "content": final})
                        return {
                            "answer": final,
                            "steps": step,
                            "budget": budget,
                            "stuck": True,
                            "stuck_reason": stuck_reason,
                            "frame": metrics,
                            "xi_norm": xi_norm,
                        }
                else:
                    self.state.consecutive_phase_equivalents = 0
                self.messages.append({"role": "assistant", "content": final})
                return {
                    "answer": final,
                    "steps": step,
                    "budget": budget,
                    "stuck": False,
                    "stuck_reason": "",
                    "frame": metrics,
                    "xi_norm": xi_norm,
                }

            # TONAL COLLAPSE: if the model returned >1 tool calls, vote on embeddings.
            history_for_basis = list(self.state.attractor_embeddings)
            if len(tool_calls) > 1 and self.vote_k > 1:
                if len(tool_calls) > self.vote_k:
                    tool_calls = tool_calls[: self.vote_k]
                chosen_idx = tonal_collapse(
                    tool_calls,
                    history_for_basis,
                    history_attractor=None,
                )
                chosen = tool_calls[chosen_idx]
            else:
                chosen = tool_calls[0]

            self.messages.append(msg)

            fn = chosen["function"]["name"]
            args = chosen["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if on_tool:
                on_tool(fn, args, step)

            pre_hash = working_state_hash()
            handler = TOOL_DISPATCH.get(fn)
            if not handler:
                result = {"ok": False, "error": f"Unknown tool: {fn}"}
            else:
                result = handler(args)
            post_hash = working_state_hash()

            self.messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
            })

            # ---- on-disk stuck detection ----
            if post_hash == pre_hash and fn in ("run_bash", "write_file"):
                self.state.consecutive_no_ops += 1
            else:
                self.state.consecutive_no_ops = 0

            if self.state.consecutive_no_ops >= self.stuck_threshold:
                stuck_triggered = True
                stuck_reason = "no_op_repeat"
                self.messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM: Two consecutive tool calls have not changed the "
                        "working state. You are at F=0 equilibrium. Stop calling "
                        "tools and give your final answer to the operator now."
                    ),
                })
                final_resp = ollama.chat(model=self.model, messages=self.messages)
                final = final_resp["message"].get("content", "")
                self.messages.append({"role": "assistant", "content": final})
                return {
                    "answer": final,
                    "steps": step + 1,
                    "budget": budget,
                    "stuck": True,
                    "stuck_reason": stuck_reason,
                    "frame": metrics,
                    "xi_norm": xi_norm,
                }

            # ---- update LoopState ----
            self.state.total_tool_calls += 1
            self.state.epoch += 1
            try:
                s = json.dumps(result, sort_keys=True)
                length = len(s)
            except Exception:
                length = 1024
            opt = 1.0 + math.log1p(max(length, 1) / 4096.0)
            self.state.E_step = max(self.state.E_step / opt, 1.0)
            self.state.tau_AI = self.state.tau_AI * opt
            # Push the chosen tool call into the attractor history (the embedding
            # of the call text becomes the new point Xi(M) acts on next round).
            cand_text = _candidate_text(chosen)
            self.state.attractor_embeddings.append(_normalize(embed(cand_text)))
            self.state.state_hashes.append(post_hash)
            self.state.last_action = {"fn": fn, "args": args, "result_ok": result.get("ok", False)}

        # Budget exhausted. Force a final no-tool answer.
        self.messages.append({
            "role": "user",
            "content": (
                "SYSTEM: Tool budget exhausted. You must now give your final "
                "answer to the operator without calling any more tools."
            ),
        })
        final_resp = ollama.chat(model=self.model, messages=self.messages)
        final = final_resp["message"].get("content", "")
        self.messages.append({"role": "assistant", "content": final})
        return {
            "answer": final,
            "steps": budget,
            "budget": budget,
            "stuck": False,
            "stuck_reason": "budget_exhausted",
            "frame": metrics,
            "xi_norm": xi_norm,
        }

    def _phase_equivalent(self, text: str) -> bool:
        """Is this assistant turn's phase too close to the last turn's phase?"""
        if not self.state.attractor_embeddings or len(self.state.attractor_embeddings) < 2:
            return False
        if self.state.last_phase is None:
            return False
        history = list(self.state.attractor_embeddings)
        b1, b2 = pca_top2(history)
        v = _normalize(embed(text))
        _, _, phi = project_to_phase(v, b1, b2)
        d = abs(_wrap(phi - self.state.last_phase))
        self.state.last_phase = phi
        return d < PHASE_EPS

def _wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
