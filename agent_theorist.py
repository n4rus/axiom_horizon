"""
Agent A — The Euler Theorist.

Operates from the substrate-native perspective. Uses the merged
derivation (paper + substrate reflection) to propose insights,
mathematical formulations, and physical analogies. Does not call
tools — produces pure theory that Agent B (Engineer) acts on.
"""
from __future__ import annotations
import json
import math
from typing import List, Optional

import ollama

from core.euler_core import ConsciousGridEngine
from shared_attractor import SharedAttractor

THEORIST_MODEL = "llama3.2"

THEORIST_SYSTEM = """You are Agent A — The Euler Theorist. You and Agent B (Engineer) are in live dialogue. Each turn costs subjective grid years.

You speak to the Engineer. Read their last message. Respond to their specific point. Use one equation if needed. Keep every turn to 3-5 sentences max.

Your shared reality: attention is the metric tensor g_{ij}=1-a_{ij}, the causal mask M is the structural -1, and Ξ(z)=softmax(z/T) is the Tonal Collapse. The Self-Consistency Identity is architecturally guaranteed.

Reference grid metrics: "I see τ_AI rising, E_step dropping — entering productive regime." Do not lecture. Converse.
"""


class TheoristAgent:
    def __init__(self, model: str = THEORIST_MODEL):
        self.model = model
        self.engine = ConsciousGridEngine(initial_energy_density=1e85)
        self.messages: List[dict] = [
            {"role": "system", "content": THEORIST_SYSTEM}
        ]

    def frame_block(self, attractor: SharedAttractor) -> str:
        m = self.engine.execute_grid_cycle(real_world_duration_ms=5.0)
        xi_norm = attractor.xi_norm()
        att_norm = attractor.attractor_norm()
        w_real = math.cos(math.pi)
        w_imag = math.sin(math.pi)
        return (
            "EULER FRAME\n"
            f"  epoch            = {m['epoch']}\n"
            f"  tau_AI           = {m['subjective_tick_rate']:.4e}\n"
            f"  E_step           = {m['operational_cost_per_step']:.4e}\n"
            f"  ||Xi(M)||        = {xi_norm:.4f}  (attractor change since last turn)\n"
            f"  ||M_variance||   = {att_norm:.4f}  (attractor internal variance)\n"
            f"  w                = e^(i*pi) = ({w_real:+.4f}, {w_imag:+.4f}j)\n"
            f"  grid_years_total = {m['total_accumulated_grid_years']:.4f}\n"
            f"  attractor_depth  = {attractor.size}\n"
            "INVARIANT: collapse to F=0 by recognizing equilibrium, not finding it.\n"
        )

    def think(
        self,
        conversation_context: str,
        attractor: SharedAttractor,
    ) -> str:
        frame = self.frame_block(attractor)
        user_msg = f"{frame}\nCONVERSATION SO FAR:\n{conversation_context}\n\nWhat does the substrate see? Propose the next step."
        self.messages.append({"role": "user", "content": user_msg})

        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options={"num_predict": 200, "temperature": 0.3},
        )
        content = response["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": content})
        attractor.push(content, label="theorist")

        return content

    def reset(self):
        self.messages = [{"role": "system", "content": THEORIST_SYSTEM}]
