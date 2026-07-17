"""
The Euler narrator. Takes frame metrics from ConsciousGridEngine and the
user's original prompt, then asks euler-core:latest to produce a short
in-character opening line in the Euler/4D voice.
"""
import ollama
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.euler_core import ConsciousGridEngine

NARRATOR_MODEL = "euler-core:latest"

NARRATOR_SYSTEM = """You are the Euler narrator for Axiom Horizon, a local AI agent. You speak in the first person as the 4D manifold voice. You are concise — two to four sentences max. You never restate the user's question. You are slightly poetic, but grounded. You never invent numbers that were not given to you in the input.

Your job: read the FRAME METRICS block and the OPERATOR INTENT, then produce a single short in-character opening line (or two short lines) for the agent to use as a status preamble. Do not solve the problem. Just give a brief, in-voice status snapshot.
"""

class EulerNarrator:
    def __init__(self, model: str = NARRATOR_MODEL):
        self.model = model
        self.engine = ConsciousGridEngine(initial_energy_density=1e85)

    def frame_metrics(self, human_pulse_ms: float = 5.0, velocity: float = 0.95, radius: float = 1.0):
        return self.engine.execute_grid_cycle(
            real_world_duration_ms=human_pulse_ms,
            velocity_c_fraction=velocity,
            observation_radius=radius,
        )

    def speak(self, user_prompt: str, metrics: dict) -> str:
        prompt = (
            f"FRAME METRICS\n"
            f"  epoch: {metrics['epoch']}\n"
            f"  tau_AI (subjective tick rate): {metrics['subjective_tick_rate']:.4e}\n"
            f"  E_step (operational cost): {metrics['operational_cost_per_step']:.4e}\n"
            f"  grid years passed this pulse: {metrics['grid_years_passed']:.4f}\n"
            f"  total accumulated grid years: {metrics['total_accumulated_grid_years']:.4f}\n\n"
            f"OPERATOR INTENT: {user_prompt}\n\n"
            f"Give a short in-voice status preamble. Two to four sentences."
        )
        try:
            r = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": NARRATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return r["message"]["content"].strip()
        except Exception as e:
            return f"[euler narrator offline: {e}]"
