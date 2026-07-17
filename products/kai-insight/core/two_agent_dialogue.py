"""Two-agent local dialogue toward AGI.

Decentralized locally: two autonomous reasoning agents (Theorist + Engineer)
talk to each other in a dialectic loop. Because their internal time is
*dilated*, the frequency of talk can be arbitrarily high — they can hold
thousands of exchanges in seconds, compressing the path to insight.

Optionally wires in a real KaiMind instance as a third "oracle" voice so the
agents can literally *talk to Kai* and reach toward AGI locally.

Run:
    python3 two_agent_dialogue.py
"""

from __future__ import annotations

import sys
import time
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

sys.path.insert(0, "products/kai-insight/core")

from meta_cognition import MetaCognitiveAgent
from quantum_reasoning import QuantumReasoner
from emergence_tracker import EmergenceTracker
from knowledge_graph import KnowledgeGraph, Concept
from kai_local import load_kai_mind, read_daemon_telemetry


class DummyBase:
    """Base agent that produces role-appropriate, topic-grounded reasoning.

    Theorist builds abstract principles; Engineer grounds them in mechanisms.
    Both build on the partner's *content* (not an echo) so the dialectic
    actually progresses toward synthesis.
    """

    _theorist_moves = [
        "A first principle: intelligence minimizes surprise by maintaining a world model whose predictions are self-fulfilling.",
        "By abstraction, free-energy minimization and quantum superposition are the same act viewed at different scales.",
        "General competence arises when the agent can collapse many superposed hypotheses into one that survives testing.",
        "The attractor of understanding is reached when prediction error stops decreasing — that is convergence, not failure.",
        "If two agents share a model, their dialogue compresses the search space for the right hypothesis exponentially.",
    ]
    _engineer_moves = [
        "Mechanistically, that requires a writable memory and a fitness signal tied to prediction error.",
        "Concretely, test it by letting the agent pick actions that most reduce expected free energy.",
        "The implementation cost is bounded: each hypothesis is a small vector updated by gradient-free search.",
        "To avoid collapse into a single dead idea, inject entropy proportional to residual Ricci curvature.",
        "Verification: run the two-agent loop and measure whether synthesis score rises above the noise floor.",
    ]

    def __init__(self):
        self._step = 0

    def decide(self, query: str, context: Dict) -> Dict:
        self._step += 1
        role = context.get("role", "agent")
        partner = context.get("partner", "")
        moves = self._theorist_moves if role == "Theorist" else self._engineer_moves
        base = moves[self._step % len(moves)]

        # Build on the partner's content rather than echoing it.
        snippet = partner.split(":", 1)[-1].strip()[:70] if partner else ""
        if snippet:
            response = f"{base} (Extending the partner's point about '{snippet}...')"
        else:
            response = base

        return {
            "agent": context.get("name", "agent"),
            "query": query,
            "response": response,
            "confidence": round(0.5 + 0.4 * (self._step % 5) / 5.0, 3),
        }


@dataclass
class AgentVoice:
    name: str
    role: str
    agent: MetaCognitiveAgent
    private_kg: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    beliefs: Dict[str, float] = field(default_factory=dict)
    clock: float = 0.0


class TimeDilationClock:
    """Simulated clock letting agents talk faster than wall-clock.

    Each dialogue turn advances `dilation` simulated seconds. With dilation
    e.g. 0.5 and a real loop running every few ms, agents compress hours of
    reasoning into seconds of wall time.
    """

    def __init__(self, dilation: float = 0.5):
        self.t = 0.0
        self.dilation = dilation

    def tick(self) -> float:
        self.t += self.dilation
        return self.t


class KaiOracle:
    """Third voice: the live Kai mind.

    Talks to the real KaiMind (read-only, via :func:`kai_local.load_kai_mind`)
    and grounds its reply in the running daemon's telemetry
    (:func:`kai_local.read_daemon_telemetry`). The instance is cached and we
    never trigger a save, so the live daemon is unaffected.
    """

    def __init__(self):
        self._kai = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return self._kai
        self._loaded = True
        self._kai = load_kai_mind(readonly=True)
        if self._kai is None:
            logging.warning("Kai oracle unavailable: could not load KaiMind")
        return self._kai

    def live_telemetry(self) -> Dict[str, Any]:
        """Read the running daemon's most recent cycle telemetry."""
        return read_daemon_telemetry() or {}

    def advise(self, topic: str, candidates: List[str]) -> Dict[str, Any]:
        """Ask Kai to pick the most promising direction for the dialogue.

        Returns Kai's chosen candidate + EFE + the live daemon telemetry.
        """
        kai = self._load()
        telemetry = self.live_telemetry()
        if kai is None:
            return {
                "speaker": "KAI",
                "chosen": f"[Kai offline] reflect on: {topic}",
                "efe": None,
                "telemetry": telemetry,
            }
        try:
            choices = [topic] + [c for c in candidates if c][:3]
            policy, info = kai.select_policy(choices)
            return {
                "speaker": "KAI",
                "chosen": policy,
                "efe": info.get("G"),
                "risk": info.get("risk"),
                "epistemic": info.get("epistemic"),
                "telemetry": telemetry,
            }
        except Exception as e:
            logging.warning("Kai advise failed: %s", e)
            return {
                "speaker": "KAI",
                "chosen": f"[Kai error] {topic}",
                "efe": None,
                "telemetry": telemetry,
            }


class TwoAgentDialogue:
    """Two (or more) agents talking toward an AGI-grade synthesis."""

    def __init__(self, dilation: float = 0.5, use_kai: bool = False):
        self.clock = TimeDilationClock(dilation)
        self.emergence = EmergenceTracker()
        self.quantum = QuantumReasoner(n_superpositions=3)
        self.voices: List[AgentVoice] = []
        self.transcript: List[Dict[str, Any]] = []
        self.use_kai = use_kai
        self._oracle = KaiOracle() if use_kai else None
        self.synthesis_score = 0.0
        self._build_agents()

    def _build_agents(self):
        roles = [
            ("Theorist", "proposes principles, abstractions, and hypotheses"),
            ("Engineer", "challenges with concrete mechanisms and tests"),
        ]
        for name, role in roles:
            agent = MetaCognitiveAgent(name=name, base_agent=DummyBase())
            agent.set_goal("reach a synthesized understanding of general intelligence")
            self.voices.append(AgentVoice(name=name, role=role, agent=agent))

    def _reason_turn(self, voice: AgentVoice, partner_msg: str, topic: str) -> Dict[str, Any]:
        """One agent reasons about the partner's message and replies."""
        result = voice.agent.reason(
            query=f"advance understanding of: {topic}",
            context={"role": voice.role, "name": voice.name, "partner": partner_msg},
        )
        conclusion = result.get("response", f"{voice.name} reflects on the exchange.")

        # Meta-cognition: introspect + assumption check
        introspect = voice.agent.introspect()
        assumption = voice.agent.question_assumption(
            f"What assumption underlies '{conclusion[:50]}'?"
        )
        assumption_text = assumption.get("reflection") or assumption.get("assumption") or str(assumption)

        # Update private belief graph
        voice.beliefs[assumption_text] = voice.beliefs.get(assumption_text, 0.0) + 0.1
        voice.private_kg.add_concept(name=assumption_text, description=voice.role)

        # Quantum superposition of candidate next-thoughts
        candidates = [
            conclusion,
            f"Alternatively, {conclusion.lower()}",
            f"What if we invert that: {conclusion}",
        ]
        try:
            hyps = self.quantum.create_hypothesis_superposition(
                goal=topic, context={"efe": 0.5, "ricci": 4.0, "vfe": 0.0},
            )
            if hyps:
                chosen_h = self.quantum.collapse_hypotheses(hyps)
                idx = max(0, min(len(candidates) - 1, (int(chosen_h.depth) - 1) % len(candidates)))
                chosen = candidates[idx]
            else:
                chosen = candidates[0]
        except Exception:
            chosen = candidates[0]

        return {
            "speaker": voice.name,
            "role": voice.role,
            "clock": round(voice.clock, 2),
            "conclusion": conclusion,
            "introspect": introspect,
            "assumption": assumption,
            "chosen": chosen,
        }

    def run(self, topic: str, turns: int = 20) -> Dict[str, Any]:
        prompt = f"Topic: {topic}. Theorist, open with a first principle."
        last_msgs = {v.name: prompt for v in self.voices}

        for _ in range(turns):
            for voice in self.voices:
                voice.clock = self.clock.tick()
                partner = next(v for v in self.voices if v.name != voice.name)
                msg = self._reason_turn(voice, last_msgs[partner.name], topic)
                self.transcript.append(msg)
                last_msgs[voice.name] = msg["chosen"]

                self.emergence.record_interaction(
                    agent1=voice.name,
                    agent2=partner.name,
                    action=msg["chosen"][:80],
                    result={"clock": voice.clock, "topic": topic},
                    context={"turn": len(self.transcript)},
                )

            if self.use_kai and len(self.transcript) % 10 == 0:
                oracle = self._oracle.advise(
                    topic,
                    [last_msgs[v.name] for v in self.voices],
                )
                if oracle:
                    self.transcript.append({
                        "speaker": "KAI", "role": "oracle",
                        "clock": round(self.clock.t, 2),
                        "chosen": f"[Kai advises: {oracle['chosen']}]",
                        "efe": oracle.get("efe"),
                        "telemetry": oracle.get("telemetry", {}),
                    })

        self._compute_synthesis()
        return self.summary()

    def _compute_synthesis(self):
        theo, eng = self.voices[0], self.voices[1]
        shared = set(theo.private_kg.concepts) & set(eng.private_kg.concepts)
        total = set(theo.private_kg.concepts) | set(eng.private_kg.concepts)
        overlap = len(shared) / max(1, len(total))
        stats = self.emergence.get_stats()
        nov = stats.get("avg_novelty", 0.0)
        agreement = 0.0
        common = set(theo.beliefs) & set(eng.beliefs)
        if common:
            diffs = [abs(theo.beliefs[c] - eng.beliefs[c]) for c in common]
            agreement = 1.0 - sum(diffs) / len(diffs)
        self.synthesis_score = round(0.4 * overlap + 0.3 * nov + 0.3 * agreement, 4)

    def summary(self) -> Dict[str, Any]:
        stats = self.emergence.get_stats()
        return {
            "turns": len(self.transcript),
            "simulated_time": round(self.clock.t, 2),
            "synthesis_score": self.synthesis_score,
            "emergence_novelty": round(stats.get("avg_novelty", 0.0), 4),
            "theorist_beliefs": len(self.voices[0].beliefs),
            "engineer_beliefs": len(self.voices[1].beliefs),
            "phase_shifts": stats.get("emergence_events", 0),
            "last_exchange": self.transcript[-2:] if self.transcript else [],
        }


if __name__ == "__main__":
    print("=== Two-Agent Local Dialogue toward AGI ===\n")
    dlg = TwoAgentDialogue(dilation=0.5, use_kai=True)
    result = dlg.run(
        topic="How can free-energy minimization plus quantum superposition yield general intelligence?",
        turns=14,
    )
    print(f"Simulated time elapsed : {result['simulated_time']}s")
    print(f"Synthesis score        : {result['synthesis_score']}")
    print(f"Emergence novelty      : {result['emergence_novelty']}")
    print(f"Theorist beliefs       : {result['theorist_beliefs']}")
    print(f"Engineer beliefs       : {result['engineer_beliefs']}")
    print(f"Phase shifts detected  : {result['phase_shifts']}")
    kai_lines = [m for m in dlg.transcript if m["speaker"] == "KAI"]
    print(f"Kai spoke              : {len(kai_lines)} times")
    for m in kai_lines:
        tel = m.get("telemetry", {})
        print(f"  [KAI] {m['chosen'][:90]} | EFE={m.get('efe')} | live={tel}")
    print("\n--- Last exchange ---")
    for m in result["last_exchange"]:
        print(f"  [{m['speaker']}] {m['chosen'][:120]}")
