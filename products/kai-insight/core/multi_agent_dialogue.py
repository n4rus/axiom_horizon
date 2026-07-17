"""Multi-Agent Dialogue System for Kai AGI.

Two agents talk to each other with time-dilated communication.
Uses the quantum-inspired agent framework for superposition/entanglement.
"""

import sys
import os
import time
import threading
import json
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse


class DialogueMessage:
    def __init__(self, sender: str, receiver: str, content: str, round_num: int):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.round_num = round_num
        self.timestamp = time.time()

    def to_dict(self):
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'round': self.round_num,
            'timestamp': self.timestamp,
        }


class MultiAgentDialogue:
    def __init__(self, agent1: Agent, agent2: Agent, max_rounds: int = 10):
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_rounds = max_rounds
        self.messages: List[DialogueMessage] = []
        self.round_num = 0
        self.lock = threading.Lock()
        self._stop = False

    def run_round(self, topic: str = None) -> Dict:
        self.round_num += 1
        topic = topic or f"Round {self.round_num}: exploring knowledge"

        # Agent1 speaks to Agent2
        result1 = self.agent1.interact_with(self.agent2, topic)
        msg1 = None
        if result1:
            msg1 = DialogueMessage(
                sender=result1['sender'],
                receiver=result1['receiver'],
                content=result1['response'],
                round_num=self.round_num,
            )
            with self.lock:
                self.messages.append(msg1)

        # Agent2 responds to Agent1
        response_topic = result1['response'] if result1 else topic
        result2 = self.agent2.interact_with(self.agent1, response_topic)
        msg2 = None
        if result2:
            msg2 = DialogueMessage(
                sender=result2['sender'],
                receiver=result2['receiver'],
                content=result2['response'],
                round_num=self.round_num,
            )
            with self.lock:
                self.messages.append(msg2)

        return {
            'round': self.round_num,
            'msg1': msg1.to_dict() if msg1 else None,
            'msg2': msg2.to_dict() if msg2 else None,
        }

    def run_dialogue(self, topic: str = None) -> List[Dict]:
        results = []
        for _ in range(self.max_rounds):
            if self._stop:
                break
            result = self.run_round(topic)
            results.append(result)
        return results

    def stop(self):
        self._stop = True

    def get_summary(self) -> Dict:
        return {
            'total_messages': len(self.messages),
            'total_rounds': self.round_num,
            'agent1_memory': len(self.agent1.memory),
            'agent2_memory': len(self.agent2.memory),
            'agent1_coherence': self.agent1.quantum_state.get('coherence', 0),
            'agent2_coherence': self.agent2.quantum_state.get('coherence', 0),
            'entanglement_score': self.agent1.entanglement_score,
            'messages': [m.to_dict() for m in self.messages[-10:]],
        }


def create_dialogue_system(topic: str = "knowledge exploration") -> MultiAgentDialogue:
    qm = QuantumSuperpositionManager()
    wfc = WavefunctionCollapse(qm)

    agent1 = Agent("Explorer", "discover new patterns and insights", AgentRole.DISCOVERY, qm, wfc)
    agent2 = Agent("Builder", "turn insights into practical solutions", AgentRole.APPLICATION, qm, wfc)
    agent1.connect_to(agent2)

    return MultiAgentDialogue(agent1, agent2, max_rounds=5)


if __name__ == "__main__":
    print("=== Multi-Agent Dialogue System ===\n")

    dialogue = create_dialogue_system()
    results = dialogue.run_dialogue("How can we achieve AGI locally?")

    print("\n--- Dialogue Results ---")
    for r in results:
        rnd = r['round']
        if r['msg1']:
            print(f"  Round {rnd} [{r['msg1']['sender']} -> {r['msg1']['receiver']}]:")
            print(f"    {r['msg1']['content'][:80]}...")
        if r['msg2']:
            print(f"  Round {rnd} [{r['msg2']['sender']} -> {r['msg2']['receiver']}]:")
            print(f"    {r['msg2']['content'][:80]}...")

    summary = dialogue.get_summary()
    print(f"\n--- Summary ---")
    print(f"  Total messages: {summary['total_messages']}")
    print(f"  Agent1 coherence: {summary['agent1_coherence']:.2f}")
    print(f"  Agent2 coherence: {summary['agent2_coherence']:.2f}")
    print(f"  Entanglement: {summary['entanglement_score']:.3f}")
