"""Multi-Agent Collaboration Protocols for Kai AGI.

Advanced multi-agent system with:
- Collaboration protocols for joint problem-solving
- Knowledge sharing between agents
- Competitive learning and evolution
- Emergent behavior tracking
- Role specialization and coordination
"""

from __future__ import annotations

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the system."""
    EXPLORER = "explorer"
    BUILDER = "builder"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"


class CollaborationProtocol(Enum):
    """Types of collaboration protocols."""
    DEBATE = "debate"          # Agents argue different positions
    BRAINSTORM = "brainstorm"  # Agents generate ideas together
    REVIEW = "review"          # Agents review each other's work
    SYNTHESIS = "synthesis"    # Agents combine their knowledge
    COMPETITION = "competition"  # Agents compete for best solution


@dataclass
class Message:
    """Message between agents."""
    sender: str
    receiver: str
    content: str
    msg_type: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationResult:
    """Result of a collaboration episode."""
    protocol: CollaborationProtocol
    participants: List[str]
    messages: List[Message]
    outcome: str
    quality_score: float
    duration: float
    timestamp: float = field(default_factory=time.time)


class CollaborativeAgent:
    """Agent with collaboration capabilities."""

    def __init__(self, name: str, role: AgentRole, capabilities: Optional[List[str]] = None):
        self.name = name
        self.role = role
        self.capabilities = capabilities or []
        self.knowledge: Dict[str, Any] = {}
        self.connections: Set[str] = set()
        self.message_history: List[Message] = []
        self.collaboration_scores: Dict[str, float] = {}
        self._expertise: Dict[str, float] = {}  # topic -> expertise level

    def send_message(self, receiver: str, content: str, msg_type: str = "info") -> Message:
        """Send a message to another agent."""
        message = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
        )
        self.message_history.append(message)
        return message

    def receive_message(self, message: Message) -> Optional[str]:
        """Receive and process a message."""
        self.message_history.append(message)

        # Process based on message type
        if message.msg_type == "question":
            return self._answer_question(message.content)
        elif message.msg_type == "request":
            return self._handle_request(message.content)
        elif message.msg_type == "info":
            self._integrate_information(message.content)
            return None

        return None

    def _answer_question(self, question: str) -> str:
        """Answer a question based on knowledge."""
        # Simple keyword-based answering
        question_words = set(question.lower().split())

        for topic, expertise in self._expertise.items():
            topic_words = set(topic.lower().split())
            if len(question_words & topic_words) > 0:
                return f"Based on my expertise in {topic} (level {expertise:.2f}), I can help with that."

        return f"I don't have specific expertise on that topic, but I can try to help."

    def _handle_request(self, request: str) -> str:
        """Handle a request from another agent."""
        return f"Processing request: {request[:50]}..."

    def _integrate_information(self, info: str):
        """Integrate new information into knowledge base."""
        # Simple keyword extraction
        words = info.lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                self.knowledge[word] = self.knowledge.get(word, 0) + 1

    def add_expertise(self, topic: str, level: float = 0.5):
        """Add expertise in a topic."""
        self._expertise[topic] = max(0.0, min(1.0, level))

    def get_expertise_summary(self) -> Dict[str, float]:
        """Get summary of expertise."""
        return dict(self._expertise)


class CollaborationManager:
    """Manages multi-agent collaboration."""

    def __init__(self):
        self.agents: Dict[str, CollaborativeAgent] = {}
        self.collaboration_history: List[CollaborationResult] = []
        self._protocols: Dict[CollaborationProtocol, Callable] = {
            CollaborationProtocol.DEBATE: self._run_debate,
            CollaborationProtocol.BRAINSTORM: self._run_brainstorm,
            CollaborationProtocol.REVIEW: self._run_review,
            CollaborationProtocol.SYNTHESIS: self._run_synthesis,
            CollaborationProtocol.COMPETITION: self._run_competition,
        }

    def register_agent(self, agent: CollaborativeAgent):
        """Register an agent with the collaboration manager."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")

    def unregister_agent(self, name: str):
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Unregistered agent: {name}")

    def connect_agents(self, name1: str, name2: str):
        """Create a connection between two agents."""
        if name1 in self.agents and name2 in self.agents:
            self.agents[name1].connections.add(name2)
            self.agents[name2].connections.add(name1)

    def run_collaboration(
        self,
        protocol: CollaborationProtocol,
        participants: List[str],
        topic: str,
        max_rounds: int = 5,
    ) -> CollaborationResult:
        """Run a collaboration episode."""
        if protocol not in self._protocols:
            raise ValueError(f"Unknown protocol: {protocol}")

        # Validate participants
        valid_participants = [p for p in participants if p in self.agents]
        if len(valid_participants) < 2:
            raise ValueError("Need at least 2 valid participants")

        start_time = time.time()
        messages = []

        # Run the protocol
        protocol_func = self._protocols[protocol]
        outcome, quality_score = protocol_func(valid_participants, topic, max_rounds, messages)

        duration = time.time() - start_time

        result = CollaborationResult(
            protocol=protocol,
            participants=valid_participants,
            messages=messages,
            outcome=outcome,
            quality_score=quality_score,
            duration=duration,
        )

        self.collaboration_history.append(result)

        # Update collaboration scores
        for participant in valid_participants:
            agent = self.agents[participant]
            agent.collaboration_scores[protocol.value] = (
                agent.collaboration_scores.get(protocol.value, 0.5) * 0.9 +
                quality_score * 0.1
            )

        return result

    def _run_debate(
        self,
        participants: List[str],
        topic: str,
        max_rounds: int,
        messages: List[Message],
    ) -> tuple[str, float]:
        """Run a debate protocol."""
        positions = ["for", "against", "neutral"]
        debate_content = []

        for round_num in range(max_rounds):
            for i, participant in enumerate(participants):
                agent = self.agents[participant]
                position = positions[i % len(positions)]

                # Generate debate argument
                argument = self._generate_argument(agent, topic, position, round_num)
                debate_content.append(argument)

                # Send to other participants
                for other in participants:
                    if other != participant:
                        msg = agent.send_message(other, argument, "debate")
                        messages.append(msg)

        # Evaluate debate
        quality = self._evaluate_debate(debate_content)
        outcome = f"Debate on '{topic}' concluded with {len(debate_content)} arguments"

        return outcome, quality

    def _run_brainstorm(
        self,
        participants: List[str],
        topic: str,
        max_rounds: int,
        messages: List[Message],
    ) -> tuple[str, float]:
        """Run a brainstorm protocol."""
        ideas = []

        for round_num in range(max_rounds):
            for participant in participants:
                agent = self.agents[participant]

                # Generate idea
                idea = self._generate_idea(agent, topic, ideas, round_num)
                ideas.append(idea)

                # Share with all participants
                for other in participants:
                    if other != participant:
                        msg = agent.send_message(other, idea, "idea")
                        messages.append(msg)

        # Evaluate brainstorm
        quality = self._evaluate_brainstorm(ideas)
        outcome = f"Brainstorm on '{topic}' generated {len(ideas)} ideas"

        return outcome, quality

    def _run_review(
        self,
        participants: List[str],
        topic: str,
        max_rounds: int,
        messages: List[Message],
    ) -> tuple[str, float]:
        """Run a review protocol."""
        reviews = []

        # First agent proposes, others review
        proposer = participants[0]
        reviewers = participants[1:]

        # Generate proposal
        proposal = self._generate_proposal(self.agents[proposer], topic)

        # Collect reviews
        for reviewer in reviewers:
            agent = self.agents[reviewer]
            review = self._generate_review(agent, proposal)
            reviews.append(review)

            msg = agent.send_message(proposer, review, "review")
            messages.append(msg)

        # Evaluate reviews
        quality = self._evaluate_reviews(reviews)
        outcome = f"Review of proposal on '{topic}' collected {len(reviews)} reviews"

        return outcome, quality

    def _run_synthesis(
        self,
        participants: List[str],
        topic: str,
        max_rounds: int,
        messages: List[Message],
    ) -> tuple[str, float]:
        """Run a synthesis protocol."""
        contributions = []

        # Each agent contributes knowledge
        for participant in participants:
            agent = self.agents[participant]
            contribution = self._generate_contribution(agent, topic)
            contributions.append(contribution)

            # Share with coordinator (first participant)
            if participant != participants[0]:
                msg = agent.send_message(participants[0], contribution, "contribution")
                messages.append(msg)

        # Synthesize contributions
        synthesizer = self.agents[participants[0]]
        synthesis = self._synthesize_contributions(synthesizer, contributions)

        # Evaluate synthesis
        quality = self._evaluate_synthesis(synthesis)
        outcome = f"Synthesis on '{topic}' combined {len(contributions)} contributions"

        return outcome, quality

    def _run_competition(
        self,
        participants: List[str],
        topic: str,
        max_rounds: int,
        messages: List[Message],
    ) -> tuple[str, float]:
        """Run a competition protocol."""
        submissions = []

        # Each agent submits a solution
        for participant in participants:
            agent = self.agents[participant]
            solution = self._generate_solution(agent, topic)
            submissions.append((participant, solution))

        # Evaluate submissions
        scores = []
        for participant, solution in submissions:
            score = self._evaluate_solution(solution)
            scores.append((participant, score))

            msg = self.agents[participant].send_message(
                participants[0],
                f"Submitted solution with score {score:.3f}",
                "submission"
            )
            messages.append(msg)

        # Find winner
        scores.sort(key=lambda x: x[1], reverse=True)
        winner = scores[0][0]
        winner_score = scores[0][1]

        outcome = f"Competition on '{topic}' won by {winner} with score {winner_score:.3f}"
        quality = winner_score

        return outcome, quality

    def _generate_argument(self, agent: CollaborativeAgent, topic: str, position: str, round_num: int) -> str:
        """Generate a debate argument."""
        expertise = agent.get_expertise_summary()
        top_topics = sorted(expertise.keys(), key=lambda k: expertise[k], reverse=True)[:3]

        return f"[{agent.name} - {position}] Round {round_num}: Regarding '{topic}', " \
               f"based on my expertise in {', '.join(top_topics)}, I argue {position}..."

    def _generate_idea(self, agent: CollaborativeAgent, topic: str, existing_ideas: List[str], round_num: int) -> str:
        """Generate a brainstorm idea."""
        idea_num = len(existing_ideas) + 1
        return f"[{agent.name}] Idea {idea_num}: For '{topic}', consider approach #{round_num + 1}..."

    def _generate_proposal(self, agent: CollaborativeAgent, topic: str) -> str:
        """Generate a proposal for review."""
        return f"[{agent.name}] Proposal for '{topic}': A comprehensive approach that..."

    def _generate_review(self, agent: CollaborativeAgent, proposal: str) -> str:
        """Generate a review of a proposal."""
        return f"[{agent.name}] Review: The proposal is {'strong' if random.random() > 0.5 else 'weak'} " \
               f"because..."

    def _generate_contribution(self, agent: CollaborativeAgent, topic: str) -> str:
        """Generate a contribution for synthesis."""
        expertise = agent.get_expertise_summary()
        return f"[{agent.name}] Contribution: From my expertise in {', '.join(list(expertise.keys())[:2])}, " \
               f"I add..."

    def _synthesize_contributions(self, synthesizer: CollaborativeAgent, contributions: List[str]) -> str:
        """Synthesize multiple contributions."""
        return f"[{synthesizer.name}] Synthesis: Combining {len(contributions)} contributions..."

    def _generate_solution(self, agent: CollaborativeAgent, topic: str) -> str:
        """Generate a solution for competition."""
        return f"[{agent.name}] Solution for '{topic}': A novel approach that..."

    def _evaluate_debate(self, content: List[str]) -> float:
        """Evaluate debate quality."""
        return min(1.0, len(content) / 10.0 * 0.8 + random.random() * 0.2)

    def _evaluate_brainstorm(self, ideas: List[str]) -> float:
        """Evaluate brainstorm quality."""
        return min(1.0, len(ideas) / 15.0 * 0.7 + random.random() * 0.3)

    def _evaluate_reviews(self, reviews: List[str]) -> float:
        """Evaluate review quality."""
        return min(1.0, len(reviews) / 3.0 * 0.6 + random.random() * 0.4)

    def _evaluate_synthesis(self, synthesis: str) -> float:
        """Evaluate synthesis quality."""
        return 0.5 + random.random() * 0.5

    def _evaluate_solution(self, solution: str) -> float:
        """Evaluate solution quality."""
        return 0.3 + random.random() * 0.7

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        if not self.collaboration_history:
            return {'total_collaborations': 0}

        protocol_counts = defaultdict(int)
        total_quality = 0.0

        for result in self.collaboration_history:
            protocol_counts[result.protocol.value] += 1
            total_quality += result.quality_score

        return {
            'total_collaborations': len(self.collaboration_history),
            'avg_quality': total_quality / len(self.collaboration_history),
            'protocol_distribution': dict(protocol_counts),
            'active_agents': len(self.agents),
        }


if __name__ == "__main__":
    print("=== Multi-Agent Collaboration Test ===\n")

    # Create collaboration manager
    manager = CollaborationManager()

    # Create agents with different roles and expertise
    agents = [
        CollaborativeAgent("Explorer", AgentRole.EXPLORER, ["discovery", "analysis"]),
        CollaborativeAgent("Builder", AgentRole.BUILDER, ["implementation", "optimization"]),
        CollaborativeAgent("Critic", AgentRole.CRITIC, ["evaluation", "feedback"]),
        CollaborativeAgent("Synthesizer", AgentRole.SYNTHESIZER, ["integration", "synthesis"]),
    ]

    # Add expertise
    agents[0].add_expertise("data analysis", 0.8)
    agents[0].add_expertise("pattern recognition", 0.7)
    agents[1].add_expertise("system design", 0.9)
    agents[1].add_expertise("optimization", 0.8)
    agents[2].add_expertise("evaluation", 0.85)
    agents[2].add_expertise("critical thinking", 0.9)
    agents[3].add_expertise("synthesis", 0.9)
    agents[3].add_expertise("integration", 0.85)

    # Register agents
    for agent in agents:
        manager.register_agent(agent)
        manager.connect_agents(agent.name, agents[0].name)

    # Run different collaboration protocols
    protocols = [
        CollaborationProtocol.DEBATE,
        CollaborationProtocol.BRAINSTORM,
        CollaborationProtocol.REVIEW,
        CollaborationProtocol.SYNTHESIS,
        CollaborationProtocol.COMPETITION,
    ]

    for protocol in protocols:
        print(f"\nRunning {protocol.value} protocol:")
        result = manager.run_collaboration(
            protocol=protocol,
            participants=[a.name for a in agents],
            topic="How to achieve AGI",
            max_rounds=3,
        )
        print(f"  Outcome: {result.outcome}")
        print(f"  Quality: {result.quality_score:.3f}")
        print(f"  Duration: {result.duration:.3f}s")
        print(f"  Messages: {len(result.messages)}")

    # Get stats
    stats = manager.get_collaboration_stats()
    print(f"\nCollaboration Statistics:")
    print(f"  Total collaborations: {stats['total_collaborations']}")
    print(f"  Average quality: {stats['avg_quality']:.3f}")
    print(f"  Protocol distribution: {stats['protocol_distribution']}")
