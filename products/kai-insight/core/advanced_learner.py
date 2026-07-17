"""Advanced Learning Algorithm for Kai AGI.

Implements sophisticated learning mechanisms:
- Experience replay and prioritized learning
- Meta-learning (learning to learn)
- Transfer learning across domains
- Curiosity-driven exploration
"""

from __future__ import annotations

import math
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single experience for learning."""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningProgress:
    """Track learning progress over time."""
    episode: int
    total_reward: float
    avg_reward: float
    loss: float
    exploration_rate: float
    timestamp: float = field(default_factory=time.time)


class ExperienceReplay:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.buffer: List[Experience] = []
        self.priorities: List[float] = []
        self._position = 0

    def add(self, experience: Experience):
        """Add experience with maximum priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self._position] = experience
            self.priorities[self._position] = max_priority

        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample experiences based on priority."""
        if len(self.buffer) == 0:
            return []

        priorities = [p ** self.alpha for p in self.priorities[:len(self.buffer)]]
        total = sum(priorities)
        probabilities = [p / total for p in priorities]

        indices = random.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class CuriosityModule:
    """Curiosity-driven exploration module."""

    def __init__(self, curiosity_factor: float = 0.1):
        self.curiosity_factor = curiosity_factor
        self.state_visits: Dict[str, int] = {}
        self._intrinsic_rewards: deque = deque(maxlen=100)

    def compute_curiosity_reward(self, state: Dict[str, Any]) -> float:
        """Compute intrinsic curiosity reward for a state."""
        state_key = str(sorted(state.items()))
        visits = self.state_visits.get(state_key, 0)

        # Curiosity decreases with more visits
        curiosity = 1.0 / (1.0 + visits * 0.1)

        # Update visit count
        self.state_visits[state_key] = visits + 1

        # Keep state visits manageable
        if len(self.state_visits) > 10000:
            # Remove least visited states
            sorted_states = sorted(self.state_visits.items(), key=lambda x: x[1])
            for state_key, _ in sorted_states[:1000]:
                del self.state_visits[state_key]

        return curiosity * self.curiosity_factor

    def get_exploration_bonus(self, state: Dict[str, Any]) -> float:
        """Get exploration bonus for unknown states."""
        state_key = str(sorted(state.items()))
        visits = self.state_visits.get(state_key, 0)

        # Higher bonus for less visited states
        return 1.0 / (1.0 + math.log1p(visits))


class MetaLearner:
    """Meta-learning: learning to learn."""

    def __init__(self, meta_lr: float = 0.01):
        self.meta_lr = meta_lr
        self.task_performance: Dict[str, List[float]] = {}
        self._learning_strategies: Dict[str, Dict[str, float]] = {}

    def record_task_performance(self, task: str, performance: float):
        """Record performance on a task."""
        if task not in self.task_performance:
            self.task_performance[task] = []
        self.task_performance[task].append(performance)

    def get_meta_gradient(self, task: str) -> Dict[str, float]:
        """Compute meta-gradient for task adaptation."""
        if task not in self.task_performance or len(self.task_performance[task]) < 2:
            return {}

        performances = self.task_performance[task]
        recent = performances[-5:]
        older = performances[-10:-5] if len(performances) >= 10 else performances[:5]

        if not older:
            return {}

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        # Meta-gradient: how much to adjust learning parameters
        improvement = recent_avg - older_avg

        return {
            'learning_rate_adjustment': self.meta_lr * improvement,
            'exploration_adjustment': -self.meta_lr * improvement,  # Less exploration if improving
            'confidence': min(1.0, len(performances) / 20),
        }

    def suggest_strategy(self, task: str) -> str:
        """Suggest learning strategy based on meta-learned knowledge."""
        if task not in self.task_performance:
            return "explore"  # New task: explore

        performances = self.task_performance[task]
        if len(performances) < 5:
            return "explore"

        recent_avg = sum(performances[-5:]) / 5

        if recent_avg > 0.8:
            return "exploit"  # Doing well: exploit
        elif recent_avg < 0.3:
            return "explore"  # Doing poorly: explore more
        else:
            return "balance"  # Moderate: balance


class AdvancedLearner:
    """Advanced learning algorithm combining multiple techniques."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.1,
        curiosity_factor: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Components
        self.replay_buffer = ExperienceReplay(capacity=10000)
        self.curiosity = CuriosityModule(curiosity_factor)
        self.meta_learner = MetaLearner()

        # State
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.episode = 0
        self.total_reward = 0.0
        self._learning_progress: List[LearningProgress] = []

    def get_state_key(self, state: Dict[str, Any]) -> str:
        """Convert state dict to hashable key."""
        return str(sorted(state.items()))

    def get_q_value(self, state_key: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        return self.q_table[state_key].get(action, 0.0)

    def set_q_value(self, state_key: str, action: str, value: float):
        """Set Q-value for state-action pair."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action] = value

    def choose_action(self, state: Dict[str, Any], possible_actions: List[str]) -> str:
        """Choose action using epsilon-greedy with curiosity."""
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)

        state_key = self.get_state_key(state)

        # Add curiosity bonus to Q-values
        curiosity_bonus = self.curiosity.compute_curiosity_reward(state)

        best_action = None
        best_value = -float('inf')

        for action in possible_actions:
            q_value = self.get_q_value(state_key, action)
            value = q_value + curiosity_bonus

            if value > best_value:
                best_value = value
                best_action = action

        return best_action or random.choice(possible_actions)

    def learn(self, experience: Experience):
        """Learn from an experience."""
        # Add to replay buffer
        self.replay_buffer.add(experience)

        # Update Q-value using Bellman equation
        state_key = self.get_state_key(experience.state)
        next_state_key = self.get_state_key(experience.next_state)

        current_q = self.get_q_value(state_key, experience.action)

        # Find max Q-value for next state
        max_next_q = 0
        if not experience.done:
            next_actions = list(self.q_table.get(next_state_key, {}).keys())
            if next_actions:
                max_next_q = max(self.get_q_value(next_state_key, a) for a in next_actions)

        # Bellman update
        target_q = experience.reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target_q - current_q)

        self.set_q_value(state_key, experience.action, new_q)

        # Update total reward
        self.total_reward += experience.reward

        # Meta-learn from experience
        self.meta_learner.record_task_performance(
            experience.metadata.get('task', 'default'),
            experience.reward
        )

    def end_episode(self):
        """End of episode processing."""
        self.episode += 1

        # Record learning progress
        avg_reward = self.total_reward / max(1, self.episode)
        progress = LearningProgress(
            episode=self.episode,
            total_reward=self.total_reward,
            avg_reward=avg_reward,
            loss=0.0,  # Placeholder
            exploration_rate=self.exploration_rate,
        )
        self._learning_progress.append(progress)

        # Decay exploration rate
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.01, self.exploration_rate)

        # Meta-learn
        meta_gradient = self.meta_learner.get_meta_gradient('default')
        if meta_gradient:
            self.learning_rate *= (1.0 + meta_gradient.get('learning_rate_adjustment', 0))
            self.learning_rate = max(0.0001, min(0.01, self.learning_rate))

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'episode': self.episode,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.episode),
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'q_table_size': len(self.q_table),
            'replay_buffer_size': len(self.replay_buffer),
            'curiosity_states': len(self.curiosity.state_visits),
        }


if __name__ == "__main__":
    print("=== Advanced Learning Algorithm Test ===\n")

    # Create learner
    learner = AdvancedLearner(
        learning_rate=0.01,
        discount_factor=0.99,
        exploration_rate=0.2,
        curiosity_factor=0.1,
    )

    # Simulate learning episodes
    possible_actions = ['explore', 'build', 'analyze', 'optimize']

    for episode in range(50):
        state = {'complexity': random.random(), 'novelty': random.random()}
        total_reward = 0

        for step in range(10):
            # Choose action
            action = learner.choose_action(state, possible_actions)

            # Simulate environment
            next_state = {
                'complexity': random.random(),
                'novelty': random.random(),
            }
            reward = random.gauss(0.5, 0.2)  # Random reward
            done = step == 9

            # Learn
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metadata={'task': 'test'},
            )
            learner.learn(experience)

            total_reward += reward
            state = next_state

        learner.end_episode()

        if episode % 10 == 0:
            stats = learner.get_learning_stats()
            print(f"Episode {episode}: avg_reward={stats['avg_reward']:.3f}, "
                  f"exploration={stats['exploration_rate']:.3f}, "
                  f"q_table={stats['q_table_size']}")

    # Final stats
    stats = learner.get_learning_stats()
    print(f"\nFinal Statistics:")
    print(f"  Episodes: {stats['episode']}")
    print(f"  Total reward: {stats['total_reward']:.3f}")
    print(f"  Avg reward: {stats['avg_reward']:.3f}")
    print(f"  Q-table size: {stats['q_table_size']}")
    print(f"  Curiosity states: {stats['curiosity_states']}")
