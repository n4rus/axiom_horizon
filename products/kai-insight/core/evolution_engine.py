"""Multi-Agent Competition and Evolution System.

Agents compete on tasks, winning strategies propagate to next generation.
Implements evolutionary selection with genetic operators.
"""

import math
import random
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AgentGenome:
    """Heritable traits for an agent."""
    exploration_bias: float = 0.5
    depth_preference: int = 2
    risk_tolerance: float = 0.3
    learning_rate: float = 0.1
    memory_retention: float = 0.8
    entanglement_weight: float = 0.5

    def mutate(self, mutation_rate: float = 0.1) -> 'AgentGenome':
        g = AgentGenome(
            exploration_bias=self._mutate_val(self.exploration_bias, 0.0, 1.0, mutation_rate),
            depth_preference=max(1, min(5, self.depth_preference + random.choice([-1, 0, 0, 1]))),
            risk_tolerance=self._mutate_val(self.risk_tolerance, 0.0, 1.0, mutation_rate),
            learning_rate=self._mutate_val(self.learning_rate, 0.01, 0.5, mutation_rate),
            memory_retention=self._mutate_val(self.memory_retention, 0.1, 1.0, mutation_rate),
            entanglement_weight=self._mutate_val(self.entanglement_weight, 0.0, 1.0, mutation_rate),
        )
        return g

    def crossover(self, other: 'AgentGenome') -> 'AgentGenome':
        return AgentGenome(
            exploration_bias=random.choice([self.exploration_bias, other.exploration_bias]),
            depth_preference=random.choice([self.depth_preference, other.depth_preference]),
            risk_tolerance=random.choice([self.risk_tolerance, other.risk_tolerance]),
            learning_rate=random.choice([self.learning_rate, other.learning_rate]),
            memory_retention=random.choice([self.memory_retention, other.memory_retention]),
            entanglement_weight=random.choice([self.entanglement_weight, other.entanglement_weight]),
        )

    @staticmethod
    def _mutate_val(val: float, lo: float, hi: float, rate: float) -> float:
        if random.random() < rate:
            return lo + random.random() * (hi - lo)
        return val


class CompetitiveAgent:
    def __init__(self, name: str, genome: AgentGenome = None):
        self.name = name
        self.genome = genome or AgentGenome()
        self.fitness: float = 0.5
        self.tasks_completed: int = 0
        self.total_reward: float = 0.0
        self.generation: int = 0
        self.memory: List[Dict] = []

    def evaluate_task(self, task: Dict) -> Dict:
        goal = task.get('goal', '')
        difficulty = task.get('difficulty', 0.5)

        success_prob = (self.genome.risk_tolerance * 0.3 +
                        self.genome.exploration_bias * 0.3 +
                        (1.0 - difficulty) * 0.4)
        success = random.random() < success_prob
        reward = success * (1.0 - difficulty) * 0.5

        self.tasks_completed += 1
        self.total_reward += reward
        self.fitness = self.fitness * 0.9 + reward * 0.1

        return {
            'agent': self.name,
            'task': goal,
            'success': success,
            'reward': reward,
            'fitness': self.fitness,
        }


class EvolutionEngine:
    def __init__(self, population_size: int = 6):
        self.population_size = population_size
        self.agents: List[CompetitiveAgent] = []
        self.generation: int = 0
        self.task_history: List[Dict] = []
        self._init_population()

    def _init_population(self):
        roles = ['explorer', 'builder', 'analyst', 'optimizer', 'coordinator', 'innovator']
        for i, role in enumerate(roles[:self.population_size]):
            genome = AgentGenome(
                exploration_bias=random.uniform(0.2, 0.8),
                risk_tolerance=random.uniform(0.1, 0.7),
                learning_rate=random.uniform(0.05, 0.3),
            )
            self.agents.append(CompetitiveAgent(f"Agent_{role}", genome))

    def run_competition(self, tasks: List[Dict]) -> List[Dict]:
        results = []
        for agent in self.agents:
            for task in tasks:
                result = agent.evaluate_task(task)
                result['generation'] = self.generation
                results.append(result)
        self.task_history.extend(results)
        return results

    def evolve(self) -> int:
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        survivors = self.agents[:max(2, self.population_size // 2)]

        new_agents = []
        while len(new_agents) < self.population_size:
            p1, p2 = random.sample(survivors, min(2, len(survivors)))
            child_genome = p1.genome.crossover(p2.genome)
            child_genome = child_genome.mutate(mutation_rate=0.2)
            child = CompetitiveAgent(
                f"Agent_gen{self.generation + 1}_{len(new_agents)}",
                child_genome,
            )
            child.generation = self.generation + 1
            new_agents.append(child)

        self.agents = new_agents
        self.generation += 1
        return self.generation

    def get_stats(self) -> Dict:
        fitnesses = [a.fitness for a in self.agents]
        return {
            'generation': self.generation,
            'population': len(self.agents),
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'max_fitness': max(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'total_tasks': sum(a.tasks_completed for a in self.agents),
            'best_agent': max(self.agents, key=lambda a: a.fitness).name if self.agents else None,
        }

    def run_evolution(self, generations: int = 5, tasks_per_gen: int = 4) -> List[Dict]:
        history = []
        for g in range(generations):
            tasks = [{'goal': f'task_{i}', 'difficulty': random.uniform(0.2, 0.8)}
                     for i in range(tasks_per_gen)]
            self.run_competition(tasks)
            stats = self.get_stats()
            history.append(stats)
            if g < generations - 1:
                self.evolve()
        return history


if __name__ == "__main__":
    print("=== Multi-Agent Competition & Evolution ===\n")

    engine = EvolutionEngine(population_size=6)
    history = engine.run_evolution(generations=5, tasks_per_gen=4)

    print(f"{'Gen':>4} {'Pop':>4} {'Avg Fit':>8} {'Max Fit':>8} {'Tasks':>6} {'Best Agent'}")
    print("-" * 60)
    for h in history:
        print(f"{h['generation']:>4} {h['population']:>4} {h['avg_fitness']:>8.3f} "
              f"{h['max_fitness']:>8.3f} {h['total_tasks']:>6} {h['best_agent']}")

    print(f"\nFinal generation: {engine.generation}")
    print(f"Best agent: {engine.get_stats()['best_agent']}")
    print(f"Best fitness: {engine.get_stats()['max_fitness']:.3f}")
