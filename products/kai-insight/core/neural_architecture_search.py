"""Neural Architecture Search for Kai AGI.

Automatically discovers optimal neural architectures:
- Search for best network topology
- Optimize hyperparameters
- Evolve architecture over time
"""

from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureNode:
    """A node in a neural architecture."""
    layer_type: str  # dense, conv, lstm, attention
    units: int
    activation: str = "relu"
    dropout: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Architecture:
    """A neural network architecture."""
    nodes: List[ArchitectureNode]
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuralArchitectureSearch:
    """Neural Architecture Search using evolutionary approach."""

    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[Architecture] = []
        self.generation = 0
        self.best_architecture: Optional[Architecture] = None
        self._layer_types = ["dense", "lstm", "attention", "conv"]
        self._activations = ["relu", "tanh", "gelu", "swish"]

    def create_random_architecture(self) -> Architecture:
        """Create a random architecture."""
        n_layers = random.randint(2, 6)
        nodes = []

        for _ in range(n_layers):
            layer_type = random.choice(self._layer_types)
            units = random.choice([32, 64, 128, 256, 512])
            activation = random.choice(self._activations)
            dropout = random.uniform(0.0, 0.5)

            node = ArchitectureNode(
                layer_type=layer_type,
                units=units,
                activation=activation,
                dropout=dropout,
            )
            nodes.append(node)

        return Architecture(nodes=nodes)

    def initialize_population(self):
        """Initialize population with random architectures."""
        self.population = []
        for _ in range(self.population_size):
            arch = self.create_random_architecture()
            self.population.append(arch)

    def evaluate_architecture(self, arch: Architecture) -> float:
        """Evaluate architecture fitness (simulated)."""
        # Simulate fitness based on architecture properties
        n_layers = len(arch.nodes)
        total_units = sum(n.layers for n in arch.nodes if hasattr(n, 'layers'))
        total_units = sum(n.units for n in arch.nodes)

        # Simple heuristic: moderate depth and width is best
        depth_score = 1.0 - abs(n_layers - 4) / 4.0
        width_score = 1.0 - abs(total_units / n_layers - 128) / 128.0

        # Regularization for complexity
        complexity_penalty = n_layers * 0.05

        fitness = (depth_score + width_score) / 2.0 - complexity_penalty
        fitness = max(0.0, min(1.0, fitness))

        # Add some randomness
        fitness += random.gauss(0, 0.05)
        fitness = max(0.0, min(1.0, fitness))

        return fitness

    def select_parents(self) -> Tuple[Architecture, Architecture]:
        """Select parents for reproduction."""
        # Tournament selection
        tournament_size = 3

        def tournament():
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            return max(candidates, key=lambda a: a.fitness)

        return tournament(), tournament()

    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover two architectures."""
        # Choose crossover point
        max_len = max(len(parent1.nodes), len(parent2.nodes))
        crossover_point = random.randint(1, max_len - 1) if max_len > 1 else 1

        # Create child
        child_nodes = []
        for i in range(max_len):
            if i < crossover_point:
                if i < len(parent1.nodes):
                    child_nodes.append(parent1.nodes[i])
                else:
                    child_nodes.append(parent2.nodes[i])
            else:
                if i < len(parent2.nodes):
                    child_nodes.append(parent2.nodes[i])
                else:
                    child_nodes.append(parent1.nodes[i])

        return Architecture(
            nodes=child_nodes,
            parent_id=f"{parent1.fitness:.3f}_{parent2.fitness:.3f}",
        )

    def mutate(self, arch: Architecture, mutation_rate: float = 0.1) -> Architecture:
        """Mutate an architecture."""
        mutated_nodes = []

        for node in arch.nodes:
            if random.random() < mutation_rate:
                # Mutate this node
                mutated_node = ArchitectureNode(
                    layer_type=random.choice(self._layer_types) if random.random() < 0.3 else node.layer_type,
                    units=random.choice([32, 64, 128, 256, 512]) if random.random() < 0.3 else node.units,
                    activation=random.choice(self._activations) if random.random() < 0.2 else node.activation,
                    dropout=min(1.0, max(0.0, node.dropout + random.gauss(0, 0.1))) if random.random() < 0.2 else node.dropout,
                )
                mutated_nodes.append(mutated_node)
            else:
                mutated_nodes.append(node)

        # Maybe add or remove a layer
        if random.random() < 0.1 and len(mutated_nodes) > 2:
            # Remove a random layer
            idx = random.randint(0, len(mutated_nodes) - 1)
            mutated_nodes.pop(idx)
        elif random.random() < 0.1 and len(mutated_nodes) < 8:
            # Add a new layer
            new_node = ArchitectureNode(
                layer_type=random.choice(self._layer_types),
                units=random.choice([32, 64, 128, 256]),
                activation=random.choice(self._activations),
            )
            idx = random.randint(0, len(mutated_nodes))
            mutated_nodes.insert(idx, new_node)

        return Architecture(nodes=mutated_nodes)

    def evolve(self):
        """Evolve one generation."""
        # Evaluate all architectures
        for arch in self.population:
            arch.fitness = self.evaluate_architecture(arch)
            arch.generation = self.generation

        # Sort by fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)

        # Update best
        if self.population:
            self.best_architecture = self.population[0]

        # Create new population
        new_population = []

        # Elitism: keep top 10%
        elite_count = max(1, self.population_size // 10)
        new_population.extend(self.population[:elite_count])

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def get_best_architecture(self) -> Optional[Architecture]:
        """Get the best architecture found."""
        return self.best_architecture

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.population:
            return {'generation': self.generation, 'population_size': 0}

        fitnesses = [a.fitness for a in self.population]
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'best_fitness': self.best_architecture.fitness if self.best_architecture else 0,
            'avg_layers': sum(len(a.nodes) for a in self.population) / len(self.population),
        }


if __name__ == "__main__":
    print("=== Neural Architecture Search Test ===\n")

    nas = NeuralArchitectureSearch(population_size=20)
    nas.initialize_population()

    # Run evolution
    for gen in range(10):
        nas.evolve()
        stats = nas.get_search_stats()
        print(f"Generation {gen}: avg_fitness={stats['avg_fitness']:.3f}, "
              f"max_fitness={stats['max_fitness']:.3f}, "
              f"avg_layers={stats['avg_layers']:.1f}")

    # Show best architecture
    best = nas.get_best_architecture()
    if best:
        print(f"\nBest Architecture:")
        print(f"  Fitness: {best.fitness:.3f}")
        print(f"  Layers: {len(best.nodes)}")
        for i, node in enumerate(best.nodes):
            print(f"    Layer {i}: {node.layer_type}({node.units}) - {node.activation}")
