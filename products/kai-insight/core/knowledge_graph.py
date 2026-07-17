"""Knowledge Graph for Kai AGI.

Implements a knowledge graph for reasoning about relationships:
- Store and query conceptual relationships
- Perform graph-based reasoning
- Support analogical reasoning
- Track concept evolution over time
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A concept in the knowledge graph."""
    id: str
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    strength: float = 1.0  # Activation strength


@dataclass
class Relationship:
    """A relationship between concepts."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    activation_count: int = 0


class KnowledgeGraph:
    """Knowledge graph for conceptual reasoning."""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.relationships: List[Relationship] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._concept_index: Dict[str, str] = {}  # name -> id

    def add_concept(self, name: str, description: str = "", properties: Optional[Dict] = None) -> Concept:
        """Add a concept to the graph."""
        concept_id = f"concept_{len(self.concepts)}"
        concept = Concept(
            id=concept_id,
            name=name,
            description=description,
            properties=properties or {},
        )
        self.concepts[concept_id] = concept
        self._concept_index[name] = concept_id
        return concept

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        concept_id = self._concept_index.get(name)
        if concept_id:
            return self.concepts.get(concept_id)
        return None

    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        weight: float = 1.0,
        properties: Optional[Dict] = None,
    ) -> Optional[Relationship]:
        """Add a relationship between concepts."""
        source = self.get_concept(source_name)
        target = self.get_concept(target_name)

        if not source or not target:
            return None

        relationship = Relationship(
            source_id=source.id,
            target_id=target.id,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
        )

        self.relationships.append(relationship)
        self._adjacency[source.id].append(target.id)
        self._adjacency[target.id].append(source.id)  # Bidirectional

        return relationship

    def get_relationships(self, concept_name: str, relation_type: Optional[str] = None) -> List[Relationship]:
        """Get relationships for a concept."""
        concept = self.get_concept(concept_name)
        if not concept:
            return []

        return [
            r for r in self.relationships
            if (r.source_id == concept.id or r.target_id == concept.id)
            and (relation_type is None or r.relation_type == relation_type)
        ]

    def get_neighbors(self, concept_name: str, max_depth: int = 1) -> List[Concept]:
        """Get neighboring concepts."""
        concept = self.get_concept(concept_name)
        if not concept:
            return []

        visited = set()
        neighbors = []
        queue = [(concept.id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            if current_id != concept.id:
                neighbor = self.concepts.get(current_id)
                if neighbor:
                    neighbors.append(neighbor)

            if depth < max_depth:
                for neighbor_id in self._adjacency.get(current_id, []):
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, depth + 1))

        return neighbors

    def find_path(self, source_name: str, target_name: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find path between two concepts."""
        source = self.get_concept(source_name)
        target = self.get_concept(target_name)

        if not source or not target:
            return None

        # BFS to find path
        queue = [(source.id, [source.name])]
        visited = {source.id}

        while queue:
            current_id, path = queue.pop(0)

            if current_id == target.id:
                return path

            if len(path) > max_depth:
                continue

            for neighbor_id in self._adjacency.get(current_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor = self.concepts.get(neighbor_id)
                    if neighbor:
                        queue.append((neighbor_id, path + [neighbor.name]))

        return None

    def compute_similarity(self, concept1_name: str, concept2_name: str) -> float:
        """Compute similarity between two concepts."""
        concept1 = self.get_concept(concept1_name)
        concept2 = self.get_concept(concept2_name)

        if not concept1 or not concept2:
            return 0.0

        # Common neighbors
        neighbors1 = set(c.id for c in self.get_neighbors(concept1_name, max_depth=2))
        neighbors2 = set(c.id for c in self.get_neighbors(concept2_name, max_depth=2))

        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)

        if union == 0:
            return 0.0

        # Jaccard similarity
        jaccard = intersection / union

        # Property similarity
        props1 = set(concept1.properties.keys())
        props2 = set(concept2.properties.keys())
        prop_intersection = len(props1 & props2)
        prop_union = len(props1 | props2)
        prop_similarity = prop_intersection / prop_union if prop_union > 0 else 0.0

        return 0.7 * jaccard + 0.3 * prop_similarity

    def analogical_reasoning(self, source_name: str, target_name: str) -> Dict[str, Any]:
        """Perform analogical reasoning between concepts."""
        source = self.get_concept(source_name)
        target = self.get_concept(target_name)

        if not source or not target:
            return {'analogy': None}

        # Find similar structures
        source_relationships = self.get_relationships(source_name)
        target_relationships = self.get_relationships(target_name)

        # Find common relationship types
        source_types = set(r.relation_type for r in source_relationships)
        target_types = set(r.relation_type for r in target_relationships)
        common_types = source_types & target_types

        # Build analogy
        analogies = []
        for rel_type in common_types:
            source_related = [
                self.concepts[r.target_id].name if r.source_id == source.id else self.concepts[r.source_id].name
                for r in source_relationships if r.relation_type == rel_type
            ]
            target_related = [
                self.concepts[r.target_id].name if r.source_id == target.id else self.concepts[r.source_id].name
                for r in target_relationships if r.relation_type == rel_type
            ]

            if source_related and target_related:
                analogies.append({
                    'relation_type': rel_type,
                    'source_related': source_related[:3],
                    'target_related': target_related[:3],
                })

        return {
            'source': source_name,
            'target': target_name,
            'common_relations': list(common_types),
            'analogies': analogies,
            'similarity': self.compute_similarity(source_name, target_name),
        }

    def activate_concept(self, concept_name: str, activation: float = 1.0):
        """Activate a concept and spread activation."""
        concept = self.get_concept(concept_name)
        if not concept:
            return

        concept.strength = min(1.0, concept.strength + activation)
        concept.access_count += 1

        # Spread activation to neighbors
        for neighbor in self.get_neighbors(concept_name, max_depth=1):
            decay = 0.5  # Activation decay
            neighbor.strength = min(1.0, neighbor.strength + activation * decay)

    def get_activated_concepts(self, threshold: float = 0.5) -> List[Concept]:
        """Get concepts with activation above threshold."""
        return [
            concept for concept in self.concepts.values()
            if concept.strength >= threshold
        ]

    def decay_activations(self, decay_rate: float = 0.1):
        """Decay all concept activations."""
        for concept in self.concepts.values():
            concept.strength = max(0.0, concept.strength - decay_rate)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'total_concepts': len(self.concepts),
            'total_relationships': len(self.relationships),
            'avg_relationships_per_concept': len(self.relationships) / max(1, len(self.concepts)),
            'unique_relation_types': len(set(r.relation_type for r in self.relationships)),
            'avg_concept_strength': sum(c.strength for c in self.concepts.values()) / max(1, len(self.concepts)),
        }


if __name__ == "__main__":
    print("=== Knowledge Graph Test ===\n")

    kg = KnowledgeGraph()

    # Add concepts
    concepts = [
        ("AGI", "Artificial General Intelligence"),
        ("machine_learning", "Machine Learning"),
        ("deep_learning", "Deep Learning"),
        ("neural_network", "Neural Network"),
        ("reinforcement_learning", "Reinforcement Learning"),
        ("quantum_computing", "Quantum Computing"),
        ("knowledge_graph", "Knowledge Graph"),
        ("reasoning", "Reasoning"),
        ("learning", "Learning"),
        ("intelligence", "Intelligence"),
    ]

    for name, desc in concepts:
        kg.add_concept(name, desc)

    # Add relationships
    relationships = [
        ("AGI", "machine_learning", "contains"),
        ("AGI", "reasoning", "requires"),
        ("AGI", "learning", "requires"),
        ("machine_learning", "deep_learning", "includes"),
        ("machine_learning", "reinforcement_learning", "includes"),
        ("deep_learning", "neural_network", "uses"),
        ("quantum_computing", "AGI", "could_enhance"),
        ("knowledge_graph", "reasoning", "supports"),
        ("knowledge_graph", "AGI", "component_of"),
        ("intelligence", "AGI", "goal_of"),
    ]

    for source, target, rel_type in relationships:
        kg.add_relationship(source, target, rel_type)

    # Test graph operations
    print("Graph Statistics:")
    stats = kg.get_graph_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test path finding
    print("\nPath from 'quantum_computing' to 'neural_network':")
    path = kg.find_path("quantum_computing", "neural_network")
    print(f"  Path: {' -> '.join(path) if path else 'No path found'}")

    # Test similarity
    print("\nSimilarity between concepts:")
    pairs = [("AGI", "intelligence"), ("machine_learning", "deep_learning"), ("reasoning", "learning")]
    for c1, c2 in pairs:
        sim = kg.compute_similarity(c1, c2)
        print(f"  {c1} <-> {c2}: {sim:.3f}")

    # Test analogical reasoning
    print("\nAnalogical Reasoning:")
    analogy = kg.analogical_reasoning("machine_learning", "quantum_computing")
    print(f"  Source: {analogy['source']}")
    print(f"  Target: {analogy['target']}")
    print(f"  Similarity: {analogy['similarity']:.3f}")
    print(f"  Common relations: {analogy['common_relations']}")

    # Test activation spreading
    print("\nActivation Spreading:")
    kg.activate_concept("AGI", 1.0)
    activated = kg.get_activated_concepts(threshold=0.3)
    print(f"  Activated concepts: {[c.name for c in activated[:5]]}")
