"""Cross-Domain Knowledge Transfer for Kai AGI.

Enables knowledge learned in one domain to transfer to another.
Uses entanglement-based similarity to map insights across domains.
"""

import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class KnowledgeNode:
    def __init__(self, concept: str, domain: str, strength: float = 1.0):
        self.concept = concept
        self.domain = domain
        self.strength = strength
        self.connections: Dict[str, float] = {}
        self.access_count: int = 0

    def connect(self, other: 'KnowledgeNode', weight: float):
        self.connections[other.concept] = weight
        other.connections[self.concept] = weight

    def decay(self, factor: float = 0.99):
        self.strength *= factor


class KnowledgeTransfer:
    def __init__(self):
        self.domains: Dict[str, Dict[str, KnowledgeNode]] = defaultdict(dict)
        self.transfer_history: List[Dict] = []
        self.cross_domain_links: List[Tuple[str, str, float]] = []

    def add_knowledge(self, concept: str, domain: str, strength: float = 1.0) -> KnowledgeNode:
        if concept in self.domains[domain]:
            self.domains[domain][concept].strength += strength * 0.1
            return self.domains[domain][concept]

        node = KnowledgeNode(concept, domain, strength)
        self.domains[domain][concept] = node

        for d, concepts in self.domains.items():
            if d != domain:
                for c, n in concepts.items():
                    sim = self._concept_similarity(concept, c)
                    if sim > 0.3:
                        node.connect(n, sim)
                        self.cross_domain_links.append((concept, c, sim))

        return node

    def _concept_similarity(self, c1: str, c2: str) -> float:
        words1 = set(c1.lower().split())
        words2 = set(c2.lower().split())
        if not words1 or not words2:
            return 0.0
        inter = len(words1 & words2)
        union = len(words1 | words2)
        return inter / union if union > 0 else 0.0

    def transfer(self, source_domain: str, target_domain: str, max_transfers: int = 3) -> List[Dict]:
        transferred = []
        if source_domain not in self.domains:
            return transferred

        source_concepts = self.domains[source_domain]
        target_concepts = self.domains.get(target_domain, {})

        candidates = []
        for sc_name, sc in source_concepts.items():
            best_sim = 0.0
            best_target = None
            for tc_name, tc in target_concepts.items():
                sim = self._concept_similarity(sc_name, tc_name)
                if sim > best_sim:
                    best_sim = sim
                    best_target = tc_name

            if best_sim < 0.5 and sc.strength > 0.5:
                transfer_score = sc.strength * (1.0 - best_sim)
                candidates.append((sc_name, transfer_score, best_sim))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for sc_name, score, existing_sim in candidates[:max_transfers]:
            new_concept = f"{sc_name} (from {source_domain})"
            self.add_knowledge(new_concept, target_domain, score * 0.5)

            transfer_record = {
                'source': sc_name,
                'source_domain': source_domain,
                'target': new_concept,
                'target_domain': target_domain,
                'score': score,
                'existing_similarity': existing_sim,
            }
            self.transfer_history.append(transfer_record)
            transferred.append(transfer_record)

        return transferred

    def get_stats(self) -> Dict:
        total_nodes = sum(len(c) for c in self.domains.values())
        total_links = len(self.cross_domain_links)
        return {
            'total_nodes': total_nodes,
            'domains': {d: len(c) for d, c in self.domains.items()},
            'cross_domain_links': total_links,
            'transfers': len(self.transfer_history),
        }


if __name__ == "__main__":
    print("=== Cross-Domain Knowledge Transfer ===\n")

    kt = KnowledgeTransfer()

    domains = {
        'physics': ['quantum mechanics', 'thermodynamics', 'relativity', 'entropy'],
        'biology': ['evolution', 'cell structure', 'neural networks', 'adaptation'],
        'computer science': ['machine learning', 'algorithms', 'optimization', 'neural networks'],
        'mathematics': ['linear algebra', 'calculus', 'probability', 'optimization'],
    }

    for domain, concepts in domains.items():
        for concept in concepts:
            kt.add_knowledge(concept, domain)

    print("Knowledge base:")
    for d, stats in kt.get_stats()['domains'].items():
        print(f"  {d}: {stats} concepts")

    print(f"\nCross-domain links: {kt.get_stats()['cross_domain_links']}")

    print("\n--- Transferring physics -> biology ---")
    transfers = kt.transfer('physics', 'biology')
    for t in transfers:
        print(f"  {t['source']} -> {t['target']} (score={t['score']:.3f})")

    print("\n--- Transferring mathematics -> computer science ---")
    transfers = kt.transfer('mathematics', 'computer science')
    for t in transfers:
        print(f"  {t['source']} -> {t['target']} (score={t['score']:.3f})")

    print(f"\nFinal stats: {kt.get_stats()}")
