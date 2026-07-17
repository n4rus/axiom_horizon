"""End-to-End Integration Test for Kai AGI System.

Validates the complete system works together, from core modules
to API layer, with proper error handling and reporting.
"""

from __future__ import annotations

import sys
import os
import time
import json
import math
import traceback
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))


class IntegrationTest:
    """End-to-end integration test suite."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def run_test(self, name: str, test_func) -> bool:
        """Run a single test and record result."""
        start = time.time()
        try:
            test_func()
            duration = time.time() - start
            self.results.append({
                'name': name,
                'status': 'passed',
                'duration': duration,
            })
            print(f"  ✓ {name} ({duration:.3f}s)")
            return True
        except Exception as e:
            duration = time.time() - start
            self.results.append({
                'name': name,
                'status': 'failed',
                'duration': duration,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
            print(f"  ✗ {name}: {e}")
            return False

    def test_adaptive_constants_pipeline(self):
        """Test adaptive constants computation and application."""
        from adaptive_constants import compute_adaptive_constants, get_default_constants

        defaults = get_default_constants()

        # Compute adaptive constants
        adapted = compute_adaptive_constants(
            time_secs=300.0,
            vfe_history=[0.5 - 0.01*i for i in range(50)],
            ricci=3.5,
            vfe_trend=-0.008,
        )

        # Verify values are within bounds
        assert 2 <= adapted.explore_span <= 20, f"explore_span out of bounds: {adapted.explore_span}"
        assert 1 <= adapted.converge_span <= 15, f"converge_span out of bounds: {adapted.converge_span}"
        assert 0.05 <= adapted.mode_eps <= 1.0, f"mode_eps out of bounds: {adapted.mode_eps}"
        assert 0.01 <= adapted.ricci_ema_lambda <= 0.2, f"ricci_ema_lambda out of bounds: {adapted.ricci_ema_lambda}"
        assert 0.001 <= adapted.wm_alpha <= 0.05, f"wm_alpha out of bounds: {adapted.wm_alpha}"
        assert 0.01 <= adapted.t_min <= 1.0, f"t_min out of bounds: {adapted.t_min}"
        assert 0.5 <= adapted.t_max <= 5.0, f"t_max out of bounds: {adapted.t_max}"
        assert 0.01 <= adapted.alpha_weight <= 0.3, f"alpha_weight out of bounds: {adapted.alpha_weight}"
        assert 0.0001 <= adapted.velocity_threshold <= 0.1, f"velocity_threshold out of bounds: {adapted.velocity_threshold}"

        # Verify conversion to/from dict
        d = adapted.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 9

    def test_health_monitor_integration(self):
        """Test health monitor with registered checks."""
        from health_monitor import HealthMonitor, HealthCheck, HealthStatus

        monitor = HealthMonitor(check_interval=5.0)

        def mock_check():
            return HealthCheck(
                component="mock",
                status=HealthStatus.HEALTHY,
                message="Mock check passed"
            )

        monitor.register_health_check(mock_check)
        results = monitor.check_health()

        assert "mock" in results
        assert results["mock"].status == HealthStatus.HEALTHY

    def test_security_validation_chain(self):
        """Test security validation pipeline."""
        from security import InputValidator, SecuritySanitizer

        # Test string validation
        result = InputValidator.validate_string("test input", max_length=100)
        assert result.is_valid
        assert result.sanitized_value == "test input"

        # Test HTML escaping
        result = InputValidator.validate_string("<script>alert('xss')</script>")
        assert result.is_valid
        assert "<script>" not in result.sanitized_value

        # Test filename sanitization
        safe = SecuritySanitizer.sanitize_filename("../../etc/passwd")
        assert "/" not in safe
        assert ".." not in safe

    def test_performance_caching(self):
        """Test LRU cache functionality."""
        from performance import LRUCache

        cache = LRUCache(max_size=3, ttl_seconds=60.0)

        # Test basic operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Test eviction
        cache.set("key4", "value4")
        assert cache.get("key1") is None  # Evicted

        # Test stats
        stats = cache.stats()
        assert stats["size"] == 3
        assert stats["hits"] > 0

    def test_quantum_reasoning_integration(self):
        """Test quantum reasoning with context."""
        from quantum_reasoning import QuantumReasoner

        qr = QuantumReasoner(n_superpositions=3)
        context = {'efe': -0.5, 'ricci': 4.0, 'vfe': -0.1, 'previous_goals': []}

        # Test reasoning
        result = qr.reason("test goal", context['efe'], context)
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Test memory
        assert len(qr.memory.states) > 0

    def test_evolution_engine_integration(self):
        """Test evolution engine with competition."""
        from evolution_engine import EvolutionEngine

        engine = EvolutionEngine(population_size=3)
        tasks = [{'goal': f'task_{i}', 'difficulty': 0.5} for i in range(3)]

        engine.run_competition(tasks)
        stats = engine.get_stats()

        assert stats['population'] == 3
        assert stats['generation'] >= 0

        # Test evolution
        engine.evolve()
        new_stats = engine.get_stats()
        assert new_stats['generation'] == stats['generation'] + 1

    def test_knowledge_transfer_integration(self):
        """Test knowledge transfer across domains."""
        from knowledge_transfer import KnowledgeTransfer

        kt = KnowledgeTransfer()
        kt.add_knowledge('quantum mechanics', 'physics')
        kt.add_knowledge('neural networks', 'computer science')

        transfers = kt.transfer('physics', 'computer science')
        stats = kt.get_stats()

        assert stats['total_nodes'] >= 2
        assert stats['cross_domain_links'] >= 0

    def test_meta_cognition_integration(self):
        """Test meta-cognition with self-awareness."""
        from meta_cognition import MetaCognitiveAgent
        from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

        qm = QuantumSuperpositionManager()
        wfc = WavefunctionCollapse(qm)
        base = Agent("Test", "goal", AgentRole.DISCOVERY, qm, wfc)

        meta = MetaCognitiveAgent("MetaTest", base)

        # Test reasoning
        meta.reason("What should we do?", {'efe': -0.4})
        meta.reason("How can we improve?", {'efe': -0.5})

        summary = meta.get_meta_summary()
        assert summary['reasoning_trace']['total_steps'] == 2

    def test_emergence_tracking_integration(self):
        """Test emergence tracking with interactions."""
        from emergence_tracker import EmergenceTracker

        tracker = EmergenceTracker(window_size=30)

        # Record interactions
        for i in range(25):
            tracker.record_interaction(
                f"Agent{i%3}",
                f"Agent{(i+1)%3}",
                "explore" if i % 2 == 0 else "build",
                {'success': True}
            )

        stats = tracker.get_stats()
        assert stats['total_interactions'] == 25
        assert stats['total_behaviors'] > 0

    def test_multi_agent_dialogue_integration(self):
        """Test multi-agent dialogue with quantum responses."""
        from multi_agent_dialogue import MultiAgentDialogue
        from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

        qm = QuantumSuperpositionManager()
        wfc = WavefunctionCollapse(qm)
        agent1 = Agent('Dialogue1', 'goal1', AgentRole.DISCOVERY, qm, wfc)
        agent2 = Agent('Dialogue2', 'goal2', AgentRole.APPLICATION, qm, wfc)

        dialogue = MultiAgentDialogue(agent1, agent2)
        result = dialogue.run_round("What is the best approach?")

        assert result is not None

    def test_logging_integration(self):
        """Test structured logging pipeline."""
        from logging_config import setup_logging, log_inference

        logger = setup_logging(level='DEBUG')
        log_inference(logger, "test", {"policy": "test", "efe": -0.5}, 42.5)

        log_file = os.path.join(os.path.dirname(__file__), '..', '..', '.axiom_state', 'logs', 'kai_insight.log')
        assert os.path.exists(log_file)

    def run_all(self):
        """Run all integration tests."""
        print("=== Kai AGI End-to-End Integration Test ===\n")

        tests = [
            ("Adaptive Constants Pipeline", self.test_adaptive_constants_pipeline),
            ("Health Monitor Integration", self.test_health_monitor_integration),
            ("Security Validation Chain", self.test_security_validation_chain),
            ("Performance Caching", self.test_performance_caching),
            ("Quantum Reasoning Integration", self.test_quantum_reasoning_integration),
            ("Evolution Engine Integration", self.test_evolution_engine_integration),
            ("Knowledge Transfer Integration", self.test_knowledge_transfer_integration),
            ("Meta-Cognition Integration", self.test_meta_cognition_integration),
            ("Emergence Tracking Integration", self.test_emergence_tracking_integration),
            ("Multi-Agent Dialogue Integration", self.test_multi_agent_dialogue_integration),
            ("Logging Integration", self.test_logging_integration),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            if self.run_test(name, test_func):
                passed += 1
            else:
                failed += 1

        total_duration = time.time() - self.start_time

        print(f"\n{'='*50}")
        print(f"Results: {passed}/{passed+failed} passed ({total_duration:.3f}s)")
        print(f"{'='*50}")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if r['status'] == 'failed':
                    print(f"  - {r['name']}: {r['error'][:80]}")

        return failed == 0


if __name__ == "__main__":
    test = IntegrationTest()
    success = test.run_all()
    sys.exit(0 if success else 1)
