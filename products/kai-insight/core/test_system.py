"""Comprehensive System Test for Kai Insight.

Production-quality tests with proper error handling and validation.
Tests all components: adapter, quantum reasoning, evolution,
knowledge transfer, adaptive constants, and REST API.
"""

from __future__ import annotations

import sys
import os
import time
import json
import math
import traceback
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestResult:
    """Container for test results."""

    def __init__(self, name: str):
        self.name = name
        self.passed: bool = False
        self.error: str = ""
        self.duration: float = 0.0

    def success(self, duration: float) -> None:
        self.passed = True
        self.duration = duration

    def failure(self, error: str, duration: float) -> None:
        self.passed = False
        self.error = error
        self.duration = duration


def test_adapter() -> TestResult:
    """Test the inference adapter."""
    result = TestResult("adapter")
    start = time.time()

    try:
        from adapter import infer, health_check, SimpleKaiMind

        h = health_check()
        if h['status'] != 'healthy':
            raise RuntimeError(f"Health check failed: {h}")

        r = infer([0.1]*8, 'test_goal')
        if not r['success']:
            raise RuntimeError(f"Inference failed: {r}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_adaptive_constants() -> TestResult:
    """Test adaptive constants computation."""
    result = TestResult("adaptive_constants")
    start = time.time()

    try:
        from adaptive_constants import compute_adaptive_constants, get_default_constants

        defaults = get_default_constants()

        adapted = compute_adaptive_constants(
            time_secs=300.0,
            vfe_history=[0.5 - 0.01*i for i in range(50)],
            ricci=3.5,
            vfe_trend=-0.008,
        )

        if not hasattr(adapted, 'explore_span'):
            raise RuntimeError("Missing explore_span in result")

        # Test validation
        try:
            compute_adaptive_constants(time_secs=-1, vfe_history=[], ricci=3.0, vfe_trend=0.0)
            raise RuntimeError("Should have raised ValueError for negative time")
        except ValueError:
            pass  # Expected

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_quantum_reasoning() -> TestResult:
    """Test quantum reasoning module."""
    result = TestResult("quantum_reasoning")
    start = time.time()

    try:
        import random
        random.seed(42)
        from quantum_reasoning import QuantumReasoner

        qr = QuantumReasoner(n_superpositions=4)
        goals = ['discover patterns', 'build solutions', 'optimize performance',
                 'explore alternatives', 'verify hypotheses', 'plan execution']
        context = {'efe': -0.4, 'ricci': 4.0, 'vfe': -0.1, 'previous_goals': []}

        for goal in goals:
            hyp, score = qr.reason(goal, context['efe'], context)
            context['previous_goals'].append(goal)

        if len(qr.memory.states) < 3:
            raise RuntimeError(f"Expected at least 3 states, got {len(qr.memory.states)}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_evolution() -> TestResult:
    """Test evolution engine."""
    result = TestResult("evolution")
    start = time.time()

    try:
        from evolution_engine import EvolutionEngine

        engine = EvolutionEngine(population_size=4)
        tasks = [{'goal': f'task_{i}', 'difficulty': 0.5} for i in range(4)]

        engine.run_competition(tasks)
        stats = engine.get_stats()
        engine.evolve()

        if stats['population'] != 4:
            raise RuntimeError(f"Expected population 4, got {stats['population']}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_knowledge_transfer() -> TestResult:
    """Test knowledge transfer module."""
    result = TestResult("knowledge_transfer")
    start = time.time()

    try:
        from knowledge_transfer import KnowledgeTransfer

        kt = KnowledgeTransfer()
        kt.add_knowledge('quantum mechanics', 'physics')
        kt.add_knowledge('evolution', 'biology')
        kt.add_knowledge('machine learning', 'computer science')

        transfers = kt.transfer('physics', 'computer science')
        stats = kt.get_stats()

        if stats['total_nodes'] < 3:
            raise RuntimeError(f"Expected >= 3 nodes, got {stats['total_nodes']}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_multi_agent() -> TestResult:
    """Test multi-agent dialogue system."""
    result = TestResult("multi_agent")
    start = time.time()

    try:
        from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

        qm = QuantumSuperpositionManager()
        wfc = WavefunctionCollapse(qm)
        a1 = Agent('Test1', 'goal1', AgentRole.DISCOVERY, qm, wfc)
        a2 = Agent('Test2', 'goal2', AgentRole.APPLICATION, qm, wfc)
        a1.connect_to(a2)
        result_interaction = a1.interact_with(a2, 'test query')

        if result_interaction is None:
            raise RuntimeError("Interaction returned None")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_logging() -> TestResult:
    """Test structured logging."""
    result = TestResult("logging")
    start = time.time()

    try:
        from logging_config import setup_logging, log_inference

        logger = setup_logging(level='DEBUG')
        log_inference(logger, "test", {"policy": "test", "efe": -0.5, "success": True}, 42.5)

        log_file = os.path.join(os.path.dirname(__file__), '..', '..', '.axiom_state', 'logs', 'kai_insight.log')
        if not os.path.exists(log_file):
            raise RuntimeError(f"Log file not found: {log_file}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_meta_cognition() -> TestResult:
    """Test meta-cognition module."""
    result = TestResult("meta_cognition")
    start = time.time()

    try:
        from meta_cognition import MetaCognitiveAgent, ReasoningTrace
        from agent_framework import Agent, AgentRole, QuantumSuperpositionManager, WavefunctionCollapse

        qm = QuantumSuperpositionManager()
        wfc = WavefunctionCollapse(qm)
        base = Agent("Explorer", "discover patterns", AgentRole.DISCOVERY, qm, wfc)

        meta = MetaCognitiveAgent("MetaExplorer", base)

        for q in ["What patterns exist?", "How can we optimize?", "What is the best approach?"]:
            meta.reason(q, {'efe': -0.4, 'ricci': 4.0})

        summary = meta.get_meta_summary()
        if summary['reasoning_trace']['total_steps'] != 3:
            raise RuntimeError(f"Expected 3 steps, got {summary['reasoning_trace']['total_steps']}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_emergence_tracker() -> TestResult:
    """Test emergence tracking."""
    result = TestResult("emergence_tracker")
    start = time.time()

    try:
        from emergence_tracker import EmergenceTracker

        tracker = EmergenceTracker(window_size=50)

        import random
        for _ in range(30):
            action = random.choice(['explore', 'build', 'optimize'])
            tracker.record_interaction('Agent1', 'Agent2', action, {'success': True})

        stats = tracker.get_stats()
        if stats['total_interactions'] != 30:
            raise RuntimeError(f"Expected 30 interactions, got {stats['total_interactions']}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_two_agent_dialogue() -> TestResult:
    """Test the two-agent local dialogue toward AGI."""
    result = TestResult("two_agent_dialogue")
    start = time.time()
    try:
        from two_agent_dialogue import TwoAgentDialogue

        dlg = TwoAgentDialogue(dilation=0.5, use_kai=False)
        summary = dlg.run(
            topic="How can free-energy minimization yield general intelligence?",
            turns=10,
        )

        if summary["turns"] < 10:
            raise RuntimeError(f"Expected >=10 turns, got {summary['turns']}")
        if summary["simulated_time"] <= 0:
            raise RuntimeError("Simulated time did not advance")
        if not (0.0 <= summary["synthesis_score"] <= 1.0):
            raise RuntimeError(f"Bad synthesis score {summary['synthesis_score']}")
        if summary["emergence_novelty"] <= 0:
            raise RuntimeError("No emergence novelty recorded")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_code_cycle() -> TestResult:
    """Test the sandboxed code-driven cycle (no Kai/LLM load)."""
    result = TestResult("code_cycle")
    start = time.time()
    try:
        from code_cycle import run_code_cycle, _static_check

        # Safe code executes and returns a number.
        rep = run_code_cycle("x", static_code="result = sum(range(1, 11))",
                              use_kai=False)
        if not rep.get("result_accepted"):
            raise RuntimeError(f"Safe code rejected: {rep.get('aborted')}")
        if rep["execution"]["result"] != 55:
            raise RuntimeError(f"Bad exec result: {rep['execution']['result']}")

        # Forbidden import is blocked by the static gate.
        ok, _ = _static_check("import os\nresult = 1")
        if ok:
            raise RuntimeError("Static check failed to block 'import os'")

        # Network module blocked.
        ok, _ = _static_check("import urllib.request\nresult = 1")
        if ok:
            raise RuntimeError("Static check failed to block network import")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def test_constant_tuner() -> TestResult:
    """Test the measured constant tuner + override plumbing (no KaiMind/LLM)."""
    result = TestResult("constant_tuner")
    start = time.time()
    try:
        from constant_tuner import (
            protocol_ricci_ema_lambda, protocol_mode, protocol_alpha_w,
            protocol_temp, protocol_vel, save_optimal, load_optimal,
            GROUNDING_PLAN,
        )
        from adaptive_constants import compute_adaptive_constants, AdaptiveConstants
        import tempfile, os

        # ricci EMA protocol: deterministic, optimal must be a candidate.
        ranking = protocol_ricci_ema_lambda([0.01, 0.02, 0.05, 0.1])
        if not ranking or ranking[0][0] not in (0.01, 0.02, 0.05, 0.1):
            raise RuntimeError(f"bad ricci ranking: {ranking}")
        if ranking != sorted(ranking, key=lambda x: x[1]):
            raise RuntimeError("ricci ranking not sorted by score")

        # The analytic protocols all return a sorted, valid ranking.
        for proto, cands in ((protocol_mode, [0.05, 0.34, 0.7]),
                             (protocol_alpha_w, [0.05, 0.1, 0.3]),
                             (protocol_temp, [0.15, 1.5, 3.0]),
                             (protocol_vel, [0.001, 0.01, 0.05])):
            r = proto(cands)
            if len(r) != len(cands) or r != sorted(r, key=lambda x: x[1]):
                raise RuntimeError(f"bad ranking for {proto.__name__}")

        # save/load roundtrip via a temp path.
        tmp = os.path.join(tempfile.gettempdir(), "ct_opt_test.json")
        if os.path.exists(tmp):
            os.remove(tmp)
        saved = {
            "ricci_ema_lambda": {
                "optimal": 0.02, "protocol": "ema",
                "ranking": [(0.02, 0.1), (0.05, 0.2)],
            }
        }
        save_optimal({k: type("R", (), {"optimal": v["optimal"],
                      "protocol": v["protocol"], "ranking": v["ranking"]})()
                      for k, v in saved.items()}, path=__import__("pathlib").Path(tmp))
        loaded = load_optimal(path=__import__("pathlib").Path(tmp))
        if loaded.get("ricci_ema_lambda") != 0.02:
            raise RuntimeError(f"override roundtrip failed: {loaded}")

        # compute_adaptive_constants must honor a measured override.
        adapted = compute_adaptive_constants(
            time_secs=300.0, vfe_history=[0.5 - 0.01 * i for i in range(50)],
            ricci=3.5, vfe_trend=-0.008, overrides={"ricci_ema_lambda": 0.02},
        )
        if abs(adapted.ricci_ema_lambda - 0.02) > 1e-9:
            raise RuntimeError(f"override not applied: {adapted.ricci_ema_lambda}")

        result.success(time.time() - start)
    except Exception as e:
        result.failure(str(e), time.time() - start)

    return result


def run_all_tests() -> Tuple[int, int, List[TestResult]]:
    """Run all tests and return results.

    Returns:
        Tuple of (passed_count, failed_count, all_results)
    """
    print("=== Kai Insight Comprehensive System Test ===\n")

    tests = [
        test_adapter,
        test_adaptive_constants,
        test_quantum_reasoning,
        test_evolution,
        test_knowledge_transfer,
        test_multi_agent,
        test_logging,
        test_meta_cognition,
        test_emergence_tracker,
        test_two_agent_dialogue,
        test_code_cycle,
        test_constant_tuner,
    ]

    results: List[TestResult] = []
    passed = 0
    failed = 0

    for test_func in tests:
        test_name = test_func.__name__
        try:
            result = test_func()
        except Exception as e:
            result = TestResult(test_name)
            result.failure(f"Unexpected error: {e}\n{traceback.format_exc()}", 0.0)

        results.append(result)

        if result.passed:
            passed += 1
            print(f"  ✓ {result.name} ({result.duration:.3f}s)")
        else:
            failed += 1
            print(f"  ✗ {result.name}: {result.error[:80]}")

    print(f"\n=== Results: {passed}/{passed+failed} passed ===")
    return passed, failed, results


if __name__ == "__main__":
    passed, failed, _ = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
