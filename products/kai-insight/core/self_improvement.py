"""Continuous Self-Improvement Loop for Kai AGI.

Monitors performance metrics and triggers adaptation cycles.
Integrates quantum reasoning, evolution, and adaptive constants.
"""

import time
import os
import sys
import json
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api', 'core'))
from adapter import get_kai_instance, health_check
from adaptive_constants import compute_adaptive_constants


class SelfImprovementLoop:
    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval
        self.metrics_history: List[Dict] = []
        self.adaptation_count: int = 0
        self.last_adaptation: float = 0
        self._running = False

    def collect_metrics(self) -> Dict:
        kai = get_kai_instance()
        health = health_check()

        metrics = {
            'timestamp': time.time(),
            'vfe': getattr(kai, 'vfe', 0.0),
            'trend': kai.vfe_trend() if hasattr(kai, 'vfe_trend') else 0.0,
            'ricci': getattr(kai, 'ricci', 0.0),
            'mode': getattr(kai, '_mode', 'unknown'),
            'healthy': health.get('status') == 'healthy',
            'fallback': health.get('fallback_mode', True),
        }

        if hasattr(kai, '_vfe_ledger'):
            metrics['ledger_size'] = len(kai._vfe_ledger)

        self.metrics_history.append(metrics)
        return metrics

    def analyze_performance(self) -> Dict:
        if len(self.metrics_history) < 2:
            return {'status': 'insufficient_data'}

        recent = self.metrics_history[-10:]
        vfe_values = [m['vfe'] for m in recent]
        trend_values = [m['trend'] for m in recent]

        avg_vfe = sum(vfe_values) / len(vfe_values)
        avg_trend = sum(trend_values) / len(trend_values)
        vfe_stability = max(vfe_values) - min(vfe_values)

        return {
            'avg_vfe': avg_vfe,
            'avg_trend': avg_trend,
            'vfe_stability': vfe_stability,
            'improving': avg_trend < -0.001,
            'stable': vfe_stability < 0.05,
            'recommendation': self._get_recommendation(avg_vfe, avg_trend, vfe_stability),
        }

    def _get_recommendation(self, avg_vfe: float, avg_trend: float, stability: float) -> str:
        if avg_trend < -0.01:
            return 'learning_well_continue'
        elif avg_trend < -0.001:
            return 'learning_slowly_increase_exploration'
        elif stability > 0.1:
            return 'unstable_reduce_learning_rate'
        elif avg_vfe > 0:
            return 'positive_vfe_investigate'
        else:
            return 'maintain_course'

    def suggest_adaptations(self, analysis: Dict) -> Dict:
        recommendation = analysis.get('recommendation', 'maintain_course')
        adaptations = {}

        if recommendation == 'learning_slowly_increase_exploration':
            adaptations['mode_eps'] = 0.5
            adaptations['exploration_boost'] = True
        elif recommendation == 'unstable_reduce_learning_rate':
            adaptations['wm_alpha'] = 0.005
            adaptations['ricci_ema_lambda'] = 0.03
        elif recommendation == 'positive_vfe_investigate':
            adaptations['t_min'] = 0.3
            adaptations['t_max'] = 2.0

        return {
            'recommendation': recommendation,
            'adaptations': adaptations,
            'timestamp': datetime.now().isoformat(),
        }

    def run_check(self) -> Dict:
        metrics = self.collect_metrics()
        analysis = self.analyze_performance()
        suggestions = self.suggest_adaptations(analysis)

        result = {
            'metrics': metrics,
            'analysis': analysis,
            'suggestions': suggestions,
        }

        if suggestions['adaptations']:
            self.adaptation_count += 1
            self.last_adaptation = time.time()

        return result


def run_self_improvement_demo():
    print("=== Continuous Self-Improvement Loop ===\n")

    loop = SelfImprovementLoop(check_interval=1)

    for i in range(5):
        print(f"--- Check {i+1} ---")
        result = loop.run_check()

        m = result['metrics']
        a = result['analysis']
        s = result['suggestions']

        print(f"  VFE: {m['vfe']:.4f}, Trend: {m['trend']:.6f}, Mode: {m['mode']}")
        print(f"  Improving: {a.get('improving', 'N/A')}, Stable: {a.get('stable', 'N/A')}")
        print(f"  Recommendation: {s['recommendation']}")
        if s['adaptations']:
            print(f"  Adaptations: {s['adaptations']}")
        print()

    print(f"Total adaptations suggested: {loop.adaptation_count}")
    print(f"Metrics collected: {len(loop.metrics_history)}")


if __name__ == "__main__":
    run_self_improvement_demo()
