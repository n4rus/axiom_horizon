"""Kai Insight REST Server - Full featured.

Endpoints:
  GET  /health              - Health check
  GET  /status              - System status
  GET  /metrics             - Real-time metrics (VFE, trend, ricci, mode)
  GET  /metrics/detail       - Measurement harness (VFE slope, learning, attractor compression/drift)
  GET  /adaptive            - Current adaptive constants
  GET  /evolution           - Evolution engine stats
  GET  /quantum             - Quantum reasoning state
  POST /infer               - Run inference
  POST /dialogue            - Run multi-agent dialogue
  GET  /dashboard.html      - Web dashboard
"""

import sys
import os
import json
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from adapter import infer, health_check, get_kai_instance, live_status

try:
    import kai_metrics
except ImportError:
    kai_metrics = None

try:
    from web_search import web_search
except ImportError:
    web_search = None

try:
    from adaptive_constants import compute_adaptive_constants, get_default_constants
except ImportError:
    compute_adaptive_constants = None

try:
    from quantum_reasoning import QuantumReasoner
except ImportError:
    QuantumReasoner = None

try:
    from evolution_engine import EvolutionEngine
except ImportError:
    EvolutionEngine = None

try:
    from multi_agent_dialogue import create_dialogue_system
except ImportError:
    create_dialogue_system = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rest_server')

PORT = int(os.environ.get('KAI_PORT', 8000))

# Global instances
_quantum_reasoner = QuantumReasoner() if QuantumReasoner else None
_evolution_engine = EvolutionEngine(population_size=4) if EvolutionEngine else None
_metrics_history = []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self._json_response(200, health_check())

        elif self.path == '/status':
            # Prefer the LIVE autonomous daemon's structured feed; fall back to a
            # locally-loaded read-only KaiMind only if the daemon isn't running.
            live = live_status()
            if live:
                self._json_response(200, {
                    'status': 'running',
                    'source': 'live_daemon',
                    'mode': live.get('mode'),
                    'vfe': live.get('vfe_after'),
                    'trend': live.get('vfe_trend'),
                    'ricci': live.get('ricci'),
                    'mode_bias': live.get('mode_bias'),
                    'w_cur': live.get('w_cur'),
                    'timestamp': datetime.now().isoformat(),
                })
            else:
                kai = get_kai_instance()
                if kai is None:
                    self._json_response(200, {'status': 'no_daemon', 'source': 'none'})
                else:
                    self._json_response(200, {
                        'status': 'running', 'source': 'local_kai',
                        'mode': getattr(kai, '_mode', 'unknown'),
                        'vfe': getattr(kai, 'vfe', 0.0),
                        'trend': kai.vfe_trend() if hasattr(kai, 'vfe_trend') else 0.0,
                        'ricci': getattr(kai, 'ricci', 0.0),
                        'timestamp': datetime.now().isoformat(),
                    })

        elif self.path == '/metrics':
            # Prefer the LIVE autonomous daemon's structured feed.
            live = live_status()
            if live:
                metrics = {
                    'vfe': live.get('vfe_after', 0.0),
                    'trend': live.get('vfe_trend', 0.0),
                    'ricci': live.get('ricci', 0.0),
                    'mode': live.get('mode', 'unknown'),
                    'mode_bias': live.get('mode_bias', 0.5),
                    'w_cur': live.get('w_cur', 0.0),
                    'source': 'live_daemon',
                    'timestamp': time.time(),
                }
            else:
                kai = get_kai_instance()
                if kai is None:
                    metrics = {'source': 'none', 'timestamp': time.time()}
                else:
                    metrics = {
                        'vfe': getattr(kai, 'vfe', 0.0),
                        'trend': kai.vfe_trend() if hasattr(kai, 'vfe_trend') else 0.0,
                        'ricci': getattr(kai, 'ricci', 0.0),
                        'mode': getattr(kai, '_mode', 'unknown'),
                        'mode_bias': getattr(kai, '_mode_bias', 0.5),
                        'cycles': getattr(kai, 'cycles', 0),
                        'tau': getattr(kai, 'tau', 0),
                        'epoch_age': getattr(kai, 'epoch_age', 0),
                        'w_novelty': getattr(kai, '_w_novelty', 0.4),
                        'w_curvature': getattr(kai, '_w_curvature', 0.3),
                        'w_pnl': getattr(kai, '_w_pnl', 0.2),
                        'w_test': getattr(kai, '_w_test', 0.1),
                        'source': 'local_kai',
                        'timestamp': time.time(),
                    }
            _metrics_history.append(metrics)
            if len(_metrics_history) > 1000:
                _metrics_history.pop(0)
            self._json_response(200, metrics)

        elif self.path == '/adaptive':
            if compute_adaptive_constants:
                kai = get_kai_instance()
                if kai is None:
                    self._json_response(200, {'error': 'kai unavailable'})
                    return
                vfe_hist = []
                if hasattr(kai, '_vfe_ledger'):
                    vfe_hist = [p[1] for p in list(kai._vfe_ledger)[-100:]]
                adapted = compute_adaptive_constants(
                    time_secs=getattr(kai, '_uptime_secs', 0) or 0,
                    vfe_history=vfe_hist,
                    ricci=getattr(kai, 'ricci', 4.0),
                    vfe_trend=kai.vfe_trend() if hasattr(kai, 'vfe_trend') else 0.0,
                )
                self._json_response(200, {
                    'adapted': adapted,
                    'defaults': get_default_constants(),
                })
            else:
                self._json_response(200, {'error': 'adaptive_constants not available'})

        elif self.path == '/evolution':
            if _evolution_engine:
                stats = _evolution_engine.get_stats()
                self._json_response(200, stats)
            else:
                self._json_response(200, {'error': 'evolution_engine not available'})

        elif self.path == '/quantum':
            if _quantum_reasoner:
                self._json_response(200, {
                    'hypothesis_memory': len(_quantum_reasoner.hypothesis_memory),
                    'entanglement_pairs': len(_quantum_reasoner.entanglement_map),
                })
            else:
                self._json_response(200, {'error': 'quantum_reasoner not available'})

        elif self.path == '/mobile':
            self._serve_html('web', 'mobile.html')

        elif self.path == '/dashboard.html':
            path = os.path.join(os.path.dirname(__file__), 'web', 'dashboard.html')
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._json_response(404, {'error': 'Dashboard not found'})

        elif self.path == '/metrics/detail':
            # Read-only measurement harness over the live daemon feed + attractor.
            # No KaiMind / Ollama load.
            if kai_metrics:
                self._json_response(200, kai_metrics.analyze(
                    n_events=400, recalibrate=False))
            else:
                self._json_response(200, {'error': 'kai_metrics not available'})

        elif self.path == '/metrics/history':
            self._json_response(200, {
                'count': len(_metrics_history),
                'recent': _metrics_history[-50:],
            })

        else:
            self._json_response(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path == '/websearch':
            if web_search is None:
                self._json_response(200, {'error': 'web_search not available'})
            else:
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length).decode() if length else '{}'
                    data = json.loads(body) if body else {}
                    q = data.get('query', '') or data.get('q', '')
                    n = int(data.get('n', 5))
                    self._json_response(200, web_search(q, n))
                except Exception as e:
                    self._json_response(500, {'error': str(e)})

        elif self.path == '/infer':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode() if length else '{}'
                data = json.loads(body) if body else {}
                goal = data.get('goal', 'Market_data_gen')
                state = data.get('state', [0.0] * 8)
                result = infer(state, goal)
                self._json_response(200, result)
            except Exception as e:
                self._json_response(500, {'error': str(e), 'success': False})

        elif self.path == '/dialogue':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode() if length else '{}'
                data = json.loads(body) if body else {}
                topic = data.get('topic', 'knowledge exploration')
                rounds = data.get('rounds', 3)

                from multi_agent_dialogue import create_dialogue_system
                dialogue = create_dialogue_system()
                dialogue.max_rounds = rounds
                results = dialogue.run_dialogue(topic)
                summary = dialogue.get_summary()
                self._json_response(200, summary)
            except Exception as e:
                self._json_response(500, {'error': str(e)})

        elif self.path == '/evolve':
            try:
                if _evolution_engine:
                    tasks = [{'goal': f'task_{i}', 'difficulty': 0.5} for i in range(4)]
                    _evolution_engine.run_competition(tasks)
                    _evolution_engine.evolve()
                    self._json_response(200, _evolution_engine.get_stats())
                else:
                    self._json_response(200, {'error': 'evolution_engine not available'})
            except Exception as e:
                self._json_response(500, {'error': str(e)})

        else:
            self._json_response(404, {'error': 'Not found'})

    def _serve_html(self, subdir, filename):
        path = os.path.join(os.path.dirname(__file__), subdir, filename)
        try:
            with open(path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._json_response(404, {'error': f'{filename} not found'})

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, fmt, *args):
        logger.debug(fmt % args)


if __name__ == '__main__':
    server = HTTPServer(('localhost', PORT), Handler)
    print(f"Kai Insight REST server on http://localhost:{PORT}")
    print(f"  GET  /health         - Health check")
    print(f"  GET  /status         - Status")
    print(f"  GET  /metrics        - Real-time metrics")
    print(f"  GET  /metrics/detail - Measurement harness")
    print(f"  GET  /adaptive       - Adaptive constants")
    print(f"  GET  /evolution      - Evolution stats")
    print(f"  GET  /quantum        - Quantum state")
    print(f"  POST /infer          - Inference")
    print(f"  POST /dialogue       - Multi-agent dialogue")
    print(f"  POST /evolve         - Run evolution step")
    print(f"  GET  /dashboard.html - Dashboard")
    print(f"  GET  /mobile         - Mobile UI")
    server.serve_forever()
