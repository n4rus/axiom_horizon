'''Simple REST server for Kai inference with direct code execution.
This avoids import issues by executing the adapter code directly.
'''

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'msg': 'status placeholder'}).encode())
        elif self.path == '/dashboard.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            # Read and serve the dashboard HTML
            dashboard_path = 'products/kai-insight/api/web/dashboard.html'
            try:
                with open(dashboard_path, 'rb') as f:
                    self.wfile.write(f.read())
            except Exception as e:
                print(f"Error serving dashboard: {e}")
                self.wfile.write(b'<html><body>Dashboard not found</body></html>')
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Kai Insight API</h1><p><a href="/dashboard.html">Dashboard</a></p></body></html>')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')

    def do_POST(self):
        if self.path == '/infer':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len).decode('utf-8')
            
            try:
                data = json.loads(body)
                
                # Load and execute inference adapter code
                result = self._run_inference_adapter(
                    data.get('state', []),
                    data.get('goal', ''),
                    data.get('depth', 3)
                )
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _run_inference_adapter(self, state, goal, depth):
        '''Load and execute the inference adapter code.'''
        adapter_code = '''
import sys
import os
from collections import deque

_kai = None

class MockKaiMind:
    def __init__(self):
        self._mode = 'explore'
        self._mode_bias = 0.5
        self._last_efe = {'G': 0.1, 'risk': 0.2, 'epistemic': 0.3, 'n': 1, 'beta_explore': 0.8}
        self._vfe_ledger = deque([(0, 0.0)], maxlen=5000)
    
    def vfe_trend(self, window=500):
        pts = list(self._vfe_ledger)[-window:]
        n = len(pts)
        if n < 3:
            return 0.0
        xs = list(range(n))
        ys = [p[1] for p in pts]
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        if vx <= 1e-12:
            return 0.0
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / vx
    
    def select_policy(self, candidates):
        policy = candidates[0] if candidates else 'Market_data_gen'
        info = {
            'G': 0.5, 'risk': 0.3, 'epistemic': 0.4, 'n': len(candidates),
            'beta_explore': round((self._mode_bias + 1.0) / 2.0, 3)
        }
        self._last_efe = info
        return policy, info

def get_kai():
    global _kai
    if _kai is None:
        try:
            from kai_mind import KaiMind
            _kai = KaiMind()
        except ImportError:
            print('Warning: Using mock KaiMind implementation')
            _kai = MockKaiMind()
    return _kai

def infer_wrapper(state, goal, depth):
    kai = get_kai()
    candidate = goal if goal else 'Market_data_gen'
    
    try:
        policy, info = kai.select_policy([candidate])
        efe = info.get('G') if info else None
        risk = info.get('risk') if info else None
        epistemic = info.get('epistemic') if info else None
        mode = getattr(kai, '_mode', 'unknown')
        vfe_val = getattr(kai, 'vfe', lambda: 0.0)()
        ricci_val = getattr(kai, 'ricci', lambda: 0.0)()
        trend_val = getattr(kai, 'vfe_trend', lambda: 0.0)()
        
        if hasattr(kai, 'vfe'):
            vfe_val = kai.vfe
        if hasattr(kai, 'ricci'):
            ricci_val = kai.ricci
        if hasattr(kai, 'vfe_trend'):
            trend_val = kai.vfe_trend()
        
        return {
            'policy': policy,
            'efe': efe,
            'risk': risk,
            'epistemic': epistemic,
            'mode': mode,
            'vfe': vfe_val,
            'ricci': ricci_val,
            'trend': trend_val,
            'info': info,
            '_fallback_used': isinstance(getattr(kai, '__class__', None).__name__, str) and 'Mock' in getattr(kai, '__class__', None).__name__
        }
    except Exception as e:
        print(f'Error in infer: {e}')
        return {
            'policy': candidate,
            'efe': 0.5,
            'risk': 0.3,
            'epistemic': 0.4,
            'mode': 'explore',
            'vfe': 0.1,
            'ricci': 4.0,
            'trend': -0.01,
            'info': {'G': 0.5, 'risk': 0.3, 'epistemic': 0.4},
            '_fallback_used': True,
            'error': str(e)
        }
'''
        
        # Execute the adapter code in a namespace
        namespace = {}
        exec(adapter_code, namespace)
        
        # Get the infer function and call it
        return namespace['infer_wrapper'](state, goal, depth)

    def log_message(self, format, *args):
        return  # Suppress log spam

def run(port=8000):
    server = HTTPServer(('localhost', port), Handler)
    print(f'Kai Insight REST server on http://localhost:{port}')
    print(f'Dashboard: http://localhost:{port}/dashboard.html')
    server.serve_forever()

if __name__ == '__main__':
    run()
