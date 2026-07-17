'''Simple REST server for Kai Insight.
This server provides a basic API for inference and status checking.
'''

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

class SimpleKaiMind:
    """Simple implementation of Kai for testing."""
    def __init__(self):
        self._mode = 'explore'
        self._mode_bias = 0.5
        self.vfe = 0.1
        self.ricci = 4.0
        
    def vfe_trend(self):
        return -0.01
        
    def select_policy(self, candidates):
        policy = candidates[0] if candidates else 'Market_data_gen'
        info = {
            'G': 0.5, 'risk': 0.3, 'epistemic': 0.4, 'n': len(candidates)
        }
        return policy, info

# Global instance
_kai = SimpleKaiMind()

def infer(state: list, goal: str, depth: int = 3) -> dict:
    '''Simple inference function that requires no imports.'''
    try:
        result = _kai.select_policy([goal if goal else 'Market_data_gen'])
        return {
            'policy': result[0],
            'efe': result[1].get('G'),
            'risk': result[1].get('risk'),
            'epistemic': result[1].get('epistemic'),
            'mode': _kai._mode,
            'vfe': _kai.vfe,
            'ricci': _kai.ricci,
            'trend': _kai.vfe_trend(),
            'info': result[1],
            '_fallback_used': False
        }
    except Exception as e:
        return {
            'policy': goal if goal else 'Market_data_gen',
            'efe': 0.5,
            'risk': 0.3,
            'epistemic': 0.4,
            'mode': _kai._mode,
            'vfe': _kai.vfe,
            'ricci': _kai.ricci,
            'trend': _kai.vfe_trend(),
            'info': {'G': 0.5, 'risk': 0.3, 'epistemic': 0.4},
            '_fallback_used': True,
            'error': str(e)
        }

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
            self.wfile.write(json.dumps({
                'mode': 'explore',
                'vfe': 0.1,
                'trend': -0.01
            }).encode())
        elif self.path == '/dashboard.html':
            # Fix the path - we're in products/kai-insight/api/
            dashboard_path = os.path.join(os.path.dirname(__file__), 'web', 'dashboard.html')
            try:
                with open(dashboard_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                print(f"Error serving dashboard: {e}")
                self.send_response(404)
                self.end_headers()
                self.wfile.write(f"Dashboard not found: {e}".encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''<html><body>
                <h1>Kai Insight Simple REST API</h1>
                <p>Adapter-based inference system</p>
                <p><a href="/dashboard.html">Dashboard</a></p>
                <p>Test inference at: /infer (POST)</p>
            </body></html>''')
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
                result = infer(
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
    
    def log_message(self, format, *args):
        pass

def run(port=8000):
    server = HTTPServer(('localhost', port), Handler)
    print(f'Kai Insight Simple REST server running on http://localhost:{port}')
    print('  - Health check: curl http://localhost:8000/health')
    print('  - Status: curl http://localhost:8000/status')
    print('  - Dashboard: http://localhost:8000/dashboard.html')
    print('  - Inference test: curl -X POST http://localhost:8000/infer -H "Content-Type: application/json" -d \'{"goal":"Market_data_gen"}\''')
    server.serve_forever()

if __name__ == '__main__':
    run()
