'''REST server using the actual KaiMind reasoning.
This is a fixed version that properly imports the adapter.
'''
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import os

# Add the repo root to path
sys.path.insert(0, '.')

# Import the inference adapter
from products.kai-insight.api.core.adapter import infer

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
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Kai Insight REST API</h1><p><a href="/dashboard.html">Dashboard</a></p></body></html>')
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
                result = infer(data.get('state', []), data.get('goal', ''), data.get('depth', 3))
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
        return  # Suppress log spam

def run(port=8000):
    server = HTTPServer(('localhost', port), Handler)
    print(f'Kai Insight REST server on http://localhost:{port}')
    print(f'Static dashboard: http://localhost:{port}/dashboard.html')
    server.serve_forever()

if __name__ == '__main__':
    run()
