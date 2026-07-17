#!/usr/bin/env python3
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Simple embedded inference function
def infer(state, goal, depth=3):
    return {
        "policy": goal if goal else "Market_data_gen",
        "efe": 0.5,
        "risk": 0.3,
        "epistemic": 0.4,
        "mode": "explore",
        "vfe": 0.1,
        "ricci": 4.0,
        "trend": -0.01,
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "mode": "explore",
                "vfe": 0.1,
                "trend": -0.01
            }).encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Kai Insight REST Server</h1><p><a href=\"/health\">Health</a></body></html>")
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/infer':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len).decode('utf-8')
            try:
                data = json.loads(body) if body else {}
                result = infer(data.get('state', []), data.get('goal', ''), data.get('depth', 3))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

def run(port=8000):
    server = HTTPServer(('localhost', port), Handler)
    print(f"Kai Insight REST server running on http://localhost:{port}")
    print(f"  - Health: http://localhost:{port}/health")
    print(f"  - Status: http://localhost:{port}/status")
    print(f"  - Infer: curl -X POST http://localhost:{port}/infer -H 'Content-Type: application/json' -d {'{\"goal\":\"Market_data_gen\"}'}")
    server.serve_forever()

if __name__ == '__main__':
    run()
