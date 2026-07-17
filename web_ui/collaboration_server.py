"""Web-based collaboration server for Axiom bounty pipeline.

Provides:
  - TerminalController: shell access, process management
  - WebAICollaborator: multi-provider AI integration
  - EnhancedCollaborationApp: Flask server with REST API
"""
from __future__ import annotations
import json, os, subprocess, sys, time, threading
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).parent.parent
STATE = BASE / '.axiom_state'


class TerminalController:
    """Shell access and process management for admin-level control."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def exec_command(self, command: str, timeout: int = 30) -> dict:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(BASE),
            )
            return {
                'stdout': result.stdout[-5000:],
                'stderr': result.stderr[-2000:],
                'returncode': result.returncode,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        except subprocess.TimeoutExpired:
            return {'error': 'timeout', 'command': command}
        except Exception as e:
            return {'error': str(e), 'command': command}

    def list_sessions(self) -> list[dict]:
        return [{'id': k, **v} for k, v in self.sessions.items()]


class WebAICollaborator:
    """Multi-provider AI integration for collaboration."""

    def __init__(self):
        self.providers = {
            'local': self._local_query,
            'ollama': self._ollama_query,
        }

    def query(self, prompt: str, provider: str = 'local') -> str:
        fn = self.providers.get(provider, self._local_query)
        return fn(prompt)

    def _local_query(self, prompt: str) -> str:
        try:
            sys.path.insert(0, str(BASE))
            from axiom_alien import AxiomAlien
            a = AxiomAlien()
            result = a.ask(prompt, max_steps=4)
            return result.get('answer', 'No response')
        except Exception as e:
            return f'Local query error: {e}'

    def _ollama_query(self, prompt: str) -> str:
        import urllib.request
        try:
            data = json.dumps({
                'model': 'qwen2.5:7b',
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
            }).encode()
            req = urllib.request.Request(
                'http://127.0.0.1:11434/api/chat',
                data=data,
                headers={'Content-Type': 'application/json'},
            )
            resp = urllib.request.urlopen(req, timeout=60)
            return json.loads(resp.read())['message']['content']
        except Exception as e:
            return f'Ollama query error: {e}'


class EnhancedCollaborationApp:
    """Flask-based collaboration server with REST API."""

    def __init__(self):
        self.terminal = TerminalController()
        self.ai = WebAICollaborator()
        self.metrics = {
            'requests': 0,
            'start_time': datetime.now(timezone.utc).isoformat(),
        }

    def health_check(self) -> dict:
        return {
            'status': 'healthy',
            'uptime': self.metrics['start_time'],
            'requests': self.metrics['requests'],
            'terminal_sessions': len(self.terminal.list_sessions()),
        }

    def run_terminal_command(self, command: str) -> dict:
        self.metrics['requests'] += 1
        return self.terminal.exec_command(command)

    def run_ai_query(self, prompt: str, provider: str = 'local') -> dict:
        self.metrics['requests'] += 1
        response = self.ai.query(prompt, provider)
        return {'response': response, 'provider': provider}

    def get_metrics(self) -> dict:
        return self.metrics

    def start(self, host: str = '127.0.0.1', port: int = 5000):
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            print('[web] Flask not installed. Install with: pip install flask')
            print('[web] Starting simple HTTP server instead...')
            self._start_simple(host, port)
            return

        app = Flask(__name__)
        collab = self

        @app.route('/api/v2/health')
        def health():
            return jsonify(collab.health_check())

        @app.route('/api/terminal/exec', methods=['POST'])
        def terminal_exec():
            data = request.json
            return jsonify(collab.run_terminal_command(data.get('command', '')))

        @app.route('/api/ai/query', methods=['POST'])
        def ai_query():
            data = request.json
            return jsonify(collab.run_ai_query(
                data.get('prompt', ''),
                data.get('provider', 'local'),
            ))

        @app.route('/api/metrics')
        def metrics():
            return jsonify(collab.get_metrics())

        print(f'[web] Collaboration server at http://{host}:{port}')
        app.run(host=host, port=port, debug=False)

    def _start_simple(self, host: str, port: int):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        collab = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/api/v2/health':
                    self._json(collab.health_check())
                elif self.path == '/api/metrics':
                    self._json(collab.get_metrics())
                else:
                    self.send_error(404)

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path == '/api/terminal/exec':
                    self._json(collab.run_terminal_command(body.get('command', '')))
                elif self.path == '/api/ai/query':
                    self._json(collab.run_ai_query(body.get('prompt', ''), body.get('provider', 'local')))
                else:
                    self.send_error(404)

            def _json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, format, *args):
                print(f'[http] {args[0]}')

        server = HTTPServer((host, port), Handler)
        print(f'[web] Simple server at http://{host}:{port}')
        server.serve_forever()


if __name__ == '__main__':
    app = EnhancedCollaborationApp()
    app.start()
