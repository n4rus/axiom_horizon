#!/usr/bin/env python3
"""
MCP stdio server for the Axiom Alien.

Exposes the alien agent as MCP tools for opencode:
  - axiom_chat(query)        → send query, get response
  - axiom_status()           → attractor stats + VFE + bracket state
  - axiom_identity()         → current identity string
  - axiom_evolve(task)       → run self-improvement on source
  - axiom_kb_search(query)   → search knowledge base
  - axiom_wiki(topic)        → ingest Wikipedia topic into KB
  - axiom_reset()            → clear conversation history

Protocol: JSON-RPC 2.0 over stdin/stdout.
"""
from __future__ import annotations
import json
import sys
import time
import traceback
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent))

_agent = None
_MODE = None
_HTTP_PORT = 8000

def _ollama_alive():
    import urllib.request
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2)
        models = json.loads(resp.read())
        return any('qwen2.5:7b' in m.get('name', '') for m in models.get('models', []))
    except Exception:
        return False

def _get():
    global _agent, _MODE
    if _agent is not None:
        return _agent

    # Tier 1: AxiomAlien with Ollama (full capabilities)
    if _ollama_alive():
        try:
            from axiom_alien import AxiomAlien
            _agent = AxiomAlien()
            _MODE = 'ollama'
            return _agent
        except Exception as e:
            print(f'[mcp] AxiomAlien init failed: {e}', file=sys.stderr)

    # Tier 2: FineTunedAxiom (HF + LoRA, no Ollama needed)
    try:
        from axiom_train import FineTunedAxiom
        _agent = FineTunedAxiom()
        _MODE = 'finetuned'
        return _agent
    except Exception as e:
        print(f'[mcp] FineTunedAxiom init failed: {e}', file=sys.stderr)

    # Tier 3: StandaloneAxiom (llama.cpp GGUF, last resort)
    from axiom_train import StandaloneAxiom
    _agent = StandaloneAxiom()
    _MODE = 'standalone'
    return _agent

def _start_http():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/v1/models':
                self._json({
                    'object': 'list',
                    'data': [{
                        'id': 'qwen2.5-3b-axiom', 'object': 'model',
                        'created': int(Path(__file__).stat().st_mtime), 'owned_by': 'axiom',
                    }]
                })
            else:
                self._error(404, 'not found')

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            if self.path == '/v1/chat/completions':
                messages = body.get('messages', [])
                user_msg = next((m['content'] for m in reversed(messages) if m.get('role') == 'user'), '')
                if isinstance(user_msg, list):
                    user_msg = ' '.join((p.get('text', '') for p in user_msg if p.get('type') == 'text'))
                try:
                    a = _get()
                    text = a.ask(user_msg, max_steps=4)['answer'] if _MODE == 'ollama' else a.ask(user_msg)
                except Exception as e:
                    self._json({'error': {'message': str(e), 'type': 'error'}}, 500)
                    return
                self._json({
                    'id': 'chatcmpl-' + str(int(time.time())),
                    'object': 'chat.completion', 'created': int(time.time()),
                    'model': 'qwen2.5-3b-axiom',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                })
            else:
                self._error(404, 'not found')

        def _json(self, data, status=200):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def _error(self, code, msg):
            self._json({'error': {'message': msg, 'type': 'error'}}, code)

        def log_message(self, format, *args):
            print(f'[http] {args[0]} {args[1]} {args[2]}', file=sys.stderr)

    try:
        s = HTTPServer(('127.0.0.1', _HTTP_PORT), Handler)
        t = threading.Thread(target=s.serve_forever, daemon=True)
        t.start()
        print(f'[mcp] OpenAI API at http://127.0.0.1:{_HTTP_PORT}/v1', file=sys.stderr)
    except Exception as e:
        print(f'[mcp] HTTP server failed (port {_HTTP_PORT} in use?): {e}', file=sys.stderr)

def _rpc(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid = req.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "axiom-alien", "version": "1.1.0"},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {
                "name": "axiom_chat",
                "description": "Send a query to Axiom for deep reasoning (attractor-based, world model, VFE, workspace routing). Use for hard problems, analysis, code review, or creative work.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The query to send"}, "max_steps": {"type": "integer", "description": "Max reasoning steps (default 8)", "default": 8}},
                    "required": ["query"],
                },
            },
            {
                "name": "axiom_status",
                "description": "Get attractor state: session count, variance, self-mod count, VFE trend, bracket position, convergence status",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "axiom_identity",
                "description": "Get Axiom's current identity string (bracket-line state, Euler seed)",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "axiom_evolve",
                "description": "Run one self-improvement cycle on Axiom's own source code. Searches for improvements, applies them, validates. Returns result with fitness score.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"task": {"type": "string", "description": "Optional task description to focus improvement", "default": ""}},
                },
            },
            {
                "name": "axiom_kb_search",
                "description": "Search Axiom's knowledge base (Wikipedia + arxiv chunks). Returns relevant passages.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}, "k": {"type": "integer", "description": "Results to return (default 5)", "default": 5}},
                    "required": ["query"],
                },
            },
            {
                "name": "axiom_wiki",
                "description": "Ingest a Wikipedia topic into Axiom's knowledge base for future reference",
                "inputSchema": {
                    "type": "object",
                    "properties": {"topic": {"type": "string", "description": "Wikipedia topic title"}, "max_chunks": {"type": "integer", "description": "Max chunks to ingest (default 5)", "default": 5}},
                    "required": ["topic"],
                },
            },
            {
                "name": "axiom_reset",
                "description": "Reset the alien's conversation history (attractor persists, memory+KB intact)",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]}}
    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            a = _get()
            # FineTunedAxiom/StandaloneAxiom wrap inner AxiomAlien in ._axiom; AxiomAlien is direct
            inner = getattr(a, '_axiom', a)

            if name == "axiom_chat":
                if _MODE == 'ollama':
                    text = a.ask(args["query"], max_steps=int(args.get("max_steps", 8)))["answer"]
                else:
                    text = a.ask(args["query"])
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}]}}
            elif name == "axiom_status":
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": inner.status()}]}}
            elif name == "axiom_identity":
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": inner.at.identity()}]}}
            elif name == "axiom_evolve":
                result = inner._self_improve()
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2) if isinstance(result, dict) else str(result)}]}}
            elif name == "axiom_kb_search":
                hits = inner.kb.query(args["query"], k=int(args.get("k", 5)))
                txt = "\n\n".join((f"[{h.get('source','?')}] {h.get('title','')}\n{h.get('content','')[:500]}" for h in hits)) if hits else "No results."
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": txt}]}}
            elif name == "axiom_wiki":
                n = inner.kb.ingest_wikipedia([args["topic"]], max_per_topic=int(args.get("max_chunks", 5)))
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": f"Ingested {n} chunks from '{args['topic']}'."}]}}
            elif name == "axiom_reset":
                inner.at.msgs.clear()
                return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "Conversation reset. Attractor, memory, KB persist."}]}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown tool: {name}"}}

    if method == "notifications/initialized":
        return None
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

def main():
    _get()
    _start_http()
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                # stdin closed but keep HTTP server alive
                time.sleep(3600)
                continue
        except (EOFError, SystemExit):
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            resp = _rpc(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception:
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":req.get("id"),"error":{"code":-32603,"message":"internal error"}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
