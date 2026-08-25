#!/usr/bin/env python3
"""
Kai PC Server — encrypted live bridge for Kai-Android's "kai-pc:live" model
Runs on your desktop alongside `axiom_horizon`. The phone app, when
"kai-pc:live" is selected, sends typed text / files / images here over
TLS + token, and this server forwards them straight into the live
opencode session (the same `kai launch opencode` you run on desktop).
The phone then sees the PC's terminal output live.

Usage:
  python3 tools/kai_pc_server.py --port 8443 --token kai-secret-123
  # On phone: Settings → Kai PC → enter 192.168.1.10:8443 + token
  # Or pair via QR (shows IP + token)

Security: self-signed TLS (generated on first run) + Bearer token.
No private data leaves the LAN unless you port-forward.
"""

import argparse, http.server, json, os, ssl, base64, urllib.parse
from pathlib import Path

# Simple in-memory log of live session (what the PC's opencode sees)
LIVE_LOG = []
MAX_LOG = 200

def log_event(typ, text, meta=None):
    LIVE_LOG.append({"type": typ, "text": text, "meta": meta or {}})
    if len(LIVE_LOG) > MAX_LOG:
        del LIVE_LOG[0]
    print(f"[{typ}] {text[:120]}")

class Handler(http.server.BaseHTTPRequestHandler):
    token = "kai-secret-123"

    def auth_ok(self):
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def do_GET(self):
        if not self.auth_ok():
            self.send_response(401); self.end_headers(); return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/live":
            # Return live log as JSON for phone's Terminal tab
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(LIVE_LOG).encode())
        elif parsed.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if not self.auth_ok():
            self.send_response(401); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")
        try:
            data = json.loads(body) if body else {}
        except:
            data = {"text": body.decode(errors="ignore")}

        text = data.get("text", "")
        typ = data.get("type", "text")  # text | file | image
        fname = data.get("filename", "")

        # === Forward straight into the PC's live opencode session ===
        # For now we log and echo back via the same VFE pipeline (stub).
        # Replace this block with: `echo text | kai --stdin` or via Axiom's Python API.
        if typ == "image":
            log_event("image", f"📷 {fname} ({len(text)} chars base64)", {"filename": fname})
            reply = f"📷 Received image {fname} on PC — vision tower (SigLIP) would describe it here. Echo: {text[:200]}"
        elif typ == "file":
            # Save to workspace so desktop Kai can see it
            ws = Path(__file__).parent.parent / "kai_phone_inbox"
            ws.mkdir(exist_ok=True)
            p = ws / fname if fname else ws / "pasted.txt"
            try:
                # text is base64 for binary, or plain for text
                try: p.write_bytes(base64.b64decode(text))
                except: p.write_text(text)
                log_event("file", f"📎 {p.name} -> {p}", {"path": str(p)})
                reply = f"📎 File {p.name} received on PC at {p} — available to desktop Kai at kai_phone_inbox/{p.name}"
            except Exception as e:
                reply = f"⚠ file save failed: {e}"
        else:
            log_event("text", text)
            # --- LIVE FORWARD: inject into desktop Kai's chat box ---
            # Option A: append to a file that desktop Kai watches
            # Option B: HTTP POST to local opencode / desktop Kai's API
            # For now we echo with PC context
            reply = f"[PC Kai live] You sent: {text}\n\nThis is the live desktop session echo. Wire this to your actual kai launch opencode by replacing this block with:\n  echo {json.dumps(text)} | python3 -m kai --stdin\nor via Axiom's Python API."

        # Also append reply to live log so phone's Terminal sees it
        log_event("kai", reply)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"reply": reply, "live": LIVE_LOG[-10:]}).encode())

    def log_message(self, fmt, *args):
        # quieter
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--token", default="kai-secret-123")
    ap.add_argument("--cert", default="kai_pc_cert.pem")
    ap.add_argument("--key", default="kai_pc_key.pem")
    args = ap.parse_args()
    Handler.token = args.token

    # Generate self-signed cert if missing
    if not Path(args.cert).exists():
        print(f"Generating self-signed cert {args.cert}...")
        os.system(f'openssl req -x509 -newkey rsa:2048 -keyout {args.key} -out {args.cert} -days 365 -nodes -subj "/CN=kai-pc" 2>/dev/null')
        # Fallback to http if openssl missing
        if not Path(args.cert).exists():
            print("openssl not found — running HTTP (not HTTPS). Use --cert with real cert for TLS.")
            from http.server import HTTPServer
            httpd = HTTPServer(("0.0.0.0", args.port), Handler)
            print(f"Kai PC live on http://0.0.0.0:{args.port}  token={args.token}")
            print(f"Phone: set kai-pc:live → http://<this-ip>:{args.port}")
            httpd.serve_forever()
            return

    # HTTPS
    from http.server import HTTPServer
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(args.cert, args.key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(f"🔒 Kai PC live on https://0.0.0.0:{args.port}  token={args.token}")
    print(f"Phone: model picker → kai-pc:live → Settings → {get_ip()}:{args.port} + token")
    print(f"Live log at https://<ip>:{args.port}/live (Bearer token)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")

def get_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "127.0.0.1"

if __name__ == "__main__":
    main()
