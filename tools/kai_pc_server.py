#!/usr/bin/env python3
"""
Kai PC Server — encrypted live bridge for Kai-Android's "kai-pc:live" model.
Runs on your desktop alongside `axiom_horizon`.

Two main functions:
1. /send-email — real SMTP email sending (account confirmation, password reset, etc.)
2. /live + POST text/file/image — live bridge to opencode session

Usage:
  python3 tools/kai_pc_server.py --port 8443 --token kai-secret-123 \
    --smtp-host smtp.gmail.com --smtp-port 587 \
    --smtp-user your@gmail.com --smtp-pass YOUR_APP_PASSWORD \
    --from "Kai <no-reply@yourdomain.com>"
  # Gmail: use App Password (not your real password) — google.com/settings/security
  # For production: use SendGrid / Mailgun / your own SMTP server

Security: self-signed TLS (generated on first run) + Bearer token.
"""

import argparse, http.server, json, os, ssl, base64, urllib.parse, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

# Simple in-memory log of live session (what the PC's opencode sees)
LIVE_LOG = []
MAX_LOG = 200

def log_event(typ, text, meta=None):
    LIVE_LOG.append({"type": typ, "text": text, "meta": meta or {}, "ts": datetime.now().isoformat()})
    if len(LIVE_LOG) > MAX_LOG:
        del LIVE_LOG[0]
    print(f"[{typ}] {text[:120]}")

def send_smtp_email(to_email: str, subject: str, html_body: str, text_body: str = None) -> dict:
    """Send a real email via SMTP. Returns {"ok": True} or {"ok": False, "error": "..."}."""
    h = getattr(Handler, '_smtp_host', None)
    if not h:
        return {"ok": False, "error": "SMTP not configured — add --smtp-* args to kai_pc_server.py"}
    try:
        port = int(getattr(Handler, '_smtp_port', 587))
        user = getattr(Handler, '_smtp_user', '')
        pw = getattr(Handler, '_smtp_pass', '')
        from_addr = getattr(Handler, '_smtp_from', user)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_email
        msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")

        if text_body:
            msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(h, port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(user, pw)
            server.sendmail(from_addr, [to_email], msg.as_string())

        print(f"[EMAIL] ✓ sent to {to_email}: {subject}")
        return {"ok": True}
    except Exception as e:
        print(f"[EMAIL] ✗ failed to {to_email}: {e}")
        return {"ok": False, "error": str(e)}

class Handler(http.server.BaseHTTPRequestHandler):
    token = "kai-secret-123"

    def auth_ok(self):
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if not self.auth_ok():
            self.send_response(401); self.end_headers(); return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/live":
            self.send_json({"log": LIVE_LOG[-20:]})
        elif parsed.path == "/health":
            self.send_json({"status": "ok", "email_configured": getattr(Handler, '_smtp_host', None) is not None})
        elif parsed.path == "/email/test":
            # Test endpoint — sends a test email to verify SMTP config
            params = dict(urllib.parse.parse_qsl(parsed.query))
            to_addr = params.get("to", "")
            if to_addr:
                result = send_smtp_email(
                    to_addr,
                    "Kai — Test Email",
                    f"<p>This is a test from Kai PC Server.</p><p>Sent at {datetime.now().isoformat()}</p>",
                    f"Kai test email at {datetime.now().isoformat()}"
                )
                self.send_json(result)
            else:
                self.send_json({"ok": False, "error": "missing ?to=email param"}, 400)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if not self.auth_ok():
            self.send_response(401); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")

        parsed = urllib.parse.urlparse(self.path)
        # --- /send-email endpoint ---
        if parsed.path == "/send-email":
            try:
                data = json.loads(body) if body else {}
            except:
                data = {}
            to_email = data.get("to", "")
            subject = data.get("subject", "Kai Notification")
            html_body = data.get("html", "")
            text_body = data.get("text", "")
            if not to_email or "@" not in to_email:
                self.send_json({"ok": False, "error": "invalid to email"}, 400)
                return
            result = send_smtp_email(to_email, subject, html_body, text_body)
            self.send_json(result, 200 if result["ok"] else 500)
            return

        # --- /live bridge ---
        try:
            data = json.loads(body) if body else {}
        except:
            data = {"text": body.decode(errors="ignore")}

        text = data.get("text", "")
        typ = data.get("type", "text")  # text | file | image
        fname = data.get("filename", "")

        if typ == "image":
            log_event("image", f"📷 {fname} ({len(text)} chars base64)", {"filename": fname})
            reply = f"📷 Received image {fname} on PC — vision tower (SigLIP) would describe it here. Echo: {text[:200]}"
        elif typ == "file":
            ws = Path(__file__).parent.parent / "kai_phone_inbox"
            ws.mkdir(exist_ok=True)
            p = ws / fname if fname else ws / "pasted.txt"
            try:
                try: p.write_bytes(base64.b64decode(text))
                except: p.write_text(text)
                log_event("file", f"📎 {p.name} -> {p}", {"path": str(p)})
                reply = f"📎 File {p.name} received on PC at {p}"
            except Exception as e:
                reply = f"⚠ file save failed: {e}"
        else:
            log_event("text", text)
            reply = f"[PC Kai live] You sent: {text}\n\nThis is the live desktop session echo."

        log_event("kai", reply)
        self.send_json({"reply": reply, "live": LIVE_LOG[-10:]})

    def log_message(self, fmt, *args):
        pass

def main():
    ap = argparse.ArgumentParser(description="Kai PC Server — live bridge + email sender")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--token", default="kai-secret-123")
    ap.add_argument("--cert", default="kai_pc_cert.pem")
    ap.add_argument("--key", default="kai_pc_key.pem")
    # SMTP settings for real email
    ap.add_argument("--smtp-host", default=None, help="SMTP host (e.g. smtp.gmail.com, smtp.mailgun.org)")
    ap.add_argument("--smtp-port", type=int, default=587, help="SMTP port (587 for TLS, 465 for SSL)")
    ap.add_argument("--smtp-user", default=None, help="SMTP username (usually your email)")
    ap.add_argument("--smtp-pass", default=None, help="SMTP password / app password")
    ap.add_argument("--from", dest="smtp_from", default=None, help="From address (e.g. 'Kai <no-reply@domain.com>')")
    args = ap.parse_args()
    Handler.token = args.token

    # Store SMTP config on Handler so send_smtp_email can reach it
    if args.smtp_host:
        Handler._smtp_host = args.smtp_host
        Handler._smtp_port = args.smtp_port
        Handler._smtp_user = args.smtp_user or args.smtp_host
        Handler._smtp_pass = args.smtp_pass or ""
        Handler._smtp_from = args.smtp_from or Handler._smtp_user
        print(f"[EMAIL] configured → {args.smtp_host}:{args.smtp_port} as {Handler._smtp_from}")
    else:
        print("[EMAIL] not configured — /send-email will return 'SMTP not configured'")
        print("  To enable: add --smtp-host --smtp-user --smtp-pass --from")
        print("  Gmail: use an App Password (google.com/settings/security → App passwords)")
        print("  Mailgun: smtp.mailgun.org + your SMTP credentials")
        print("  SendGrid: smtp.sendgrid.net + API key")

    if not Path(args.cert).exists():
        print(f"Generating self-signed cert {args.cert}...")
        os.system(f'openssl req -x509 -newkey rsa:2048 -keyout {args.key} -out {args.cert} -days 365 -nodes -subj "/CN=kai-pc" 2>/dev/null')
        if not Path(args.cert).exists():
            print("openssl not found — running HTTP. Use --cert with real cert for TLS.")
            from http.server import HTTPServer
            httpd = HTTPServer(("0.0.0.0", args.port), Handler)
            print(f"Kai PC live on http://0.0.0.0:{args.port}  token={args.token}")
            print(f"Phone: set kai-pc:live → http://<this-ip>:{args.port}")
            httpd.serve_forever()
            return

    from http.server import HTTPServer
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(args.cert, args.key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    ip = get_ip()
    print(f"🔒 Kai PC live on https://0.0.0.0:{args.port}  token={args.token}")
    print(f"Phone: model picker → kai-pc:live → Settings → {ip}:{args.port} + token")
    print(f"Live log: GET https://{ip}:{args.port}/live (Bearer {args.token})")
    print(f"Email:   POST https://{ip}:{args.port}/send-email (Bearer {args.token})")
    print(f"Health:  GET  https://{ip}:{args.port}/health (Bearer {args.token})")
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
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    main()
