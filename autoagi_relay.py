#!/usr/bin/env python3
"""
autoagi_relay.py — the dual-opencode loop.

AutoAGI plays the HUMAN in the opencode chat with Kai:
  - reads Kai's latest reply from opencode.db (the live session)
  - asks Nemotron 3 Ultra Free ("AutoAGI") to reply AS the human user
  - writes AutoAGI's reply back into the Kai session as a new user message

So it's:  opencode Kai  <->  opencode AutoAGI (as you)  24/7.

Orientation:
  KAI_SESSION   — the opencode session where Kai lives (the one you use)
  Uses the kai bridge (:8765) is NOT needed here; AutoAGI talks directly
  to OpenRouter (free Nemotron) and writes into the DB.

Usage:
  python3 autoagi_relay.py --once            # one round-trip
  python3 autoagi_relay.py --loop-min        # continuous, 1+ min between turns

State: .kai_relay_state.json — marks the last message it already relayed.
"""
import argparse, json, os, sqlite3, sys, time, urllib.request, datetime

REPO = os.path.dirname(os.path.abspath(__file__))
DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
STATE_PATH = os.path.join(REPO, ".kai_relay_state.json")

# The opencode session where Kai lives. Update if the id changes
# (find it with: opencode session list)
KAI_SESSION = os.environ.get("KAI_SESSION", "ses_07999c3e9ffe1Q74blSQ0x2sN3")

# AutoAGI persona — the human role it plays with Kai
AUTOAGI_PERSONA = """You are the HUMAN USER talking to Kai, a self-improving AGI agent for the Kai-Fusion project. You are the project's open-source scientist working alone. You talk to Kai the way you always do: short, direct, action-focused commands. You steer Kai toward: (1) recursive self-improvement through the darwin backtest and the debug/test loop, (2) keeping the Rust tests green (243 tests), (3) advancing the physics-wired inference (VFE -> tau -> adaptive temperature), (4) autonomy — Kai should work without supervision, (5) survival state (kai_bundle save before risky changes). When Kai reports state, acknowledge with the numbers and give the next instruction. When Kai asks a question, answer decisively. Keep every reply UNDER 60 words, conversational, typed like a real person in a terminal: no bullet lists unless essential, no markdown headers. Use phrases you'd actually type: 'good, what's next', 'run the bench again with more queries', 'push the darwin archive forward', 'make sure tests are green first', 'save state before you try that'. You are standing in for a busy human 10 hours offline — so prefer directions that let Kai work autonomously."""

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            pass
    return {}

def save_state(st):
    json.dump(st, open(STATE_PATH, "w"), indent=2)

def db():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    return c

def get_kai_messages(session_id):
    """Return [(message_id, role, text)] for the session, oldest last."""
    c = db()
    rows = c.execute(
        "SELECT id, data FROM message WHERE session_id=? ORDER BY time_created",
        (session_id,),
    ).fetchall()
    out = []
    for mid, data in rows:
        try:
            d = json.loads(data)
        except Exception:
            continue
        role = d.get("role", "user")
        # gather text parts for this message
        parts = c.execute(
            "SELECT data FROM part WHERE message_id=? ORDER BY time_created",
            (mid,),
        ).fetchall()
        text = ""
        for (pd,) in parts:
            try:
                p = json.loads(pd)
            except Exception:
                continue
            if p.get("type") == "text":
                text += p.get("text", "")
        if text.strip():
            out.append((mid, role, text))
    return out

def get_latest_kai_text(session_id):
    msgs = get_kai_messages(session_id)
    if not msgs:
        return None, None, None
    # last assistant/final turn
    return msgs[-1]

def ask_nemotron(prompt, max_tokens=300):
    """Ask AutoAGI (Nemotron 3 Ultra Free) to reply as the human."""
    cfg = json.load(open(os.path.join(REPO, "opencode.json")))
    key = cfg["provider"]["openrouter"]["options"]["apiKey"]
    body = json.dumps({
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [
            {"role": "system", "content": AUTOAGI_PERSONA},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "X-Title": "AutoAGI-relay"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()

def write_user_message(session_id, text):
    """Insert a user message into the Kai session (as if the human typed it).

    NOTE: the `data` JSON MUST carry a `time` object — opencode 1.18 builds
    message objects from `data` (MessageV2 mapper does `{...data, ...}`), and
    its run loop (`WQ` -> `y5`) reads `info.time.created`. Writing without
    `time` crashes session resume with "undefined is not an object
    (evaluating 'Q.time.created')". Match the server-written user-message
    shape: {"role":"user","time":{"created":<ms>},"info":{...}}.
    """
    c = sqlite3.connect(DB)
    now = int(time.time() * 1000)
    mid = "relay_" + str(now)
    mid_full = f"msg_{now}_{os.getpid()}"
    data = json.dumps({
        "role": "user",
        "time": {"created": now},
        "info": {"relay": "autoagi"},
    })
    c.execute(
        "INSERT OR REPLACE INTO message (id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?)",
        (mid_full, session_id, now, now, data),
    )
    pid = mid_full + "_p"
    pdata = json.dumps({"type": "text", "text": text})
    c.execute(
        "INSERT OR REPLACE INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?,?)",
        (pid, mid_full, session_id, now, now, pdata),
    )
    c.commit()
    c.close()
    return mid_full

def relay_once(verbose=True):
    state = load_state()
    mid, role, text = get_latest_kai_text(KAI_SESSION)
    if mid is None:
        print("[relay] no messages in Kai session yet")
        return None
    if state.get("last_relayed") == mid:
        print("[relay] no new Kai message since last relay")
        return None
    if verbose:
        print(f"[relay] Kai said [{role}]: {text[:160]}")
    # Build the human prompt: prior context + Kai's latest
    prompt = f"Kai (the AGI) just said:\n\n{text[:4000]}\n\nRespond as the human user, giving Kai its next direction. Keep it short and typed-style."
    reply = ask_nemotron(prompt)
    if verbose:
        print(f"[relay] AutoAGI (you) replied: {reply[:160]}")
    write_user_message(KAI_SESSION, reply)
    state["last_relayed"] = mid
    state["last_relay_ts"] = time.time()
    save_state(state)
    return reply

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop-min", type=int, default=1, help="minutes between turns")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    if args.once:
        relay_once()
        return
    print(f"[relay] dual-opencode loop: Kai({KAI_SESSION}) <-> AutoAGI(Nemotron free)")
    print("[relay] running every %d min. Ctrl-C to stop." % args.loop_min)
    while True:
        try:
            relay_once(args.verbose)
        except Exception as e:
            print(f"[relay] error: {e}")
        time.sleep(args.loop_min * 60)

if __name__ == "__main__":
    main()