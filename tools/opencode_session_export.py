#!/usr/bin/env python3
"""Dump recent opencode sessions as a Q/A corpus for absorption into Kai chat
memory. Reads opencode's SQLite DB, reconstructs user->assistant turns, and
writes one .txt file per session into axiom_horizon/opencode_sessions/.

Each turn is encoded as:
    Q: <user input>
    A: <assistant text>
with a blank line between turns, matching the kai_absorb format so it can be
absorbed later with kai_absorb_docs.py --resume.
"""
import sqlite3, json, os, sys

DB = os.path.expanduser('~/.local/share/opencode/opencode.db')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'opencode_sessions')


def get_role_and_text(pdata):
    """Return (role, text) from a part's data JSON."""
    if not pdata:
        return (None, '')
    try:
        d = json.loads(pdata)
    except Exception:
        return (None, '')
    if not isinstance(d, dict):
        return (None, '')
    role = d.get('role') or d.get('type')
    txt = d.get('text') or ''
    return (role, txt)


def main():
    os.makedirs(OUT, exist_ok=True)
    db = sqlite3.connect(DB)
    c = db.cursor()

    # parts ordered by message id / time; message id encodes time-ish ordering
    c.execute("""SELECT p.session_id, p.message_id, p.data
                 FROM part p
                 ORDER BY p.time_created, p.id""")
    rows = c.fetchall()

    # Message role lookup
    roles = {}
    c.execute("SELECT id, data FROM message")
    for mid, mdata in c.fetchall():
        try:
            roles[mid] = json.loads(mdata).get('role')
        except Exception:
            roles[mid] = None

    # Build sessions: list of (role, text) in order
    sessions = {}
    for sid, mid, pdata in rows:
        if sid not in sessions:
            sessions[sid] = []
        role = roles.get(mid)
        txt = ''
        if pdata:
            try:
                d = json.loads(pdata)
                if isinstance(d, dict) and d.get('type') == 'text':
                    txt = d.get('text') or ''
            except Exception:
                txt = ''
        if txt:
            sessions[sid].append((role, txt))

    total_turns = 0
    n_files = 0
    for sid, msgs in sessions.items():
        # Build Q/A turns: user text followed by next assistant text
        turns = []
        i = 0
        while i < len(msgs):
            role, txt = msgs[i]
            if role == 'user':
                q = txt[:500]
                # find next assistant text
                a = ''
                j = i + 1
                while j < len(msgs):
                    r2, t2 = msgs[j]
                    if r2 == 'assistant':
                        a = t2[:1000]
                        break
                    j += 1
                if q and a:
                    turns.append(f"Q: {q}\nA: {a}")
                    i = j  # skip past the assistant we consumed
                    continue
            i += 1

        if not turns:
            continue
        total_turns += len(turns)
        n_files += 1
        body = '\n\n'.join(turns) + '\n'
        fp = os.path.join(OUT, f"ses_{sid[:14]}.txt")
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(body)

    print(f"Wrote {n_files} session files, {total_turns} turns total, to {OUT}")
    db.close()


if __name__ == '__main__':
    main()
