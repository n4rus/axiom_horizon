#!/usr/bin/env python3
"""
opencode_history_fullexport.py — dump the ENTIRE opencode history as Q/A text.

Unlike opencode_session_export.py this does NOT truncate and does NOT drop
sessions: it walks every session/message, reconstructs ordered user->assistant
turns with FULL text, and writes one .txt per session into a target dir that
kai_absorb_docs.py can then ingest.

Usage:
  python3 tools/opencode_history_fullexport.py [--out opencode_history] [--min_q 20]
"""
import sqlite3, os, json, sys, argparse, time

DB = os.path.expanduser('~/.local/share/opencode/opencode.db')


def load_sessions():
    db = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
    c = db.cursor()

    # message id -> role (from message.data JSON)
    roles = {}
    c.execute("SELECT id, data FROM message")
    for mid, mdata in c.fetchall():
        try:
            d = json.loads(mdata)
            roles[mid] = d.get('role') if isinstance(d, dict) else None
        except Exception:
            roles[mid] = None

    # text parts ordered by time (join part -> message for role)
    c.execute("""SELECT p.session_id, p.message_id, p.data, p.time_created
                 FROM part p ORDER BY p.time_created, p.id""")
    parts = c.fetchall()

    # build per-session ordered list of (role, text)
    sessions = {}                      # sid -> [(role,text)]
    for sid, mid, pdata, tc in parts[::1]:
        if not pdata:
            continue
        try:
            d = json.loads(pdata)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if d.get('type') != 'text':
            continue
        txt = (d.get('text') or '').strip()
        if not txt:
            continue
        role = roles.get(mid)
        sessions.setdefault(sid, []).append((role, txt))
    db.close()
    return sessions


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="opencode_history_full")
    ap.add_argument("--min-q", type=int, default=20, help="min user msg chars")
    args = ap.parse_args()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.out)
    os.makedirs(out, exist_ok=True)
    sessions = load_sessions()

    n_turns = n_files = 0
    t0 = time.time()
    for sid, msgs in sessions.items():
        turns = []
        i = 0
        while i < len(msgs):
            role, txt = msgs[i]
            if role == 'user' and len(txt) >= args.min_q:
                a = ''
                j = i + 1
                while j < len(msgs):
                    r2, t2 = msgs[j]
                    if r2 == 'assistant':
                        a = t2
                        break
                    j += 1
                # never pair two users with no answer
                turns.append(f"Q: {txt}\n\nA: {a}" if a else f"Q: {txt}\n\nA: (no response)")
                i = j if a else i + 1
                continue
            i += 1
        if not turns:
            continue
        n_files += 1
        n_turns += len(turns)
        fn = f"{sid[:14]}.txt" if sid else f"ses_{n_files}.txt"
        with open(os.path.join(out, fn), 'w', encoding='utf-8') as f:
            f.write("\n\n---\n\n".join(turns) + "\n")

    print(f"Wrote {n_files} session files, {n_turns} turns total at {out} in "
          f"{time.time()-t0:.1f}s")
    return n_turns


if __name__ == '__main__':
    main()