"""physics_absorb.py — autonomous sequential physics study driver.

Reads Halliday parts in order (resuming from progress.json), asks the LOCAL
LLM to reason through each part with explicit connection to prior material,
appends the reasoning to kai_physics_log.md, and maintains a rolling
'running summary' so each part connects to everything before it. Fully
resumable: kill/restart anytime; it continues from next_part.

Run:  python3 physics_absorb.py      (background with nohup recommended)
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

BASE = Path(__file__).parent / '.axiom_state' / 'physics'
PARTS = BASE / 'parts'
LOG = BASE / 'kai_physics_log.md'
PROGRESS = BASE / 'progress.json'
RUNSUM = BASE / 'running_summary.md'
OLLAMA = 'http://localhost:11434/api/chat'
MODEL = 'qwen2.5:7b'


def load_progress() -> dict:
    return json.loads(PROGRESS.read_text())


def save_progress(d: dict):
    PROGRESS.write_text(json.dumps(d, indent=2))


def get_runsum() -> str:
    if RUNSUM.exists():
        return RUNSUM.read_text()
    # seed from the log's existing "Running Understanding" block if present
    txt = LOG.read_text() if LOG.exists() else ''
    m = re.search(r'\*After parts.*?\*(.*?)\n\*Bridge to the AGI project:\*', txt, re.S)
    if m:
        return "Prior understanding (seeded):\n" + m.group(1).strip()
    return "(no prior summary yet)"


def call_llm(system: str, user: str) -> str:
    import requests
    try:
        r = requests.post(OLLAMA, json={
            'model': MODEL,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'stream': False,
            'options': {'temperature': 0.0, 'num_predict': 1100},
        }, timeout=180)
        r.raise_for_status()
        return r.json()['message']['content']
    except Exception as e:
        return f'[LLM ERROR: {e}]'


def main():
    d = load_progress()
    start = d.get('next_part', 11)
    total = d.get('total_parts', 182)
    runsum = get_runsum()
    print(f'absorb: starting at part {start} of {total}', flush=True)
    for n in range(start, total + 1):
        part = PARTS / f'part_{n:03d}.txt'
        if not part.exists():
            print(f'  skip missing part_{n:03d}', flush=True)
            continue
        text = part.read_text(errors='replace')
        m = re.search(r'part \d+: pages (\d+)-(\d+)', text)
        pages = f"{m.group(1)}-{m.group(2)}" if m else '?'
        system = ('You are a physics reasoning engine doing a persistent, sequential study of '
                  'Halliday & Resnick, Fundamentals of Physics. You read the book in order and reason '
                  'through each part, explicitly CONNECTING it to earlier material. Write a concise, '
                  'substantive reasoning section (a few bullets): what the part covers, the key physics '
                  'ideas/equations, and how it CONNECTS to and EXTENDS prior chapters (use the running '
                  'summary). Then on a new line write "RUNNING SUMMARY:" followed by an UPDATED compact '
                  'running summary (the connective tissue) for the next part. Be precise.')
        user = (f'RUNNING SUMMARY (prior understanding):\n{runsum}\n\n'
                f'PART {n} (pages {pages}) TEXT (may be truncated):\n{text[:14000]}\n\n'
                'Produce the reasoning section, then RUNNING SUMMARY:.')
        out = call_llm(system, user)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(f'\n## Part {n:03d}  (pages {pages})\n\n{out}\n')
        if 'RUNNING SUMMARY:' in out:
            runsum = out.split('RUNNING SUMMARY:', 1)[1].strip()
            RUNSUM.write_text(runsum)
        d['next_part'] = n + 1
        d['last_done'] = n
        d['reader'] = 'local_model_auto'
        save_progress(d)
        print(f'  done part {n:03d} (pages {pages}); next={d["next_part"]}', flush=True)
        time.sleep(0.5)
    print('absorb: ALL PARTS DONE', flush=True)


if __name__ == '__main__':
    main()
