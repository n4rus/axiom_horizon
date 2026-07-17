"""split_parts.py — prepare Halliday.pdf text for sequential persistent study.

Splits the extracted textbook into ordered parts (pages_per_part pages each),
so each part can be fed to Kai in order with reasoning that connects to the
previous part (physics is sequential/connected). Progress is tracked in
progress.json so the study resumes exactly where it stopped.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(__file__).parent / '.axiom_state' / 'physics'
SRC = BASE / 'halliday_text.txt'
PARTS = BASE / 'parts'
PARTS.mkdir(parents=True, exist_ok=True)
PROGRESS = BASE / 'progress.json'

PAGES_PER_PART = 8  # ~50KB/part — sized for one reasoning pass with connection


def split():
    text = SRC.read_text(encoding='utf-8', errors='replace')
    pages = text.split('\x0c')
    # drop the leading watermark junk page if it's tiny/garbage
    pages = [p for p in pages]
    n = len(pages)
    idx = 1
    for start in range(0, n, PAGES_PER_PART):
        chunk = '\n'.join(pages[start:start + PAGES_PER_PART])
        # skip fully empty parts
        if not chunk.strip():
            continue
        out = PARTS / f'part_{idx:03d}.txt'
        meta = (f'<!-- part {idx}: pages {start+1}-{min(start+PAGES_PER_PART, n)} '
                f'of {n} -->\n\n')
        out.write_text(meta + chunk, encoding='utf-8')
        idx += 1
    PROGRESS.write_text(json.dumps({
        'next_part': 1,
        'pages_per_part': PAGES_PER_PART,
        'total_pages': n,
        'total_parts': idx - 1,
    }, indent=2))
    print(f'split {n} pages into {idx-1} parts ({PAGES_PER_PART} pp each) -> {PARTS}')


if __name__ == '__main__':
    split()
