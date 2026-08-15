#!/usr/bin/env python3
"""Count chunks with the wiki absorb settings (6000 chars / 300 overlap)."""
import glob
import sys

sys.path.insert(0, "tools")
import kai_absorb_docs as kad

kad.CHUNK_SIZE = 6000
kad.CHUNK_OVERLAP = 300
from kai_absorb_docs import chunk_text

total = 0
arts = 0
for fp in sorted(glob.glob("wiki_filtered/chunks/articles.*.txt")):
    title = None
    body = []
    c = 0
    for line in open(fp, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if title is None:
            title = line
        elif line == "---":
            continue
        elif line == "<<<END>>>":
            b = "\n".join(body)
            body = []
            if b:
                ch = chunk_text(b)
                c += len(ch)
                total += len(ch)
                arts += 1
        else:
            body.append(line)
    print(fp, "chunks", c)
print("TOTAL chunks:", total, "articles:", arts)
