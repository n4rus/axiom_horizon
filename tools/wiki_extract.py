#!/usr/bin/env python3
"""Stream-parse the enwiki multistream XML dump and emit plaintext articles
filtered to math/physics/CS/hardware domains.

Uses lxml.etree.iterparse for fast streaming: only one <page> subtree is in
memory at a time, and is dropped immediately after processing.
"""
import argparse, bz2, os, re, sys, time
from lxml import etree

# ── Keyword hints ───────────────────────────────────────────────────────
# Positive: titles containing any of these pass the title filter.
KEYWORD_HINTS = [
    # math
    "mathematics", "algebra", "geometry", "topology", "analysis",
    "calculus", "number theory", "combinatorics", "probability",
    "statistics", "set theory", "group theory", "linear algebra",
    "differential", "tensor", "manifold", "category theory",
    "logic", "theorem", "lemma", "proof", "graph theory",
    # physics
    "physics", "quantum", "relativity", "mechanics", "thermodynamics",
    "electromagnetism", "particle", "atomic", "nuclear", "optics",
    "cosmology", "astrophysics", "gravity", "field theory",
    "string theory", "statistical mechanics", "condensed matter",
    "superconduct", "plasma", "semiconductor", "wave", "energy",
    "force", "momentum", "entropy", "hamiltonian", "lagrangian",
    # software / CS
    "algorithm", "data structure", "computer science", "programming",
    "compiler", "operating system", "database", "network",
    "software", "kernel", "linux", "unix", "windows", "macos",
    "python", "java", "rust ", "c++", "javascript", "typescript",
    "machine learning", "neural network", "deep learning",
    "artificial intelligence", "computer architecture",
    "instruction set", "memory (computing)", "cpu", "gpu",
    "distributed system", "cloud computing", "cryptography",
    "hash function", "encryption", "decryption",
    "regular expression", "automata", "complexity theory",
    "turing machine", "lambda calculus", "type theory",
    "functional programming", "object-oriented",
    # computer hardware / electronics
    "microprocessor", "transistor", "logic gate", "circuit",
    "integrated circuit", "semiconductor device", "diode",
    "mosfet", "cmos", "flip-flop", "register", "alu",
    "bus (computing)", "motherboard", "ram", "rom",
    "hard disk", "ssd", "solid-state drive", "flash memory",
    "graphics card", "sound card", "network card",
    "computer hardware", "electronic engineering",
    "printed circuit", "fpga", "asic", "vlsi",
    "signal processing", "digital signal", "analog",
    "oscilloscope", "multimeter",
]

# Negative: titles containing any of these are skipped (avoid biographies,
# entertainment, geography noise that shares words with our domains).
NEGATIVE_HINTS = [
    "television series", "album by", "song by", "film starring",
    "married to", "born in", "singer", "actor", "actress",
    "footballer", "cricketer", "politician",
    "village in", "town in", "district of",
]

# ── Wikitext cleanup ────────────────────────────────────────────────────
WIKI_PATTERNS = [
    (re.compile(r"\{\{[^{}]*\}\}"), ""),                # simple templates
    (re.compile(r"\[\[File:[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[Image:[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[Category:[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[([^|\]]*?\|)?([^\]]+?)\]\]"), r"\2"),
    (re.compile(r"'''([^']+)'''"), r"\1"),
    (re.compile(r"''([^']+)''"), r"\1"),
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I), ""),
    (re.compile(r"<[^>]+>"), ""),
    (re.compile(r"^=+\s*([^=]+?)\s*=+$", re.M), r"\n\n## \1\n\n"),
    (re.compile(r"^\*+", re.M), ""),
    (re.compile(r"^#+\s*", re.M), ""),
    (re.compile(r"^\{\|.*?\|\}", re.S | re.M), ""),
    (re.compile(r"^---+$", re.M), "\n"),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&#\d+;"), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]

def wikitext_to_text(s: str) -> str:
    for pat, repl in WIKI_PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()

def title_matches(title: str) -> bool:
    t = title.lower()
    if any(neg in t for neg in NEGATIVE_HINTS):
        return False
    return any(kw in t for kw in KEYWORD_HINTS)

# ── Main ───────────────────────────────────────────────────────────────
def open_input(path):
    if path == "-":
        return sys.stdin.buffer
    if path.endswith(".bz2"):
        return bz2.open(path, "rb")
    return open(path, "rb")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to enwiki .xml or .xml.bz2 (- for stdin)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--min-chars", type=int, default=400,
                    help="skip articles whose cleaned text is shorter than this")
    ap.add_argument("--progress", type=int, default=20000)
    args = ap.parse_args()

    stats = {"pages": 0, "matched": 0, "emitted": 0, "redirects": 0, "chars": 0}
    t0 = time.time()
    src = open_input(args.input)
    with src, open(args.output, "w", encoding="utf-8") as out:
        context = etree.iterparse(src, events=("end",), tag="{*}page")
        for _, page in context:
            stats["pages"] += 1
            if stats["pages"] % args.progress == 0:
                elapsed = time.time() - t0
                rate = stats["pages"] / elapsed
                print(f"[wiki_extract] pages={stats['pages']} "
                      f"matched={stats['matched']} emitted={stats['emitted']} "
                      f"chars={stats['chars']/1e6:.1f}MB "
                      f"rate={rate:.0f}pg/s elapsed={elapsed:.0f}s",
                      file=sys.stderr, flush=True)
            # Pull the cheap fields
            title_el = page.find("{*}title")
            ns_el = page.find("{*}ns")
            if title_el is None or ns_el is None:
                page.clear(); continue
            title = title_el.text or ""
            if ns_el.text != "0":
                page.clear(); continue
            if page.find("{*}redirect") is not None:
                stats["redirects"] += 1
                page.clear(); continue
            if not title_matches(title):
                page.clear(); continue
            stats["matched"] += 1
            # Pull text
            text_el = page.find(".//{*}text")
            wikitext = text_el.text if text_el is not None else ""
            if not wikitext:
                page.clear(); continue
            clean = wikitext_to_text(wikitext)
            if len(clean) < args.min_chars:
                page.clear(); continue
            stats["emitted"] += 1
            out.write(title)
            out.write("\n---\n")
            out.write(clean)
            out.write("\n<<<END>>>\n")
            stats["chars"] += len(clean)
            # CRITICAL: free the subtree to keep memory bounded
            page.clear()
            # also drop preceding siblings to keep the tree small
            while page.getprevious() is not None:
                del page.getparent()[0]

    elapsed = time.time() - t0
    rate = stats["pages"] / elapsed if elapsed > 0 else 0
    print(f"\nDONE in {elapsed:.1f}s ({rate:.0f} pages/s)")
    print(f"  pages scanned: {stats['pages']}")
    print(f"  redirects:     {stats['redirects']}")
    print(f"  matched:       {stats['matched']}")
    print(f"  emitted:       {stats['emitted']}")
    print(f"  total chars:   {stats['chars']/1e6:.1f} MB")
    print(f"  output:        {os.path.getsize(args.output)/1e6:.1f} MB")

if __name__ == "__main__":
    main()