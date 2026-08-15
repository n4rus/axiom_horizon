#!/usr/bin/env python3
"""
kai_ingest.py — Seed Kai's attractor memory from the axiom_horizon corpus.

Strategy (from KAI_SCALE_PROPOSAL.md):
  295k files → ~500 directory-level attractor entries
  VFE controller routes queries to the right neighborhood.

Phases:
  1. walk tree → build directory index + file-type inventory
  2. for each significant dir, read key files → generate compact summary
  3. embed summary via Ollama (nomic-embed-text) → store in attractor JSON
  4. write .kai_ingest_progress.json for resume across sessions

Usage:
  python3 tools/kai_ingest.py [--force] [--phase 1|2|3]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

ROOT = "/home/l/Desktop/AxiomTree/axiom_horizon"
PROGRESS_FILE = os.path.join(ROOT, ".kai_ingest_progress.json")
ATTRACTOR_FILE = os.path.join(ROOT, ".kai_corpus_attractor.json")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Directories to skip (build artifacts, caches, etc.)
SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".hg", ".svn",
    "target", "build", "build_llvm", "_deps", "googletest-src",
    ".cache", "blobs", "dist", "out",
    "unsloth_compiled_cache", "cmake-build-debug",
}

# Source file extensions (what we actually summarize)
SOURCE_EXTS = {
    ".py", ".rs", ".go", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx",
    ".rs", ".toml", ".yaml", ".yml", ".json", ".md", ".txt",
    ".ll", ".mlir", ".td", ".mir",
    ".sh", ".bash", ".zsh",
    ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".scss",
    ".proto", ".thrift",
    ".cmake", ".mk", ".gn", ".gni",
    ".cl", ".hlsl", ".metal",
    ".f90", ".f95", ".f03", ".f",
}

# Important root-level files to always include
ROOT_PRIORITY = {
    "AGI_PLAN.md", "KAI_SCALE_PROPOSAL.md", "KAI_LLAMACPP_PLAN.md",
    "KAI_DISSOLUTION_REDO.md", "KAI_FUSION_ARCHITECTURE.md",
    "opencode.json", "kai_bridge.py", "Cargo.toml",
}


def ollama_embed(text: str) -> list:
    """Embed text via Ollama nomic-embed-text. Returns 768-dim vector."""
    data = json.dumps({
        "model": EMBED_MODEL,
        "prompt": text[:8192],  # truncate to context
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_EMBED_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["embedding"]
    except Exception as e:
        print(f"  [WARN] embed failed: {e}", file=sys.stderr)
        return []


def summarize_file(path: str, max_lines: int = 20) -> str:
    """Read key content from a source file for summarization."""
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except:
        return ""
    
    if not lines:
        return ""
    
    # First line (often has a comment/docstring describing the file)
    header = ""
    for i in range(min(3, len(lines))):
        stripped = lines[i].strip()
        if stripped:
            header = stripped[:200]
            break
    
    # Look for docstrings/comments in first 10 lines
    doc_lines = []
    for line in lines[:15]:
        s = line.strip()
        if s.startswith("// ") or s.startswith("# ") or s.startswith("/*") or s.startswith("* "):
            doc_lines.append(s[:150])
    
    # Take a sample of content lines
    content_sample = []
    # Skip comment header
    start = 0
    for i, line in enumerate(lines[:30]):
        s = line.strip()
        if s and not s.startswith("//") and not s.startswith("#") and not s.startswith("/*") and not s.startswith("*"):
            start = i
            break
    
    # Grab key structural lines (function defs, class defs, imports)
    struct_lines = []
    for line in lines[start:min(start + max_lines, len(lines))]:
        s = line.strip()
        if any(kw in s for kw in ("def ", "fn ", "func ", "class ", "struct ",
                                  "impl ", "pub ", "pub fn", "pub struct",
                                  "import ", "from ", "use ", "mod ",
                                  "int ", "float ", "void ", "static ")):
            struct_lines.append(s[:120])
    
    parts = []
    if header:
        parts.append(f"header: {header}")
    if doc_lines:
        parts.append("docs: " + " | ".join(doc_lines[:3]))
    if struct_lines:
        parts.append("defs: " + " | ".join(struct_lines[:8]))
    
    return "\n".join(parts)


def summarize_directory(dirpath: str) -> dict:
    """Generate a compact summary of a directory."""
    rel = os.path.relpath(dirpath, ROOT)
    entries = []
    try:
        for name in os.listdir(dirpath):
            fp = os.path.join(dirpath, name)
            if os.path.isfile(fp):
                entries.append(name)
            elif os.path.isdir(fp):
                if name not in SKIP_DIRS:
                    entries.append(name + "/")
    except PermissionError:
        entries = ["[permission denied]"]
    
    # Categorize
    sources = [e for e in entries if os.path.splitext(e)[1].lower() in SOURCE_EXTS]
    dirs = [e for e in entries if e.endswith("/")]
    
    # Read first-level summary info
    readme_content = ""
    for readme_name in ("README.md", "README.txt", "README", "readme.md"):
        rp = os.path.join(dirpath, readme_name)
        if os.path.isfile(rp):
            try:
                with open(rp, "r") as f:
                    readme_content = f.read(2000)
            except:
                pass
            break
    
    # Sample key files
    key_files = {}
    # Priority to Cargo.toml, CMakeLists.txt, package.json, etc.
    for priority_name in ("Cargo.toml", "CMakeLists.txt", "package.json",
                          "pyproject.toml", "go.mod", "BUILD", "BUILD.gn",
                          "AGENTS.md", "Makefile", "main.py", "main.rs", "main.go",
                          "mod.rs", "lib.rs", "lib.py"):
        if priority_name in entries:
            kf = summarize_file(os.path.join(dirpath, priority_name), max_lines=15)
            if kf:
                key_files[priority_name] = kf
    
    # Sample a few source files
    sampled = []
    for src in sorted(sources)[:5]:
        sf = summarize_file(os.path.join(dirpath, src), max_lines=10)
        if sf:
            sampled.append(sf)
    
    summary = {
        "path": rel,
        "total_entries": len(entries),
        "source_files": len(sources),
        "subdirs": len(dirs),
        "readme": readme_content[:500] if readme_content else "",
        "key_files": key_files,
        "sample_sources": sampled[:3],
        "dir_list": dirs[:20],
        "source_list": sorted(sources)[:20],
    }
    return summary


def generate_summary_text(summary: dict) -> str:
    """Convert a directory summary dict to a compact text for embedding."""
    parts = [f"directory: {summary['path']}"]
    parts.append(f"entries: {summary['total_entries']} files, {summary['subdirs']} subdirs")
    if summary['source_files'] > 0:
        parts.append(f"sources: {summary['source_files']}")
    if summary['readme']:
        parts.append(f"readme: {summary['readme'][:300]}")
    for name, content in summary.get('key_files', {}).items():
        parts.append(f"{name}: {content[:200]}")
    for s in summary.get('sample_sources', []):
        parts.append(f"sample: {s[:200]}")
    if summary.get('dir_list'):
        parts.append("subdirs: " + ", ".join(summary['dir_list'][:15]))
    if summary.get('source_list'):
        parts.append("files: " + ", ".join(summary['source_list'][:15]))
    return "\n".join(parts)


def phase1_walk():
    """Phase 1: Walk tree, build directory index, identify all dirs to summarize."""
    print("=== Phase 1: Walking corpus tree ===", file=sys.stderr)
    
    all_dirs = []
    root_entries = []
    
    # Top-level files
    try:
        for name in os.listdir(ROOT):
            fp = os.path.join(ROOT, name)
            if os.path.isfile(fp) and not name.startswith("."):
                root_entries.append(name)
    except PermissionError:
        pass
    
    # Walk directories
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Skip hidden and build dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
        
        if dirpath == ROOT:
            continue  # skip root itself (handled separately)
        
        rel = os.path.relpath(dirpath, ROOT)
        
        # Only include directories with source files
        source_count = sum(1 for f in filenames 
                          if os.path.splitext(f)[1].lower() in SOURCE_EXTS)
        if source_count > 0 or any(f in filenames for f in ("Cargo.toml", "CMakeLists.txt", 
                                                              "go.mod", "package.json")):
            all_dirs.append({
                "path": rel,
                "files": len(filenames),
                "sources": source_count,
                "dirs": len(dirnames),
            })
    
    # Save directory index
    index = {
        "root_entries": sorted(root_entries),
        "directories": sorted(all_dirs, key=lambda x: x["path"]),
        "total_dirs_found": len(all_dirs),
        "phase": "complete",
    }
    
    progress = load_progress()
    progress["phase1"] = index
    save_progress(progress)
    
    print(f"  Found {len(root_entries)} root files", file=sys.stderr)
    print(f"  Found {len(all_dirs)} source directories", file=sys.stderr)
    return index


def phase2_summarize(limit: int = None):
    """Phase 2: Generate summaries for each significant directory."""
    print("=== Phase 2: Generating directory summaries ===", file=sys.stderr)
    
    progress = load_progress()
    if "phase1" not in progress:
        print("  ERROR: Run phase 1 first", file=sys.stderr)
        return None
    
    dirs = progress["phase1"]["directories"]
    if limit:
        dirs = dirs[:limit]
    
    summaries = progress.get("phase2", {}).get("summaries", {})
    resume_from = len(summaries)
    
    print(f"  Total dirs: {len(dirs)}, already done: {resume_from}", file=sys.stderr)
    
    for i, d in enumerate(dirs):
        if i < resume_from:
            continue
        
        # Map directory path to screen
        dir_path = os.path.join(ROOT, d["path"])
        summary = summarize_directory(dir_path)
        text = generate_summary_text(summary)
        summaries[d["path"]] = {
            "text": text[:2000],
            "file_count": d["files"],
            "source_count": d["sources"],
        }
        
        if (i + 1) % 50 == 0:
            print(f"  Summarized {i+1}/{len(dirs)} dirs...", file=sys.stderr)
            progress["phase2"] = {"summaries": summaries, "phase": "in_progress"}
            save_progress(progress)
    
    progress["phase2"] = {"summaries": summaries, "phase": "complete"}
    save_progress(progress)
    print(f"  Done: {len(summaries)} directory summaries", file=sys.stderr)
    return summaries


def phase3_embed():
    """Phase 3: Embed summaries and build attractor JSON."""
    print("=== Phase 3: Embedding and writing attractor ===", file=sys.stderr)
    
    progress = load_progress()
    if "phase2" not in progress or "summaries" not in progress.get("phase2", {}):
        print("  ERROR: Run phase 2 first", file=sys.stderr)
        return None
    
    summaries = progress["phase2"]["summaries"]
    
    # Also include root-level priority files
    root_entries = []
    for fname in ROOT_PRIORITY:
        fp = os.path.join(ROOT, fname)
        if os.path.isfile(fp):
            try:
                with open(fp, "r") as f:
                    root_entries.append({
                        "path": fname,
                        "text": f.read(2000),
                    })
            except:
                pass
    
    attractor_entries = {}
    embedded = progress.get("phase3", 0)
    
    # Embed each summary
    for i, (path, summary) in enumerate(summaries.items()):
        if i < embedded:
            continue
        
        text = summary["text"]
        if not text.strip():
            attractor_entries[path] = {"embedding": [], "text": text, "error": "empty"}
            continue
        
        embedding = ollama_embed(f"directory: {path}\n{text}")
        if embedding:
            attractor_entries[path] = {
                "embedding": embedding,
                "text": text[:500],
                "file_count": summary["file_count"],
                "source_count": summary["source_count"],
            }
        else:
            attractor_entries[path] = {"embedding": [], "text": text[:500], "error": "embed_failed"}
        
        if (i + 1) % 25 == 0:
            print(f"  Embedded {i+1}/{len(summaries)}...", file=sys.stderr)
            progress["phase3"] = i + 1
            save_progress(progress)
    
    # Embed root entries
    for entry in root_entries:
        embedding = ollama_embed(f"file: {entry['path']}\n{entry['text']}")
        attractor_entries[f"__root__/{entry['path']}"] = {
            "embedding": embedding if embedding else [],
            "text": entry['text'][:500],
        }
    
    # Write attractor JSON
    attractor_data = {
        "version": 1,
        "generated": time.time(),
        "total_entries": len(attractor_entries),
        "embed_dim": 768,
        "source_corpus": "axiom_horizon",
        "entries": attractor_entries,
    }
    
    with open(ATTRACTOR_FILE, "w") as f:
        json.dump(attractor_data, f, indent=1)
    
    progress["phase3"] = len(summaries)
    progress["complete"] = True
    save_progress(progress)
    
    print(f"  Written to {ATTRACTOR_FILE}", file=sys.stderr)
    print(f"  Total entries: {len(attractor_entries)}", file=sys.stderr)
    return attractor_data


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description="Kai corpus ingestion")
    parser.add_argument("--force", action="store_true", help="Restart from scratch")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1=walk, 2=summarize, 3=embed, 0=all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit directories processed (for testing)")
    args = parser.parse_args()
    
    if args.force and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Removed progress file, starting fresh", file=sys.stderr)
    
    if args.phase == 0 or args.phase == 1:
        print("\n" + "=" * 60, file=sys.stderr)
        phase1_walk()
    
    if args.phase == 0 or args.phase == 2:
        print("\n" + "=" * 60, file=sys.stderr)
        phase2_summarize(limit=args.limit)
    
    if args.phase == 0 or args.phase == 3:
        print("\n" + "=" * 60, file=sys.stderr)
        phase3_embed()
    
    print("\n=== Done ===", file=sys.stderr)
    progress = load_progress()
    if progress.get("phase1"):
        print(f"  Dirs found: {progress['phase1']['total_dirs_found']}", file=sys.stderr)
    if progress.get("phase2", {}).get("summaries"):
        print(f"  Dirs summarized: {len(progress['phase2']['summaries'])}", file=sys.stderr)
    if progress.get("complete"):
        print(f"  Attractor written to: {ATTRACTOR_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
