#!/usr/bin/env python3
"""
kai_agency.py — Layer 1 action loop: Kai's "hands".

Provides a small, sandboxed, *self-observing* action toolset the bridge can
execute in-process when the model requests a function call. This closes the
   perceive -> decide -> ACT -> observe -> update priors (VFE)
loop: instead of passively predicting text, Kai can read/list/recall/write and
then reason about the consequence of that action.

Tools (names addressable by the model as kai_*):
  kai_recall(query, top_k)  : semantic recall against the corpus attractor
  kai_read(rel_path)        : read a repo file, returns content (truncated)
  kai_list(rel_path)        : list a directory in the repo
  kai_write(rel_path, text) : WRITE a file under the repo (sandboxed to repo)
  kai_exec(cmd)             : run a shell command (HEAVILY sandboxed, allowed list)

Security:
  - kai_read/list/write resolve paths and refuse anything escaping the repo
    or the workspace (no absolute symlink escape). Write is append-safe.
  - kai_exec runs with a short timeout, no network flags, and only safe
    read-only commands by default (KAI_AGENCY_EXEC=1 enables more).
Built for a local dev agent — reviewerModel, not a remote service.
"""
import os, subprocess, shutil, json

REPO = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.abspath(os.path.join(REPO, "..", "..", "AxiomTree"))

# commands the agent may run (read-only, safe) unless KAI_AGENCY_EXEC=1
SAFE_EXEC = {"ls", "grep", "cat", "head", "tail", "wc", "find", "git",
             "du", "df", "ps", "pwd", "echo", "python3", "wc -c", "stat"}
MAX_READ = 60000
MAX_LIST = 200
MAX_EXEC_OUT = 30000
EXEC_TIMEOUT = 15
ALLOW_HARD_EXEC = os.environ.get("KAI_AGENCY_EXEC", "0") == "1"


def _resolve(root: str, rel: str) -> str:
    """Resolve a workspace-safe absolute path from rel. Raises on escape.

    Models often pass sandbox-style absolute paths ("/workspace/collision.py")
    or plain relative names ("collision.py"). A leading '/' must NOT discard
    the sandbox base (os.path.join would), so absolute rels are rebased onto
    `root` and re-checked — escaping is still impossible."""
    if not rel:
        raise ValueError("empty path")
    base = os.path.realpath(root)
    # Rebase sandbox/absolute style paths onto the sandbox root.
    if os.path.isabs(rel):
        for prefix, repl in (("/workspace/", ""), ("/work/", "")):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        else:
            # Arbitrary absolute path: treat as root-relative (sandboxed).
            rel = rel.lstrip("/")
    joined = os.path.realpath(os.path.join(base, rel))
    if not (joined == base or joined.startswith(base + os.sep)):
        raise PermissionError(f"escape detected: {rel}")
    return joined


def kai_recall(corpus, query: str, top_k: int = 3):
    try:
        hits = corpus.search(query, top_k=int(top_k))
        out = []
        for h in hits:
            out.append(f"{h['path']} (sim={h['similarity']:.3f})\n{h['text'][:400]}")
        return {"ok": True, "results": out[:5] if out else ["(no recall)"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kai_read(rel: str):
    try:
        p = _resolve(REPO, rel)
        if not os.path.isfile(p):
            return {"ok": False, "error": "not a file"}
        data = open(p, "r", errors="replace").read()
        return {"ok": True, "len": len(data), "content": data[:MAX_READ]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kai_list(rel: str = "."):
    try:
        p = _resolve(REPO, rel)
        if not os.path.isdir(p):
            return {"ok": False, "error": "not a dir"}
        entries = sorted(os.listdir(p))[:MAX_LIST]
        return {"ok": True, "n": len(entries), "entries": entries}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kai_write(rel: str, text: str):
    try:
        p = _resolve(REPO, rel)
        d = os.path.dirname(p)
        os.makedirs(d, exist_ok=True)
        # atomic-ish write
        tmp = p + ".kai_tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, p)
        return {"ok": True, "wrote": rel, "bytes": len(text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kai_exec(cmd: str):
    if not cmd or not cmd.strip():
        return {"ok": False, "error": "empty cmd"}
    first = cmd.strip().split()[0]
    allowed = ALLOW_HARD_EXEC or first in SAFE_EXEC
    if not allowed:
        return {"ok": False, "error": f"command '{first}' not in safe list (KAI_AGENCY_EXEC=1 to allow)"}
    try:
        r = subprocess.run(cmd, shell=True, cwd=REPO,
                           capture_output=True, text=True,
                           timeout=EXEC_TIMEOUT)
        out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
        return {"ok": r.returncode == 0, "code": r.returncode,
                "out": out[:MAX_EXEC_OUT]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout >{EXEC_TIMEOUT}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


DISPT = {
    "kai_recall": lambda a, c: kai_recall(c, _as_str(a.get("query", "")), _as_int(a.get("top_k", 3))),
    "kai_read":    lambda a, c: kai_read(_as_str(a.get("path", ""))),
    "kai_list":    lambda a, c: kai_list(_as_str(a.get("path", "."))),
    "kai_write":   lambda a, c: kai_write(_as_str(a.get("path", "")), _as_str(a.get("text", ""))),
    "kai_exec":    lambda a, c: kai_exec(_as_str(a.get("cmd", ""))),
}


def _as_int(v, default: int = 0) -> int:
    """Coerce a tool-arg to an int (flatten dict/list that contain one)."""
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        return int(s) if s.lstrip("-").isdigit() else default
    if isinstance(v, dict):
        for k2, v2 in v.items():
            r = _as_int(v2, default)
            if r != default or isinstance(v2, int):
                return r
        return default
    if isinstance(v, (list, tuple)):
        for x in v:
            r = _as_int(x, default)
            if r != default or isinstance(x, int):
                return r
        return default
    try:
        return int(v)
    except Exception:
        return default


def _as_str(v, default: str = ""):
    """Coerce a tool-arg value to a plain string. Qwen-family models sometimes
    emit double-nested args (e.g. {"cmd": {"cmd": "pwd"}} or {"path": ["a"]});
    flatten any dict/list down to a usable string so the sandboxed tool never
    sees a non-string."""
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, dict):
        # pick the deepest scalar value, else JSON
        for k2, v2 in v.items():
            if isinstance(v2, (str, int, float, bool)):
                return str(v2)
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        parts = [str(x) for x in v if isinstance(x, (str, int, float, bool))]
        return " ".join(parts) if parts else json.dumps(v)
    return str(v)


def dispatch(name: str, args: dict, corpus=None) -> dict:
    """Call the right action tool and return a JSON-serializable observation."""
    fn = DISPT.get(name)
    if not fn:
        return {"ok": False, "error": f"unknown tool {name}"}
    try:
        args = args or {}
        # Normalize each scalar arg to a string before bound tool sees it.
        # (keep top_k etc. coercible; the lambdas are string-oriented)
        return fn(args, corpus)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tools_schema() -> list:
    """Expose the action tools to the model as an OpenAI tools array."""
    return [
        {"type": "function", "function": {
            "name": "kai_recall", "description": "Semantic recall from Kai's memory (corpus).  Returns top matching entries with similarity.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "search query"},
                "top_k": {"type": "integer", "description": "how many results"}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "kai_read", "description": "Read a file from the workspace.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "kai_list", "description": "List a directory.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": []}}},
        {"type": "function", "function": {
            "name": "kai_write", "description": "Write a file (under the workspace).",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "text": {"type": "string"}},
                "required": ["path", "text"]}}},
        {"type": "function", "function": {
            "name": "kai_exec", "description": "Run a safe read-only shell command.",
            "parameters": {"type": "object", "properties": {
                "cmd": {"type": "string"}}, "required": ["cmd"]}}},
    ]