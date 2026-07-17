"""
Linux shell tools for the Axiom Horizon agent.

Tools are exposed to qwen3-coder via JSON schemas. All destructive or
network-active actions are gated through confirm_destructive() which asks
the human operator before execution.
"""
import os
import shlex
import subprocess
import json
import re
import hashlib
from pathlib import Path

CWD = Path.cwd()

DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf?\b",
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bchmod\s+-R\s+777\b",
    r"\bchown\s+-R\b",
    r">\s*/dev/(sd|nvme|hd)",
    r"\bcurl\b.*\|\s*(ba)?sh",
    r"\bwget\b.*\|\s*(ba)?sh",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\binit\s+0\b",
    r":\(\)\s*\{.*:\|:.*\}\s*;:",  # fork bomb
]

SAFE_BASH_PREFIX = f"set -u; export PS1='$ '; cd {shlex.quote(str(CWD))} && "

def confirm_destructive(command: str) -> bool:
    """Print a red warning and require 'y' to proceed. Used for risky patterns."""
    red = "\033[91m"
    yellow = "\033[93m"
    reset = "\033[0m"
    print(f"{red}[DESTRUCTIVE COMMAND DETECTED]{reset}")
    print(f"{yellow}  $ {command}{reset}")
    try:
        answer = input("Execute? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer == "y"

def is_destructive(command: str) -> bool:
    for pat in DESTRUCTIVE_PATTERNS:
        if re.search(pat, command):
            return True
    return False

def run_bash(command: str, timeout: int = 30) -> dict:
    """Execute a shell command, capture output, enforce safety on risky patterns."""
    if is_destructive(command):
        if not confirm_destructive(command):
            return {"ok": False, "blocked": True, "stdout": "", "stderr": "User declined destructive command."}
    try:
        result = subprocess.run(
            ["bash", "-c", SAFE_BASH_PREFIX + command],
            capture_output=True, text=True, timeout=timeout, cwd=str(CWD)
        )
        out = result.stdout
        if len(out) > 8000:
            out = out[:8000] + f"\n... [truncated, {len(result.stdout)} total chars]"
        err = result.stderr
        if len(err) > 2000:
            err = err[:2000] + f"\n... [truncated, {len(result.stderr)} total chars]"
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": out,
            "stderr": err,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True, "stdout": "", "stderr": f"Command timed out after {timeout}s."}
    except Exception as e:
        return {"ok": False, "exception": str(e), "stdout": "", "stderr": ""}

def read_file(path: str, max_bytes: int = 50000) -> dict:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"ok": False, "error": f"No such file: {p}"}
        if p.is_dir():
            return {"ok": False, "error": f"Is a directory: {p}"}
        size = p.stat().st_size
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_bytes)
        truncated = size > max_bytes
        return {"ok": True, "path": str(p), "size": size, "truncated": truncated, "content": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def write_file(path: str, content: str) -> dict:
    p = Path(path).expanduser().resolve()
    if p.exists() and not confirm_destructive(f"overwrite {p}"):
        return {"ok": False, "blocked": True, "error": "User declined overwrite."}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(p), "bytes": len(content)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def list_dir(path: str = ".") -> dict:
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            return {"ok": False, "error": f"Not a directory: {p}"}
        entries = []
        for child in sorted(p.iterdir()):
            try:
                st = child.stat()
                entries.append({
                    "name": child.name,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": st.st_size if child.is_file() else None,
                })
            except (PermissionError, FileNotFoundError):
                entries.append({"name": child.name, "kind": "?", "size": None})
        return {"ok": True, "path": str(p), "entries": entries[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def grep_files(pattern: str, path: str = ".", max_matches: int = 80) -> dict:
    try:
        cmd = ["grep", "-rn", "--color=never", pattern, path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(CWD))
        out = result.stdout
        lines = out.splitlines()
        truncated = len(lines) > max_matches
        if truncated:
            lines = lines[:max_matches]
        return {"ok": True, "matches": lines, "truncated": truncated, "total_lines": len(result.stdout.splitlines())}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "grep timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

STATE_EXCLUDE_DIRS = {".venv", ".git", "__pycache__", "node_modules", ".cache"}
STATE_EXCLUDE_EXTS = {".pyc", ".so", ".o", ".a", ".log", ".tmp", ".swp"}

def working_state_hash(root: str | None = None, max_files: int = 4000) -> str:
    """
    Hash the on-disk working state. Walks CWD (or root), hashes each file's
    path + mtime + size. Used to detect when tool calls have stopped changing
    anything — i.e. the agent is in F=0 equilibrium.
    """
    root = Path(root or CWD)
    h = hashlib.sha256()
    count = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in STATE_EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix in STATE_EXCLUDE_EXTS:
            continue
        try:
            st = p.stat()
            h.update(str(p.relative_to(root)).encode())
            h.update(str(st.st_mtime_ns).encode())
            h.update(str(st.st_size).encode())
        except (OSError, PermissionError):
            continue
        count += 1
        if count >= max_files:
            h.update(b"[truncated]")
            break
    h.update(str(count).encode())
    return h.hexdigest()[:16]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command in the local Linux environment. Captures stdout, stderr, and return code. Destructive patterns (rm -rf, sudo, dd, mkfs, fork bomb, etc.) will be confirmed with the user before running.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                    "timeout": {"type": "integer", "description": "Max seconds to wait. Default 30.", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Files larger than 50KB are truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file. Will ask user before overwriting an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to current working directory."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search for a regex pattern recursively using grep. Returns matching lines with file:line:content format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern (grep -E compatible)."},
                    "path": {"type": "string", "description": "Directory or file to search. Defaults to current dir."},
                },
                "required": ["pattern"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "run_bash": lambda args: run_bash(args["command"], args.get("timeout", 30)),
    "read_file": lambda args: read_file(args["path"]),
    "write_file": lambda args: write_file(args["path"], args["content"]),
    "list_dir": lambda args: list_dir(args.get("path", ".")),
    "grep_files": lambda args: grep_files(args["pattern"], args.get("path", ".")),
}
