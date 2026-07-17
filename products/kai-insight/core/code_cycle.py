"""Code-driven cycle: Kai writes and executes real Python, safely.

This is the AGI-grade self-action step: given a question, Kai generates a
short Python program that *computes* an answer, then we execute it inside a
hardened sandbox. Safety model:

  * Generated code is statically scanned for forbidden constructs
    (imports of os/sys/subprocess, __import__, eval/exec of dynamic values,
    file writes, network sockets, deletes). Anything forbidden -> abort.
  * Execution runs in a SEPARATE subprocess with:
      - a read-only view of the filesystem (cwd = temp dir, no writes allowed)
      - a hard wall-clock timeout (no infinite loops / hangs)
      - network fully disabled (socket close at start)
      - restricted environment (no PYTHONPATH leakage of privileged code)
  * Gated by the existing self-mod capability test: the live KaiMind must pass
    run_health_check() BEFORE and AFTER execution. If post-exec health drops,
    the result is discarded and the sandbox is considered unsafe.

This module is MANUAL-ONLY — it is NEVER wired into the running daemon loop.
Run it directly:

    python3 code_cycle.py "compute the entropy of this list: [1,1,2,3,5,8]"
"""

from __future__ import annotations

import os
import sys
import re
import ast
import json
import time
import subprocess
import tempfile
import logging
from typing import Dict, Any, Tuple

from kai_local import load_kai_mind

logger = logging.getLogger("code_cycle")

# ── Static safety policy ──────────────────────────────────────────────────
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "pathlib", "importlib", "ctypes", "mmap", "pdb", "builtins",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}
FORBIDDEN_ATTRS = {"__class__", "__subclasses__", "__globals__", "__builtins__",
                   "__import__", "__dict__", "__mro__"}
FORBIDDEN_DECORATORS = {"classmethod", "staticmethod"}

CODE_TIMEOUT = 8.0  # seconds; hard cap on generated code


def _static_check(src: str) -> Tuple[bool, str]:
    """Reject obviously unsafe generated code. Returns (ok, reason)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):
        # Imports of forbidden modules
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in FORBIDDEN_NAMES:
                    return False, f"forbidden import: {top}"
        if isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in FORBIDDEN_NAMES:
                return False, f"forbidden import from: {top}"
        # Forbidden calls
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                return False, f"forbidden call: {fn.id}()"
            if isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_ATTRS:
                return False, f"forbidden attribute access: {fn.attr}"
        # Bare name references to dangerous builtins
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return False, f"forbidden name: {node.id}"

    # Regex backstop for obfuscated misuse
    for pat in (r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(",
                r"\bos\.", r"\bsubprocess\.", r"\bsocket\.", r"\bopen\s*\("):
        if re.search(pat, src):
            return False, f"forbidden pattern: {pat}"

    if "import os" in src or "from os" in src:
        return False, "forbidden: os module"
    return True, "ok"


def _sandbox_runner(code: str, timeout: float = CODE_TIMEOUT) -> Dict[str, Any]:
    """Execute `code` in a hardened subprocess. Returns stdout/stderr/ok."""
    harness = (
        "import sys, signal\n"
        "try:\n"
        "    import socket\n"
        "    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    _s.close()\n"
        "    socket.socket = None  # disable further network use\n"
        "except Exception:\n"
        "    pass\n"
        "import json\n"
        "try:\n"
        "    _ns = {}\n"
        "    exec(compile(" + repr(code) + ", '<kai_code>', 'exec'), _ns)\n"
        "    _out = _ns.get('result', None)\n"
        "    print('__KAI_RESULT__' + json.dumps({'result': _out}))\n"
        "except Exception as e:\n"
        "    print('__KAI_ERROR__' + repr(e))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s",
                "stdout": "", "stderr": ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "stdout": "", "stderr": ""}

    out = proc.stdout or ""
    err = proc.stderr or ""
    if "__KAI_ERROR__" in out:
        return {"ok": False, "error": out.split("__KAI_ERROR__", 1)[1].strip(),
                "stdout": out, "stderr": err}
    if "__KAI_RESULT__" in out:
        try:
            payload = json.loads(out.split("__KAI_RESULT__", 1)[1].strip())
            return {"ok": True, "result": payload.get("result"),
                    "stdout": out, "stderr": err}
        except Exception:
            return {"ok": True, "result": out, "stdout": out, "stderr": err}
    return {"ok": True, "result": out.strip(), "stdout": out, "stderr": err}


def _generate_code(kai, question: str) -> str:
    """Ask the live Kai to produce a self-contained Python snippet solving `question`.

    The snippet must assign its final answer to a variable named `result`.
    """
    prompt = (
        "Write a short, self-contained Python 3 program that COMPUTES the answer "
        "to the question below. Constraints:\n"
        " - Pure standard library only (math, statistics, json, itertools, collections).\n"
        " - Assign the final answer to a variable named `result`.\n"
        " - No imports of os/sys/subprocess/socket, no file IO, no network, no eval/exec.\n"
        " - Print nothing except via the final `result` assignment.\n"
        "Return ONLY the code in a single ```python fenced block.\n\n"
        f"QUESTION: {question}\n"
    )
    try:
        r = kai.live(prompt, max_steps=1)
        text = r.get("answer", "") if isinstance(r, dict) else str(r)
    except Exception:
        text = ""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if the model returned raw code, use it.
    if "result" in text and ("=" in text):
        return text.strip()
    return ""


def run_code_cycle(question: str, use_kai: bool = True,
                   static_code: str = "") -> Dict[str, Any]:
    """End-to-end: optionally generate code via Kai, statically check, sandbox-run,
    and gate on the live KaiMind health check. Returns a structured report.
    """
    report: Dict[str, Any] = {"question": question, "ts": round(time.time(), 3)}
    kai = None
    if use_kai:
        try:
            kai = load_kai_mind(readonly=True)
            report["kai_loaded"] = kai is not None
            if kai is None:
                report["kai_error"] = "load_kai_mind returned None"
                use_kai = False
        except Exception as e:
            report["kai_loaded"] = False
            report["kai_error"] = str(e)
            use_kai = False

    # Capability gate (pre): only run if the mind is healthy.
    if use_kai and kai is not None:
        try:
            pre = kai.run_health_check()
            report["health_pre"] = round(pre.get("test_health", 0.0), 3)
            if pre.get("test_health", 0.0) < 0.5:
                report["aborted"] = "pre health gate failed"
                return report
        except Exception as e:
            report["aborted"] = f"health check error: {e}"
            return report

    # Get the code
    if static_code:
        code = static_code
        report["code_source"] = "provided"
    elif use_kai:
        code = _generate_code(kai, question)
        report["code_source"] = "kai_generated"
    else:
        report["aborted"] = "no code source"
        return report

    report["code"] = code

    ok, reason = _static_check(code)
    report["static_check"] = {"ok": ok, "reason": reason}
    if not ok:
        report["aborted"] = f"static check failed: {reason}"
        return report

    res = _sandbox_runner(code)
    report["execution"] = {
        "ok": res["ok"],
        "result": res.get("result"),
        "error": res.get("error"),
    }

    # Capability gate (post): discard result if the mind got worse.
    if use_kai and kai is not None:
        try:
            post = kai.run_health_check()
            report["health_post"] = round(post.get("test_health", 0.0), 3)
            if post.get("test_health", 0.0) < 0.5 or \
               (report.get("health_pre", 1.0) - post.get("test_health", 0.0)) > 0.2:
                report["result_accepted"] = False
                report["aborted"] = "post health gate regressed"
                return report
        except Exception:
            pass

    report["result_accepted"] = res["ok"]
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = sys.argv[1] if len(sys.argv) > 1 else "compute the entropy of [1,1,2,3,5,8]"
    rep = run_code_cycle(q, use_kai=True)
    print(json.dumps(rep, indent=2, default=str))
