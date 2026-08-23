#!/usr/bin/env python3
"""Kai Live Loop — AGI bet item 1: closed perceive->act->perceive vs raw prompting.

Real external task gauntlet (python-humanize/humanize, both live at HEAD):
  BUG A (#376): ordinal() wrong suffixes for negative integers.
  BUG B (#375): naturaldelta/naturaltime/precisedelta raise OverflowError on +/-inf.

Arm A (closed loop): up to N cycles of [issues+code+pytest-failure feedback] ->
    model patches -> splice -> repo's own suite judges -> failures fed back.
Arm B (raw baseline): identical first prompt, NO feedback, one effective attempt.

Judge: harness-written acceptance tests + full existing repo suite green.
Ledger: JSONL. Inference: local ollama :11434. Same model both arms.
"""
import difflib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/tmp/opencode/humanize_live")
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"
LEDGER = Path("/home/l/Desktop/AxiomTree/axiom_horizon/.kai_live_loop.humanize_gauntlet.jsonl")
MAX_CYCLES = 20
LAST_ERR = ""

# name -> (file, regex extracting the full top-level function)
FUNCS = {
    "ordinal": (REPO / "src/humanize/number.py", r"^def ordinal\(.*?(?=^def |\Z)"),
    "naturaldelta": (REPO / "src/humanize/time.py", r"^def naturaldelta\(.*?(?=^def |\Z)"),
    "naturaltime": (REPO / "src/humanize/time.py", r"^def naturaltime\(.*?(?=^def |\Z)"),
    "precisedelta": (REPO / "src/humanize/time.py", r"^def precisedelta\(.*?(?=^def |\Z)"),
}
ACCEPT = REPO / "tests/test_gauntlet_kai.py"

ISSUE = """Two open bugs in humanize (both confirmed at HEAD). Fix BOTH.

BUG A (issue #376): ordinal() computes wrong suffixes for negative integers.
>>> humanize.ordinal(-21)
'-21th'     # expected '-21st'
>>> humanize.ordinal(-1)
'-1th'      # expected '-1st'
Cause: value % 100 / value % 10 misbehave for negatives (Python % is non-negative),
so the suffix digit is wrong for negative inputs.

BUG B (issue #375): naturaldelta(), naturaltime() and precisedelta() raise an
uncaught OverflowError for float('inf') / float('-inf') instead of returning the
value unchanged -- inconsistent with other humanize functions and with their own
float('nan') handling.
>>> humanize.naturaldelta(float("inf"))
OverflowError: cannot convert float infinity to integer   # expected: 'inf'
Cause: int()/round() raise OverflowError (not ValueError/TypeError) on infinite
floats; the existing except clauses don't catch it.

Constraints: MINIMAL diffs — change ONLY what is needed for the bugs;
zero behavior change for finite/valid inputs (do NOT rewrite working branches);
keep docstrings verbatim; pure Python; no new dependencies."""

ACCEPT_SRC = '''"""Gauntlet acceptance tests (harness-written, both arms judged identically)."""
import humanize

def test_bugA_negative_ordinals():
    assert humanize.ordinal(-1) == "-1st"
    assert humanize.ordinal(-2) == "-2nd"
    assert humanize.ordinal(-3) == "-3rd"
    assert humanize.ordinal(-4) == "-4th"
    assert humanize.ordinal(-11) == "-11th"
    assert humanize.ordinal(-12) == "-12th"
    assert humanize.ordinal(-13) == "-13th"
    assert humanize.ordinal(-21) == "-21st"
    assert humanize.ordinal(-101) == "-101st"

def test_bugA_positive_ordinals_unchanged():
    assert humanize.ordinal(1) == "1st"
    assert humanize.ordinal(2) == "2nd"
    assert humanize.ordinal(3) == "3rd"
    assert humanize.ordinal(4) == "4th"
    assert humanize.ordinal(11) == "11th"
    assert humanize.ordinal(21) == "21st"
    assert humanize.ordinal(101) == "101st"
    assert humanize.ordinal(111) == "111th"
    assert humanize.ordinal("x") == "x"

def test_bugB_nonfinite():
    assert humanize.naturaldelta(float("inf")) == "inf"
    assert humanize.naturaldelta(float("-inf")) == "-inf"
    assert humanize.naturaltime(float("inf")) == "inf"
    assert humanize.naturaltime(float("-inf")) == "-inf"

def test_bugB_finite_unchanged():
    assert humanize.naturaldelta(45) == "45 seconds"
    assert humanize.naturaldelta(3600) == "an hour"
    assert humanize.naturaltime(0) == "now"
'''


def log(event: dict) -> None:
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with LEDGER.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[ledger] {event.get('event')} {event.get('arm', '')} ok={event.get('ok', '')} cycle={event.get('cycle', '')}")


def chat(messages: list[dict], temperature: float) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 4500, "num_ctx": 16384},
        "keep_alive": "15m",
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2400) as r:
        return json.load(r)["message"]["content"]


def current_sources() -> dict[str, str]:
    out = {}
    for name, (path, rx) in FUNCS.items():
        m = re.search(rx, path.read_text(), re.S | re.M)
        out[name] = m.group(0) if m else ""
    return out


def strip_fences(code: str) -> str:
    code = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", code.strip())
    # nested fences echoed from docstrings/diffs: drop any line that is ONLY a fence
    code = re.sub(r"(?m)^\s*`{3,}[a-zA-Z]*\s*$", "", code)
    # stray marker echoes inside code
    code = re.sub(r"(?m)^\s*===(BEGIN|END)[^=]*===\s*$", "", code)
    return code.strip()


def parse_patches(text: str) -> dict[str, str]:
    """Parse ===BEGIN <name>=== ... ===END=== blocks; fallback: any fenced block defining a known function."""
    patches = {}
    for m in re.finditer(r"===BEGIN\s+([a-zA-Z_]+)\s*===\s*(.*?)===END[^=]*===", text, re.S):
        name = m.group(1).strip()
        if name in FUNCS:
            code = strip_fences(m.group(2))
            if f"def {name}" in code:
                patches[name] = code
    if not patches:
        for m in re.finditer(r"```[a-zA-Z]*\s*(.*?)```", text, re.S):
            code = m.group(1)
            for name in FUNCS:
                if re.search(rf"^def {name}\(", code, re.M) and name not in patches:
                    patches[name] = code.strip()
                    break
    return patches


def splice(patches: dict[str, str]) -> bool:
    """Compile-check each target file after splicing; write only if ALL compile."""
    touched: dict[Path, str] = {}
    for name, code in patches.items():
        path, rx = FUNCS[name]
        src = touched.get(path, path.read_text())
        m = re.search(rx, src, re.S | re.M)
        if not m:
            return False
        touched[path] = src[:m.start()] + code.rstrip() + "\n\n\n" + src[m.end():]
    global LAST_ERR
    for path, new_src in touched.items():
        try:
            compile(new_src, str(path), "exec")
        except SyntaxError as e:
            LAST_ERR = f"{path.name}:{e.lineno} {e.msg} | ctx={e.text!r}"
            return False
    for path, new_src in touched.items():
        path.write_text(new_src)
    return True


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "--ignore=tests/test_benchmarks.py",
         "-q", "--tb=line", "--maxfail=8"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=300,
    )
    out = (r.stdout + "\n" + r.stderr)[-2600:]
    passed = re.search(r"(\d+) passed", out)
    ok = bool(passed) and " failed" not in out and "error" not in out.lower()
    return ok, out


def reset_repo() -> None:
    subprocess.run(["git", "checkout", "--", "src/humanize/number.py", "src/humanize/time.py"],
                   cwd=REPO, check=True)
    ACCEPT.write_text(ACCEPT_SRC)


def build_prompt(sources: dict[str, str], feedback: str = "") -> str:
    blocks = "\n\n".join(
        f"--- current `{name}` ({FUNCS[name][0].name}) ---\n```python\n{src}\n```"
        for name, src in sources.items()
    )
    return f"""You are fixing two real bugs in the open-source project `humanize`.

{issue_block}

{blocks}

{feedback}For EVERY function listed above that needs a change (and only those), return its
complete corrected source wrapped like:

===BEGIN ordinal===
<complete corrected function incl. def line and docstring>
===END ordinal===

If a function needs no change, omit it entirely. Return nothing else outside the blocks."""


issue_block = ISSUE  # kept separate so build_prompt stays simple



def grade_bugs() -> dict:
    """Grade each bug independently against current repo state."""
    import importlib, sys
    for mod in ["humanize"]:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(REPO / "src"))
    import humanize  # noqa
    out = {}
    try:
        out["bugA"] = humanize.ordinal(-21) == "-21st" and humanize.ordinal(-1) == "-1st"
    except Exception:
        out["bugA"] = False
    try:
        ok = True
        for fn in ("naturaldelta", "naturaltime"):
            for v, exp in ((float("inf"), "inf"), (float("-inf"), "-inf")):
                if getattr(humanize, fn)(v) != exp:
                    ok = False
        out["bugB"] = ok
    except Exception:
        out["bugB"] = False
    sys.path.remove(str(REPO / "src"))
    return out

def arm(tag: str, loop: bool) -> dict:
    reset_repo()
    sources = current_sources()
    prev_applied: dict[str, str] = {}
    result = {"arm": tag, "model": MODEL, "cycles": 0, "ok": False, "history": []}
    messages = [{"role": "user", "content": build_prompt(sources)}]
    for cycle in range(1, MAX_CYCLES + 1):
        raw = chat(messages, temperature=0.2)
        patches = parse_patches(raw)
        if not patches:
            result["history"].append({"cycle": cycle, "error": "unparseable"})
            log({"event": "cycle", "arm": tag, "cycle": cycle, "ok": False, "err": "parse", "raw_len": len(raw)})
            break
        if not splice(patches):
            result["cycles"] = cycle
            result["history"].append({"cycle": cycle, "error": "splice/compile"})
            log({"event": "cycle", "arm": tag, "cycle": cycle, "ok": False,
                 "err": "compile", "compile_err": LAST_ERR,
                 "names": sorted(patches),
                 "lens": {k: len(v) for k, v in patches.items()}})
            if not loop:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                "The patched files did not compile. Re-send corrected complete functions "
                "in the same ===BEGIN <name>=== format for every function you changed.")})
            continue
        # restore any functions the model omitted (they are untouched by splice) — ok
        ok, out = run_tests()
        result["cycles"] = cycle
        result["history"].append({"cycle": cycle, "ok": ok, "tail": out[-500:]})
        log({"event": "cycle", "arm": tag, "cycle": cycle, "ok": ok, "patched": sorted(patches)})
        if ok:
            result["ok"] = True
            break
        if not loop:
            break
        messages.append({"role": "assistant", "content": raw})
        # which functional areas are still failing?
        failed_A = "test_bugA" in out
        failed_B = "test_bugB" in out
        untouched = [n for n in FUNCS if n not in prev_applied]
        nudge = ""
        if failed_B and not {"naturaldelta", "naturaltime", "precisedelta"} & set(patches):
            nudge = ("NOTE: you have not modified naturaldelta/naturaltime/precisedelta at all, "
                     "yet their tests are failing. You MUST include ===BEGIN naturaldelta===, "
                     "===BEGIN naturaltime=== and ===BEGIN precisedelta=== blocks with "
                     "corrected complete functions this time.\n\n")
        elif failed_A and "ordinal" not in patches:
            nudge = ("NOTE: ordinal tests are still failing and you omitted ordinal. Include a "
                     "===BEGIN ordinal=== block.\n\n")
        elif (not failed_A) and failed_B:
            nudge = ("Bug A (ordinal) is now FIXED and passing - do NOT include ordinal anymore. "
                     "Only Bug B remains. Return ===BEGIN naturaldelta===, ===BEGIN naturaltime=== "
                     "and ===BEGIN precisedelta=== blocks. Reminder: your previous infinity check used `is`, which compares object "
                     "identity and never fires for floats - equality or a finiteness test "
                     "is required. Also "
                     "and do not modify any finite-input code paths.\n\n")
        diffs = []
        for name, new_code in patches.items():
            old_code = prev_applied.get(name) or sources.get(name, "")
            d = "\n".join(difflib.unified_diff(
                old_code.splitlines(), new_code.splitlines(),
                fromfile=f"{name}:before", tofile=f"{name}:after (yours)", lineterm=""))
            diffs.append(d[:1200])
        diff_txt = "\n\n".join(diffs) or "(no textual change detected)"
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": (
            nudge +
            "Your patch did NOT pass the repository test suite. Pytest:\n\n"
            f"```\n{out}\n```\n\n"
            "Diff of what you just changed:\n\n"
            f"```diff\n{diff_txt}\n```\n\n"
            "Read the assertions carefully: LEFT side is what your code actually "
            "returned, RIGHT side is expected. This gauntlet has TWO independent bugs (ordinal negatives AND inf handling in naturaldelta/naturaltime/precisedelta). Fix ALL failing tests without breaking "
            "anything else. Return complete corrected functions in the same "
            "===BEGIN <name>=== format.")})
        prev_applied.update(patches)
        prev_applied.update(patches)
    result["bugs"] = grade_bugs()
    log({"event": "arm_done", **{k: v for k, v in result.items() if k != "history"}})
    return result


def main():
    LEDGER.parent.mkdir(exist_ok=True)
    log({"event": "start", "task": "humanize-gauntlet-376+375", "model": MODEL, "max_cycles": MAX_CYCLES})
    base = arm("B_raw_singleshot", loop=False)
    loopr = arm("A_closed_loop", loop=True)
    if base["ok"] == loopr["ok"]:
        verdict = f"tie:{loopr['ok']}"
    else:
        verdict = "LOOP_WINS" if loopr["ok"] else "BASELINE_WINS"
    summary = {"event": "summary", "baseline_ok": base["ok"], "baseline_cycles": base["cycles"],
               "loop_ok": loopr["ok"], "loop_cycles": loopr["cycles"], "verdict": verdict,
               "baseline_bugs": base.get("bugs"), "loop_bugs": loopr.get("bugs")}
    log(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
