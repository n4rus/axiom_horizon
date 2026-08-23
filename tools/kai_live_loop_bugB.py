#!/usr/bin/env python3
"""Lean Bug-B-only gauntlet: naturaldelta/naturaltime/precisedelta inf handling.

Same harness as kai_live_loop.py but ONLY those three functions in context,
small prompt (~3k tokens vs 6k gauntlet), 7b-friendly. Judge = Bug B only.
"""
import json, re, subprocess, urllib.request, difflib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/tmp/opencode/humanize_live")
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
LEDGER = Path("/home/l/Desktop/AxiomTree/axiom_horizon/.kai_live_loop.bugB_lean.jsonl")
MAX_CYCLES = 12
FUNCS = {
    "naturaldelta": (REPO / "src/humanize/time.py", r"^def naturaldelta\(.*?(?=^def |\Z)"),
    "naturaltime": (REPO / "src/humanize/time.py", r"^def naturaltime\(.*?(?=^def |\Z)"),
    "precisedelta": (REPO / "src/humanize/time.py", r"^def precisedelta\(.*?(?=^def |\Z)"),
}
ACCEPT = REPO / "tests/test_gauntlet_kai.py"  # reuse same path; overwritten per run

ISSUE = """Bug: naturaldelta(), naturaltime(), precisedelta() raise an uncaught
OverflowError for float('inf') / float('-inf') instead of returning the value
unchanged. Consistent with how these three already handle float('nan') and how
every other humanize function handles non-finite input.

Repro:
>>> humanize.naturaldelta(float("inf"))
OverflowError: cannot convert float infinity to integer   # expected: 'inf'
>>> humanize.naturaltime(float('-inf'))
OverflowError                                          # expected: '-inf'

Cause: int()/round() raise OverflowError on infinite floats, but the existing
try/except clauses only catch (ValueError, TypeError).

Fix must preserve ALL finite-input behavior."""

ACCEPT_SRC = '''import humanize
def test_bugB_nonfinite():
    assert humanize.naturaldelta(float("inf")) == "inf"
    assert humanize.naturaldelta(float("-inf")) == "-inf"
    assert humanize.naturaltime(float("inf")) == "inf"
    assert humanize.naturaltime(float("-inf")) == "-inf"
def test_bugB_finite():
    assert humanize.naturaldelta(45) == "45 seconds"
    assert humanize.naturaldelta(3600) == "an hour"
    assert humanize.naturaltime(0) == "now"
'''

def log(e):
    e["ts"] = datetime.now(timezone.utc).isoformat()
    with LEDGER.open("a") as f: f.write(json.dumps(e)+"\n")
    print(f"[B] {e.get('event')} {e.get('arm','')} ok={e.get('ok','')} cyc={e.get('cycle','')}")

def chat(msgs, t=0.2):
    body = json.dumps({"model": MODEL, "messages": msgs, "stream": False,
        "options": {"temperature": t, "num_predict": 3200, "num_ctx": 10240},
        "keep_alive": "10m"}).encode()
    r = urllib.request.urlopen(urllib.request.Request(OLLAMA, data=body, headers={"Content-Type":"application/json"}), timeout=2400)
    return json.load(r)["message"]["content"]

def current_sources():
    return {n: re.search(rx, p.read_text(), re.S|re.M).group(0) for n,(p,rx) in FUNCS.items()}

def strip_fences(c):
    c = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", c.strip())
    c = re.sub(r"(?m)^\s*`{3,}[a-zA-Z]*\s*$", "", c)
    c = re.sub(r"(?m)^\s*===(BEGIN|END)[^=]*===\s*$", "", c)
    return c.strip()

def parse_patches(t):
    out={}
    for m in re.finditer(r"===BEGIN\s+([a-zA-Z_]+)\s*===\s*(.*?)===END[^=]*===", t, re.S):
        n,c = m.group(1).strip(), strip_fences(m.group(2))
        if n in FUNCS and f"def {n}" in c: out[n]=c
    if not out:
        for m in re.finditer(r"```[a-zA-Z]*\s*(.*?)```", t, re.S):
            c=m.group(1)
            for n in FUNCS:
                if re.search(rf"^def {n}\(", c, re.M) and n not in out:
                    out[n]=c.strip(); break
    return out

LAST_ERR=""
def splice(patches):
    global LAST_ERR
    touched={}
    for n,c in patches.items():
        p,rx = FUNCS[n]
        src=touched.get(p, p.read_text())
        m=re.search(rx, src, re.S|re.M)
        if not m: return False
        touched[p]=src[:m.start()]+c.rstrip()+"\n\n\n"+src[m.end():]
    for p,s in touched.items():
        try: compile(s, str(p), "exec")
        except SyntaxError as e:
            LAST_ERR=f"{p.name}:{e.lineno} {e.msg}|{e.text!r}"; return False
    for p,s in touched.items(): p.write_text(s)
    return True

def run_tests():
    r=subprocess.run(["python3","-m","pytest","tests/","--ignore=tests/test_benchmarks.py","-q","--tb=line","--maxfail=8"],
        cwd=REPO, capture_output=True, text=True, env={"PYTHONPATH":"src","PATH":"/usr/bin:/bin:/usr/local/bin"}, timeout=300)
    out=(r.stdout+"\n"+r.stderr)[-2600:]
    return bool(re.search(r"(\d+) passed", out)) and " failed" not in out and "error" not in out.lower(), out

def reset():
    subprocess.run(["git","checkout","--","src/humanize/time.py","src/humanize/number.py"], cwd=REPO, check=True)
    ACCEPT.write_text(ACCEPT_SRC)

def build_prompt(sources, fb=""):
    blocks="\n\n".join(f"--- `{n}` ({FUNCS[n][0].name}) ---\n```python\n{s}\n```" for n,s in sources.items())
    return f"You are fixing one real bug in `humanize`.\n\n{ISSUE}\n\n{blocks}\n\n{fb}For EVERY function that needs a change, return its complete corrected source in:\n\n===BEGIN naturaldelta===\n<complete function>\n===END naturaldelta===\n\nOmit unchanged functions. Nothing else outside blocks."

def arm(tag, loop):
    reset(); srcs=current_sources()
    res={"arm":tag,"model":MODEL,"cycles":0,"ok":False}; msgs=[{"role":"user","content":build_prompt(srcs)}]
    prev={}
    for cyc in range(1, MAX_CYCLES+1):
        raw=chat(msgs)
        patches=parse_patches(raw)
        if not patches:
            log({"event":"cycle","arm":tag,"cycle":cyc,"ok":False,"err":"parse","raw_len":len(raw)}); break
        if not splice(patches):
            res["cycles"]=cyc
            log({"event":"cycle","arm":tag,"cycle":cyc,"ok":False,"err":"compile","compile_err":LAST_ERR,"names":sorted(patches)}); 
            if not loop: break
            msgs+=[{"role":"assistant","content":raw},{"role":"user","content":"Patch did not compile. Re-send corrected blocks."}]; continue
        ok,out=run_tests(); res["cycles"]=cyc
        log({"event":"cycle","arm":tag,"cycle":cyc,"ok":ok,"patched":sorted(patches)})
        if ok: res["ok"]=True; break
        if not loop: break
        nudge=""
        if not {"naturaldelta","naturaltime","precisedelta"} & set(patches):
            nudge="You omitted the failing functions. You MUST include blocks for naturaldelta/naturaltime/precisedelta.\n\n"
        diff="\n\n".join("\n".join(difflib.unified_diff((prev.get(n) or srcs.get(n,"")).splitlines(), patches[n].splitlines(), fromfile=n+":before", tofile=n+":after", lineterm=""))[:1200] for n in patches)
        msgs+=[{"role":"assistant","content":raw},{"role":"user","content": nudge+"Your patch did NOT pass. Pytest:\n```\n"+out+"\n```\n\nDiff of your last change:\n```diff\n"+diff+"\n```\nFix every failure; left=actual, right=expected."}]
        prev.update(patches)
    log({"event":"arm_done","arm":tag,"cycles":res["cycles"],"ok":res["ok"]}); return res

def main():
    LEDGER.parent.mkdir(exist_ok=True)
    log({"event":"start","task":"humanize-bugB-lean","model":MODEL,"max_cycles":MAX_CYCLES})
    b=arm("B_raw_singleshot", loop=False)
    a=arm("A_closed_loop", loop=True)
    verdict="LOOP_WINS" if a["ok"] and not b["ok"] else ("tie:"+str(a["ok"]) if a["ok"]==b["ok"] else "BASELINE_WINS")
    s={"event":"summary","baseline_ok":b["ok"],"loop_ok":a["ok"],"verdict":verdict}
    log(s); print(s)

if __name__=="__main__": main()
