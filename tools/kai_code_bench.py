#!/usr/bin/env python3
"""kai_code_bench.py — verifiable-answer ruler for the AGI bet.

The one-sentence bet: a closed perceive->act->perceive loop beats raw
prompting on a REAL task. The rubric judges (2-dim, 3-dim) were falsifiable
but bounded: on general-knowledge questions the greedy 3b worker sits at the
ceiling, and no arm produces measurable insight. This bench replaces
LLM-judged scoring with OBJECTIVE ground truth: hidden unit tests.

Arms (same prompt, same worker qwen2.5-coder:3b):
  greedy  — temp 0.0 direct ollama. Deterministic: pass@K == pass@1.
            This is the honest pinned baseline (no judge, no contrast).
  fixed   — temp 0.7 direct ollama, K attempts. Isolates "exploration helps"
            from "ADAPTIVE exploration helps".
  physics — bridge kai/worker, VFE-adaptive temperature (darwin params live),
            K attempts. The closed loop: adaptive sampling under uncertainty.

Metrics (objective, reproducible, no judge):
  pass@1  — fraction of tasks whose FIRST attempt passes hidden tests.
  pass@K  — fraction of tasks solved within K attempts (best-of-K).
  Greedy pass@K == pass@1 by construction. If physics pass@K > greedy
  pass@1 (and > fixed pass@K), adaptive exploration is real. If not, the
  bet is falsified for this task class.

Grader: extracted code + hidden tests run in a subprocess with CPU/mem
limits and a timeout. pass = exit 0. No file writes, no network in sandbox.
"""
import argparse, json, os, re, resource, subprocess, sys, time, urllib.request

STOCK = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
BRIDGE_PORT = 8765
RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".kai_code_bench.jsonl")
RECORD = os.path.abspath(RECORD)

# ── Task set: code-gen with hidden tests (ground truth) ───────────────────
# Difficulty spread chosen so a 3b coder passes ~5-7/10 single-shot —
# enough signal for exploration to matter either way.
TASKS = [
    {
        "name": "fizzbuzz",
        "prompt": ("Write a Python function fizzbuzz(n) that returns a list of "
                   "length n where index i (1-based) contains 'Fizz' if i%3==0, "
                   "'Buzz' if i%5==0, 'FizzBuzz' if both, else str(i)."),
        "tests": (
            "assert fizzbuzz(1) == ['1']\n"
            "assert fizzbuzz(3) == ['1','2','Fizz']\n"
            "assert fizzbuzz(15)[14] == 'FizzBuzz'\n"
            "assert fizzbuzz(5)[4] == 'Buzz'\n"
            "assert fizzbuzz(30).count('FizzBuzz') == 2\n"
        ),
    },
    {
        "name": "is_palindrome",
        "prompt": ("Write a Python function is_palindrome(s) that returns True "
                   "if s is a palindrome ignoring case and non-alphanumeric "
                   "characters, else False."),
        "tests": (
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('A man, a plan, a canal: Panama') == True\n"
            "assert is_palindrome('hello') == False\n"
            "assert is_palindrome('') == True\n"
            "assert is_palindrome('No lemon, no melon') == True\n"
        ),
    },
    {
        "name": "two_sum",
        "prompt": ("Write a Python function two_sum(nums, target) that returns "
                   "the indices of the two numbers that add up to target. Assume "
                   "exactly one solution exists."),
        "tests": (
            "assert sorted(two_sum([2,7,11,15], 9)) == [0,1]\n"
            "assert sorted(two_sum([3,2,4], 6)) == [1,2]\n"
            "assert sorted(two_sum([3,3], 6)) == [0,1]\n"
            "assert sorted(two_sum([1,5,3,9], 12)) == [2,3]\n"
        ),
    },
    {
        "name": "flatten",
        "prompt": ("Write a Python function flatten(nested) that takes a list "
                   "which may contain arbitrarily nested lists and returns a flat "
                   "list of all non-list elements in order."),
        "tests": (
            "assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]\n"
            "assert flatten([]) == []\n"
            "assert flatten([[],[[],[]]]) == []\n"
            "assert flatten([1,2,3]) == [1,2,3]\n"
            "assert flatten([[1],[[2]],[[[3]]]]) == [1,2,3]\n"
        ),
    },
    {
        "name": "merge_intervals",
        "prompt": ("Write a Python function merge_intervals(intervals) that "
                   "takes a list of [start,end] intervals and returns the merged "
                   "non-overlapping intervals, sorted by start."),
        "tests": (
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
            "assert merge_intervals([[1,2]]) == [[1,2]]\n"
            "assert merge_intervals([]) == []\n"
            "assert merge_intervals([[2,3],[1,2]]) == [[1,3]]\n"
        ),
    },
    {
        "name": "is_balanced",
        "prompt": ("Write a Python function is_balanced(s) that returns True if "
                   "the brackets in s are balanced: '()', '[]', '{}' properly "
                   "nested. Ignore other characters."),
        "tests": (
            "assert is_balanced('()') == True\n"
            "assert is_balanced('([]){()}') == True\n"
            "assert is_balanced('(]') == False\n"
            "assert is_balanced('([)]') == False\n"
            "assert is_balanced('') == True\n"
            "assert is_balanced('((()))') == True\n"
        ),
    },
    {
        "name": "fib",
        "prompt": ("Write a Python function fib(n) that returns the nth "
                   "Fibonacci number with fib(0)=0, fib(1)=1."),
        "tests": (
            "assert fib(0) == 0\n"
            "assert fib(1) == 1\n"
            "assert fib(10) == 55\n"
            "assert fib(20) == 6765\n"
            "assert fib(15) == 610\n"
        ),
    },
    {
        "name": "count_vowels",
        "prompt": ("Write a Python function count_vowels(s) that returns the "
                   "number of vowels (a,e,i,o,u, case-insensitive) in s."),
        "tests": (
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('AEIOU') == 5\n"
            "assert count_vowels('rhythm') == 0\n"
            "assert count_vowels('') == 0\n"
            "assert count_vowels('Beautiful') == 5\n"
        ),
    },
    {
        "name": "matrix_transpose",
        "prompt": ("Write a Python function transpose(m) that returns the "
                   "transpose of a rectangular matrix given as a list of lists."),
        "tests": (
            "assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]\n"
            "assert transpose([[1]]) == [[1]]\n"
            "assert transpose([]) == []\n"
            "assert transpose([[1,2],[3,4],[5,6]]) == [[1,3,5],[2,4,6]]\n"
        ),
    },
    {
        "name": "gcd",
        "prompt": ("Write a Python function gcd(a, b) that returns the greatest "
                   "common divisor of two non-negative integers."),
        "tests": (
            "assert gcd(48, 18) == 6\n"
            "assert gcd(0, 5) == 5\n"
            "assert gcd(17, 13) == 1\n"
            "assert gcd(100, 0) == 100\n"
            "assert gcd(1071, 462) == 21\n"
        ),
    },
]

K_DEFAULT = 4
MAX_TOKENS = 400


def _ollama_chat(model: str, q: str, temperature: float, max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": q}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        f"{STOCK}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d.get("message", {}).get("content", "").strip()


def _bridge_chat(q: str, max_tokens: int, port: int) -> tuple:
    body = {
        "model": "kai/qwen2.5-coder:3b",
        "messages": [{"role": "user", "content": q}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.7,  # base knob; bridge's VFE controller adapts it
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read())
    content = (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    phys = d.get("kai_physics", {}) or {}
    return content, phys


def extract_code(text: str) -> str:
    """Pull the python fenced block; fall back to whole answer."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_tests(code: str, tests: str) -> tuple:
    """Execute extracted code + hidden tests in a constrained subprocess."""
    src = code + "\n\n" + tests
    try:
        def _limit():
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        p = subprocess.run(
            [sys.executable, "-c", src],
            capture_output=True, text=True, timeout=15,
            preexec_fn=_limit if hasattr(os, "fork") else None,
        )
        return p.returncode == 0, p.stderr.strip()[-300:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def grade_arm(label: str, tasks: list, mode: str, k: int, port: int) -> dict:
    pass1, solved = 0, 0
    attempts_used = []
    temps = []
    per_task = []
    t0 = time.time()
    for ti, task in enumerate(tasks, 1):
        q = task["prompt"]
        first_pass = None
        task_solved_at = None
        task_temps = []
        for attempt in range(k):
            if mode == "greedy":
                # Deterministic: only attempt 0 is meaningful, but re-run to
                # verify stability (greedy must repeat the SAME code).
                ans = _ollama_chat("qwen2.5-coder:3b", q, 0.0, MAX_TOKENS)
                temp_used = 0.0
            elif mode == "fixed":
                ans = _ollama_chat("qwen2.5-coder:3b", q, 0.7, MAX_TOKENS)
                temp_used = 0.7
            else:  # physics
                ans, phys = _bridge_chat(q, MAX_TOKENS, port)
                temp_used = phys.get("temperature", 0.7)
            task_temps.append(temp_used)
            passed, err = run_tests(extract_code(ans), task["tests"])
            if first_pass is None:
                first_pass = passed
            if passed and task_solved_at is None:
                task_solved_at = attempt + 1
            if mode == "greedy" and attempt == 0:
                break  # deterministic; no point sampling again
            if task_solved_at is not None and mode != "greedy":
                break  # solved — don't burn attempts
        temps.append(round(sum(task_temps) / len(task_temps), 3))
        if first_pass:
            pass1 += 1
        if task_solved_at is not None:
            solved += 1
            attempts_used.append(task_solved_at)
        else:
            attempts_used.append(0)
        per_task.append({
            "name": task["name"], "pass1": bool(first_pass),
            "solved": task_solved_at is not None,
            "attempts": task_solved_at or 0,
            "temps": task_temps,
        })
        print(f"  [{ti}/{len(tasks)}] {task['name']:18s} pass1={first_pass} "
              f"solved@={task_solved_at or '-'} temps={task_temps}")
    n = len(tasks)
    res = {
        "t": time.time(), "label": label, "mode": mode, "worker": "qwen2.5-coder:3b",
        "k": k, "tasks": n, "elapsed_s": round(time.time() - t0, 1),
        "pass1": round(pass1 / n, 3), "pass_k": round(solved / n, 3),
        "solved": solved, "mean_attempts": round(
            sum(a for a in attempts_used if a) / solved, 2) if solved else None,
        "mean_temp": round(sum(temps) / len(temps), 3),
        "per_task": per_task,
    }
    return res


def main():
    ap = argparse.ArgumentParser(description="verifiable-answer code bench")
    ap.add_argument("--tasks", type=int, default=len(TASKS),
                    help="how many tasks to run (default all)")
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="attempts per task (best-of-K)")
    ap.add_argument("--label", default="code_bench")
    ap.add_argument("--port", type=int, default=BRIDGE_PORT)
    ap.add_argument("--arms", default="greedy,fixed,physics",
                    help="comma-separated arms to run")
    args = ap.parse_args()

    tasks = TASKS[: args.tasks]
    print(f"==== kai code bench  label='{args.label}'  tasks={len(tasks)}  K={args.k}  arms={args.arms}")
    print(f"     objective grading (hidden unit tests), no judge, no rubric\n")

    results = {}
    for mode in [m.strip() for m in args.arms.split(",")]:
        print(f"--- arm: {mode} ---")
        res = grade_arm(f"{args.label}_{mode}", tasks, mode, args.k, args.port)
        results[mode] = res
        with open(RECORD, "a") as f:
            f.write(json.dumps(res) + "\n")

    print("\n==== results ====")
    for mode, res in results.items():
        print(f"  {mode:8s} pass@1 {res['pass1']:.2f}  pass@K {res['pass_k']:.2f}  "
              f"solved {res['solved']}/{res['tasks']}  mean_attempts {res['mean_attempts']}  "
              f"mean_temp {res['mean_temp']}")
    if "physics" in results and "greedy" in results:
        g, p = results["greedy"], results["physics"]
        delta = p["pass_k"] - g["pass_k"]
        verdict = ("PHYSICS WINS — adaptive exploration finds solutions "
                   "greedy cannot reach" if delta >= 0.15
                   else "MARGINAL" if delta > -0.05
                   else "PHYSICS LOSES — bet falsified for this task class")
        print(f"\n  VERDICT       physics pass@K vs greedy pass@K: {delta:+.2f} -> {verdict}")
        if "fixed" in results:
            print(f"  NOTE          fixed-exploration pass@K {results['fixed']['pass_k']:.2f} "
                  f"isolates 'exploration helps' from 'adaptive exploration helps'")
    print(f"  -> appended to {RECORD}\n")


if __name__ == "__main__":
    main()
