#!/usr/bin/env python3
"""kai_test_coverage.py — measure hidden-test strength via mutation testing.

The reverse_words blind spot (probe verified=True, hidden tests PASSED, but
real-world hidden tests failed) exposed that reference-anchoring fixes assert
*hallucination* but not assert *coverage*. This tool measures coverage
directly: for every bench task with a reference implementation, inject
hand-crafted bugs (mutants) representing the failure modes the bench must
catch — empty inputs, leading/trailing whitespace, off-by-one, duplicates,
negatives, unary minus, disconnected components, self-loops, O(n) blowups.
A hidden-test suite's strength = fraction of mutants it KILLS (fails).

  mutation score 1.00 — tests catch every bug class: strong
  mutation score < 1 — the suite lets some bug class through: the exact
                        blind spot the auto-verify probe would also miss.

Usage:
  python3 tools/kai_test_coverage.py              # all tasks, full report
  python3 tools/kai_test_coverage.py --tasks can_finish,reverse_words
  python3 tools/kai_test_coverage.py --json       # machine-readable
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import importlib.util
spec = importlib.util.spec_from_file_location("kcb", os.path.join(HERE, "kai_code_bench.py"))
kcb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kcb)
spec2 = importlib.util.spec_from_file_location("seed", os.path.join(HERE, "kai_seed_code_memory.py"))
seed = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(seed)


def run_grader(code: str, tests: str) -> bool:
    return kcb.run_tests(code, tests)[0]


# ── Mutant factory: name -> (base_reference, replacement) ─────────────────
# Each mutant replaces one line/expression of the reference with a buggy
# variant that a real model plausibly writes. The hidden tests must catch it.
MUTANTS = {
    # --- base tier ---
    "fizzbuzz": [
        ("Fizz before FizzBuzz (15 -> Fizz)",
         "if i % 15 == 0: out.append('FizzBuzz')\n        elif i % 3 == 0: out.append('Fizz')",
         "if i % 3 == 0: out.append('Fizz')\n        elif i % 15 == 0: out.append('FizzBuzz')"),
    ],
    "is_palindrome": [
        ("missing case-insensitivity",
         "f = [c.lower() for c in s if c.isalnum()]",
         "f = [c for c in s if c.isalnum()]"),
        ("missing punctuation strip",
         "f = [c.lower() for c in s if c.isalnum()]",
         "f = [c.lower() for c in s]"),
    ],
    "two_sum": [
        ("returns values not indices",
         "return [seen[target - v], i]",
         "return [target - v, v]"),
    ],
    "flatten": [
        ("only one level deep",
         "out.extend(flatten(x))",
         "out.extend(x)"),
    ],
    "merge_intervals": [
        ("forgets sorting",
         "intervals = sorted(intervals)",
         "pass"),
        ("overlap not transitive",
         "out[-1][1] = max(out[-1][1], b)",
         "out.append([a, b])"),
    ],
    "is_balanced": [
        ("forgets to pop",
         "elif c in ')]}':\n            if not st or st.pop() != pairs[c]: return False",
         "elif c in ')]}':\n            if not st: return False"),
    ],
    "fib": [
        ("off-by-one: returns fib(n+1)",
         "a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
         "a, b = 0, 1\n    for _ in range(n + 1):\n        a, b = b, a + b\n    return a"),
    ],
    "count_vowels": [
        ("forgets case-insensitive",
         "sum(1 for c in s.lower() if c in 'aeiou')",
         "sum(1 for c in s if c in 'aeiou')"),
    ],
    "matrix_transpose": [
        ("swapped dims (IndexError on rectangular)",
         "return [list(r) for r in zip(*m)]",
         "return [[m[j][i] for j in range(len(m[0]))] for i in range(len(m))]"),
    ],
    "gcd": [
        ("ignores zero",
         "while b:\n        a, b = b, a % b",
         "while b and a % b:\n        a, b = b, a % b"),
    ],
"reverse_words": [
        ("single-space split (the real blind spot)",
         "' '.join(s.split()[::-1])",
         "' '.join(s.split(' ')[::-1]).strip()"),
        ("no collapse of multi-space gaps",
         "return ' '.join(s.split()[::-1])",
         "return s.strip().split()[::-1]"),
    ],
    "max_subarray": [
        ("all-negative returns 0",
         "cur = max(x, cur + x)",
         "cur = max(0, cur + x)"),
    ],
    "longest_common_prefix": [
        ("empty list crash",
         "if not strs: return ''",
         "pass"),
    ],
    "find_missing": [
        ("assumes 1-based range (off-by-one)",
         "return n * (n + 1) // 2 - sum(nums)",
         "return (n + 1) * (n + 2) // 2 - sum(nums)"),
    ],
    "is_anagram": [
        ("case-sensitive (fails mixed case)",
         "sorted(c.lower() for c in s if c != ' ')",
         "sorted(c for c in s if c != ' ')"),
    ],
    "remove_duplicates": [
        ("removes ALL occurrences (count filter)",
         "if not out or x != out[-1]:\n            out.append(x)",
         "out = [x for x in nums if nums.count(x) == 1]"),
    ],
    "sqrt_int": [
        ("off-by-one (returns ceil)",
         "return hi",
         "return lo"),
    ],
    "is_power_of_two": [
        ("accepts 0",
         "return n > 0 and (n & (n - 1)) == 0",
         "return (n & (n - 1)) == 0"),
    ],
    "next_greater": [
        ("strictly greater vs greater-or-equal",
         "while st and nums[st[-1]] < v:",
         "while st and nums[st[-1]] <= v:"),
    ],
    "factorial": [
        ("0! == 0",
         "r = 1",
         "r = 0"),
    ],
    "count_words": [
        ("counts spaces not words",
         "return len(s.split())",
         "return s.count(' ') + 1 if s.strip() else 0"),
    ],
    "rotate_list": [
        ("negative k crash",
         "k %= len(nums)",
         "pass"),
    ],
    "majority_element": [
        ("returns middle element (fails ties)",
         "return cand",
         "return nums[len(nums) // 2]"),
    ],
"sum_digits": [
        ("sums only first digit",
         "sum(int(d) for d in str(n))",
         "sum(int(d) for d in str(n)[:1])"),
    ],
    "first_uniq_char": [
        ("returns count not index",
         "return i",
         "return cnt[c]"),
    ],
    "min_cost_stairs": [
        ("forgets start at 0 or 1",
         "a, b = cost[0], cost[1]",
         "a, b = 0, cost[0]"),
    ],
    "valid_parentheses": [
        ("ignores nesting (counts only)",
         "if not st: return False\n            st.pop()",
         "if not st: return False"),
    ],
    "merge_sorted": [
        ("appends rest of both without compare",
         "out.extend(a[i:]); out.extend(b[j:])",
         "out = a + b"),
    ],
    "climbing_stairs": [
        ("off-by-one",
         "return b",
         "return a"),
    ],
    # --- capacity tier ---
    "lru_cache": [
        ("no recency refresh on get",
         "self.d.move_to_end(key)\n        return self.d[key]",
         "return self.d[key]"),
        ("evicts wrong (no move_to_end on put-exists)",
         "if key in self.d: self.d.move_to_end(key)\n        self.d[key] = value",
         "self.d[key] = value"),
    ],
    "can_finish": [
        ("only catches direct mutual prereqs (misses longer cycles)",
         "state = [0] * n\n    def dfs(u):\n        state[u] = 1\n        for v in adj[u]:\n            if state[v] == 1: return False\n            if state[v] == 0 and not dfs(v): return False\n        state[u] = 2\n        return True\n    for u in range(n):\n        if state[u] == 0 and not dfs(u): return False\n    return True",
         "for a, b in prereqs:\n        if [b, a] in prereqs: return False\n    return True"),
    ],
    "lis": [
        ("non-strict (allows equal)",
         "if nums[j] < nums[i]:",
         "if nums[j] <= nums[i]:"),
    ],
    "word_ladder": [
        ("empty words crash",
         "if end not in ws: return 0",
         "pass"),
    ],
    "edit_distance": [
        ("forgets replace cost",
         "dp[i-1][j-1] + (a[i-1] != b[j-1])",
         "dp[i-1][j-1] + 1"),
    ],
    "spiral_order": [
        ("single-column crash",
         "if t <= b:",
         "if True:"),
    ],
    "num_decodings": [
        ("leading-zero crash",
         "if not s or s[0] == '0': return 0",
         "pass"),
        ("allows 0x pairs",
         "and int(s[i-2:i]) <= 26",
         "and int(s[i-2:i]) <= 26 and s[i-2] != '0'" if False else "pass"),
    ],
    "is_bipartite": [
        ("forgets disconnected components",
         "for s in range(n):\n        if color[s]: continue",
         "color[0] = 1\n        q = deque([0])"),
    ],
    "max_area": [
        ("brute force off-by-one",
         "best = max(best, min(h[l], h[r])*(r-l))",
         "best = max(best, min(h[l], h[r])*(r-l+1))"),
    ],
    "group_anagrams": [
        ("sorts letters instead of grouping (returns letters)",
         "d[tuple(sorted(w))].append(w)",
         "return [list(w) for w in words]"),
    ],
    "find_duplicate": [
        ("count-based (misses O(1) space, returns wrong for dup>1)",
         "return slow",
         "return nums[-1]"),
    ],
    "lps": [
        ("odd-length centers only (misses even palindromes)",
         "for lo, hi in ((c, c), (c, c+1)):",
         "for lo, hi in ((c, c),):"),
    ],
    "num_islands": [
        ("counts diagonal as same island",
         "for di, dj in ((1,0),(-1,0),(0,1),(0,-1)): dfs(i+di, j+dj)",
         "for di, dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)): dfs(i+di, j+dj)"),
    ],
    "word_break": [
        ("greedy single pass",
         "dp[i] = True",
         "dp[i] = True\n                break"),
    ],
    "calculator": [
        ("ignores operator precedence",
         "v = term()",
         "v = factor()"),
    ],
}


def apply_mutant(ref: str, old: str, new: str) -> str:
    """Replace the mutant target in the reference source. If the marker isn't
    found, the mutant is INVALID (skip)."""
    if old not in ref:
        return None
    return ref.replace(old, new, 1)


def grade_mutant(name: str, ref: str, mutant: str, tests: str):
    """Return 'killed' (tests catch it), 'survived' (blind spot), 'invalid',
    or 'timeout'."""
    try:
        passed = run_grader(mutant, tests)
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "timeout"
    return "survived" if passed else "killed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    all_tasks = kcb.TASKS + kcb.CAPACITY_TASKS
    by_name = {t["name"]: t for t in all_tasks}
    if args.tasks == "all":
        names = sorted(MUTANTS.keys())
    else:
        names = [n.strip() for n in args.tasks.split(",") if n.strip() in MUTANTS]

    report = {}
    for name in names:
        task = by_name.get(name)
        if task is None:
            continue
        ref = seed.reference_for(name)
        if ref is None:
            continue
        killed = survived = invalid = 0
        details = []
        for mlabel, old, new in MUTANTS[name]:
            mutant = apply_mutant(ref, old, new)
            if mutant is None:
                invalid += 1
                details.append({"mutant": mlabel, "verdict": "invalid"})
                continue
            verdict = grade_mutant(name, ref, mutant, task["tests"])
            if verdict == "killed": killed += 1
            elif verdict == "survived": survived += 1
            else: invalid += 1
            details.append({"mutant": mlabel, "verdict": verdict})
        n = killed + survived
        score = round(killed / n, 3) if n else None
        report[name] = {
            "tier": "capacity" if name in {t["name"] for t in kcb.CAPACITY_TASKS} else "base",
            "mutants": killed + survived,
            "killed": killed, "survived": survived, "invalid": invalid,
            "mutation_score": score,
            "survivors": [d["mutant"] for d in details if d["verdict"] == "survived"],
            "details": details,
        }

    if args.json:
        print(json.dumps(report, indent=1))
        return

    print(f"{'task':22s} {'tier':9s} {'score':6s} {'killed':7s} {'survived':9s} survivors")
    for name, r in sorted(report.items()):
        surv = ", ".join(r["survivors"]) or "-"
        print(f"{name:22s} {r['tier']:9s} {str(r['mutation_score']):6s} "
              f"{r['killed']:3d}/{r['mutants']:2d}     {r['survived']:2d}      {surv[:70]}")
    scores = [r["mutation_score"] for r in report.values() if r["mutation_score"] is not None]
    if scores:
        print(f"\nmean mutation score: {sum(scores)/len(scores):.3f} "
              f"(1.000 = all bug classes caught)")
    weak = [n for n, r in report.items() if r["mutation_score"] is not None and r["mutation_score"] < 1.0]
    if weak:
        print(f"\nWEAK SUITES (blind spots): {', '.join(weak)}")
        print("  -> these are exactly the bug classes the auto-verify probe "
              "would also miss")


if __name__ == "__main__":
    main()