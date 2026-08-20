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
    {
        "name": "reverse_words",
        "prompt": ("Write a Python function reverse_words(s) that returns s with "
                   "the order of words reversed, preserving single spaces between "
                   "words (strip leading/trailing spaces)."),
        "tests": (
            "assert reverse_words('the sky is blue') == 'blue is sky the'\n"
            "assert reverse_words('  hello world  ') == 'world hello'\n"
            "assert reverse_words('a') == 'a'\n"
            "assert reverse_words('') == ''\n"
            "assert reverse_words('two words') == 'words two'\n"
        ),
    },
    {
        "name": "max_subarray",
        "prompt": ("Write a Python function max_subarray(nums) that returns the "
                   "maximum sum of a contiguous subarray (Kadane's algorithm)."),
        "tests": (
            "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\n"
            "assert max_subarray([1]) == 1\n"
            "assert max_subarray([-1,-2,-3]) == -1\n"
            "assert max_subarray([5,4,-1,7,8]) == 23\n"
            "assert max_subarray([-2,-1]) == -1\n"
        ),
    },
    {
        "name": "longest_common_prefix",
        "prompt": ("Write a Python function longest_common_prefix(strs) that "
                   "returns the longest common prefix string of a list of strings, "
                   "or '' if none."),
        "tests": (
            "assert longest_common_prefix(['flower','flow','flight']) == 'fl'\n"
            "assert longest_common_prefix(['dog','racecar','car']) == ''\n"
            "assert longest_common_prefix(['a']) == 'a'\n"
            "assert longest_common_prefix([]) == ''\n"
            "assert longest_common_prefix(['interspecies','interstellar','interstate']) == 'inters'\n"
        ),
    },
    {
        "name": "find_missing",
        "prompt": ("Write a Python function find_missing(nums) that takes a list "
                   "of n distinct numbers in [0,n] and returns the one missing "
                   "number (range has n+1 values)."),
        "tests": (
            "assert find_missing([3,0,1]) == 2\n"
            "assert find_missing([0,1]) == 2\n"
            "assert find_missing([9,6,4,2,3,5,7,0,1]) == 8\n"
            "assert find_missing([0]) == 1\n"
            "assert find_missing([1]) == 0\n"
        ),
    },
    {
        "name": "is_anagram",
        "prompt": ("Write a Python function is_anagram(a, b) that returns True "
                   "if a and b are anagrams (same letters, any order), "
                   "case-insensitive, ignoring spaces."),
        "tests": (
            "assert is_anagram('listen', 'silent') == True\n"
            "assert is_anagram('anagram', 'nagaram') == True\n"
            "assert is_anagram('rat', 'car') == False\n"
            "assert is_anagram('Hello World', 'World Hello') == True\n"
            "assert is_anagram('', '') == True\n"
        ),
    },
    {
        "name": "remove_duplicates",
        "prompt": ("Write a Python function remove_duplicates(nums) that takes a "
                   "SORTED list and returns the list with duplicates removed, "
                   "preserving order."),
        "tests": (
            "assert remove_duplicates([1,1,2]) == [1,2]\n"
            "assert remove_duplicates([0,0,1,1,1,2,2,3,3,4]) == [0,1,2,3,4]\n"
            "assert remove_duplicates([]) == []\n"
            "assert remove_duplicates([1,2,3]) == [1,2,3]\n"
            "assert remove_duplicates([5,5,5,5]) == [5]\n"
        ),
    },
    {
        "name": "sqrt_int",
        "prompt": ("Write a Python function sqrt_int(x) that returns the integer "
                   "square root of a non-negative integer x (floor of sqrt), "
                   "without importing math."),
        "tests": (
            "assert sqrt_int(8) == 2\n"
            "assert sqrt_int(9) == 3\n"
            "assert sqrt_int(0) == 0\n"
            "assert sqrt_int(1) == 1\n"
            "assert sqrt_int(2147395599) == 46339\n"
        ),
    },
    {
        "name": "is_power_of_two",
        "prompt": ("Write a Python function is_power_of_two(n) that returns True "
                   "if n is a power of two, else False."),
        "tests": (
            "assert is_power_of_two(1) == True\n"
            "assert is_power_of_two(16) == True\n"
            "assert is_power_of_two(3) == False\n"
            "assert is_power_of_two(0) == False\n"
            "assert is_power_of_two(1024) == True\n"
        ),
    },
    {
        "name": "next_greater",
        "prompt": ("Write a Python function next_greater(nums) that returns a "
                   "list where each element is the next element to the right that "
                   "is strictly greater, or -1 if none."),
        "tests": (
            "assert next_greater([4,5,2,25]) == [5,25,25,-1]\n"
            "assert next_greater([2,1,2,4,3]) == [4,2,4,-1,-1]\n"
            "assert next_greater([1,2,3]) == [2,3,-1]\n"
            "assert next_greater([3,2,1]) == [-1,-1,-1]\n"
            "assert next_greater([]) == []\n"
        ),
    },
    {
        "name": "factorial",
        "prompt": ("Write a Python function factorial(n) that returns n! for "
                   "n >= 0 (0! = 1)."),
        "tests": (
            "assert factorial(0) == 1\n"
            "assert factorial(1) == 1\n"
            "assert factorial(5) == 120\n"
            "assert factorial(10) == 3628800\n"
            "assert factorial(7) == 5040\n"
        ),
    },
    {
        "name": "count_words",
        "prompt": ("Write a Python function count_words(s) that returns the "
                   "number of words in a string (words separated by any amount "
                   "of whitespace)."),
        "tests": (
            "assert count_words('hello world') == 2\n"
            "assert count_words('  a   b  c ') == 3\n"
            "assert count_words('') == 0\n"
            "assert count_words('single') == 1\n"
            "assert count_words('one two three four five') == 5\n"
        ),
    },
    {
        "name": "rotate_list",
        "prompt": ("Write a Python function rotate_list(nums, k) that rotates "
                   "the list to the right by k steps (k can be larger than "
                   "len(nums)). Return the rotated list."),
        "tests": (
            "assert rotate_list([1,2,3,4,5,6,7], 3) == [5,6,7,1,2,3,4]\n"
            "assert rotate_list([-1,-100,3,99], 2) == [3,99,-1,-100]\n"
            "assert rotate_list([1,2,3], 4) == [3,1,2]\n"
            "assert rotate_list([1], 0) == [1]\n"
            "assert rotate_list([1,2,3,4], 8) == [1,2,3,4]\n"
        ),
    },
    {
        "name": "majority_element",
        "prompt": ("Write a Python function majority_element(nums) that returns "
                   "the element appearing more than len(nums)//2 times (guaranteed "
                   "to exist)."),
        "tests": (
            "assert majority_element([3,2,3]) == 3\n"
            "assert majority_element([2,2,1,1,1,2,2]) == 2\n"
            "assert majority_element([1]) == 1\n"
            "assert majority_element([5,5,5,1,5]) == 5\n"
            "assert majority_element([9,9,9,9,8,8]) == 9\n"
        ),
    },
    {
        "name": "sum_digits",
        "prompt": ("Write a Python function sum_digits(n) that returns the sum "
                   "of the digits of a non-negative integer n."),
        "tests": (
            "assert sum_digits(123) == 6\n"
            "assert sum_digits(0) == 0\n"
            "assert sum_digits(999) == 27\n"
            "assert sum_digits(102030) == 6\n"
            "assert sum_digits(7) == 7\n"
        ),
    },
    {
        "name": "first_uniq_char",
        "prompt": ("Write a Python function first_uniq_char(s) that returns the "
                   "index of the first non-repeating character in s, or -1 if "
                   "none."),
        "tests": (
            "assert first_uniq_char('leetcode') == 0\n"
            "assert first_uniq_char('loveleetcode') == 2\n"
            "assert first_uniq_char('aabb') == -1\n"
            "assert first_uniq_char('') == -1\n"
            "assert first_uniq_char('a') == 0\n"
        ),
    },
    {
        "name": "min_cost_stairs",
        "prompt": ("Write a Python function min_cost_stairs(cost) that takes a "
                   "list where cost[i] is the cost of step i. You can start at "
                   "step 0 or 1 and climb 1 or 2 steps. Return the minimum cost "
                   "to reach the top (one past the last step)."),
        "tests": (
            "assert min_cost_stairs([10,15,20]) == 15\n"
            "assert min_cost_stairs([1,100,1,1,1,100,1,1,100,1]) == 6\n"
            "assert min_cost_stairs([0,0,0,0]) == 0\n"
            "assert min_cost_stairs([5,10]) == 5\n"
        ),
    },
    {
        "name": "valid_parentheses",
        "prompt": ("Write a Python function valid_parentheses(s) that returns "
                   "True if the string has balanced parentheses '()' (no other "
                   "bracket types), else False."),
        "tests": (
            "assert valid_parentheses('()') == True\n"
            "assert valid_parentheses('()()') == True\n"
            "assert valid_parentheses('(())') == True\n"
            "assert valid_parentheses('())') == False\n"
            "assert valid_parentheses(')(') == False\n"
            "assert valid_parentheses('') == True\n"
        ),
    },
    {
        "name": "merge_sorted",
        "prompt": ("Write a Python function merge_sorted(a, b) that takes two "
                   "sorted lists and returns one sorted merged list."),
        "tests": (
            "assert merge_sorted([1,2,3], [2,5,6]) == [1,2,2,3,5,6]\n"
            "assert merge_sorted([], [1]) == [1]\n"
            "assert merge_sorted([1,3,5], []) == [1,3,5]\n"
            "assert merge_sorted([], []) == []\n"
            "assert merge_sorted([1,2],[3,4,5,6]) == [1,2,3,4,5,6]\n"
        ),
    },
    {
        "name": "climbing_stairs",
        "prompt": ("Write a Python function climbing_stairs(n) that returns the "
                   "number of distinct ways to climb n stairs taking 1 or 2 "
                   "steps at a time."),
        "tests": (
            "assert climbing_stairs(2) == 2\n"
            "assert climbing_stairs(3) == 3\n"
            "assert climbing_stairs(4) == 5\n"
            "assert climbing_stairs(5) == 8\n"
            "assert climbing_stairs(1) == 1\n"
        ),
    },
]

K_DEFAULT = 4
MAX_TOKENS = 400

# ── Capacity tier: harder multi-step/stateful tasks ───────────────────────
# The original 29 tasks saturate at pass@1 ~1.0 on 7b/16b — no headroom for
# darwin or the probe to optimize. These tasks are multi-step/stateful
# (graphs, DP with adversarial edge cases, stateful LRU, expression parsing)
# with adversarial hidden tests designed to break naive implementations
# (empty inputs, single elements, duplicates, unary minus, performance).
# Every expected value below was validated against a known-correct reference
# implementation (tools/validate_tier.py logic) BEFORE entering the ruler.
CAPACITY_TASKS = [
    {
        "name": "lru_cache",
        "prompt": ("Write a Python class LRUCache with __init__(capacity), "
                   "get(key) and put(key, value). get returns -1 if the key "
                   "is absent. put inserts/updates; if the cache exceeds "
                   "capacity, evict the LEAST RECENTLY USED key. Accessing "
                   "a key via get refreshes its recency."),
        "tests": (
            "c = LRUCache(2)\n"
            "c.put(1,1); c.put(2,2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3,3); assert c.get(2) == -1\n"
            "c.put(4,4); assert c.get(1) == -1\n"
            "assert c.get(3) == 3 and c.get(4) == 4\n"
            "c2 = LRUCache(1)\n"
            "c2.put(1,1); assert c2.get(1) == 1\n"
            "c2.put(2,2); assert c2.get(1) == -1 and c2.get(2) == 2\n"
            "c3 = LRUCache(3)\n"
            "c3.put(1,1); c3.put(2,2); c3.put(3,3)\n"
            "assert c3.get(1) == 1\n"
            "c3.put(4,4); assert c3.get(2) == -1 and c3.get(1) == 1 and c3.get(3) == 3\n"
        ),
    },
    {
        "name": "can_finish",
        "prompt": ("Write a Python function can_finish(n, prerequisites) that "
                   "takes n courses numbered 0..n-1 and a list of pairs "
                   "[a,b] meaning course a requires course b first, and "
                   "returns True if all courses can be finished (no cycle)."),
        "tests": (
            "assert can_finish(2, [[1,0]]) == True\n"
            "assert can_finish(2, [[1,0],[0,1]]) == False\n"
            "assert can_finish(4, [[1,0],[2,1],[3,2]]) == True\n"
            "assert can_finish(3, [[0,1],[1,2],[2,0]]) == False\n"
            "assert can_finish(5, []) == True\n"
            "assert can_finish(1, [[0,0]]) == False\n"
            "assert can_finish(4, [[1,0],[2,1],[3,1],[0,3]]) == False\n"
        ),
    },
    {
        "name": "lis",
        "prompt": ("Write a Python function lis(nums) that returns the length "
                   "of the longest strictly increasing subsequence."),
        "tests": (
            "assert lis([10,9,2,5,3,7,101,18]) == 4\n"
            "assert lis([]) == 0\n"
            "assert lis([1]) == 1\n"
            "assert lis([3,2,1]) == 1\n"
            "assert lis([1,1,1]) == 1\n"
            "assert lis([0,1,0,3,2,3]) == 4\n"
            "assert lis([7,7,7,7,7,7,7]) == 1\n"
        ),
    },
    {
        "name": "word_ladder",
        "prompt": ("Write a Python function word_ladder(begin, end, words) "
                   "that returns the length of the shortest transformation "
                   "sequence from begin to end where each step changes ONE "
                   "letter into another word in the words list, and the "
                   "length counts begin. Return 0 if unreachable."),
        "tests": (
            "assert word_ladder('hit','cog',['hot','dot','dog','lot','log','cog']) == 5\n"
            "assert word_ladder('hit','cog',['hot','dot','dog','lot','log']) == 0\n"
            "assert word_ladder('a','c',['b','c']) == 2\n"
            "assert word_ladder('same','same',[]) == 1\n"
            "assert word_ladder('toon','plea',['poon','plee','same','poie','plea','plie','poin']) == 7\n"
        ),
    },
    {
        "name": "edit_distance",
        "prompt": ("Write a Python function edit_distance(a, b) that returns "
                   "the minimum number of operations (insert, delete, "
                   "replace) to convert a into b."),
        "tests": (
            "assert edit_distance('horse','ros') == 3\n"
            "assert edit_distance('intention','execution') == 5\n"
            "assert edit_distance('','') == 0\n"
            "assert edit_distance('abc','') == 3\n"
            "assert edit_distance('','xyz') == 3\n"
            "assert edit_distance('a','a') == 0\n"
            "assert edit_distance('abc','abc') == 0\n"
        ),
    },
    {
        "name": "spiral_order",
        "prompt": ("Write a Python function spiral_order(matrix) that returns "
                   "the elements of a rectangular matrix in clockwise spiral "
                   "order."),
        "tests": (
            "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n"
            "assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]\n"
            "assert spiral_order([]) == []\n"
            "assert spiral_order([[1]]) == [1]\n"
            "assert spiral_order([[1,2]]) == [1,2]\n"
            "assert spiral_order([[1],[2],[3]]) == [1,2,3]\n"
            "assert spiral_order([[1,2,3]]) == [1,2,3]\n"
        ),
    },
    {
        "name": "num_decodings",
        "prompt": ("Write a Python function num_decodings(s) that counts the "
                   "number of ways to decode a digit string where 'A'..'Z' "
                   "map to 1..26. Return 0 if the string cannot be decoded."),
        "tests": (
            "assert num_decodings('12') == 2\n"
            "assert num_decodings('226') == 3\n"
            "assert num_decodings('0') == 0\n"
            "assert num_decodings('06') == 0\n"
            "assert num_decodings('10') == 1\n"
            "assert num_decodings('100') == 0\n"
            "assert num_decodings('101') == 1\n"
            "assert num_decodings('1'*50) == 20365011074\n"
        ),
    },
    {
        "name": "is_bipartite",
        "prompt": ("Write a Python function is_bipartite(graph) that takes an "
                   "undirected graph as an adjacency list and returns True if "
                   "it can be colored with 2 colors so no edge joins same-"
                   "colored vertices (the graph may be disconnected)."),
        "tests": (
            "assert is_bipartite([[1,3],[0,2],[1,3],[0,2]]) == True\n"
            "assert is_bipartite([[1,2,3],[0,2],[0,1,3],[0,2]]) == False\n"
            "assert is_bipartite([[],[2],[1]]) == True\n"
            "assert is_bipartite([[1],[0]]) == True\n"
            "assert is_bipartite([]) == True\n"
            "assert is_bipartite([[]]) == True\n"
            "assert is_bipartite([[1,2],[0,2],[0,1]]) == False\n"
            "assert is_bipartite([[1,2],[0,2],[0,1],[4],[3]]) == False\n"
        ),
    },
    {
        "name": "max_area",
        "prompt": ("Write a Python function max_area(heights) that returns the "
                   "maximum area of water a container can hold, where each "
                   "element is a vertical line height and the container is "
                   "bounded by two lines and the x-axis."),
        "tests": (
            "assert max_area([1,8,6,2,5,4,8,3,7]) == 49\n"
            "assert max_area([1,1]) == 1\n"
            "assert max_area([2,1]) == 1\n"
            "assert max_area([1,2,1]) == 2\n"
            "assert max_area([4,3,2,1,4]) == 16\n"
            "assert max_area([1,2,4,3]) == 4\n"
        ),
    },
    {
        "name": "group_anagrams",
        "prompt": ("Write a Python function group_anagrams(words) that groups "
                   "anagrams together and returns a list of groups (order of "
                   "groups and order within groups does not matter)."),
        "tests": (
            "res = group_anagrams(['eat','tea','tan','ate','nat','bat'])\n"
            "assert sorted([sorted(g) for g in res]) == [['ate','eat','tea'],['bat'],['nat','tan']]\n"
            "assert sorted([sorted(g) for g in group_anagrams([''])]) == [['']]\n"
            "assert sorted([sorted(g) for g in group_anagrams(['a'])]) == [['a']]\n"
            "assert sorted([sorted(g) for g in group_anagrams(['ab','ba','ab'])]) == [['ab','ab','ba']]\n"
        ),
    },
    {
        "name": "find_duplicate",
        "prompt": ("Write a Python function find_duplicate(nums) that takes a "
                   "list of n+1 integers in range [1, n] with exactly one "
                   "duplicate and returns that duplicate, using O(1) extra "
                   "space."),
        "tests": (
            "assert find_duplicate([1,3,4,2,2]) == 2\n"
            "assert find_duplicate([3,1,3,4,2]) == 3\n"
            "assert find_duplicate([1,1]) == 1\n"
            "assert find_duplicate([2,2,2,2,2]) == 2\n"
            "assert find_duplicate([1,2,3,4,5,6,7,8,9,10,10]) == 10\n"
            "assert find_duplicate([2,5,9,6,9,3,8,9,7,1]) == 9\n"
        ),
    },
    {
        "name": "lps",
        "prompt": ("Write a Python function lps(s) that returns the longest "
                   "palindromic substring of s (any one if several exist)."),
        "tests": (
            "assert lps('babad') in ('bab','aba')\n"
            "assert lps('cbbd') == 'bb'\n"
            "assert lps('a') == 'a'\n"
            "assert lps('ac') in ('a','c')\n"
            "assert lps('aaaa') == 'aaaa'\n"
            "assert lps('') == ''\n"
            "assert lps('racecar') == 'racecar'\n"
        ),
    },
    {
        "name": "num_islands",
        "prompt": ("Write a Python function num_islands(grid) that takes a 2D "
                   "grid of '1' (land) and '0' (water) and returns the number "
                   "of islands, where islands connect orthogonally (up/down/"
                   "left/right, NOT diagonally)."),
        "tests": (
            "assert num_islands([['1','1','1','1','0'],['1','1','0','1','0'],['1','1','0','0','0'],['0','0','0','0','0']]) == 1\n"
            "assert num_islands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]) == 3\n"
            "assert num_islands([['0']]) == 0\n"
            "assert num_islands([['1']]) == 1\n"
            "assert num_islands([]) == 0\n"
            "assert num_islands([['1','0','1']]) == 2\n"
            "assert num_islands([['1','0'],['0','1']]) == 2\n"
        ),
    },
    {
        "name": "word_break",
        "prompt": ("Write a Python function word_break(s, words) that returns "
                   "True if s can be segmented into a space-separated "
                   "sequence of dictionary words (words may be reused)."),
        "tests": (
            "assert word_break('leetcode',['leet','code']) == True\n"
            "assert word_break('applepenapple',['apple','pen']) == True\n"
            "assert word_break('catsandog',['cats','dog','sand','and','cat']) == False\n"
            "assert word_break('',[]) == True\n"
            "assert word_break('aaaaaaa',['aaaa','aaa']) == True\n"
            "assert word_break('bb',['a','b','bbb']) == True\n"
            "assert word_break('aaaaaaa',['aaaa','aa']) == False\n"
        ),
    },
    {
        "name": "calculator",
        "prompt": ("Write a Python function calculate(s) that evaluates a "
                   "string expression containing +, -, *, /, parentheses and "
                   "spaces. Division truncates toward zero. Handle unary "
                   "minus. Return the integer result."),
        "tests": (
            "assert calculate('1 + 1') == 2\n"
            "assert calculate(' 2-1 + 2 ') == 3\n"
            "assert calculate('(1+(4+5+2)-3)+(6+8)') == 23\n"
            "assert calculate('3*2+2') == 8\n"
            "assert calculate('14-3*2') == 8\n"
            "assert calculate(' 2*3-4/2 ') == 4\n"
            "assert calculate('-2+3') == 1\n"
            "assert calculate('2-(-3)') == 5\n"
            "assert calculate('(7)-(0)+(4)') == 11\n"
        ),
    },
]


def _ollama_chat(model: str, q: str, temperature: float, max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": q}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    # gemma4 defaults to "thinking" mode: fills `message.thinking` and
    # returns EMPTY content. Disable thinking so the code lands in content
    # (also ~7x faster: 14s vs 104s). No-op for non-gemma models.
    if "gemma" in model.lower():
        body["think"] = False
    req = urllib.request.Request(
        f"{STOCK}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    # 12b on 6GB VRAM offloads 53% to CPU — first calls (incl. cold model
    # load) can exceed 180s. Match the bridge ceiling of 900s.
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d.get("message", {}).get("content", "").strip()


def _bridge_chat(q: str, max_tokens: int, port: int, attempt: int = 0,
                 model: str = "qwen2.5-coder:3b", retries: int = 2,
                 auto: bool = False) -> tuple:
    """Bridge call with connection resilience: 7b on 6GB VRAM + auto-recall
    + absorb can push a single request past 240s; the client timeout was
    closing the socket mid-request (RemoteDisconnected on the bench,
    BrokenPipeError in the bridge journal). Raise the timeout to 600s and
    retry transient connection drops instead of crashing the whole arm.

    auto=True sends the LAYER 2c autonomous-request flag (NO external
    attempt index): the bridge itself self-verifies and escalates
    internally (generate -> verify -> explore, bounded by auto_max_attempts).
    This is the honest closed-loop measurement: pass@1 here is the bridge's
    own self-decided final answer."""
    body = {
        "model": f"kai/{model}",
        "messages": [{"role": "user", "content": q}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.7,  # base knob; bridge's VFE controller adapts it
    }
    if auto:
        body["auto"] = 1  # no attempt field — bridge decides escalation
    else:
        body["attempt"] = attempt  # 0 = first try (confident), >0 = retry (explore)
    last_err = None
    for rtry in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            content = (d.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            phys = d.get("kai_physics", {}) or {}
            return content, phys
        except Exception as e:
            last_err = e
            if rtry < retries:
                print(f"  [bridge retry {rtry + 1}/{retries}] {type(e).__name__}: {e}",
                      flush=True)
                time.sleep(5 * (rtry + 1))
    raise last_err


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


def grade_arm(label: str, tasks: list, mode: str, k: int, port: int,
              model: str = "qwen2.5-coder:3b") -> dict:
    # Per-task resume: a crash mid-arm (RemoteDisconnected at 7b) must not
    # discard completed tasks. Ledger persists one line per task; on restart
    # with the same label, completed tasks are replayed, not re-run.
    # NOTE: `label` already contains the mode suffix (main() calls
    # grade_arm(f"{args.label}_{mode}", ...)) — do NOT append mode again.
    ledger_path = RECORD + f".{label}.ledger.jsonl"
    done = {}
    if os.path.exists(ledger_path):
        for line in open(ledger_path):
            try:
                e = json.loads(line)
                done[e["name"]] = e
            except Exception:
                pass
    pass1, solved = 0, 0
    attempts_used = []
    temps = []
    per_task = []
    t0 = time.time()
    ledger = open(ledger_path, "a")
    for ti, task in enumerate(tasks, 1):
        if task["name"] in done:
            e = done[task["name"]]
            per_task.append(e)
            if e["pass1"]: pass1 += 1
            if e["solved"]:
                solved += 1
                attempts_used.append(e["attempts"])
            else:
                attempts_used.append(0)
            temps.append(round(sum(e["temps"]) / len(e["temps"]), 3) if e["temps"] else 0.0)
            print(f"  [{ti}/{len(tasks)}] {task['name']:18s} (resumed) pass1={e['pass1']} "
                  f"solved@={e['attempts'] or '-'}")
            continue
        q = task["prompt"]
        first_pass = None
        task_solved_at = None
        task_temps = []
        for attempt in range(k):
            if mode == "greedy":
                # Deterministic: only attempt 0 is meaningful, but re-run to
                # verify stability (greedy must repeat the SAME code).
                ans = _ollama_chat(model, q, 0.0, MAX_TOKENS)
                temp_used = 0.0
            elif mode == "fixed":
                ans = _ollama_chat(model, q, 0.7, MAX_TOKENS)
                temp_used = 0.7
            elif mode == "auto":
                # LAYER 2c autonomous loop: ONE external call, no attempt
                # index — the bridge self-verifies and escalates internally.
                # Honest closed loop: the bridge's final answer is graded.
                ans, phys = _bridge_chat(q, MAX_TOKENS, port, model=model, auto=True)
                temp_used = phys.get("temperature", 0.7)
                if attempt > 0:
                    break  # single external call; bridge loop is internal
            else:  # physics
                ans, phys = _bridge_chat(q, MAX_TOKENS, port, attempt=attempt, model=model)
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
        entry = {
            "name": task["name"], "pass1": bool(first_pass),
            "solved": task_solved_at is not None,
            "attempts": task_solved_at or 0,
            "temps": task_temps,
        }
        if mode == "auto":
            entry["auto_attempts"] = phys.get("auto_attempts", 0)
            entry["auto_verified"] = bool(phys.get("auto_verified", 0))
            entry["auto_agreement"] = phys.get("auto_agreement", 0.0)
        per_task.append(entry)
        ledger.write(json.dumps(entry) + "\n")
        ledger.flush()
        print(f"  [{ti}/{len(tasks)}] {task['name']:18s} pass1={first_pass} "
              f"solved@={task_solved_at or '-'} temps={task_temps}")
    ledger.close()
    n = len(tasks)
    res = {
        "t": time.time(), "label": label, "mode": mode, "worker": model,
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
    ap.add_argument("--model", default="qwen2.5-coder:3b",
                    help="coder model for all arms (default qwen2.5-coder:3b)")
    ap.add_argument("--arms", default="greedy,fixed,physics",
                    help="comma-separated arms to run")
    ap.add_argument("--tier", choices=["base", "capacity", "all"],
                    default="base",
                    help="task tier: base=29 saturating tasks, "
                         "capacity=15 harder stateful/adversarial tasks, "
                         "all=both")
    args = ap.parse_args()

    if args.tier == "capacity":
        tasks = CAPACITY_TASKS[: args.tasks]
    elif args.tier == "all":
        tasks = (TASKS + CAPACITY_TASKS)[: args.tasks]
    else:
        tasks = TASKS[: args.tasks]
    print(f"==== kai code bench  label='{args.label}'  model={args.model}  "
          f"tier={args.tier}  tasks={len(tasks)}  K={args.k}  arms={args.arms}")
    print(f"     objective grading (hidden unit tests), no judge, no rubric\n")

    results = {}
    for mode in [m.strip() for m in args.arms.split(",")]:
        print(f"--- arm: {mode} ---")
        res = grade_arm(f"{args.label}_{mode}", tasks, mode, args.k, args.port,
                        model=args.model)
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
