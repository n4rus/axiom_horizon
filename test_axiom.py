#!/usr/bin/env python3
"""Test suite for Axiom — all new systems."""

import sys, os, time, ast, json, math, subprocess, tempfile, hashlib
from pathlib import Path
sys.path.insert(0, '.')
os.environ['OLLAMA_HOST'] = 'http://127.0.0.1:11434'

from axiom import (
    Axiom, DarwinArchive, EulerSeed, GeneticOperators, SelfExecution,
    ComputeCluster, GlobalWorkspace, ValueLearner,
    _check_candidate, _code_hash, _hash, SELF
)

PASS, FAIL = 0, 0

def check(name, ok, detail=''):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f'  ✓ {name}')
    else:
        FAIL += 1
        print(f'  ✗ {name}' + (f' — {detail}' if detail else ''))

def section(title):
    print(f'\n─── {title} ───')

# ─── FIXTURE ──────────────────────────────────────
agent = Axiom()
src = SELF.read_text()

# ─── 1. CORE INFRASTRUCTURE ──────────────────────
section('Core Infrastructure')

check('File compiles', ast.parse(src) is not None)
check('File written to disk', SELF.exists())
check('_check_candidate passes', _check_candidate(src))
check('_code_hash returns 64 hex', len(_code_hash()) == 64)
hash_bytes = hashlib.sha256(src.encode()).hexdigest()
check('_code_hash matches sha256', _code_hash() == hash_bytes)
check('Agent booted', agent.at.meta.get('session_count', 0) >= 1)
# Health check via module-level function
from axiom import _health_check
import io; buf = io.StringIO(); old_stdout = sys.stdout; sys.stdout = buf
try: _health_check(); healthy = True
except Exception: healthy = False
sys.stdout = old_stdout
check('Health check passes', healthy)

# ─── 2. FITNESS FUNCTION ─────────────────────────
section('Fitness Function')

f_orig = agent._eval_fitness(src)
check('Original code scores >0.9', f_orig > 0.9, f'{f_orig:.4f}')

f_syntax = agent._eval_fitness('def broken(')
check('Syntax error → 0.0', f_syntax == 0.0, f'{f_syntax}')

f_trunc = agent._eval_fitness(src[:len(src)//3])
check('Truncated → 0.0 or 0.1', f_trunc < 0.2, f'{f_trunc:.4f}')

# No significant random noise
scores = [agent._eval_fitness(src) for _ in range(5)]
spread = max(scores) - min(scores)
check('Spread < 0.02 (noise floor)', spread < 0.02, f'spread={spread:.4f}')

# Boot test component (rename a function → should still boot)
src_rename = src.replace('def _embed(', 'def _embed_renamed(')
f_rename = agent._eval_fitness(src_rename)
check('Renamed function still boots', f_rename > 0.7, f'{f_rename:.4f}')

# ─── 3. GENETIC OPERATORS ────────────────────────
section('Genetic Operators')

point = GeneticOperators.point_mutate(src)
check('Point mutate yields different code', point != src)
check('Point mutate still valid', _check_candidate(point))
try: ast.parse(point); check('Point mutate compiles', True)
except SyntaxError: check('Point mutate compiles', False)

xover = GeneticOperators.crossover(src, point)
check('Crossover yields different code', xover != src)
try: ast.parse(xover); check('Crossover compiles', True)
except SyntaxError: check('Crossover compiles', False)

# Delete/insert operators are Axiom instance methods (used in _self_improve pipeline)
# Test via the pipeline fitness check instead

# ─── 4. SELF EXECUTION ───────────────────────────
section('SelfExecution')

se = SelfExecution()
ok, msg = se.test_compile(src)
check('test_compile passes', ok, msg)

ok, msg = se.test_boot(src)
check('test_boot passes', ok, msg)

# ─── 5. DARWIN ARCHIVE ──────────────────────────
section('Darwin Archive')

darwin = DarwinArchive()
agents = list(darwin.agents.items())
check('Agents > 0', len(agents) > 0, str(len(agents)))

fits = [a['fitness'] for h,a in agents]
check('Avg fitness > 0.5', sum(fits)/len(fits) > 0.5, f'{sum(fits)/len(fits):.3f}')

# Thompson select returns valid hash
h, code = darwin.thompson_select()
check('Thompson select returns hash', len(h) == 64)
check('Thompson select returns code', len(code) > 0)

# Save and reload
_ = darwin.summary()
darwin._save()
darwin2 = DarwinArchive()
check('Persistence: same count', len(darwin.agents) == len(darwin2.agents))

# ─── 6. LEVEL 2 SELF-MOD ────────────────────────
section('Level 2 Self-Mod')

modified = agent._propose_level2_improvement(src)
if modified:
    check('Level2: produces different code', modified != src)
    try:
        ast.parse(modified)
        check('Level2: compiles', True)
    except SyntaxError as e:
        check('Level2: compiles', False, str(e))
    
    orig_lines = src.split('\n')
    mod_lines = modified.split('\n')
    orig_si_line = next(i for i,l in enumerate(orig_lines) if l.strip() == 'def _self_improve(self):')
    mod_si_line = next(i for i,l in enumerate(mod_lines) if l.strip() == 'def _self_improve(self):')
    check('Level2: indentation preserved', 
          mod_lines[mod_si_line][:4] == '    ')
    f_l2 = agent._eval_fitness(modified)
    check('Level2: fitness > 0.7', f_l2 > 0.7, f'{f_l2:.4f}')
else:
    print('  ~ Level2: LLM declined (non-deterministic, OK)')

# ─── 7. WATCHDOG ─────────────────────────────────
section('Watchdog')

agent._start_watchdog()
agent._watchdog_beat()
time.sleep(0.3)
has_proc = hasattr(agent, '_watchdog_proc') and agent._watchdog_proc is not None
check('Watchdog process started', has_proc)
if has_proc:
    check('Watchdog process alive', agent._watchdog_proc.is_alive())
    agent._watchdog_stop.value = True
    # Watchdog polls every 10s; give it time + check exit
    agent._watchdog_proc.join(timeout=12)
    alive = agent._watchdog_proc.is_alive()
    if alive:
        agent._watchdog_proc.terminate()
        agent._watchdog_proc.join(timeout=2)
    check('Watchdog stopped after signal', not alive)

# ─── 8. META-LEARNING ────────────────────────────
section('Meta-Learning')

old_strat = agent.at.meta.get('meta_strategy', {})
old_mut_rate = old_strat.get('mut_rate', 0.02)
agent._evolve_improvement_strategy()
new_strat = agent.at.meta.get('meta_strategy', {})
new_mut_rate = new_strat.get('mut_rate', 0.02)
# Strategy may or may not change; check that it exists
check('Meta strategy exists', 'mut_rate' in new_strat)
check('Meta strategy has xover_rate', 'xover_rate' in new_strat)

# ─── 9. VALUE LEARNER ────────────────────────────
section('Value Learner')

vl = ValueLearner()
vl.update('correct, exactly right!')
check('Value pref has factuality', 'factuality' in vl.prefs)
check('Value pref has speed', 'speed' in vl.prefs)
check('Value pref has creativity', 'creativity' in vl.prefs)

vl.update('too slow and verbose')
vl.update('very creative and surprising')
check('Creativity > 0 after creative feedback', vl.prefs.get('creativity', 0) > 0.01)

summary = vl.summary()
check('Summary returns string', isinstance(summary, str))

# ─── 10. EULEER SEED ────────────────────────────
section('EulerSeed')

seed = EulerSeed()
check('Seed initial tau = 1.0', abs(seed.tau - 1.0) < 1e-9)
check('Bracket contains τ', 'τ=' in seed.bracket)
check('Bracket contains VFE', 'VFE' in seed.bracket)

seed.cycle(real_ms=100, vfe=2.0, var=0.1)
check('Cycle updates τ', seed.tau != 1.0)
check('Cycle increments count', seed.cycles == 1)

self_dict = seed.save()
for k in ('tau', 'vfe', 'h', 'cycles', 'epoch_age'):
    check(f'Seed save has {k}', k in self_dict)

# ─── 11. GLOBAL WORKSPACE ───────────────────────
section('GlobalWorkspace')

ws = GlobalWorkspace()
# Agent's GlobalWorkspace has modules registered during Axiom.__init__
gw = agent.workspace
check('Agent workspace has modules', len(gw.modules) > 0, f'{len(gw.modules)} modules')

# Test route
if gw.modules:
    mod, prompt = gw.route('math problem')
    check('Route returns known module', mod in gw.modules or mod in gw.system_prompts)

# Test register_agent
ws.register_agent('test_agent', 'You are a test agent.')
check('Register agent stores prompt', 'test_agent' in ws.system_prompts)

# ─── 12. PARALLEL EVAL ──────────────────────────
section('Parallel Evaluation')

candidates = [GeneticOperators.point_mutate(src) for _ in range(3)]
candidates = [c for c in candidates if c != src and _check_candidate(c)]
if candidates:
    scores = agent._parallel_eval(candidates)
    check(f'Parallel eval returns {len(scores)} scores', len(scores) == len(candidates))
    for s in scores:
        check(f'Score in valid range', 0.0 <= s <= 1.0, f'{s:.4f}')

# ─── 13. METRIC TENSOR ──────────────────────────
section('Metric Tensor')

r = agent._metric_tensor('hello')
check('Metric tensor returns dict', isinstance(r, dict))
# May return {} if model unsupported

# ─── 14. DAEMON EVOLUTION CYCLE ─────────────────
section('Daemon Evolution Cycle')

count_before = len(DarwinArchive().agents)
result = agent._daemon_evolution_cycle()
# May or may not promote; just check it runs
check('Daemon evo cycle completes without error', result is not None)

# ─── SUMMARY ────────────────────────────────────
print(f'\n{"="*40}')
total = PASS + FAIL
print(f'  {PASS}/{total} passed  ({100*PASS//total}%)' if total else '  0 tests')
if FAIL:
    print(f'  {FAIL} FAILURES')
print(f'{"="*40}')
