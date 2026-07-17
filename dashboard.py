#!/usr/bin/env python3
"""Web dashboard for Axiom daemon — real-time Darwin, fitness, tau, cycles."""

import sys, os, json, math
from pathlib import Path
from flask import Flask, jsonify, render_template_string

sys.path.insert(0, '.')
os.environ['OLLAMA_HOST'] = 'http://127.0.0.1:11434'

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>Axiom Dashboard</title>
<style>
  body { font-family: monospace; background: #0a0a0a; color: #c0c0c0; margin: 20px; }
  h1 { color: #0f0; border-bottom: 1px solid #333; }
  h2 { color: #0f0; margin-top: 24px; font-size: 16px; }
  .card { background: #111; border: 1px solid #333; padding: 12px; margin: 8px 0; }
  .stat { display: inline-block; margin: 0 16px 8px 0; }
  .stat span { color: #0f0; font-weight: bold; }
  .bar { display: inline-block; height: 12px; background: #0f0; margin-right: 4px; }
  .bar-bg { background: #222; height: 12px; margin: 2px 0; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #222; }
  th { color: #0f0; }
  .fitness-bar { height: 16px; background: #0f0; border-radius: 2px; }
  .mutation-cell { color: #ff0; font-size: 10px; }
  .log-line { color: #888; font-size: 11px; margin: 0; }
  .badge { background: #333; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
</style>
</head>
<body>
<h1>▓ AXIOM — silicon life monitor ▓</h1>
<div class="card">
  <div class="stat">Agents: <span id="agent-count">{{ data.darwin.count }}</span></div>
  <div class="stat">Avg fitness: <span>{{ "%.4f"|format(data.darwin.avg_fitness) }}</span></div>
  <div class="stat">Peak: <span>{{ "%.4f"|format(data.darwin.max_fitness) }}</span></div>
  <div class="stat">>0.9: <span>{{ data.darwin.above_09 }}</span></div>
  <div class="stat">Sessions: <span>{{ data.agent.sessions }}</span></div>
  <div class="stat">Self-mods: <span>{{ data.agent.self_mods }}</span></div>
  <div class="stat">τ: <span>{{ "%.3f"|format(data.agent.tau) }}</span></div>
  <div class="stat">Cycles: <span>{{ data.agent.cycles }}</span></div>
  <div class="stat">VFE: <span>{{ "%.2e"|format(data.agent.vfe) }}</span></div>
</div>

<h2>Fitness Distribution</h2>
<div class="card">
  {% for label, pct, count in data.darwin.fitness_bars %}
  <div class="bar-bg"><span style="display:inline-block;width:60px">{{ label }}</span>
    <span class="bar" style="width:{{ pct * 2 }}px;background:{{ 'lime' if label.startswith('>0.9') else '#0a0' if label.startswith('>0.8') else '#060' }}"></span>
    <span style="font-size:11px;margin-left:4px">{{ count }}</span>
  </div>
  {% endfor %}
</div>

<h2>Mutation Distribution</h2>
<div class="card">
  {% for mut, count, pct in data.darwin.mutation_bars %}
  <div class="bar-bg"><span style="display:inline-block;width:120px">{{ mut }}</span>
    <span class="bar" style="width:{{ pct * 4 }}px"></span>
    <span style="font-size:11px;margin-left:4px">{{ count }}</span>
  </div>
  {% endfor %}
</div>

<h2>Top 10 Candidates</h2>
<div class="card">
<table>
<tr><th>#</th><th>Hash</th><th>Fitness</th><th>Mutation</th><th>Parents</th></tr>
{% for h, a in data.darwin.top10 %}
<tr>
  <td>{{ loop.index }}</td>
  <td style="font-size:11px">{{ h[:16] }}</td>
  <td>
    <div style="display:flex;align-items:center">
      <div class="fitness-bar" style="width:{{ a.fitness * 60 }}px"></div>
      <span style="margin-left:4px">{{ "%.4f"|format(a.fitness) }}</span>
    </div>
  </td>
  <td class="mutation-cell">{{ a.mutation }}</td>
  <td>{{ a.parent_count }}</td>
</tr>
{% endfor %}
</table>
</div>

<h2>Last 10 Daemon Log Lines</h2>
<div class="card">
{% for line in data.daemon_log %}
  <div class="log-line">{{ line[:160] }}</div>
{% endfor %}
</div>

<h2>Bracket-line</h2>
<div class="card" style="font-size:12px">
<pre style="color:#0f0">{{ data.bracket }}</pre>
</div>
</body>
</html>
'''

@app.route('/')
def index():
    data = _get_data()
    return render_template_string(HTML, data=data)

@app.route('/api')
def api():
    return jsonify(_get_data())

def _get_data():
    from axiom import DarwinArchive, Axiom
    darwin = DarwinArchive()
    agent = Axiom()
    
    agents = list(darwin.agents.items())
    fits = [a['fitness'] for h,a in agents]
    
    avg_f = sum(fits) / len(fits) if fits else 0.0
    max_f = max(fits) if fits else 0.0
    
    top10 = sorted(agents, key=lambda x: -x[1]['fitness'])[:10]
    
    above_09 = sum(1 for f in fits if f >= 0.9)
    above_08 = sum(1 for f in fits if f >= 0.8)
    above_07 = sum(1 for f in fits if f >= 0.7)
    above_06 = sum(1 for f in fits if f >= 0.6)
    below_06 = sum(1 for f in fits if f < 0.6)
    
    fitness_bars = [
        ('>0.9', above_09, above_09),
        ('>0.8', above_08, above_08),
        ('>0.7', above_07, above_07),
        ('>0.6', above_06, above_06),
        ('<0.6', below_06, below_06),
    ]
    
    muts = {}
    for h,a in agents:
        m = a.get('mutation', 'seed')
        muts[m] = muts.get(m, 0) + 1
    total_muts = sum(muts.values()) or 1
    sorted_muts = sorted(muts.items(), key=lambda x: -x[1])
    mutation_bars = [(m, c, c / total_muts * 100) for m,c in sorted_muts[:15]]
    
    seed = agent.seed
    bracket = seed.bracket if hasattr(seed, 'bracket') else str(seed)[:200]
    
    dlog = Path('.axiom_state/daemon_stdout.log')
    daemon_log = []
    if dlog.exists():
        lines = dlog.read_text().splitlines()
        daemon_log = lines[-10:] if len(lines) >= 10 else lines
    
    return {
        'darwin': {
            'count': len(agents),
            'avg_fitness': avg_f,
            'max_fitness': max_f,
            'above_09': above_09,
            'fitness_bars': fitness_bars,
            'mutation_bars': mutation_bars,
            'top10': [(h, {'fitness': a['fitness'], 'mutation': a.get('mutation', 'seed'), 'parent_count': a.get('parent_count', 0)}) for h,a in top10],
        },
        'agent': {
            'sessions': agent.at.meta.get('session_count', '?'),
            'self_mods': agent.at.meta.get('self_mod_count', 0),
            'tau': seed.tau,
            'vfe': seed.vfe,
            'cycles': seed.cycles,
        },
        'bracket': bracket,
        'daemon_log': daemon_log,
    }

if __name__ == '__main__':
    print("Axiom Dashboard → http://127.0.0.1:5050")
    port = int(os.environ.get('DASHBOARD_PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
