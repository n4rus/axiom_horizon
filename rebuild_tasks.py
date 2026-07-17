"""Rebuild .axiom_state/tasks.json from the ai-research bounty repo's open
issues, preserving any progress flags already on disk."""
import os, sys, json, re
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for line in open('/home/l/Desktop/AxiomTree/scanner/.env'):
    line = line.strip()
    if line.startswith('GITHUB_TOKEN='):
        os.environ['GITHUB_TOKEN'] = line.split('=', 1)[1].strip().strip('"')

import urllib.request
import task_ingestion as ti

TOKEN = os.environ['GITHUB_TOKEN']
REPO = 'zhangjiayang6835-cyber/ai-research'
STATE = HERE / '.axiom_state' / 'tasks.json'

# Load existing progress (submitted/completed) if any survived
prev = {}
if STATE.exists():
    try:
        d = json.loads(STATE.read_text())
        for t in d.get('tasks', []) + d.get('queue', []):
            prev[t['id']] = t
    except Exception:
        pass

# Fetch open issues
issues = []
page = 1
while True:
    url = f'https://api.github.com/repos/{REPO}/issues?state=open&per_page=100&page={page}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {TOKEN}', 'User-Agent': 'AxiomAGI/1.0'})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    if not data:
        break
    issues.extend(data)
    page += 1
    if len(data) < 100:
        break

# Filter out PRs (issues endpoint returns PRs too)
issues = [i for i in issues if 'pull_request' not in i]

# Keep only bounty-like issues (dollar reward or [BUG]/bounty/vulnerability)
def is_bounty(title):
    return bool(re.search(r'\$\d', title) or re.search(r'\[BUG\]|bounty|vulnerab|CVE-', title, re.I))

issues = [i for i in issues if is_bounty(i['title'])]

def parse_reward(title):
    m = re.search(r'\$?(\d+(?:\.\d+)?)', title)
    return float(m.group(1)) if m else 0.0

tasks = []
for i in issues:
    iid = i['id']
    num = i['number']
    title = i['title']
    url = i['html_url']
    desc = i.get('body', '') or ''
    reward_usd = parse_reward(title)
    # preserve progress
    p = prev.get(f'github_{iid}', {})
    t = ti.Task(
        id=f'github_{iid}',
        source='github',
        title=title,
        description=desc,
        reward=title,
        reward_usd=reward_usd,
        url=url,
        tags=['security', 'coding', 'audit'],
        difficulty=0.6,
        submitted=bool(p.get('submitted', False)),
        completed=bool(p.get('completed', False)),
        payment_tx=p.get('payment_tx', ''),
        fetched_at=i.get('created_at', '') or 0.0,
    )
    tasks.append(t)

print(f'fetched {len(tasks)} issues')
# Mark issues/792 as already submitted (PRs #848/#850 already opened)
for t in tasks:
    if t.url.rstrip('/').endswith('/issues/792'):
        t.submitted = True
pipeline = ti.TaskPipeline()
pipeline.tasks = tasks
# set active to first unsubmitted
pipeline.active_task = None
pipeline.queue = []
pipeline.ingest_cycle()  # scores, builds queue, advances active
pipeline.save_state()
print('active:', pipeline.active_task.title[:50] if pipeline.active_task else None)
print('queue len:', len(pipeline.queue))
