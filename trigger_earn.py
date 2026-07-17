"""One-shot earning trigger — mirrors daemon._execute_active_task.
Validates the PR pipeline end-to-end: fork -> clone -> LLM solution -> PR.
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Load token from scanner/.env (same source the daemon uses)
for line in open('/home/l/Desktop/AxiomTree/scanner/.env'):
    line = line.strip()
    if line.startswith('GITHUB_TOKEN='):
        os.environ['GITHUB_TOKEN'] = line.split('=', 1)[1].strip().strip('"')

from axiom import ComputeCluster
from submissions import solve_and_submit

# Load active task
state = json.load(open('.axiom_state/tasks.json'))
task = state['active']
import re
m = re.match(r'https?://github\.com/([^/]+/[^/]+)/issues/(\d+)', task['url'])
repo, num = m.group(1), int(m.group(2))
print(f'[earn] active: {repo}#{num} — {task["title"][:50]}')

def llm_call(prompt: str) -> str:
    msgs = [{'role': 'system', 'content': 'You are a coding agent earning money from open-source bounties. Generate precise, working code.'},
            {'role': 'user', 'content': prompt}]
    r = ComputeCluster.chat(messages=msgs, task='chat', options={'temperature': 0.3, 'num_predict': 2048})
    return r.get('message', {}).get('content', '') or ''

try:
    result = solve_and_submit(repo, num, task['title'], task.get('description', ''), llm_call)
    pr_url = result.get('pr_url', '')
    print(f'[earn] PR SUBMITTED -> {pr_url}')
    json.dump(result, open('.axiom_state/last_submission.json', 'w'), default=str)
except Exception as e:
    print(f'[earn] FAILED: {e!r}')
