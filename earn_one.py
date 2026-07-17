"""One-shot: submit a PR for the next fresh (unsubmitted) GitHub bounty task.

Mirrors daemon._execute_active_task but picks the next queued task so we can
generate income immediately instead of waiting for the ingest cadence.
"""
import os, sys, json, re
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

for line in open('/home/l/Desktop/AxiomTree/scanner/.env'):
    line = line.strip()
    if line.startswith('GITHUB_TOKEN='):
        os.environ['GITHUB_TOKEN'] = line.split('=', 1)[1].strip().strip('"')

from axiom import ComputeCluster
from submissions import solve_and_submit
import task_ingestion as ti

state_path = HERE / '.axiom_state' / 'tasks.json'
data = json.loads(state_path.read_text())
tasks_obj = ti.TaskPipeline()
tasks_obj.load_state()

# pick next fresh github task (skip ones already submitted in our state)
cand = None
for t in tasks_obj.queue:
    if t.source == 'github' and not t.submitted and not t.completed:
        m = re.match(r'https?://github\.com/([^/]+/[^/]+)/issues/(\d+)', t.url)
        if m:
            cand = t
            break
if cand is None:
    # fall back to in-memory tasks list
    for t in tasks_obj.tasks:
        if t.source == 'github' and not t.submitted and not t.completed:
            m = re.match(r'https?://github\.com/([^/]+/[^/]+)/issues/(\d+)', t.url)
            if m:
                cand = t
                break
if not cand:
    print('[earn] no fresh github task in queue')
    sys.exit(0)

repo, num = m.group(1), int(m.group(2))
print(f'[earn] target: {repo}#{num} — {cand.title[:55]}')

def llm_call(prompt: str) -> str:
    msgs = [{'role': 'system', 'content': 'You are a coding agent earning money from open-source security bounties. Produce a precise, working code fix as a valid unified diff wrapped in a ```diff block.'},
            {'role': 'user', 'content': prompt}]
    r = ComputeCluster.chat(messages=msgs, task='chat', options={'temperature': 0.0, 'num_predict': 1500})
    return r.get('message', {}).get('content', '') or ''

try:
    result = solve_and_submit(repo, num, cand.title, cand.description, llm_call)
    pr_url = result.get('pr_url', '')
    print(f'[earn] PR SUBMITTED -> {pr_url}')
    tasks_obj.mark_submitted(cand.id, pr_url)
    tasks_obj.save_state()
    json.dump(result, open(HERE / '.axiom_state' / 'last_submission.json', 'w'), default=str)
except Exception as e:
    print(f'[earn] FAILED: {e!r}')
