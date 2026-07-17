"""Validate the coder model produces an applicable diff for a bounty — without
pushing or opening a PR (avoids duplicate submissions with the live daemon)."""
import os, sys, re
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for line in open('/home/l/Desktop/AxiomTree/scanner/.env'):
    line = line.strip()
    if line.startswith('GITHUB_TOKEN='):
        os.environ['GITHUB_TOKEN'] = line.split('=', 1)[1].strip().strip('"')

from axiom import ComputeCluster
import submissions as S

REPO = 'zhangjiayang6835-cyber/ai-research'
ISSUE = 732  # ERC-777 reentrancy (current active)

def llm_call(prompt: str) -> str:
    msgs = [{'role': 'system', 'content': 'You are a coding agent earning money from open-source security bounties. Produce a precise, working code fix as a valid unified diff wrapped in a ```diff block.'},
            {'role': 'user', 'content': prompt}]
    r = ComputeCluster.chat(messages=msgs, task='chat', options={'temperature': 0.0, 'num_predict': 1500})
    return r.get('message', {}).get('content', '') or ''

workdir = HERE / '.axiom_state' / 'validate_732'
workdir.mkdir(parents=True, exist_ok=True)
print('forking...')
fork = S.fork_repo(REPO)
print('clone...')
S.clone_repo(fork, workdir)
ctx = S.read_issue_context(workdir, '')
# minimal issue body from API
import urllib.request, json
req = urllib.request.Request(f'https://api.github.com/repos/{REPO}/issues/{ISSUE}', headers={'Authorization': f'Bearer {os.environ["GITHUB_TOKEN"]}', 'User-Agent': 'x'})
iss = json.loads(urllib.request.urlopen(req, timeout=20).read())
print('issue:', iss['title'])
targets = S.find_target_files(workdir, iss['title'], iss.get('body', ''))
target_file = targets[0] if targets else ''
target_content = ''
if target_file:
    target_content = (workdir / target_file).read_text(errors='replace')
    print('target file:', target_file)
target_func = S.locate_target_function(target_content, iss['title'], iss.get('body',''))
cm, pb, patch = S.generate_function_fix(target_file, target_content, iss['title'], iss.get('body',''), llm_call, target_func=target_func)
applied = False
if patch.strip():
    if S.check_patch(workdir, patch) or S.apply_patch(workdir, patch):
        applied = True
if not applied:
    cm, pb, new_text = S.generate_rewrite(target_file, target_content, iss['title'], iss.get('body',''), llm_call)
    if new_text.strip() and new_text.strip() != target_content.strip():
        patch = S.build_diff(target_file, target_content, new_text)
        if S.check_patch(workdir, patch) or S.apply_patch(workdir, patch):
            applied = True
print('commit:', cm[:70])
print('patch length:', len(patch))
print('--- patch head ---')
print('\n'.join(patch.splitlines()[:25]))
ok = bool(patch.strip()) and S.check_patch(workdir, patch)
print('APPLIES CLEANLY (git apply --check):', ok)
if applied:
    print('REAL CODE FIX APPLIED -> would earn')
else:
    fuzzed = S.apply_patch(workdir, patch)
    print('APPLIES WITH FUZZ FALLBACK:', fuzzed)
    if not fuzzed:
        print('(falls back to doc-only PR — will not earn)')
    else:
        print('REAL CODE FIX APPLIED -> would earn')
