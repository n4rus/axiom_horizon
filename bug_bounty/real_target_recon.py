import os, json, urllib.request, subprocess, re

os.environ['GITHUB_TOKEN'] = 'ghp_REDACTED'
token = os.environ['GITHUB_TOKEN']

# Find a small repo with a bug bounty program
# Look for repos with "bounty" label issues and actual code
print("=== Finding Real Targets ===\n")

# Search for repos with bounty issues that have actual code
url = 'https://api.github.com/search/repositories?q=label:bounty+stars:50..500+language:python&sort=updated&order=desc&per_page=10'
req = urllib.request.Request(url, headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'})
resp = urllib.request.urlopen(req, timeout=15)
data = json.loads(resp.read())

targets = []
for repo in data.get('items', []):
    name = repo['full_name']
    stars = repo['stargazers_count']
    lang = repo.get('language', 'unknown')
    desc = (repo.get('description', '') or '')[:100]
    url = repo['html_url']
    
    # Check if it has bounty issues
    issues_url = f'https://api.github.com/repos/{name}/issues?labels=bounty&state=open&per_page=3'
    req2 = urllib.request.Request(issues_url, headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'})
    try:
        resp2 = urllib.request.urlopen(req2, timeout=10)
        issues = json.loads(resp2.read())
        if issues:
            targets.append({
                'name': name,
                'stars': stars,
                'lang': lang,
                'desc': desc,
                'url': url,
                'issues': len(issues),
                'issue_title': issues[0]['title'][:60] if issues else ''
            })
    except:
        pass

print(f"Found {len(targets)} potential targets:\n")
for t in targets:
    print(f"  {t['name']} ({t['stars']}★) [{t['lang']}]")
    print(f"    {t['desc']}")
    print(f"    Bounty issue: {t['issue_title']}")
    print(f"    {t['url']}")
    print()

# Pick the best target and run recon
if targets:
    best = targets[0]
    print(f"\n=== Starting Recon on {best['name']} ===")
    
    # Clone and analyze
    workdir = f"/tmp/bounty_target_{best['name'].replace('/', '_')}"
    subprocess.run(['rm', '-rf', workdir], capture_output=True)
    result = subprocess.run(['git', 'clone', '--depth=1', f"https://github.com/{best['name']}.git", workdir], 
                          capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
        print(f"  Cloned to {workdir}")
        
        # List files
        files_result = subprocess.run(['find', workdir, '-name', '*.py', '-type', 'f'], 
                                     capture_output=True, text=True)
        py_files = [f for f in files_result.stdout.strip().split('\n') if f]
        print(f"  Python files: {len(py_files)}")
        
        # Quick security scan
        print("\n  Quick security scan:")
        for f in py_files[:10]:
            try:
                content = open(f).read()
                rel_path = f.replace(workdir + '/', '')
                
                # Check for common issues
                issues = []
                if 'eval(' in content:
                    issues.append('eval() usage')
                if 'exec(' in content:
                    issues.append('exec() usage')
                if 'subprocess' in content and 'shell=True' in content:
                    issues.append('shell=True in subprocess')
                if 'pickle' in content:
                    issues.append('pickle usage (deserialization)')
                if 'yaml.load(' in content and 'Loader' not in content:
                    issues.append('yaml.load without Loader')
                if re.search(r'password\s*=\s*["\']', content, re.I):
                    issues.append('hardcoded password')
                if re.search(r'api[_-]?key\s*=\s*["\']', content, re.I):
                    issues.append('hardcoded API key')
                    
                if issues:
                    print(f"    {rel_path}: {', '.join(issues)}")
            except:
                pass
        
        print(f"\n  Target ready for deeper analysis!")
    else:
        print(f"  Clone failed: {result.stderr[:200]}")
