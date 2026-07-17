#!/usr/bin/env python3
"""
HackerOne Workflow for Smart Contract Bounties
Finds targets, runs scans, generates reports
"""
import subprocess
import os
import json
import glob
import re
import urllib.request
from datetime import datetime

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', 'ghp_REDACTED')

# HackerOne Programs with Smart Contract Scope
H1_PROGRAMS = [
    {
        "name": "Coinbase",
        "handle": "coinbase",
        "scope": "*.coinbase.com, Smart Contracts",
        "min_bounty": 200,
        "max_bounty": 100000,
        "type": "Crypto Exchange",
        "url": "https://hackerone.com/coinbase",
        "focus": ["DeFi", "Wallet", "Bridge", "L2 Base"]
    },
    {
        "name": "Kraken",
        "handle": "kraken",
        "scope": "kraken.com, Smart Contracts",
        "min_bounty": 250,
        "max_bounty": 50000,
        "type": "Crypto Exchange",
        "url": "https://hackerone.com/kraken",
        "focus": ["Staking", "DeFi", "Wallet"]
    },
    {
        "name": "GitLab",
        "handle": "gitlab",
        "scope": "gitlab.com",
        "min_bounty": 100,
        "max_bounty": 20000,
        "type": "Web App",
        "url": "https://hackerone.com/gitlab",
        "focus": ["Web", "API", "Auth"]
    },
    {
        "name": "Shopify",
        "handle": "shopify",
        "scope": "*.myshopify.com, shopify.com",
        "min_bounty": 500,
        "max_bounty": 30000,
        "type": "E-commerce",
        "url": "https://hackerone.com/shopify",
        "focus": ["Web", "API", "Payment"]
    },
    {
        "name": "Uber",
        "handle": "uber",
        "scope": "*.uber.com",
        "min_bounty": 500,
        "max_bounty": 20000,
        "type": "Tech",
        "url": "https://hackerone.com/uber",
        "focus": ["Web", "API", "Mobile"]
    },
    {
        "name": "Blockstream",
        "handle": "blockstream",
        "scope": "*.blockstream.com, Liquid Network",
        "min_bounty": 500,
        "max_bounty": 100000,
        "type": "Bitcoin",
        "url": "https://hackerone.com/blockstream",
        "focus": ["Bitcoin", "Liquid", "Sidechains"]
    },
]

class HackerOneHunter:
    def __init__(self):
        self.results = []
    
    def scan_web_target(self, target):
        """Basic web security scan"""
        print(f"\nScanning {target['name']}...")
        findings = []
        
        # Check for common endpoints
        endpoints = ['/robots.txt', '/.well-known/security.txt', '/api', '/graphql', '/admin']
        
        for endpoint in endpoints:
            try:
                url = f"https://{target['scope'].split(',')[0].strip().replace('*.', '')}{endpoint}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=10)
                status = resp.getcode()
                if status == 200:
                    findings.append({
                        'type': 'Endpoint Found',
                        'severity': 'Info',
                        'url': url,
                        'status': status
                    })
            except:
                pass
        
        return findings
    
    def scan_smart_contracts(self, repo_url):
        """Scan smart contracts for vulnerabilities"""
        print(f"\nScanning contracts: {repo_url}")
        
        work_dir = f"/tmp/h1_{repo_url.split('/')[-1]}"
        subprocess.run(['rm', '-rf', work_dir], capture_output=True)
        
        result = subprocess.run(
            ['git', 'clone', '--depth=1', repo_url, work_dir],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            return []
        
        sol_files = glob.glob(os.path.join(work_dir, '**/*.sol'), recursive=True)
        sol_files = [f for f in sol_files if 'test' not in f.lower() and 'mock' not in f.lower()]
        
        findings = []
        
        for sol_file in sol_files:
            try:
                with open(sol_file, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Critical patterns
                patterns = [
                    (r'delegatecall\(.*\)', 'Delegatecall', 'High'),
                    (r'\.call\{.*value.*\}.*\n.*state', 'Reentrancy', 'Critical'),
                    (r'tx\.origin', 'tx.origin', 'Medium'),
                    (r'selfdestruct\(.*\)', 'Selfdestruct', 'High'),
                    (r'ecrecover\(.*\)', 'Signature', 'Medium'),
                    (r'block\.timestamp', 'Timestamp', 'Low'),
                    (r'pragma\s+solidity\s+\^', 'Floating Pragma', 'Low'),
                ]
                
                for pattern, name, severity in patterns:
                    if re.search(pattern, content):
                        findings.append({
                            'type': name,
                            'severity': severity,
                            'file': sol_file.replace(work_dir + '/', ''),
                            'repo': repo_url
                        })
            except:
                pass
        
        return findings
    
    def generate_report(self, program, findings):
        """Generate HackerOne report"""
        report = []
        report.append("=" * 60)
        report.append(f"HACKERONE REPORT: {program['name']}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        report.append(f"\nProgram: {program['name']}")
        report.append(f"URL: {program['url']}")
        report.append(f"Scope: {program['scope']}")
        report.append(f"Bounty Range: ${program['min_bounty']} - ${program['max_bounty']:,}")
        report.append(f"Focus Areas: {', '.join(program['focus'])}")
        
        if findings:
            report.append(f"\nFindings: {len(findings)}")
            for f in findings[:20]:
                report.append(f"\n  [{f['severity']}] {f['type']}")
                if 'file' in f:
                    report.append(f"    File: {f['file']}")
                if 'url' in f:
                    report.append(f"    URL: {f['url']}")
        else:
            report.append("\nNo findings yet. Manual testing required.")
        
        report.append("\n" + "=" * 60)
        report.append("NEXT STEPS:")
        report.append("1. Go to " + program['url'])
        report.append("2. Read scope carefully")
        report.append("3. Test for OWASP Top 10")
        report.append("4. Write clear report with PoC")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def hunt(self):
        """Main hunt loop"""
        print("HACKERONE BOUNTY HUNTER")
        print("=" * 60)
        
        for program in H1_PROGRAMS[:3]:
            print(f"\n{'='*60}")
            print(f"TARGET: {program['name']}")
            print(f"{'='*60}")
            
            findings = self.scan_web_target(program)
            report = self.generate_report(program, findings)
            
            report_path = f'/home/l/Desktop/AxiomTree/axiom_horizon/bug_bounty/H1_{program["handle"]}.md'
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"Report: {report_path}")
            print(report[:1500])

if __name__ == '__main__':
    hunter = HackerOneHunter()
    hunter.hunt()
