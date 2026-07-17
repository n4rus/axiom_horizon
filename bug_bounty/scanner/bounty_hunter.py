#!/usr/bin/env python3
"""
Smart Contract Bounty Hunter - Complete Workflow
Finds targets, scans for vulnerabilities, generates reports
"""
import subprocess
import os
import json
import glob
import re
import urllib.request
from datetime import datetime

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', 'ghp_REDACTED')
PATH = os.environ.get('PATH', '')
os.environ['PATH'] = f"{os.path.expanduser('~')}/.foundry/bin:{os.path.expanduser('~')}/.local/bin:{PATH}"

# Top Immunefi Programs with Active Bounties
TARGETS = [
    {"name": "Morpho Blue", "repo": "morpho-org/morpho-blue", "bounty": 250000, "type": "Lending"},
    {"name": "Pendle Core", "repo": "pendle-finance/pendle-core-v2-public", "bounty": 500000, "type": "DeFi"},
    {"name": "Gearbox V3", "repo": "Gearbox-protocol/gearbox-v3", "bounty": 500000, "type": "Leverage"},
    {"name": "Ambire Wallet", "repo": "AmbireWallet/ambire-common", "bounty": 100000, "type": "Wallet"},
    {"name": "EigenLayer", "repo": "Layr-Labs/eigenlayer-contracts", "bounty": 1000000, "type": "Restaking"},
]

class BountyHunter:
    def __init__(self):
        self.results = []
    
    def scan_target(self, target):
        """Scan a single target"""
        print(f"\n{'='*60}")
        print(f"SCANNING: {target['name']}")
        print(f"Repository: {target['repo']}")
        print(f"Max Bounty: ${target['bounty']:,}")
        print(f"{'='*60}")
        
        repo_url = f"https://github.com/{target['repo']}"
        work_dir = f"/tmp/bounty_{target['repo'].replace('/', '_')}"
        
        # Clone
        subprocess.run(['rm', '-rf', work_dir], capture_output=True)
        result = subprocess.run(
            ['git', 'clone', '--depth=1', repo_url, work_dir],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr[:200]}")
            return None
        
        # Find Solidity files
        sol_files = glob.glob(os.path.join(work_dir, '**/*.sol'), recursive=True)
        sol_files = [f for f in sol_files if not any(skip in f for skip in ['node_modules', 'test/', 'mock', 'example', '.git'])]
        
        print(f"  Found {len(sol_files)} Solidity contracts")
        
        findings = []
        
        # Run Slither
        print("\n  Running Slither...")
        for sol_file in sol_files[:5]:
            try:
                result = subprocess.run(
                    ['slither', sol_file, '--json', '-', '--checklist'],
                    capture_output=True, text=True, timeout=120
                )
                if result.stdout:
                    data = json.loads(result.stdout)
                    for det in data.get('results', {}).get('detectors', []):
                        if det.get('impact', '').lower() in ['high', 'critical']:
                            findings.append({
                                'tool': 'Slither',
                                'severity': det.get('impact', 'High'),
                                'title': det.get('check', 'Unknown'),
                                'description': det.get('description', '')[:200],
                                'file': sol_file.replace(work_dir + '/', ''),
                            })
            except:
                pass
        
        # Manual review for high-value patterns
        print("  Running manual review...")
        for sol_file in sol_files:
            try:
                with open(sol_file, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Check for critical patterns
                patterns = [
                    (r'delegatecall\(.*\)', 'Delegatecall Usage', 'High'),
                    (r'\.call\{.*value.*\}.*\n.*state', 'Reentrancy Risk', 'Critical'),
                    (r'tx\.origin', 'tx.origin Usage', 'Medium'),
                    (r'selfdestruct\(.*\)', 'Selfdestruct Usage', 'High'),
                    (r'ecrecover\(.*\)', 'Signature Verification', 'Medium'),
                    (r'block\.timestamp', 'Timestamp Dependence', 'Low'),
                ]
                
                for pattern, name, severity in patterns:
                    if re.search(pattern, content):
                        findings.append({
                            'tool': 'Manual',
                            'severity': severity,
                            'title': name,
                            'file': sol_file.replace(work_dir + '/', ''),
                        })
            except:
                pass
        
        print(f"\n  Found {len(findings)} high/critical findings")
        
        return {
            'target': target,
            'findings': findings,
            'sol_files': len(sol_files),
            'work_dir': work_dir,
        }
    
    def generate_report(self, results):
        """Generate final report"""
        report = []
        report.append("=" * 70)
        report.append("SMART CONTRACT BOUNTY HUNTER - SCAN RESULTS")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        total_findings = sum(len(r['findings']) for r in results)
        report.append(f"\nTotal Targets Scanned: {len(results)}")
        report.append(f"Total Findings: {total_findings}")
        
        for r in results:
            target = r['target']
            findings = r['findings']
            
            report.append(f"\n{'='*70}")
            report.append(f"TARGET: {target['name']}")
            report.append(f"Repository: {target['repo']}")
            report.append(f"Max Bounty: ${target['bounty']:,}")
            report.append(f"Contracts: {r['sol_files']}")
            report.append(f"Findings: {len(findings)}")
            
            if findings:
                report.append("\nHigh/Critical Findings:")
                for f in findings[:10]:
                    report.append(f"  [{f['severity']}] {f['title']}")
                    report.append(f"    File: {f.get('file', 'N/A')}")
                    if 'description' in f:
                        report.append(f"    Description: {f['description'][:150]}")
        
        report.append("\n" + "=" * 70)
        report.append("NEXT STEPS:")
        report.append("1. Review each finding manually")
        report.append("2. Check Immunefi scope for each program")
        report.append("3. Write PoC exploit if vulnerability confirmed")
        report.append("4. Submit report via Immunefi")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def hunt(self):
        """Main hunt loop"""
        print("SMART CONTRACT BOUNTY HUNTER")
        print("=" * 70)
        print(f"Targets: {len(TARGETS)}")
        print(f"Total Potential Bounties: ${sum(t['bounty'] for t in TARGETS):,}")
        
        results = []
        for target in TARGETS[:3]:  # Scan top 3
            result = self.scan_target(target)
            if result:
                results.append(result)
        
        report = self.generate_report(results)
        
        report_path = '/home/l/Desktop/AxiomTree/axiom_horizon/bug_bounty/scanner/BOUNTY_HUNT_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
        print("\n" + report[:3000])
        
        return results

if __name__ == '__main__':
    hunter = BountyHunter()
    results = hunter.hunt()
