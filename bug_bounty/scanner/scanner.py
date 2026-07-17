#!/usr/bin/env python3
"""
Smart Contract Vulnerability Scanner
Runs Slither + Mythril against repos to find real vulnerabilities
"""
import subprocess
import os
import json
import glob
import re
from datetime import datetime

PATH = os.environ.get('PATH', '')
os.environ['PATH'] = f"{os.path.expanduser('~')}/.foundry/bin:{PATH}"

SLITHER_PATH = '/home/l/.local/bin/slither'
MYTHRIL_PATH = '/home/l/.local/bin/myth'

class ContractScanner:
    def __init__(self, repo_url, work_dir=None):
        self.repo_url = repo_url
        self.repo_name = repo_url.split('/')[-1].replace('.git', '')
        self.work_dir = work_dir or f"/tmp/scanner_{self.repo_name}"
        self.results = {
            'repo': self.repo_url,
            'scan_time': datetime.now().isoformat(),
            'slither_findings': [],
            'mythril_findings': [],
            'manual_findings': [],
            'files_scanned': 0,
            'contracts_found': 0,
        }
    
    def clone_repo(self):
        """Clone the target repository"""
        print(f"\n[1/5] Cloning {self.repo_url}...")
        subprocess.run(['rm', '-rf', self.work_dir], capture_output=True)
        result = subprocess.run(
            ['git', 'clone', '--depth=1', self.repo_url, self.work_dir],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr[:200]}")
            return False
        print(f"  Cloned to {self.work_dir}")
        return True
    
    def find_solidity_files(self):
        """Find all Solidity files in the repo"""
        pattern = os.path.join(self.work_dir, '**/*.sol')
        files = glob.glob(pattern, recursive=True)
        # Filter out node_modules, test, and mock files
        real_files = []
        for f in files:
            rel = f.replace(self.work_dir + '/', '')
            if any(skip in rel.lower() for skip in ['node_modules', 'test/', 'mock', 'example', '.git']):
                continue
            real_files.append(f)
        return real_files
    
    def run_slither(self, sol_files):
        """Run Slither static analysis"""
        print("\n[2/5] Running Slither...")
        if not os.path.exists(SLITHER_PATH):
            print("  Slither not found, skipping")
            return
        
        for sol_file in sol_files[:5]:  # Limit to 5 files
            rel_path = sol_file.replace(self.work_dir + '/', '')
            print(f"  Scanning {rel_path}...")
            
            result = subprocess.run(
                [SLITHER_PATH, sol_file, '--json', '-', '--checklist', '--markdown-root', self.work_dir],
                capture_output=True, text=True, timeout=120
            )
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    detectors = data.get('results', {}).get('detectors', [])
                    for det in detectors:
                        finding = {
                            'type': 'Slither',
                            'impact': det.get('impact', 'unknown'),
                            'confidence': det.get('confidence', 'unknown'),
                            'check': det.get('check', 'unknown'),
                            'description': det.get('description', '')[:200],
                            'file': rel_path,
                            'function': det.get('extra', {}).get('function', 'N/A'),
                        }
                        self.results['slither_findings'].append(finding)
                except json.JSONDecodeError:
                    pass
            
            # Also capture stderr for errors
            if result.returncode != 0 and result.stderr:
                if 'error' in result.stderr.lower():
                    print(f"    Slither error: {result.stderr[:100]}")
        
        print(f"  Found {len(self.results['slither_findings'])} Slither findings")
    
    def run_mythril(self, sol_files):
        """Run Mythril symbolic execution"""
        print("\n[3/5] Running Mythril...")
        if not os.path.exists(MYTHRIL_PATH):
            print("  Mythril not found, skipping")
            return
        
        for sol_file in sol_files[:3]:  # Limit to 3 files (slower)
            rel_path = sol_file.replace(self.work_dir + '/', '')
            print(f"  Analyzing {rel_path}...")
            
            result = subprocess.run(
                [MYTHRIL_PATH, 'analyze', sol_file, '--execution-timeout', '60', '--json', '-o', '-'],
                capture_output=True, text=True, timeout=90
            )
            
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    issues = data.get('issues', [])
                    for issue in issues:
                        finding = {
                            'type': 'Mythril',
                            'title': issue.get('title', 'Unknown'),
                            'severity': issue.get('severity', 'unknown'),
                            'description': issue.get('description', '')[:200],
                            'file': rel_path,
                            'line': issue.get('line', 0),
                            'swc_id': issue.get('swc-id', 'N/A'),
                        }
                        self.results['mythril_findings'].append(finding)
                except json.JSONDecodeError:
                    pass
        
        print(f"  Found {len(self.results['mythril_findings'])} Mythril findings")
    
    def manual_review(self, sol_files):
        """Manual code review for common vulnerabilities"""
        print("\n[4/5] Running manual review...")
        
        vulnerability_patterns = [
            # Reentrancy
            (r'\.call\{.*value.*\}.*\n.*(?:state|balance|mapping)', 'Reentrancy', 'Critical'),
            (r'\.transfer\(.*\)\s*;', 'Reentrancy (transfer)', 'Medium'),
            (r'\.send\(.*\)\s*;', 'Reentrancy (send)', 'Medium'),
            
            # Access Control
            (r'function\s+\w+.*public\s+.*returns.*\{', 'Public function without access control', 'Medium'),
            (r'function\s+\w+.*external\s+.*returns.*\{', 'External function without access control', 'Medium'),
            
            # Integer Overflow
            (r'\+\+\s*|-\-\s*|\+\s*=\s*\d+\s*\*\s*\d+', 'Potential integer overflow', 'Low'),
            
            # Unchecked Return
            (r'\.call\(.*\)\s*;', 'Unchecked call return', 'Medium'),
            
            #tx.origin
            (r'tx\.origin', 'tx.origin usage', 'Medium'),
            
            # Floating pragma
            (r'pragma\s+solidity\s+\^', 'Floating pragma', 'Low'),
            
            # Delegatecall
            (r'delegatecall\(.*\)', 'Delegatecall usage', 'High'),
            
            # Selfdestruct
            (r'selfdestruct\(.*\)', 'Selfdestruct usage', 'High'),
            
            # Low-level calls
            (r'\.call\(.*\)', 'Low-level call', 'Medium'),
            
            # Unchecked math
            (r'unchecked\s*\{', 'Unchecked math', 'Low'),
            
            # Block timestamp dependence
            (r'block\.timestamp', 'Block timestamp dependence', 'Low'),
            
            # Missing event
            (r'function\s+\w+.*\bemit\b', 'Event emission', 'Info'),
        ]
        
        for sol_file in sol_files:
            rel_path = sol_file.replace(self.work_dir + '/', '')
            try:
                with open(sol_file, 'r', errors='ignore') as f:
                    content = f.read()
                
                for pattern, name, severity in vulnerability_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        context = content[max(0, match.start()-50):match.end()+50].strip()
                        
                        finding = {
                            'type': 'Manual',
                            'title': name,
                            'severity': severity,
                            'file': rel_path,
                            'line': line_num,
                            'context': context[:200],
                        }
                        self.results['manual_findings'].append(finding)
            except Exception as e:
                print(f"  Error reading {rel_path}: {e}")
        
        print(f"  Found {len(self.results['manual_findings'])} manual findings")
    
    def generate_report(self):
        """Generate final vulnerability report"""
        print("\n[5/5] Generating report...")
        
        report = []
        report.append("=" * 60)
        report.append("SMART CONTRACT VULNERABILITY REPORT")
        report.append(f"Repository: {self.repo_url}")
        report.append(f"Scan Time: {self.results['scan_time']}")
        report.append("=" * 60)
        
        # Summary
        slither_count = len(self.results['slither_findings'])
        mythril_count = len(self.results['mythril_findings'])
        manual_count = len(self.results['manual_findings'])
        total = slither_count + mythril_count + manual_count
        
        report.append(f"\n## SUMMARY")
        report.append(f"Total Findings: {total}")
        report.append(f"  Slither: {slither_count}")
        report.append(f"  Mythril: {mythril_count}")
        report.append(f"  Manual: {manual_count}")
        
        # Critical/High findings
        critical = []
        for f in self.results['slither_findings'] + self.results['mythril_findings']:
            if f.get('impact', '').lower() in ['high', 'critical'] or f.get('severity', '').lower() in ['high', 'critical']:
                critical.append(f)
        
        if critical:
            report.append(f"\n## CRITICAL/HIGH FINDINGS ({len(critical)})")
            report.append("-" * 40)
            for f in critical:
                report.append(f"\n  {f.get('type', 'Unknown')}: {f.get('check', f.get('title', 'N/A'))}")
                report.append(f"    Impact: {f.get('impact', f.get('severity', 'N/A'))}")
                report.append(f"    File: {f.get('file', 'N/A')}:{f.get('line', 'N/A')}")
                report.append(f"    Description: {f.get('description', f.get('context', 'N/A'))[:150]}")
        
        # All findings by severity
        for severity in ['Critical', 'High', 'Medium', 'Low']:
            sev_findings = []
            for f in self.results['manual_findings']:
                if f.get('severity', '').lower() == severity.lower():
                    sev_findings.append(f)
            
            if sev_findings:
                report.append(f"\n## {severity.upper()} FINDINGS ({len(sev_findings)})")
                report.append("-" * 40)
                for f in sev_findings[:10]:  # Limit output
                    report.append(f"\n  {f['title']}")
                    report.append(f"    File: {f['file']}:{f['line']}")
                    report.append(f"    Context: {f['context'][:100]}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.work_dir, 'VULNERABILITY_REPORT.md')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save JSON
        json_path = os.path.join(self.work_dir, 'scan_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nReport saved to: {report_path}")
        print(f"JSON saved to: {json_path}")
        
        return report_text
    
    def scan(self):
        """Full scan pipeline"""
        if not self.clone_repo():
            return None
        
        sol_files = self.find_solidity_files()
        if not sol_files:
            print("\nNo Solidity files found!")
            return None
        
        self.results['contracts_found'] = len(sol_files)
        self.results['files_scanned'] = len(sol_files)
        print(f"\nFound {len(sol_files)} Solidity contracts to scan")
        
        self.run_slither(sol_files)
        self.run_mythril(sol_files)
        self.manual_review(sol_files)
        
        return self.generate_report()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 scanner.py <repo_url>")
        print("Example: python3 scanner.py https://github.com/lidofinance/lido-dao")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    scanner = ContractScanner(repo_url)
    report = scanner.scan()
    
    if report:
        print("\n" + report[:3000])


if __name__ == '__main__':
    main()
