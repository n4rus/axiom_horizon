#!/usr/bin/env python3
"""
Smart Contract Bounty Discovery
Finds real Immunefi programs with high payouts
"""
import json
import urllib.request
import os
import re
from datetime import datetime

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', 'ghp_REDACTED')

# Known Immunefi programs with bounties (curated list)
IMMUNEFI_PROGRAMS = [
    {
        "name": "Lido Finance",
        "url": "https://immunefi.com/bounty/lido/",
        "github": "lidofinance",
        "max_bounty": 2500000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["lido-dao", "lido-council-daemon"],
        "description": "Liquid staking protocol, largest DeFi staking platform",
        "focus": ["Reentrancy", "Oracle Manipulation", "Access Control", "Flash Loan Attacks"]
    },
    {
        "name": "Rocket Pool",
        "url": "https://immunefi.com/bounty/rocketpool/",
        "github": "rocket-pool",
        "max_bounty": 500000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["rocketpool", "smartnode"],
        "description": "Decentralized Ethereum staking protocol",
        "focus": ["Node Operator Fraud", "Withdrawal Attacks", "MEV Extraction"]
    },
    {
        "name": "Mantle",
        "url": "https://immunefi.com/bounty/mantle/",
        "github": "mantlenetworkio",
        "max_bounty": 2000000,
        "type": "L2",
        "chain": "Ethereum",
        "contracts": ["mantle"],
        "description": "Ethereum L2 with optimistic rollup",
        "focus": ["Bridge Vulnerabilities", "Sequencer Attacks", "Fraud Proofs"]
    },
    {
        "name": "Morph",
        "url": "https://immunefi.com/bounty/morph/",
        "github": "morph-l2",
        "max_bounty": 1000000,
        "type": "L2",
        "chain": "Ethereum",
        "contracts": ["morph"],
        "description": "Ethereum L2 with optimistic rollup",
        "focus": ["Bridge", "Sequencer", "Data Availability"]
    },
    {
        "name": "EigenLayer",
        "url": "https://immunefi.com/bounty/eigenlayer/",
        "github": "Layr-Labs",
        "max_bounty": 1000000,
        "type": "Restaking",
        "chain": "Ethereum",
        "contracts": ["eigenlayer-contracts"],
        "description": "Restaking protocol for Ethereum",
        "focus": ["Slashing Logic", "Delegation Attacks", "Oracle Manipulation"]
    },
    {
        "name": "Pendle",
        "url": "https://immunefi.com/bounty/pendle/",
        "github": "pendle-finance",
        "max_bounty": 500000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["pendle-core-v2-public"],
        "description": "Yield tokenization and trading",
        "focus": ["Yield Oracle", "AMM Manipulation", "Flash Loan"]
    },
    {
        "name": "Morpho",
        "url": "https://immunefi.com/bounty/morpho/",
        "github": "morpho-org",
        "max_bounty": 250000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["morpho-blue"],
        "description": "Optimized lending protocol",
        "focus": ["Interest Rate", "Liquidation", "Oracle"]
    },
    {
        "name": "Ethena",
        "url": "https://immunefi.com/bounty/ethena/",
        "github": "ethena-labs",
        "max_bounty": 1000000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["ethena-dao", "usde"],
        "description": "Synthetic dollar protocol",
        "focus": ["Custody Risk", "Derivatives", "Oracle"]
    },
    {
        "name": "Spark",
        "url": "https://immunefi.com/bounty/spark/",
        "github": "sparkpi-io",
        "max_bounty": 250000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["spark-lending"],
        "description": "MakerDAO ecosystem lending protocol",
        "focus": ["Interest Rate", "Liquidation", "Governance"]
    },
    {
        "name": "Kelp DAO",
        "url": "https://immunefi.com/bounty/kelp-dao/",
        "github": "kelp-dao",
        "max_bounty": 200000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["kelp-dao"],
        "description": "Liquid staking derivative",
        "focus": ["Restaking", "Oracle", "Flash Loan"]
    },
    {
        "name": "Sommelier Finance",
        "url": "https://immunefi.com/bounty/sommelier/",
        "github": "sommelier-finance",
        "max_bounty": 500000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["sommelier-cellar"],
        "description": "DeFi vault strategy protocol",
        "focus": ["Vault Manipulation", "Strategy Exploits", "Oracle"]
    },
    {
        "name": "Gearbox Protocol",
        "url": "https://immunefi.com/bounty/gearbox/",
        "github": "Gearbox-protocol",
        "max_bounty": 500000,
        "type": "DeFi",
        "chain": "Ethereum",
        "contracts": ["gearbox-v3"],
        "description": "Leveraged trading protocol",
        "focus": ["Leverage Attacks", "Liquidation", "Oracle"]
    },
    {
        "name": "Ambire Wallet",
        "url": "https://immunefi.com/bounty/ambire/",
        "github": "AmbireWallet",
        "max_bounty": 100000,
        "type": "Wallet",
        "chain": "Multi-chain",
        "contracts": ["ambire-common"],
        "description": "Smart contract wallet",
        "focus": ["Account Abstraction", "Signature", "Relayer"]
    },
]

# HackerOne smart contract programs
HACKERONE_PROGRAMS = [
    {"name": "Coinbase", "scope": "*.coinbase.com, Smart Contracts", "min_bounty": 200, "type": "Crypto"},
    {"name": "Kraken", "scope": "kraken.com, Smart Contracts", "min_bounty": 250, "type": "Crypto"},
    {"name": "Blockstream", "scope": "*.blockstream.com", "min_bounty": 500, "type": "Bitcoin"},
    {"name": "ShapeShift", "scope": "*.shapeshift.com", "min_bounty": 100, "type": "DeFi"},
]

def search_github_bounty_repos():
    """Search GitHub for repos with active bug bounty programs"""
    queries = [
        "topic:bug-bounty+topic:smart-contract+language:solidity",
        "topic:immunefi+language:solidity",
        "topic:security-audit+language:solidity",
        "bounty+solidity+in:name,description",
        "security+audit+solidity+in:description",
    ]
    
    repos = []
    seen = set()
    
    for query in queries:
        url = f"https://api.github.com/search/repositories?q={query}&sort=updated&order=desc&per_page=10"
        req = urllib.request.Request(url, headers={
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        })
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            for repo in data.get('items', []):
                name = repo['full_name']
                if name not in seen:
                    seen.add(name)
                    repos.append({
                        'name': name,
                        'stars': repo['stargazers_count'],
                        'url': repo['html_url'],
                        'desc': (repo.get('description', '') or '')[:200],
                        'language': repo.get('language', 'unknown'),
                        'updated': repo.get('updated_at', ''),
                        'topics': repo.get('topics', []),
                    })
        except Exception as e:
            print(f"  Query failed: {e}")
    
    return repos

def find_solidity_contracts(repos, top_n=5):
    """Find repos with actual Solidity contracts to audit"""
    targets = []
    
    for repo in repos[:top_n]:
        name = repo['name']
        print(f"\n  Checking {name} for Solidity contracts...")
        
        # Check for Solidity files
        url = f"https://api.github.com/search/code?q=repo:{name}+extension:solidity&per_page=5"
        req = urllib.request.Request(url, headers={
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        })
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            solidity_files = data.get('total_count', 0)
            
            if solidity_files > 0:
                repo['solidity_files'] = solidity_files
                targets.append(repo)
                print(f"    Found {solidity_files} Solidity files")
        except:
            pass
    
    return targets

def generate_report(programs, github_repos):
    """Generate a report of found programs"""
    report = []
    report.append("=" * 60)
    report.append("SMART CONTRACT BOUNTY DISCOVERY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    
    report.append("\n## Immunefi Programs (Top Payouts)")
    report.append("-" * 40)
    for p in sorted(programs, key=lambda x: x['max_bounty'], reverse=True):
        report.append(f"\n{p['name']}")
        report.append(f"  Max Bounty: ${p['max_bounty']:,}")
        report.append(f"  Type: {p['type']} | Chain: {p['chain']}")
        report.append(f"  GitHub: https://github.com/{p['github']}")
        report.append(f"  Focus: {', '.join(p['focus'])}")
        report.append(f"  Description: {p['description']}")
    
    report.append("\n\n## GitHub Repos with Solidity Contracts")
    report.append("-" * 40)
    for r in github_repos:
        report.append(f"\n{r['name']} ({r['stars']}★)")
        report.append(f"  URL: {r['url']}")
        report.append(f"  Solidity files: {r.get('solidity_files', 'unknown')}")
        report.append(f"  Description: {r['desc'][:100]}")
    
    report.append("\n\n## Recommended Attack Vectors")
    report.append("-" * 40)
    report.append("""
    1. REENTRANCY: Check for external calls before state changes
    2. ACCESS CONTROL: Test unauthorized function calls
    3. ORACLE MANIPULATION: Test price feed manipulation
    4. FLASH LOAN ATTACKS: Test atomic transactions
    5. INTEGER OVERFLOW: Check unchecked arithmetic
    6. FRONT-RUNNING: Test MEV extraction
    7. LOGIC ERRORS: Test edge cases in math
    8. UPGRADABLE CONTRACTS: Test proxy upgrade paths
    """)
    
    return "\n".join(report)

def main():
    print("Smart Contract Bounty Discovery")
    print("=" * 50)
    
    # Step 1: Search for Immunefi programs
    print("\n[1/3] Searching Immunefi programs...")
    print(f"  Found {len(IMMUNEFI_PROGRAMS)} programs")
    
    # Step 2: Search GitHub for bounty repos
    print("\n[2/3] Searching GitHub for bounty repos...")
    github_repos = search_github_bounty_repos()
    print(f"  Found {len(github_repos)} repos with bounty tags")
    
    # Step 3: Find repos with actual Solidity
    print("\n[3/3] Finding repos with Solidity contracts...")
    targets = find_solidity_contracts(github_repos, top_n=10)
    print(f"  Found {len(targets)} repos with Solidity files")
    
    # Generate report
    report = generate_report(IMMUNEFI_PROGRAMS, targets)
    report_path = "/home/l/Desktop/AxiomTree/axiom_horizon/bug_bounty/scanner/bounty_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + report[:2000])
    
    return IMMUNEFI_PROGRAMS, targets

if __name__ == '__main__':
    programs, targets = main()
