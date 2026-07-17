"""wallet.py — crypto wallet monitoring + funding path for the AGI earning loop.

The validation of AGI is not just "PRs opened" but "wallet climbs": real ETH
received. This module checks the Base-mainnet balance of the agent's wallet and
documents the funding path (Sepolia testnet faucet -> real ETH).
"""

from __future__ import annotations
import json, os, urllib.request
from pathlib import Path

# From NEW_PR session: agent wallet + payout destination.
WALLET = os.environ.get('AGI_WALLET', '0x8f36105eE73b4Aadc0Cf5301A756378F49eB0eb5')
DEST = os.environ.get('AGI_DEST', '0xD0b864545a5b6CA2654e46bEcb5602946E3AEbb7')

# Public Base RPC endpoints (no key required for eth_getBalance / eth_call).
# mainnet.base.org returns 403 to generic clients, so we prefer ones that work.
BASE_RPC = os.environ.get('BASE_RPC', 'https://1rpc.io/base')
_RPC_FALLBACKS = [
    'https://1rpc.io/base',
    'https://base.meowrpc.com',
    'https://rpc.ankr.com/base',
    'https://mainnet.base.org',
]

# Base Sepolia faucet (testnet ETH) — first step before real mainnet funding.
SEPOLIA_FAUCET = 'https://www.coinbase.com/faucets/base-sepolia-faucet'


def _rpc(method: str, params: list) -> object:
    payload = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
    endpoints = [BASE_RPC] + [e for e in _RPC_FALLBACKS if e != BASE_RPC]
    last_err = None
    for ep in endpoints:
        try:
            req = urllib.request.Request(
                ep, data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json',
                         'User-Agent': 'AxiomAGI/1.0'}, method='POST')
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read())
            if 'result' in resp:
                return resp['result']
            last_err = resp.get('error')
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise RuntimeError(str(last_err))
    return None


def balance_eth(addr: str = WALLET) -> float:
    """Return native ETH balance of an address on Base (mainnet)."""
    try:
        hexbal = _rpc('eth_getBalance', [addr, 'latest'])
        if not hexbal:
            return 0.0
        wei = int(hexbal, 16)
        return wei / 1e18
    except Exception:
        return -1.0  # signal error


def status() -> dict:
    return {
        'wallet': WALLET,
        'dest': DEST,
        'network': 'base-mainnet',
        'balance_eth': balance_eth(),
        'rpc': BASE_RPC,
        'sepolia_faucet': SEPOLIA_FAUCET,
    }


def funding_path() -> str:
    """The concrete steps to make the wallet climb from zero."""
    return (
        f"1. Get Base-Sepolia testnet ETH from the faucet: {SEPOLIA_FAUCET}\n"
        f"2. Bridge Sepolia -> Base mainnet (or fund mainnet directly).\n"
        f"3. Agent wallet ({WALLET}) receives ETH as bounties/services pay out.\n"
        f"4. Payouts forward to dest ({DEST}).\n"
        f"5. Validation = balance_eth() > 0 on Base mainnet."
    )


if __name__ == '__main__':
    s = status()
    print(json.dumps(s, indent=2))
    print()
    print(funding_path())
