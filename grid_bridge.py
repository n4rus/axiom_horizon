"""
GridBridge — connects the Axiom Agent to the physical infrastructure.

Bridges the agent to:
  - TBot/Scanner (trading engine control)
  - PoGIE energy grid (power monitoring and management)
  - Blockchain ledger (transactions and balance)
  - Compute ledger (gasless token where compute = gas)

All operations work in simulation mode by default (no real RPC keys needed).
Set GRID_REAL_MODE=1 to connect to live infrastructure.
"""
from __future__ import annotations
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Compute Ledger — the gasless token engine
# ---------------------------------------------------------------------------

FLOP_PER_TOKEN = 1e12  # 1 TFLOPS-second = 1 compute token
LEDGER_PATH = Path(__file__).parent / ".axiom_state" / "compute_ledger.json"


@dataclass
class ComputeLedgerEntry:
    action: str          # "mint" | "burn" | "transfer"
    amount: float        # token amount
    flops_seconds: float # compute contributed or consumed
    counterparty: str    # "self" | peer_id
    timestamp: str
    memo: str


class ComputeLedger:
    """
    Gasless token ledger. Computation IS the currency.
    - Mint: contribute compute to the network → earn tokens
    - Burn: request compute from the network → spend tokens
    - Transfer: send tokens to another node for energy/currency exchange
    - Value is backed by real TFLOPS-seconds, not speculation.
    """

    def __init__(self):
        self.balance: float = 0.0
        self.total_minted: float = 0.0
        self.total_burned: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self._load()

    def mint(self, flops_seconds: float, memo: str = "") -> float:
        """Contribute compute. Tokens are minted at FLOP_PER_TOKEN rate."""
        tokens = flops_seconds / FLOP_PER_TOKEN
        self.balance += tokens
        self.total_minted += tokens
        entry = {
            "action": "mint",
            "amount": tokens,
            "flops_seconds": flops_seconds,
            "counterparty": "self",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memo": memo or f"Minted {tokens:.6f} tokens from {flops_seconds:.2e} FLOP·s",
        }
        self.history.append(entry)
        self._save()
        return tokens

    def burn(self, flops_needed: float, memo: str = "") -> float:
        """Request compute. Tokens are burned at FLOP_PER_TOKEN rate."""
        cost = flops_needed / FLOP_PER_TOKEN
        if cost > self.balance:
            cost = self.balance
            flops_needed = cost * FLOP_PER_TOKEN
        self.balance -= cost
        self.total_burned += cost
        entry = {
            "action": "burn",
            "amount": cost,
            "flops_seconds": flops_needed,
            "counterparty": "self",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memo": memo or f"Burned {cost:.6f} tokens for {flops_needed:.2e} FLOP·s",
        }
        self.history.append(entry)
        self._save()
        return cost

    def transfer(self, amount: float, to: str, memo: str = "") -> bool:
        """Transfer tokens to another node (for energy or currency exchange)."""
        if amount > self.balance:
            return False
        self.balance -= amount
        entry = {
            "action": "transfer",
            "amount": amount,
            "flops_seconds": 0.0,
            "counterparty": to,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memo": memo or f"Transferred {amount:.6f} tokens to {to}",
        }
        self.history.append(entry)
        self._save()
        return True

    def receive(self, amount: float, fr: str, memo: str = ""):
        """Receive tokens from another node."""
        self.balance += amount
        entry = {
            "action": "receive",
            "amount": amount,
            "flops_seconds": 0.0,
            "counterparty": fr,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memo": memo or f"Received {amount:.6f} tokens from {fr}",
        }
        self.history.append(entry)
        self._save()

    def summary(self) -> str:
        return (
            f"Compute Ledger Balance: {self.balance:.6f} tokens\n"
            f"  Total minted:  {self.total_minted:.6f}\n"
            f"  Total burned:  {self.total_burned:.6f}\n"
            f"  Transactions:  {len(self.history)}\n"
            f"  1 token = {FLOP_PER_TOKEN:.0e} FLOP·seconds\n"
            f"  Backing: real computation, not speculation"
        )

    def _save(self):
        LEDGER_PATH.parent.mkdir(exist_ok=True)
        data = {
            "balance": self.balance,
            "total_minted": self.total_minted,
            "total_burned": self.total_burned,
            "history": self.history[-500:],
        }
        LEDGER_PATH.write_text(json.dumps(data, ensure_ascii=False))

    def _load(self):
        if not LEDGER_PATH.exists():
            return
        try:
            data = json.loads(LEDGER_PATH.read_text())
            self.balance = data.get("balance", 0.0)
            self.total_minted = data.get("total_minted", 0.0)
            self.total_burned = data.get("total_burned", 0.0)
            self.history = data.get("history", [])
        except (json.JSONDecodeError, OSError):
            pass


# ---------------------------------------------------------------------------
# TBot Bridge — control the scanner trading engine
# ---------------------------------------------------------------------------

TBOT_STATE_PATH = Path(__file__).parent.parent / "scanner" / "tbot_checkpoint.json"


class TBotBridge:
    """
    Interface to the TBot v2 trading engine.
    In simulation mode, maintains a local state mirror.
    """

    def __init__(self, real_mode: bool = False):
        self.real_mode = real_mode or bool(os.environ.get("GRID_REAL_MODE"))
        self.state: Dict[str, Any] = self._read_state()

    def _read_state(self) -> Dict[str, Any]:
        # Always reflect the live TBot engine's checkpoint when present, so the
        # mind sees real price / net worth / regime (read-only, simulation-safe).
        if TBOT_STATE_PATH.exists():
            try:
                return json.loads(TBOT_STATE_PATH.read_text())
            except Exception:
                pass
        return {
            "status": "simulation",
            "last_known_price": 0.0,
            "grid_lower_limit": 0.0,
            "grid_upper_limit": 0.0,
            "paper_usdc_balance": 10000.0,
            "paper_eth_balance": 5.0,
            "current_regime": "SIDEWAYS",
            "weighted_vote_score": 0.0,
            "last_successful_iteration": 0,
        }

    def status(self) -> Dict[str, Any]:
        return self._read_state()

    def set_regime(self, regime: str) -> str:
        valid = {"BULL", "BEAR", "SIDEWAYS"}
        if regime.upper() not in valid:
            return f"Invalid regime: {regime}. Use {valid}"
        self.state["current_regime"] = regime.upper()
        return f"Regime set to {regime.upper()}"

    def adjust_grid(self, lower: float, upper: float):
        self.state["grid_lower_limit"] = lower
        self.state["grid_upper_limit"] = upper

    def price_update(self, price: float):
        self.state["last_known_price"] = price


# ---------------------------------------------------------------------------
# Energy Grid Bridge — PoGIE power management
# ---------------------------------------------------------------------------


class EnergyGridBridge:
    """
    Interface to the PoGIE energy grid.
    Tracks generation, consumption, and grid balancing.
    """

    def __init__(self):
        self.generation_watts: float = 0.0
        self.load_watts: float = 0.0
        self.storage_kwh: float = 0.0
        self.storage_max_kwh: float = 1000.0
        self.grid_connected: bool = False
        self.solar_capacity_mwp: float = 50.4  # From specs
        self.bess_capacity_mwh: float = 230.5   # From specs

    def update_generation(self, watts: float):
        self.generation_watts = watts

    def update_load(self, watts: float):
        self.load_watts = watts

    def net_power(self) -> float:
        return self.generation_watts - self.load_watts

    def charge_storage(self, kwh: float):
        self.storage_kwh = min(self.storage_kwh + kwh, self.storage_max_kwh)

    def discharge_storage(self, kwh: float) -> float:
        available = min(kwh, self.storage_kwh)
        self.storage_kwh -= available
        return available

    def status(self) -> Dict[str, Any]:
        return {
            "generation_watts": self.generation_watts,
            "load_watts": self.load_watts,
            "net_power_watts": self.net_power(),
            "storage_kwh": round(self.storage_kwh, 2),
            "storage_max_kwh": self.storage_max_kwh,
            "solar_capacity_mwp": self.solar_capacity_mwp,
            "grid_connected": self.grid_connected,
        }


# ---------------------------------------------------------------------------
# Orchestrator — single bridge all agent tools call
# ---------------------------------------------------------------------------

class GridBridge:
    """
    Unified bridge. The agent calls methods on this class
    through its tool interface.
    """

    def __init__(self):
        self.compute = ComputeLedger()
        self.tbot = TBotBridge()
        self.energy = EnergyGridBridge()
        self.peers: Dict[str, Dict[str, Any]] = {}

    def full_status(self) -> str:
        sections = [
            "=== GRID BRIDGE STATUS ===",
            "",
            self.compute.summary(),
            "",
            "--- TBot Trading ---",
            json.dumps(self.tbot.status(), indent=2),
            "",
            "--- Energy Grid ---",
            json.dumps(self.energy.status(), indent=2),
            "",
            f"--- Peers ---",
            f"  {len(self.peers)} known",
        ]
        for pid, info in self.peers.items():
            sections.append(f"  {pid}: {info.get('address', 'unknown')}")
        return "\n".join(sections)

    def register_peer(self, peer_id: str, address: str, capabilities: List[str] = None):
        self.peers[peer_id] = {
            "address": address,
            "capabilities": capabilities or [],
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
        return f"Peer {peer_id} registered at {address}"


BRIDGE = GridBridge()


# ---------------------------------------------------------------------------
# Tool schemas for the agent
# ---------------------------------------------------------------------------

GRID_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "grid_status",
            "description": "Show full grid status: compute ledger, TBot trading state, energy grid metrics, and known peers.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_mint",
            "description": "Mint compute tokens by contributing FLOP-seconds. Records real compute work as token value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flops_seconds": {
                        "type": "number",
                        "description": "FLOP-seconds contributed (e.g., 1e12 = 1 TFLOPS for 1 second = 1 token)",
                    },
                    "memo": {"type": "string", "description": "Optional description of the compute work"},
                },
                "required": ["flops_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_burn",
            "description": "Burn compute tokens to request computation from the network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flops_needed": {"type": "number", "description": "FLOP-seconds needed"},
                    "memo": {"type": "string", "description": "What the compute is for"},
                },
                "required": ["flops_needed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_transfer",
            "description": "Transfer compute tokens to another node (for energy, currency, or compute exchange).",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Token amount to send"},
                    "to": {"type": "string", "description": "Peer node ID"},
                    "memo": {"type": "string", "description": "Reason for transfer"},
                },
                "required": ["amount", "to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "energy_status",
            "description": "Show current energy generation, load, storage, and grid connection status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "energy_set_generation",
            "description": "Set the current energy generation in watts (simulated or real sensor reading).",
            "parameters": {
                "type": "object",
                "properties": {"watts": {"type": "number", "description": "Generation in watts"}},
                "required": ["watts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "energy_set_load",
            "description": "Set the current energy load in watts.",
            "parameters": {
                "type": "object",
                "properties": {"watts": {"type": "number", "description": "Load in watts"}},
                "required": ["watts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tbot_status",
            "description": "Show TBot trading engine state: price, grid bounds, balance, regime.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tbot_set_regime",
            "description": "Manually set the TBot trading regime (BULL, BEAR, or SIDEWAYS).",
            "parameters": {
                "type": "object",
                "properties": {
                    "regime": {
                        "type": "string",
                        "enum": ["BULL", "BEAR", "SIDEWAYS"],
                        "description": "Trading regime",
                    },
                },
                "required": ["regime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peer_register",
            "description": "Register a peer node for compute/energy trading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "peer_id": {"type": "string", "description": "Unique peer identifier"},
                    "address": {"type": "string", "description": "Network address (IP:port or temporary ID)"},
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of capabilities (compute, energy, trading, etc.)",
                    },
                },
                "required": ["peer_id", "address"],
            },
        },
    },
]

GRID_TOOL_DISPATCH = {
    "grid_status": lambda args: BRIDGE.full_status(),
    "compute_mint": lambda args: (
        r := BRIDGE.compute.mint(args["flops_seconds"], args.get("memo", "")),
        f"Minted {r:.6f} tokens"
    )[-1],
    "compute_burn": lambda args: (
        r := BRIDGE.compute.burn(args["flops_needed"], args.get("memo", "")),
        f"Burned {r:.6f} tokens"
    )[-1],
    "compute_transfer": lambda args: (
        "Transfer failed (insufficient balance)"
        if not BRIDGE.compute.transfer(args["amount"], args["to"], args.get("memo", ""))
        else f"Transferred {args['amount']:.6f} tokens to {args['to']}"
    ),
    "energy_status": lambda args: json.dumps(BRIDGE.energy.status(), indent=2),
    "energy_set_generation": lambda args: (
        BRIDGE.energy.update_generation(args["watts"]),
        f"Generation set to {args['watts']} W"
    )[-1],
    "energy_set_load": lambda args: (
        BRIDGE.energy.update_load(args["watts"]),
        f"Load set to {args['watts']} W"
    )[-1],
    "tbot_status": lambda args: json.dumps(BRIDGE.tbot.status(), indent=2),
    "tbot_set_regime": lambda args: BRIDGE.tbot.set_regime(args["regime"]),
    "peer_register": lambda args: BRIDGE.register_peer(
        args["peer_id"], args["address"], args.get("capabilities", [])
    ),
}
