"""
NodeProtocol — peer-to-peer cluster chain for the Axiom Network.

Defines how agents discover each other, negotiate compute/energy trades,
and settle using the compute-backed gasless token.

Protocol primitives (all operations use local simulation by default):
  - DISCOVER: find peers on the network
  - OFFER: advertise available compute/power
  - REQUEST: ask a peer for compute/power
  - TRADE: execute a swap (compute ↔ tokens ↔ energy ↔ currency)
  - SETTLE: finalize a trade on both ledgers

When connected to real infrastructure, these map to UDP broadcasts
matching the PoGIE 128-byte frame format (defined in the hardware manual).
"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from grid_bridge import BRIDGE

# ---------------------------------------------------------------------------
# Protocol types
# ---------------------------------------------------------------------------


class MessageType(Enum):
    DISCOVER = "discover"
    OFFER = "offer"
    REQUEST = "request"
    TRADE = "trade"
    SETTLE = "settle"
    ACK = "ack"


class ResourceType(Enum):
    COMPUTE = "compute"       # TFLOPS-seconds
    ENERGY = "energy"         # kWh
    TOKENS = "tokens"         # compute-backed tokens
    CURRENCY = "currency"     # fiat/stablecoin (future)


@dataclass
class NodeIdentity:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    address: str = "localhost"
    port: int = 0
    capabilities: List[str] = field(default_factory=lambda: ["compute", "energy", "trading"])
    public_key: str = ""  # ECDSA key from ATECC608A (future)


@dataclass
class TradeOffer:
    offer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    resource: ResourceType = ResourceType.COMPUTE
    quantity: float = 0.0  # TFLOPS-seconds or kWh
    price_tokens: float = 0.0
    node_id: str = ""
    ttl_seconds: int = 60


@dataclass
class TradeAgreement:
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    seller: str = ""
    buyer: str = ""
    resource: ResourceType = ResourceType.COMPUTE
    quantity: float = 0.0
    price: float = 0.0
    status: str = "pending"  # pending | active | settled | disputed
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Local cluster state
# ---------------------------------------------------------------------------

_local_identity = NodeIdentity()
_pending_offers: Dict[str, TradeOffer] = {}
_active_trades: Dict[str, TradeAgreement] = {}
_known_nodes: Dict[str, NodeIdentity] = {}


# ---------------------------------------------------------------------------
# Protocol operations
# ---------------------------------------------------------------------------


def set_identity(node_id: str = "", address: str = "localhost"):
    global _local_identity
    if node_id:
        _local_identity.node_id = node_id
    _local_identity.address = address


def my_id() -> str:
    return _local_identity.node_id


def discover() -> List[Dict[str, Any]]:
    """Return all known peer nodes."""
    return [
        {"node_id": nid, "address": n.address, "capabilities": n.capabilities}
        for nid, n in _known_nodes.items()
    ]


def register_peer(node_id: str, address: str, capabilities: List[str] = None) -> str:
    _known_nodes[node_id] = NodeIdentity(
        node_id=node_id,
        address=address,
        capabilities=capabilities or ["compute"],
    )
    BRIDGE.register_peer(node_id, address, capabilities)
    return f"Peer {node_id} known. Total: {len(_known_nodes)}"


def offer_compute(tflops: float, duration_hours: float, price_per_tflop_hour: float) -> str:
    """Advertise available compute for trade."""
    o = TradeOffer(
        resource=ResourceType.COMPUTE,
        quantity=tflops * duration_hours,
        price_tokens=price_per_tflop_hour * tflops * duration_hours,
        node_id=_local_identity.node_id,
    )
    _pending_offers[o.offer_id] = o
    return (
        f"Offering {tflops} TFLOPS for {duration_hours}h "
        f"at {price_per_tflop_hour} tokens/TFLOPS-hour "
        f"[offer {o.offer_id}]"
    )


def offer_energy(kwh: float, price_per_kwh: float) -> str:
    """Advertise available energy for trade."""
    o = TradeOffer(
        resource=ResourceType.ENERGY,
        quantity=kwh,
        price_tokens=price_per_kwh * kwh,
        node_id=_local_identity.node_id,
    )
    _pending_offers[o.offer_id] = o
    return f"Offering {kwh} kWh at {price_per_kwh} tokens/kWh [offer {o.offer_id}]"


def request_compute(tflops: float, duration_hours: float, max_price: float) -> List[Dict[str, Any]]:
    """Find offers matching a compute request."""
    needed = tflops * duration_hours
    matches = []
    for oid, o in _pending_offers.items():
        if o.resource == ResourceType.COMPUTE and o.quantity >= needed:
            if o.price_tokens <= max_price * needed:
                matches.append({
                    "offer_id": o.offer_id,
                    "node": o.node_id,
                    "quantity_TFLOPS_hours": o.quantity,
                    "price_tokens": o.price_tokens,
                })
    return matches


def execute_trade(offer_id: str, buyer_id: str = "") -> str:
    """Accept an offer and settle the trade."""
    offer = _pending_offers.get(offer_id)
    if not offer:
        return f"Offer {offer_id} not found"

    buyer = buyer_id or _local_identity.node_id
    trade = TradeAgreement(
        seller=offer.node_id,
        buyer=buyer,
        resource=offer.resource,
        quantity=offer.quantity,
        price=offer.price_tokens,
        status="active",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Simulate settlement: buyer pays seller
    if trade.price > 0:
        if buyer == _local_identity.node_id:
            ok = BRIDGE.compute.transfer(trade.price, offer.node_id, f"Trade {trade.trade_id}")
            if not ok:
                trade.status = "failed"
                return f"Insufficient tokens for trade {trade.trade_id}"
        # If buyer is remote, we'd notify them (future)

    _pending_offers.pop(offer_id, None)
    _active_trades[trade.trade_id] = trade

    return (
        f"Trade {trade.trade_id}: {trade.quantity} {trade.resource.value} "
        f"for {trade.price} tokens. "
        f"Status: {trade.status}"
    )


def settle_trade(trade_id: str, proof: str = "") -> str:
    """Finalize a completed trade."""
    trade = _active_trades.get(trade_id)
    if not trade:
        return f"Trade {trade_id} not found"
    trade.status = "settled"
    return f"Trade {trade_id} settled."


def pending_offers() -> List[Dict[str, Any]]:
    return [
        {"offer_id": oid, "node": o.node_id, "resource": o.resource.value,
         "quantity": o.quantity, "price": o.price_tokens}
        for oid, o in _pending_offers.items()
    ]


def active_trades() -> List[Dict[str, Any]]:
    return [
        {"trade_id": tid, "seller": t.seller, "buyer": t.buyer,
         "resource": t.resource.value, "quantity": t.quantity,
         "price": t.price, "status": t.status}
        for tid, t in _active_trades.items()
    ]


def full_status() -> str:
    lines = [
        "=== NODE PROTOCOL STATUS ===",
        f"  Node ID:   {_local_identity.node_id}",
        f"  Address:   {_local_identity.address}:{_local_identity.port}",
        f"  Capabilities: {', '.join(_local_identity.capabilities)}",
        f"  Known peers:  {len(_known_nodes)}",
        f"  Offers open:  {len(_pending_offers)}",
        f"  Active trades:{len(_active_trades)}",
        "",
    ]
    for nid, n in _known_nodes.items():
        lines.append(f"  Peer {nid}: {n.address} [{', '.join(n.capabilities)}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schemas for the agent
# ---------------------------------------------------------------------------

NODE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "node_discover",
            "description": "List all known peer nodes on the cluster chain.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "node_register_peer",
            "description": "Register a peer node on the cluster chain for compute/energy trading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Peer node identifier"},
                    "address": {"type": "string", "description": "Network address"},
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Capabilities: compute, energy, trading",
                    },
                },
                "required": ["node_id", "address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "offer_compute",
            "description": "Advertise available compute power to the cluster for tokens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tflops": {"type": "number", "description": "TFLOPS available"},
                    "duration_hours": {"type": "number", "description": "Hours available"},
                    "price_per_tflop_hour": {"type": "number", "description": "Token price per TFLOPS-hour"},
                },
                "required": ["tflops", "duration_hours", "price_per_tflop_hour"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "offer_energy",
            "description": "Advertise available energy (kWh) for trade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kwh": {"type": "number", "description": "kWh available"},
                    "price_per_kwh": {"type": "number", "description": "Token price per kWh"},
                },
                "required": ["kwh", "price_per_kwh"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_compute",
            "description": "Search for available compute matching your needs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tflops": {"type": "number", "description": "TFLOPS needed"},
                    "duration_hours": {"type": "number", "description": "Hours needed"},
                    "max_price": {"type": "number", "description": "Max tokens per TFLOPS-hour"},
                },
                "required": ["tflops", "duration_hours", "max_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_trade",
            "description": "Accept an offer and execute a trade. Tokens transfer automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_id": {"type": "string", "description": "Offer ID to accept"},
                },
                "required": ["offer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "node_status",
            "description": "Show cluster chain status: node ID, peers, open offers, active trades.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

NODE_TOOL_DISPATCH = {
    "node_discover": lambda args: json.dumps(discover(), indent=2) or "No peers found.",
    "node_register_peer": lambda args: register_peer(
        args["node_id"], args["address"], args.get("capabilities", [])
    ),
    "offer_compute": lambda args: offer_compute(
        args["tflops"], args["duration_hours"], args["price_per_tflop_hour"]
    ),
    "offer_energy": lambda args: offer_energy(args["kwh"], args["price_per_kwh"]),
    "request_compute": lambda args: (
        r := request_compute(args["tflops"], args["duration_hours"], args["max_price"]),
        json.dumps(r, indent=2) if r else "No matching offers found."
    )[-1],
    "execute_trade": lambda args: execute_trade(args["offer_id"]),
    "node_status": lambda args: full_status(),
}
