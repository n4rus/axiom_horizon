#!/usr/bin/env python3
"""graph_tools.py — live Subgraph tools for the Axiom MCP server (ETHOnline 2026).

Continuity delta (Sept 7-13, 2026): natural-language questions over live
blockchain data via The Graph. No mocks, no static datasets — every answer
executes against a live Subgraph endpoint.

Auth: Subgraph Studio API key from the GRAPH_API_KEY environment variable.
The key is never committed (see .gitignore: .env).

Endpoint pattern:
    https://gateway-arbitrum.network.thegraph.com/api/<KEY>/subgraphs/id/<ID>
"""
from __future__ import annotations
import json
import os
import urllib.request

GRAPH_GATEWAY = "https://gateway-arbitrum.network.thegraph.com/api"


def _key() -> str:
    key = os.environ.get("GRAPH_API_KEY", "")
    if not key:
        raise RuntimeError("GRAPH_API_KEY not set — export it (never commit it)")
    return key


def graph_query(subgraph_id: str, graphql: str) -> dict:
    """Execute raw GraphQL against a live Subgraph. Returns parsed JSON."""
    url = f"{GRAPH_GATEWAY}/{_key()}/subgraphs/id/{subgraph_id}"
    payload = json.dumps({"query": graphql}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def graph_nl(question: str, subgraph_id: str, model_fn=None) -> dict:
    """Natural-language question -> GraphQL (via local model) -> live answer.

    model_fn: callable(prompt) -> graphql string. Wired to the local
    Ollama-backed agent in axiom_mcp_server.py. Returns the answer plus
    the generated query and reasoning for judge reproducibility.
    """
    if model_fn is None:
        raise RuntimeError("model_fn not wired yet (Day 2-3)")
    graphql = model_fn(
        f"Write a GraphQL query for The Graph Subgraph {subgraph_id} "
        f"that answers: {question}. Return only the query."
    )
    answer = graph_query(subgraph_id, graphql)
    return {"question": question, "graphql": graphql, "answer": answer}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: graph_tools.py <SUBGRAPH_ID> <GRAPHQL...>  (needs GRAPH_API_KEY)")
        sys.exit(2)
    print(json.dumps(graph_query(sys.argv[1], " ".join(sys.argv[2:])), indent=2)[:2000])
