"""Web-search capability for Kai (keyless by default).

Gives the Kai Insight product (and optionally the KaiMind daemon) the ability to
query the live web for current information — raising Kai's effective capacity
beyond its local training/embeddings.

Backends (select via ``KAI_WEBSEARCH_PROVIDER`` env var):
  - ``duckduckgo`` (default) — HTML endpoint, **no API key required**.
  - ``exa`` / ``tavily`` / ``brave`` — hosted providers; supply the matching
    ``EXA_API_KEY`` / ``TAVILY_API_KEY`` / ``BRAVE_API_KEY`` env var.

Read-only: performs only outbound GET requests; never mutates local state.
Network failures degrade gracefully to an error result (never raises).
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List

PROVIDER = os.environ.get("KAI_WEBSEARCH_PROVIDER", "duckduckgo").lower()

_UA = {"User-Agent": "Mozilla/5.0 (KaiInsight/1.0)"}


def _http_get(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


def _duckduckgo(query: str, n: int) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    html = _http_get(url)
    links = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
    out: List[Dict[str, str]] = []
    for i, (href, title) in enumerate(links[:n]):
        snip = _clean(snippets[i]) if i < len(snippets) else ""
        out.append({
            "title": _clean(title),
            "url": urllib.parse.unquote(href),
            "snippet": snip,
        })
    return out


_WIKI_UA = "KaiInsight/1.0 (https://example.com/kai; contact@example.com)"


def _wikipedia(query: str, n: int) -> List[Dict[str, str]]:
    """Keyless fallback (no challenge): Wikipedia full-text search API."""
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
           "&format=json&srlimit=" + str(n) + "&srsearch=" +
           urllib.parse.quote(query))
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8", "ignore"))
    out: List[Dict[str, str]] = []
    for it in data.get("query", {}).get("search", [])[:n]:
        out.append({
            "title": it.get("title", ""),
            "url": "https://en.wikipedia.org/wiki/" +
                  urllib.parse.quote(it.get("title", "").replace(" ", "_")),
            "snippet": _clean(it.get("snippet", "")),
        })
    return out


def _hosted_json(provider: str, endpoint: str, key: str, query: str,
                 n: int, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={**_UA, "Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=12) as r:
        data = json.loads(r.read().decode("utf-8", "ignore"))
    # Normalise per provider.
    if provider == "exa":
        items = data.get("results", [])
        return [{"title": it.get("title", ""),
                 "url": it.get("url", ""),
                 "snippet": _clean(it.get("text", ""))} for it in items[:n]]
    if provider == "tavily":
        items = data.get("results", [])
        return [{"title": it.get("title", ""),
                 "url": it.get("url", ""),
                 "snippet": it.get("content", "")} for it in items[:n]]
    if provider == "brave":
        items = data.get("web", {}).get("results", [])
        return [{"title": it.get("title", ""),
                 "url": it.get("url", ""),
                 "snippet": it.get("description", "")} for it in items[:n]]
    return []


def web_search(query: str, n: int = 5) -> Dict[str, Any]:
    """Return ``{"query", "provider", "results": [...], "error": ...}``.

    The ``duckduckgo`` provider transparently falls back to the keyless
    Wikipedia API when DuckDuckGo returns a bot-challenge (common on throttled
    IPs), so the capability always returns something useful without a key.
    """
    try:
        provider = PROVIDER
        if PROVIDER == "duckduckgo":
            results = _duckduckgo(query, n)
            if not results:  # challenged/empty -> fall back to Wikipedia
                results = _wikipedia(query, n)
                provider = "wikipedia(fallback)"
        elif PROVIDER == "wikipedia":
            results = _wikipedia(query, n)
        elif PROVIDER == "exa":
            key = os.environ.get("EXA_API_KEY", "")
            results = _hosted_json("exa",
                "https://api.exa.ai/search",
                key, query, n,
                {"query": query, "numResults": n})
        elif PROVIDER == "tavily":
            key = os.environ.get("TAVILY_API_KEY", "")
            results = _hosted_json("tavily",
                "https://api.tavily.com/search",
                key, query, n,
                {"query": query, "max_results": n})
        elif PROVIDER == "brave":
            key = os.environ.get("BRAVE_API_KEY", "")
            results = _hosted_json("brave",
                "https://api.search.brave.com/res/v1/web/search",
                key, query, n,
                {"q": query, "count": n})
        else:
            results = _duckduckgo(query, n)
        return {"query": query, "provider": provider, "results": results}
    except Exception as e:  # network/parse failure -> graceful degrade
        return {"query": query, "provider": PROVIDER,
                "results": [], "error": str(e)}
