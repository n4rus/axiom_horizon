# Build with CMC: API Hackathon — Plan (DoraHacks, live door)

**Entrant:** n4rus / AxiomTree (solo, pseudonymous)
**Event:** Build with CMC: API Hackathon — `dorahacks.io/hackathon/coinmarketcap-api-202609`
**Window:** submissions open Wed 9 Sept 2026 00:00 UTC → close Wed 30 Sept 2026 23:59 UTC (21 days)
**Format:** virtual, worldwide, solo allowed. Judging 1–16 Oct, results 19 Oct 2026.
**Repo:** github.com/n4rus/axiom_horizon (public, MIT) + event-period commits
**Application date:** 2026-09-07

---

## TL;DR

Extend the Axiom MCP server with live CoinMarketCap data tools
(`tools/cmc_tools.py`, same pattern as the proven `tools/graph_tools.py`):
natural-language questions over live market data — screeners, agent tools,
MCP integrations — executed against the real CMC API, never mocked.
Primary track: **AI Agents and Automation**. No stake, no entry fee, free
Startup-tier API key. Solo pace, MIT licensed, everything reproducible.

---

## Why this fits

- The stack already exists: `axiom_mcp_server.py` (MCP, Ollama-backed) +
  `tools/graph_tools.py` (live-API tool pattern, proven against The Graph
  gateway: block 503,788,784 returned).
- The delta is integration surface, not engine core: new data source behind
  the same MCP interface. Crown jewels (invariant loop, attractor internals)
  stay pre-existing prior work, documented as such.
- Judging weights Technicality / Originality / Practicality / Usability / WOW.
  WOW entry: local-first AI (no cloud API) putting live market data in front
  of any agent that speaks MCP.

---

## Tracks (one selected)

1. **AI Agents and Automation** (PRIMARY): agent tools, MCP integrations,
   workflow automations over live CMC data.
2. Held in reserve: Markets and Trading Tools (screener/alert bot reuses the
   same tools module), Data and Visualisation (only if days go spare).

---

## Build plan (21 days, solo pace)

| Window | Deliverable |
|---|---|
| Days 1–3 (Sep 9–11) | Free CMC account + Startup-tier key (`CMC_API_KEY`, env only); `tools/cmc_tools.py` scaffold; first live call proven |
| Days 4–10 | NL→query via local model, answer formatting, MCP wiring into `axiom_mcp_server.py` |
| Days 11–15 | Polish, edge cases, AI-use attribution file (per hackathon rules) |
| Days 16–18 | Screen-record demo (voice over screen, no camera), README/SKILL so judges can run it |
| Days 19–21 (by Sep 30) | Buffer + submit via DoraHacks (public repo, demo, endpoints named, X post with hashtag) |

All event work committed incrementally with dated history (disqualification
risk otherwise). Pre-existing work documented in submission; only event-period
delta is judged.

---

## Rules compliance (disqualification risks)

- [ ] Event-only delta, incremental commits Sept 9–30 (no single dump).
- [ ] Live CMC data via Studio key (never mocked, never static).
- [ ] AI-use attribution in repo (which files AI-assisted).
- [ ] Public repo + demo/screen recording + endpoints named explicitly.
- [ ] X/Twitter post with submission link, demo, and hashtag (pseudonymous handle).
- [ ] One track selected.

---

## Scoring (what judges weight)

Technicality / Originality / Practicality / Usability / WOW.
Our WOW: local-first AI querying live market data through an MCP any agent
can call — no cloud bill, no vendor key, reproducible on commodity hardware.

---

## Payout

DoraHacks prize disbursement per event rules (USDT/BNB-chain standard;
confirm at submission). No entry stake was paid; nothing to recover.

---

*Supersedes shelved proposals (kept locally): the live ask is this build, not
a grant application.*
