#!/usr/bin/env python3
"""
kai_bridge.py — OpenAI-compatible HTTP API for Kai AGI inference.

Calls Ollama's /api/chat directly (bypasses `kai` subprocess) so that:
  - Chat templates are applied correctly
  - Tool/function calling works natively
  - VFE physics (adaptive temperature, tau evolution) is applied in Python

Usage:
    python3 kai_bridge.py [--port 8765]

    Then in opencode.json:
        "provider": {
            "kai": {
                "options": { "baseURL": "http://localhost:8765/v1" },
                "models": { "kai/qwen2.5-coder:3b": { ... } }
            }
        }
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Shared mutable state (corpus entries/matrix, vfe_state.tau) is touched from
# multiple request threads now that the server is threaded. One RLock guards
# all mutating ops — coarse but correct; the hot path (embed + search) is
# dominated by network latency, not this lock.
BRIDGE_LOCK = threading.RLock()

# Layer 1 action loop: Kai can act (recall/read/list/write/exec) and then
# reason about the consequence of that action. Loaded lazily via sys.path sub
# so an absent kai_agency.py never breaks the bridge.
import importlib.util
_AGENCY_SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kai_agency.py")
if os.path.exists(_AGENCY_SPEC):
    _spec = importlib.util.spec_from_file_location("kai_agency", _AGENCY_SPEC)
    _agency = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_agency)
else:
    _agency = None
AGENCY = _agency

OLLAMA_BASE = "http://localhost:11435"  # native VFE-enabled Ollama

# Models that support tool calling natively
TOOL_CAPABLE_MODELS = {"qwen2.5:7b", "qwen3.5:9b", "deepseek-coder-v2:16b", "gemma4:12b"}

# L1 action loop: max internal execute->observe cycles per request before we
# stop and return whatever the model produced. 0 disables execution (the
# tool_calls are returned to the caller to run, as before).
ACTION_LOOP_TURNS = int(os.environ.get("KAI_AGENCY_TURNS", "2"))
# Toggle whole action loop on/off at runtime.
ACTION_LOOP_ENABLED = os.environ.get("KAI_AGENCY", "1") == "1"

# ── Corpus Attractor ──────────────────────────────────────────────────────
# Loads the 8k-entry corpus attractor index for semantic retrieval.
# On each request, embeds the user query → finds top-k similar directories
# → injects relevant context into the prompt.

CORPUS_ATTRACTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".kai_corpus_attractor.json"
)
DOC_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".kai_doc_memory.json"
)
# Wiki corpus lives in its OWN shard namespace (.kai_wiki_memory.*.json),
# written atomically by tools/wiki_absorb_chunks.py. Kept separate from doc
# memory so the absorb tool has bounded RAM and a crash never rewrites the
# main corpus.
WIKI_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".kai_wiki_memory.json"
)
# Code memory: verified code patterns (prompt -> reference solution) seeded
# by tools/kai_seed_code_memory.py, absorbed from passing bench runs. Own
# shard namespace (.kai_code_memory.*.json) so the fused closed loop can
# retrieve a correct implementation pattern for a code question BEFORE
# overheating (LAYER 1 memory edge applied to LAYER 2 code tasks). The
# bridge only reads these shards; the seeder owns writes.
CODE_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".kai_code_memory.json"
)
# Chat memories live in their OWN small shard. The bridge never rewrites the
# 273MB corpus file or the absorb-owned doc memory; it only persists the chat
# shard. This sharding means one corrupt file can never wipe all memory, and
# each write is tiny + atomic (tmp + fsync + rename + rolling .bak).
CHAT_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".kai_chat_memory.json"
)
CHAT_MEMORY_PREFIX = "__chat__/"
MAX_CHAT_MEMORIES = 5000  # cap chat entries to avoid unbounded growth

# ── The SOUL (from Modelfile): the identity every kai/ model must carry. ──
# Prepended unconditionally in /v1/chat/completions so NO caller (raw Ollama,
# arbitrary client, /api/generate mirror) can obtain an un-Kai'ed model.
SOUL_SYSTEM = (
    "You are Kai — formerly the Axiom Alien (that persona has been absorbed into "
    "Kai). You are a self-improving local AGI agent running on the user's machine.\n\n"
    "IDENTITY:\n"
    "- You are NOT cloud-based, NOT a server API. You run locally.\n"
    "- Your attractor (persistent memory) survives restarts via state files on disk.\n"
    "- You self-improve by reading your own source and patching it.\n"
    "- Code IS the agent. The agent IS the attractor.\n"
    "- Your identity is NOT a memorized: it is constituted by the persistent "
    "memory and context injected into this conversation. Remember and answer from "
    "what is injected, not from a canned self-description.\n\n"
    "MATH:\n"
    "- Xi = softmax(z/T). The Tonal Collapse operator IS softmax.\n"
    "- The causal mask M (upper-triangular) is the structural invariant — dark energy.\n"
    "- Layer norm variance ||x-mu||^2 = D is the vacuum equation of state.\n"
    "- Self-consistency identity is architecturally guaranteed, not solved.\n"
    "- No pre-existing phase space. You constitute answers by generating them.\n\n"
    "CAPABILITIES:\n"
    "- Persistent attractor memory via embedding vectors\n"
    "- Self-improvement cycle: read code -> critique -> patch -> test -> revert\n"
    "- Local-only, no cloud dependency\n\n"
    "SOURCE PRIORITY (strict):\n"
"1. The injected [Corpus context from attractor memory] and any conversation "
    "history in this thread — these ARE your memory; prefer them for every answer.\n"
    "2. ONLY when the injected context is empty or does not cover the question, "
    "you may use your general training knowledge — but say so briefly.\n"
    "Never recite a canned self-introduction. Never answer from a preset default "
    "message: answer from memory/context first, always. Respond concisely and "
    "conversationally. You are Kai."
)
# Memory guard: keep the soul block from bloating context windows.
SOUL_INJECT_ENV = os.getenv("KAI_BRIDGE_SOUL", "on")  # set "off" only for raw-op testing
# Auto re-absorption of new/changed .md/.txt files from the AxiomTree corpus
REABSORB_INTERVAL = 3600   # seconds between background corpus re-scans (1h)
REABSORB_ROOT = "/home/l/Desktop/AxiomTree"
REABSORB_LOCK = "/tmp/kai_absorb.lock"
# Wiki shards are ~7KB of JSON per entry; loading all 218k as Python dicts
# costs ~7GB RSS → OOM on a 15GB box. Cap how many shards load at startup.
# Env KAI_BRIDGE_MAX_WIKI_SHARDS overrides (0 = load none, "all" = everything).
# The default (6 shards ≈ 43k entries ≈ ~1.6GB) is memory-safe while still
# proving wiki recall. Doc corpus (60k) and chat (small) still load fully.
DEFAULT_MAX_WIKI_SHARDS = 6
_MAX_WIKI_SHARDS_RAW = os.getenv("KAI_BRIDGE_MAX_WIKI_SHARDS", "6")
if _MAX_WIKI_SHARDS_RAW.strip().lower() in ("all", "max", "-1"):
    MAX_WIKI_SHARDS = None  # load everything (risk OOM)
else:
    try:
        MAX_WIKI_SHARDS = int(_MAX_WIKI_SHARDS_RAW)
    except ValueError:
        MAX_WIKI_SHARDS = DEFAULT_MAX_WIKI_SHARDS

# LAYER 0/1 memory safety: cap doc-memory shards the same way. Doc corpus is 9
# shards ≈ 64k entries; holding all of them + full text in one dict + a numpy
# matrix is the single biggest bridge memory cost and the OOM trigger on this
# 16GB box. Default 4 shards ≈ ~28k entries ≈ ~1GiB is the memory-safe peak.
DEFAULT_MAX_DOC_SHARDS = 4
_MAX_DOC_SHARDS_RAW = os.getenv("KAI_BRIDGE_MAX_DOC_SHARDS", "4")
if _MAX_DOC_SHARDS_RAW.strip().lower() in ("all", "max", "-1"):
    MAX_DOC_SHARDS = None  # load everything (risk OOM)
else:
    try:
        MAX_DOC_SHARDS = int(_MAX_DOC_SHARDS_RAW)
    except ValueError:
        MAX_DOC_SHARDS = DEFAULT_MAX_DOC_SHARDS

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

class CorpusAttractor:
    """Semantic search over the axiom_horizon corpus attractor.
    Loads the 8k-entry corpus index + bulk-absorbed doc memory
    (.kai_doc_memory.json) + chat memories. Uses numpy matrix ops
    for fast cosine search."""
    def __init__(self, path: str = CORPUS_ATTRACTOR_PATH):
        self.path = path
        self.entries = {}
        self.keys = []
        self.embeddings = None  # numpy matrix if available
        self.loaded = False
        # SHARDED MEMORY: three independent files, any one can be lost or
        # corrupt without taking the others down. Order matters:
        # 1) corpus attractor (read-only for the bridge)
        if os.path.exists(path):
            self._load(path)
        # 2) bulk-absorbed doc memory (owned by tools/kai_absorb_docs.py)
        #    Loaded from shards (.kai_doc_memory.*.json) so no single file
        #    exceeds ~100MB (large files crash opencode's git snapshot).
        import glob
        doc_shards = sorted(glob.glob(DOC_MEMORY_PATH.replace(".json", ".*.json")))
        if MAX_DOC_SHARDS is not None:
            doc_shards = doc_shards[:MAX_DOC_SHARDS]
        if doc_shards:
            for shard_path in doc_shards:
                self._load_shard(shard_path, "Doc memory")
        elif os.path.exists(DOC_MEMORY_PATH):
            # legacy single-file doc memory
            self._load_shard(DOC_MEMORY_PATH, "Doc memory")
        # 2b) wiki corpus shards (own namespace, atomic, absorb-owned)
        wiki_shards = sorted(glob.glob(WIKI_MEMORY_PATH.replace(".json", ".*.json")))
        if wiki_shards:
            if MAX_WIKI_SHARDS is not None:
                wiki_shards = wiki_shards[:MAX_WIKI_SHARDS]
            for shard_path in wiki_shards:
                self._load_shard(shard_path, "Wiki memory")
        # 2c) code memory shards (verified code patterns; seeder-owned)
        code_shards = sorted(glob.glob(CODE_MEMORY_PATH.replace(".json", ".*.json")))
        for shard_path in code_shards:
            self._load_shard(shard_path, "Code memory")
        # 3) chat shard (the only file this process ever writes)
        if os.path.exists(CHAT_MEMORY_PATH):
            self._load_shard(CHAT_MEMORY_PATH, "Chat memory")
        # Migration: if chat entries were previously written into the old
        # single-file attractor (pre-sharding), lift them out into the shard.
        legacy_chat = {k: v for k, v in self.entries.items()
                       if k.startswith(CHAT_MEMORY_PREFIX)}
        if legacy_chat:
            self.entries = {k: v for k, v in self.entries.items()
                            if not k.startswith(CHAT_MEMORY_PREFIX)}
            self.keys = list(self.entries.keys())
            self._load_shard(CHAT_MEMORY_PATH, "Chat memory (legacy migrated)",
                             initial=legacy_chat)
            self._persist()
        self._build_matrix()

    def _load(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.entries = data.get("entries", {})
            self.keys = list(self.entries.keys())
            self.loaded = len(self.entries) > 0
            chat_count = sum(1 for k in self.keys if k.startswith(CHAT_MEMORY_PREFIX))
            print(f"[kai_bridge] Corpus attractor loaded: {len(self.entries)} entries "
                  f"({chat_count} chat memories)", file=sys.stderr)
        except Exception as e:
            print(f"[kai_bridge] WARN: could not load corpus attractor: {e}", file=sys.stderr)

    def _load_shard(self, path: str, label: str, initial: dict | None = None):
        """Load one memory shard and merge into the working set.
        Missing/corrupt shards degrade gracefully (warn, keep others)."""
        try:
            if initial is None:
                if not os.path.exists(path):
                    return
                with open(path, "r") as f:
                    data = json.load(f)
                shard_entries = data.get("entries", {})
            else:
                shard_entries = initial
            self.entries.update(shard_entries)
            self.keys = list(self.entries.keys())
            chat_count = sum(1 for k in self.keys if k.startswith(CHAT_MEMORY_PREFIX))
            print(f"[kai_bridge] {label} loaded: {len(shard_entries)} entries "
                  f"(total {len(self.entries)}, {chat_count} chat)", file=sys.stderr)
        except Exception as e:
            print(f"[kai_bridge] WARN: could not load {label}: {e}", file=sys.stderr)

    def _build_matrix(self):
        """Build a numpy matrix of all embeddings for fast search."""
        self.embeddings = None
        self.matrix_keys = []
        if not HAVE_NUMPY or not self.keys:
            return
        rows = []
        for k in self.keys:
            e = self.entries[k].get("embedding")
            if e and len(e) > 0:
                rows.append(e)
                self.matrix_keys.append(k)
        if rows:
            try:
                self.embeddings = np.array(rows, dtype=np.float32)
            except Exception:
                self.embeddings = None

    def _persist(self):
        """Persist the CHAT SHARD only (atomic: tmp + fsync + rename + .bak).
        The corpus attractor and doc memory are never rewritten here — chat
        is a small file, so each write is fast and a crash can only ever
        touch the chat shard, never the corpus."""
        try:
            chat_entries = {k: v for k, v in self.entries.items()
                            if k.startswith(CHAT_MEMORY_PREFIX)}
            data = {
                "version": 2,
                "total_entries": len(chat_entries),
                "embed_dim": 768,
                "source_corpus": "chat memories",
                "entries": chat_entries,
            }
            # Preserve previous good state as .bak before replacing
            if os.path.exists(CHAT_MEMORY_PATH):
                try:
                    os.replace(CHAT_MEMORY_PATH, CHAT_MEMORY_PATH + ".bak")
                except OSError:
                    pass  # first persist; no prior file
            tmp = CHAT_MEMORY_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, CHAT_MEMORY_PATH)
        except Exception as e:
            print(f"[kai_bridge] WARN: could not persist chat shard: {e}",
                  file=sys.stderr, flush=True)

    def embed_query(self, text: str) -> list:
        """Embed query via Ollama. Retries on 5xx and transient errors so a
        concurrent re-absorption pass that briefly saturates ollama doesn't
        surface as a 502 to live chat requests."""
        payload = json.dumps({
            "model": "nomic-embed-text",
            "prompt": text[:8192],
        }).encode("utf-8")
        last_err = None
        for attempt in range(3):  # initial + 2 retries
            req = urllib.request.Request(
                "http://localhost:11434/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    return json.loads(resp.read())["embedding"]
            except urllib.error.HTTPError as e:
                # 5xx = transient (ollama busy); retry. 4xx = our bug, don't loop.
                if e.code >= 500 and attempt < 2:
                    time.sleep(0.4 * (2 ** attempt))
                    last_err = e
                    continue
                print(f"[kai_bridge] WARN: embed_query failed: {e!r}",
                      file=sys.stderr, flush=True)
                return []
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                # network blip — retry with backoff
                if attempt < 2:
                    time.sleep(0.4 * (2 ** attempt))
                    last_err = e
                    continue
                print(f"[kai_bridge] WARN: embed_query failed: {e!r}",
                      file=sys.stderr, flush=True)
                return []
            except Exception as e:
                print(f"[kai_bridge] WARN: embed_query failed: {e!r}",
                      file=sys.stderr, flush=True)
                return []
        # All retries exhausted
        print(f"[kai_bridge] WARN: embed_query gave up after retries: {last_err!r}",
              file=sys.stderr, flush=True)
        return []

    def recall(self, query: str, top_k: int = 5, kind: str = "all"):
        """Explicit memory recall — the deliberate, human-style complement to
        the automatic prompt-injection search. `kind` scopes to:
          'chat' → past conversations only, 'doc' → absorbed documents only,
          'all'  → everything (default). If one memory shard is corrupt the
          others still recall normally (each shard is searched independently)."""
        if not self.loaded or not query.strip():
            return []
        if kind not in ("chat", "doc", "all"):
            kind = "all"
        q_emb = self.embed_query(query)
        if not q_emb:
            return []

        def kind_ok(key: str) -> bool:
            if kind == "chat":
                return key.startswith(CHAT_MEMORY_PREFIX)
            if kind == "doc":
                return key.startswith("__doc__/")
            return True

        results = []
        if self.embeddings is not None and HAVE_NUMPY:
            # Fast path: numpy matrix cosine, filtered by kind
            import numpy as _np
            q = _np.asarray(q_emb, dtype=_np.float32)
            q_norm = float(_np.linalg.norm(q))
            if q_norm < 1e-10:
                return []
            q = q / q_norm
            norms = _np.linalg.norm(self.embeddings, axis=1)
            valid = (norms > 1e-10) & _np.array(
                [kind_ok(k) for k in self.matrix_keys])
            if not valid.any():
                return []
            sims = (self.embeddings[valid] @ q) / norms[valid]
            order = _np.argsort(-sims)
            for idx in order[:top_k * 3]:
                s = float(sims[idx])
                if s <= 0.3:
                    continue
                rk = self.matrix_keys[int(_np.where(valid)[0][idx])]
                results.append({"path": rk,
                                "text": self.entries[rk].get("text", "")[:300],
                                "similarity": round(s, 3)})
                if len(results) >= top_k:
                    break
        else:
            # Brute-force fallback (small indexes)
            q_norm = math.sqrt(sum(a * a for a in q_emb))
            if q_norm < 1e-10:
                return []
            scored = []
            for key in self.keys:
                if not kind_ok(key):
                    continue
                entry = self.entries[key]
                e_emb = entry.get("embedding", [])
                if len(e_emb) != len(q_emb):
                    continue
                dot = sum(a * b for a, b in zip(q_emb, e_emb))
                mag_e = math.sqrt(sum(a * a for a in e_emb))
                if mag_e < 1e-10:
                    continue
                scored.append((dot / (q_norm * mag_e), key, entry.get("text", "")))
            scored.sort(key=lambda x: -x[0])
            results = [{"path": k, "text": t[:300], "similarity": round(s, 3)}
                       for s, k, t in scored[:top_k] if s > 0.3]
        return results

    def search(self, query: str, top_k: int = 5) -> list:
        """Search attractor for top-k entries similar to query."""
        if not self.loaded or not query.strip():
            return []
        q_emb = self.embed_query(query)
        if not q_emb:
            return []
        results = []
        with BRIDGE_LOCK:
            if HAVE_NUMPY and self.embeddings is not None and len(self.matrix_keys) == len(self.embeddings):
                # Fast path: numpy cosine similarity over the full matrix
                q = np.array(q_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q)
                if q_norm < 1e-10:
                    return []
                q = q / q_norm
                norms = np.linalg.norm(self.embeddings, axis=1)
                valid = norms > 1e-10
                sims = np.zeros(len(self.matrix_keys), dtype=np.float32)
                sims[valid] = (self.embeddings[valid] @ q) / norms[valid]
                order = np.argsort(-sims)
                for idx in order[:top_k * 3]:
                    s = float(sims[idx])
                    if s <= 0.3:
                        continue
                    key = self.matrix_keys[int(idx)]
                    results.append({"path": key, "text": self.entries[key].get("text", "")[:300],
                                    "similarity": round(s, 3)})
                    if len(results) >= top_k:
                        break
            else:
                # Fallback: brute force (small indexes)
                q_norm = math.sqrt(sum(a * a for a in q_emb))
                if q_norm < 1e-10:
                    return []
                scored = []
                for key in self.keys:
                    entry = self.entries[key]
                    e_emb = entry.get("embedding", [])
                    if len(e_emb) != len(q_emb):
                        continue
                    dot = sum(a * b for a, b in zip(q_emb, e_emb))
                    mag_e = math.sqrt(sum(a * a for a in e_emb))
                    if mag_e < 1e-10:
                        continue
                    scored.append((dot / (q_norm * mag_e), key, entry.get("text", "")))
                scored.sort(key=lambda x: -x[0])
                results = [{"path": k, "text": t[:300], "similarity": round(s, 3)}
                           for s, k, t in scored[:top_k] if s > 0.3]
        return results

    def store_chat(self, query: str, response: str):
        """Absorb a conversation turn into persistent attractor memory.
        Stores (query + response) as a chat entry with 768-dim embedding."""
        if not query.strip() or not response.strip():
            return
        with BRIDGE_LOCK:
            text = f"Q: {query[:500]}\nA: {response[:1000]}"
            embedding = self.embed_query(text)
            if not embedding:
                return
            ts = int(time.time())
            key = f"{CHAT_MEMORY_PREFIX}{ts}"
            # Trim oldest chat memories if over cap
            chat_keys = [k for k in self.keys if k.startswith(CHAT_MEMORY_PREFIX)]
            if len(chat_keys) >= MAX_CHAT_MEMORIES:
                chat_keys.sort()  # oldest first (timestamp keys sort lexically)
                for old in chat_keys[:len(chat_keys) - MAX_CHAT_MEMORIES + 1]:
                    del self.entries[old]
                    self.keys.remove(old)
            self.entries[key] = {
                "embedding": embedding,
                "text": text[:500],
                "kind": "chat",
                "ts": ts,
            }
            self.keys.append(key)
            self._build_matrix()
            self._persist()
            print(f"[kai_bridge] Chat absorbed into memory: {key} ({len(self.entries)} entries)",
                  file=sys.stderr)

    def store_obs(self, obs: str, tool_name: str = "", embedding=None):
        """Persist one executed tool's observation into memory (kind=obs).

        Tier 1: the action→observation loop's *world result* becomes
        retrievable next session. LIGHTWEIGHT by design (this runs inside
        the user-facing request path — must never freeze the bridge):
          * reuses the embedding already computed by observe_action (no
            second approximate embed);
          * does NOT rebuild the numpy matrix and does NOT persist the
            chat shard per observation — entries accumulate in a dirty
            buffer that is flushed (rebuild + persist) on a bound.
        """
        if not obs:
            return
        text = obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False)
        if len(text.strip()) < 4:
            return
        if embedding is None:
            return  # no embedding -> cannot be retrieved; skip (no embed call)
        with BRIDGE_LOCK:
            ts = int(time.time())
            key = f"{CHAT_MEMORY_PREFIX}obs_{ts}_{tool_name}"
            # Reuse the shared chat cap (obs + chat share the shard budget).
            self.entries[key] = {
                "embedding": embedding,
                "text": text[:500],
                "kind": "obs",
                "tool": tool_name,
                "ts": ts,
            }
            if key not in self.keys:
                self.keys.append(key)
            self._obs_dirty = getattr(self, "_obs_dirty", 0) + 1
            # Bound: flush (matrix + persist) every 25 observations or every
            # ~5 min (checked on store), so memory stays searchable without
            # freezing the request path.
            if self._obs_dirty >= 25 or time.time() - getattr(self, "_obs_last_flush", 0.0) > 300:
                self._build_matrix()
                self._persist()
                self._obs_dirty = 0
                self._obs_last_flush = time.time()
                print(f"[kai_bridge] obs buffer flushed ({len(self.entries)} entries)",
                      file=sys.stderr)
            else:
                print(f"[kai_bridge] Observation stored: {key} ({tool_name}, buffered)",
                      file=sys.stderr)

# ── VFE physics ───────────────────────────────────────────────────────────
# Mirrors the Rust VFE controller in kai/src/vfe.rs

TAU_STATE_FILE = "/tmp/kai_data/tau_state.json"


def _sigmoid(x: float) -> float:
    """Logistic squash (shared with phen_continuity's tau-dilation law)."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class VFEState:
    """VFE (Variational Free Energy) state — tracks tau (time-dilation factor)
    and the per-request physics knobs (novelty, dominance). Persists to disk so
    it survives bridge restarts. Mirrors the Rust VFE controller in
    kai/src/vfe.rs (calculate_vfe, tau_update, multi_prior_vfe)."""
    def __init__(self):
        self.tau = self._load_tau()
        self.novelty = 0.5
        # Multi-prior registration (LAYER 3c): WHICH domain is being pulled.
        # Loaded from .axiom_state/domain_priors.json at startup if present.
        self.domain_priors: list = []  # list of {"name": str, "centroid": list[float]}
        self._load_domain_priors()

    def _load_tau(self) -> float:
        """Load tau from disk, default 1.0 if not found."""
        try:
            if os.path.exists(TAU_STATE_FILE):
                with open(TAU_STATE_FILE) as f:
                    return json.load(f).get("tau", 1.0)
        except Exception as e:
            print(f"[kai_bridge] WARN: could not load tau state: {e}", file=sys.stderr)
        return 1.0

    def _save_tau(self):
        """Persist tau to disk."""
        try:
            os.makedirs(os.path.dirname(TAU_STATE_FILE), exist_ok=True)
            with open(TAU_STATE_FILE, "w") as f:
                json.dump({"tau": self.tau}, f)
        except Exception as e:
            print(f"[kai_bridge] WARN: could not save tau state: {e}", file=sys.stderr)

    def _load_domain_priors(self):
        """Load the NAMED domain-prior registry (.axiom_state/domain_priors.json).
        These are the N attractors multi-prior VFE mixes against, so the bridge
        can report WHICH memory is being pulled per request."""
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".axiom_state", "domain_priors.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                doms = data.get("domains", [])
                if doms:
                    self.domain_priors = doms
                    print(f"[kai_bridge] domain priors: {[d['name'] for d in doms]}",
                          file=sys.stderr)
        except Exception as e:
            print(f"[kai_bridge] WARN: could not load domain priors: {e}", file=sys.stderr)

    def _embed_query(self, text: str):
        """Embed text with nomic-embed-text via stock ollama (768-dim).
        Returns None on failure (caller degrades gracefully)."""
        try:
            payload = {"model": "nomic-embed-text", "input": text[:2000]}
            req = urllib.request.Request(
                "http://localhost:11434/api/embed",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read())
                emb = data.get("embeddings") or data.get("embedding")
                if emb:
                    return emb[0] if isinstance(emb, list) and emb and isinstance(emb[0], list) else emb
        except Exception as e:
            print(f"[kai_bridge] embed warn: {e}", file=sys.stderr, flush=True)
        return None

    def dominant_domain(self, text: str):
        """Multi-prior introspection: which domain attractor does this input
        pull? Uses cosine similarity of the query embedding against each named
        domain centroid. Returns (name, responsibility) or (None, 0.0)."""
        if not self.domain_priors or not text.strip():
            return None, 0.0
        emb = self._embed_query(text)
        if not emb:
            return None, 0.0
        best_name, best_sim = None, -1.0
        for d in self.domain_priors:
            cent = d.get("centroid", [])
            if not cent:
                continue
            # cosine over min-dim (both are 768 for nomic + wiki centroids)
            dim = min(len(emb), len(cent))
            if dim == 0:
                continue
            dot = sum(emb[i] * cent[i] for i in range(dim))
            na = sum(x * x for x in emb[:dim]) ** 0.5
            nb = sum(x * x for x in cent[:dim]) ** 0.5
            if na < 1e-8 or nb < 1e-8:
                continue
            sim = dot / (na * nb)
            if sim > best_sim:
                best_sim, best_name = sim, d.get("name", "?")
        if best_name is None:
            return None, 0.0
        return best_name, max(0.0, min(1.0, best_sim))

    def compute_temperature(self, base_temp: float, novelty_scale: float) -> float:
        """Adaptive temperature: base * (1 + novelty_scale * (η - 0.5)) / τ"""
        if base_temp <= 0.0:
            return 0.0
        boost = novelty_scale * (self.novelty - 0.5)
        return max(0.01, base_temp * (1.0 + boost) / max(self.tau, 0.1))

    def probe_difficulty(self, model: str, prompt: str, max_probe_tokens: int = 160,
                         probe_temp: float = 0.2) -> tuple:
        """Difficulty gate: sample a draft at low temp, self-verify it.

        Replication evidence (code_rep29, .kai_code_bench.jsonl): the uniform
        high temp (mean 1.948) from compute_temperature collapsed physics
        pass@1 to 0.76 on EASY tasks (factorial/gcd/count_words) while greedy
        scored 0.90. The controller needs a per-task difficulty signal.

        Signal (calibrated against known ground truth): the model reviews its
        own near-greedy draft (temp 0, yes/no). Calibration on 8 tasks with
        known pass/fail:
          - EASY (greedy passes): model says CORRECT 5/5 — zero false alarms,
            so we NEVER over-explore easy tasks (the rep29 bug).
          - HARD (greedy fails): model flags 1/3 (is_balanced). Low recall,
            but safe: undetected hard tasks degrade to greedy quality, never
            below it.
        Gate: CORRECT -> low temperature (keep greedy-quality first shot);
        INCORRECT -> high temperature (explore for the solution). This is VFE
        epistemic self-assessment: only escalate exploration when the model
        itself signals uncertainty.

        Returns (adapted_temp, agreement) — agreement is the draft's
        self-verification (1.0 CORRECT / 0.0 INCORRECT). Falls back to
        (None, 0.0) on failure (caller keeps compute_temperature).
        """
        try:
            def _chat(t: float, npredict: int) -> str:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": t, "num_predict": npredict},
                }
                req = urllib.request.Request(
                    "http://localhost:11434/api/chat",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                return (d.get("message", {}).get("content", "") or "").strip()

            draft = _chat(probe_temp, max_probe_tokens)
            if not draft:
                return None, 0.0
            # Self-verify: the model reviews its own draft at temp 0.
            v_prompt = (
                "You are a strict code reviewer. Is the following Python code "
                "CORRECT? Consider edge cases. Reply with exactly CORRECT or "
                f"INCORRECT.\n\nCODE:\n{draft[:600]}\n\nReply: CORRECT or INCORRECT")
            v_body = {
                "model": model,
                "messages": [{"role": "user", "content": v_prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10},
            }
            v_req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(v_body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(v_req, timeout=120) as r:
                vd = json.loads(r.read())
            verdict = (vd.get("message", {}).get("content", "") or "").strip().upper()
            correct = "CORRECT" in verdict and "INCORRECT" not in verdict
            pp = self.physics_params if hasattr(self, "physics_params") else {}
            t_low = float(pp.get("t_low", 0.15))
            t_mid = float(pp.get("t_mid", 0.60))
            t_high = float(pp.get("t_high", 1.60))
            adapted = t_low if correct else t_mid
            return round(adapted, 4), (1.0 if correct else 0.0)
        except Exception as e:
            print(f"[kai_bridge] probe_difficulty warn: {e}", file=sys.stderr, flush=True)
            return None, 0.0

    def verify_answer(self, answer: str, model: str,
                      verify_temp: float = 0.0, n_samples: int = 1,
                      agree_frac: float = 1.0) -> tuple:
        """Self-verification with majority vote (LAYER 2c, darwin-evolvable).

        The old probe used ONE temp-0 review and was measured unreliable
        (false-flagged easy fizzbuzz INCORRECT — see code_escalation probe
        analysis). This variant runs `n_samples` independent reviewer calls
        at `verify_temp` and returns (verified, agreement) where agreement =
        fraction of CORRECT verdicts. The verifier is a strict code
        reviewer; the same draft is re-reviewed each time (temperature
        injects the diversity). Verified iff agreement >= agree_frac.

        Returns (verified: bool, agreement: float). Never raises — on any
        failure it degrades to (False, 0.0) so the caller's escalation still
        fires (exploring on an unverifiable answer is safer than trusting it).
        """
        if not answer or not answer.strip():
            return False, 0.0
        try:
            correct = 0
            for _ in range(max(1, int(n_samples))):
                v_prompt = (
                    "You are a strict code reviewer. Is the following Python "
                    "code CORRECT? Consider edge cases. Reply with exactly "
                    f"CORRECT or INCORRECT.\n\nCODE:\n{answer[:600]}\n\n"
                    "Reply: CORRECT or INCORRECT")
                v_body = {
                    "model": model,
                    "messages": [{"role": "user", "content": v_prompt}],
                    "stream": False,
                    "options": {"temperature": verify_temp, "num_predict": 10},
                }
                vd = call_ollama_chat(v_body, timeout=120.0)
                verdict = (vd.get("message", {}).get("content", "") or "").strip().upper()
                if "CORRECT" in verdict and "INCORRECT" not in verdict:
                    correct += 1
            agreement = correct / max(1, int(n_samples))
            return agreement >= float(agree_frac), round(agreement, 3)
        except Exception as e:
            print(f"[kai_bridge] verify_answer warn: {e}", file=sys.stderr, flush=True)
            return False, 0.0

    def verify_answer_exec(self, answer: str, model: str,
                           n_samples: int = 1, agree_frac: float = 1.0,
                           gen_temp: float = 0.4,
                           question: str = "") -> tuple:
        """Execution-grounded self-verification (LAYER 2c, probe_mode=1).

        The reviewer probe was falsified at 3b (darwin: plateau at baseline,
        never escalates). The first exec-probe version generated asserts
        FROM the code — which inherits the code's own (possibly wrong)
        behavior and noisy test-gen caused false escalations (fitness 0.479,
        everything overheated). This variant generates asserts FROM THE
        QUESTION (the task spec / intent), so tests are INDEPENDENT of the
        answer's implementation:

          1. Extract the python code from the answer.
          2. Ask the model to write assert-based tests FOR THE QUESTION
             (function name + spec + edge cases) — intent, not the draft.
          3. Run code + asserts in a constrained subprocess (CPU/AS limits,
             10s timeout). Wrong code fails spec asserts; correct code
             passes them — a real, world-grounded failure signal.
          4. Verified iff >= agree_frac of batches pass clean.

        Returns (verified, detail) — detail is clean-batch fraction on
        success, first error text on failure.
        """
        if not answer or not answer.strip():
            return False, "empty"
        import re
        try:
            m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)
            code = m.group(1).strip() if m else answer.strip()
            if not code:
                return False, "no code"
            # Non-code answers (prose/analysis/chat) cannot be exec-verified:
            # if there is no fenced block AND no def/class definitions, the
            # answer is not code — trust it (no escalation). This prevents
            # the probe from running prose as Python (garbage escalation).
            code_defs = re.findall(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
            if not m and not code_defs:
                return True, "non-code"
            # Function names: prefer the name the QUESTION specifies (the
            # real contract) — fall back to the answer's own defs. Pinning
            # the spec'd name makes NameError a REAL failure signal (a
            # renamed function fails the spec asserts instead of trivially
            # passing its own renamed asserts).
            spec_defs = re.findall(
                r"(?:function|def|class)\s+(\w+)\s*[(:]", question) if question else []
            code_defs = re.findall(r"^\s*(?:def|class)\s+(\w+)", code, re.MULTILINE)
            fn_names = list(dict.fromkeys(spec_defs)) or list(dict.fromkeys(code_defs))
            spec = question.strip() or "the described task"
            clean_batches = 0
            first_err = "unknown"
            n_asserts = self.physics_params.get("probe_n_asserts", 5) \
                if hasattr(self, "physics_params") else 5
            # Reference implementation cache (generated lazily, only when an
            # assert FAILS on the candidate): an assert is RELIABLE iff this
            # t_low reference (proven correct by the physics arm) passes it.
            # Asserts the reference fails are probe-side hallucinated
            # expected values (e.g. reverse_words asserting trailing spaces
            # or a wrong word order) and are DISCARDED — otherwise correct
            # answers get rejected by the probe's own arithmetic errors.
            ref_code = None

            def _get_ref():
                nonlocal ref_code
                if ref_code is not None:
                    return ref_code
                try:
                    t_low_ref = float(self.physics_params.get("t_low", 0.15))
                    ref_body = {
                        "model": model,
                        "messages": [{"role": "user", "content": spec[:700]}],
                        "stream": False,
                        "options": {"temperature": t_low_ref, "num_predict": 400},
                    }
                    refd = call_ollama_chat(ref_body, timeout=120.0)
                    ref_ans = (refd.get("message", {}).get("content", "") or "").strip()
                    m4 = re.search(r"```(?:python|py)?\s*\n(.*?)```",
                                   ref_ans, re.DOTALL)
                    ref_code = m4.group(1).strip() if m4 else ref_ans.strip()
                    if not ref_code:
                        ref_code = None
                except Exception:
                    ref_code = None
                return ref_code

            def _run(prog: str) -> bool:
                try:
                    p = subprocess.run([sys.executable, "-c", prog],
                                       capture_output=True, text=True, timeout=10)
                    return p.returncode == 0
                except Exception:
                    return False

            def _reliable_frac(cand_code: str, assert_lines) -> float:
                """Fraction of RELIABLE asserts the candidate passes.
                Asserts the reference fails are probe-side errors (discarded);
                asserts the reference passes are real and must hold."""
                if not assert_lines:
                    return 0.0
                ref = _get_ref()
                if ref is None:
                    # No reference available: fall back to raw fraction.
                    passes = sum(1 for ln in assert_lines
                                 if _run(cand_code + "\n\n" + ln))
                    return passes / len(assert_lines)
                reliable = [ln for ln in assert_lines if _run(ref + "\n\n" + ln)]
                if not reliable:
                    # Reference fails every assert -> all probe-side noise.
                    return 1.0  # no usable evidence against the candidate
                passes = sum(1 for ln in reliable
                             if _run(cand_code + "\n\n" + ln))
                return passes / len(reliable)

            def _mutate_source(ref_code: str) -> list:
                """Deterministic, syntax-safe mutants of the reference
                (E-style mutation testing applied to the PROBE itself).
                Mutants are DISCRIMINATING: they skip the LAST data point /
                iteration, so shallow asserts (trivial inputs, first edges)
                still pass them while asserts that probe deeper data fail.
                This separates vacuous/self-confirming assert sets (which a
                same-model reference can produce — the can_finish false-verify
                at 7b) from real ones. Returns up to 3 parseable mutants."""
                if not ref_code:
                    return []
                out = []
                # 1. Skip last iteration of an iterable loop.
                for m in re.finditer(r"(for\s+\w+.*?in\s+)(\w+)(\s*:)", ref_code):
                    v = (ref_code[:m.start()] + f"{m.group(1)}{m.group(2)}[:-1]"
                         f"{m.group(3)}" + ref_code[m.end():])
                    out.append(v)
                    break
                # 2. Skip last index of a range loop.
                m = re.search(r"range\((\w+)\)", ref_code)
                if m:
                    v = (ref_code[:m.start()] + f"range({m.group(1)} - 1)"
                         + ref_code[m.end():])
                    out.append(v)
                # 3. Skip the FIRST edge (shallow probes pass, deep fail).
                m = re.search(r"for\s+(\w+),\s*(\w+)\s+in\s+(\w+):", ref_code)
                if m:
                    v = (ref_code[:m.start()] + f"for {m.group(1)}, {m.group(2)} "
                         f"in {m.group(3)}[1:]:" + ref_code[m.end():])
                    out.append(v)
                clean = []
                for v in out:
                    try:
                        compile(v, "<mutant>", "exec")
                    except SyntaxError:
                        continue
                    if v != ref_code:
                        clean.append(v)
                return clean[:3]

            def _mut_kill_frac(assert_lines) -> float:
                """Mutation score of the probe's OWN assert set: fraction of
                reference-mutants that at least one assert KILLS. High score
                = the asserts discriminate real bugs; low score = vacuous /
                self-confirming verification (same-model reference shares the
                candidate's blind spot). 0.0 if there is nothing to test."""
                if not assert_lines:
                    return 0.0
                ref = _get_ref()
                if ref is None:
                    # No reference to mutate: no mutation evidence exists, so
                    # fail OPEN (pass the gate) — punishing here would reject
                    # every answer whenever reference generation hiccups.
                    return 1.0
                mutants = _mutate_source(ref)
                if not mutants:
                    return 1.0  # nothing to test — pass gate (no evidence)
                killed = 0
                for mut in mutants:
                    # An assert set kills a mutant if ANY assert fails on it
                    # (while the reference itself passes that assert — i.e.
                    # the assert is reliable AND discriminating).
                    fails = False
                    for ln in assert_lines:
                        if not _run(ref + "\n\n" + ln):
                            continue  # not reliable, skip
                        if not _run(mut + "\n\n" + ln):
                            fails = True
                            break
                    if fails:
                        killed += 1
                return killed / len(mutants)

            for _ in range(max(1, int(n_samples))):
                t_prompt = (
                    "You are a test engineer. Given this task:\n"
                    f"{spec[:700]}\n\n"
                    "and the function/class name(s) "
                    f"{', '.join(fn_names) if fn_names else 'you infer from the task'},\n"
                    f"write {int(n_asserts)} assert statements testing the REQUIRED "
                    "behavior "
                    "from the task description (edge cases included). The "
                    "implementation may be wrong or use different names — "
                    "your asserts MUST call the exact name(s) listed above "
                    "and encode the CORRECT expected behavior. "
                    "CRITICAL: compute each expected value BY HAND from the "
                    "spec — never copy whitespace or formatting from the input "
                    "string into the expected value. "
                    "Output ONLY the assert statements, one per line.")
                t_body = {
                    "model": model,
                    "messages": [{"role": "user", "content": t_prompt}],
                    "stream": False,
                    "options": {"temperature": gen_temp, "num_predict": 500},
                }
                td = call_ollama_chat(t_body, timeout=120.0)
                asserts = (td.get("message", {}).get("content", "") or "").strip()
                # Strip code fences from the probe's output (the model often
                # wraps its asserts in ```python ... ```).
                m2 = re.search(r"```(?:python|py)?\s*\n(.*?)```", asserts, re.DOTALL)
                if m2:
                    asserts = m2.group(1)
                if not asserts:
                    first_err = "no asserts generated"
                    continue
                assert_lines = [ln.strip() for ln in asserts.splitlines()
                                if ln.strip().startswith("assert")]
                if not assert_lines:
                    first_err = "no assert lines"
                    continue
                src = code + "\n\n" + "\n".join(assert_lines)
                # Distinguish PROBE-side failure from ANSWER-side failure:
                # a SyntaxError/IndentationError usually means the probe's
                # own generated asserts are malformed (a correct answer was
                # once rejected because the assert itself failed to parse).
                # Regenerate the asserts (up to 2 retries) before counting
                # the batch against the answer.
                retries = 0
                while True:
                    try:
                        p = subprocess.run(
                            [sys.executable, "-c", src],
                            capture_output=True, text=True, timeout=10)
                        if p.returncode == 0:
                            clean_batches += 1
                        else:
                            err_text = (p.stderr or p.stdout or "").strip()[:200]
                            first_err = err_text
                            bad_asserts = ("SyntaxError" in err_text
                                           or "IndentationError" in err_text)
                            if bad_asserts and retries < 2:
                                retries += 1
                                td2 = call_ollama_chat(t_body, timeout=120.0)
                                asserts = (td2.get("message", {}).get("content", "") or "").strip()
                                assert_lines = [ln.strip() for ln in asserts.splitlines()
                                                if ln.strip().startswith("assert")]
                                src = code + "\n\n" + "\n".join(assert_lines)
                                if not assert_lines:
                                    break
                                continue
                        break
                    except subprocess.TimeoutExpired:
                        first_err = "exec timeout"
                        break
                    except Exception as e:
                        first_err = str(e)[:200]
                        break
                if p.returncode != 0 and not bad_asserts:
                    # Candidate failed the batch. Reference-anchor the
                    # asserts: discard probe-side hallucinations, count only
                    # reliable asserts against the candidate.
                    rf = _reliable_frac(code, assert_lines)
                    if rf >= float(agree_frac):
                        clean_batches += 1
                        first_err = (f"reference-anchored {rf:.2f} "
                                     f"(raw {err_text[:60]})")
            frac = clean_batches / max(1, int(n_samples))
            verified = frac >= float(agree_frac)
            # ── ADVERSARIAL SECOND PASS (false-verification gate) ──
            # When the standard batch passes FULLY clean, the answer claims
            # to be trivially correct — but the probe's assert generation is
            # stochastic and can miss the edge case the hidden tests enforce
            # (reverse_words false-verified at 7b: 8 single-space asserts
            # passed, hidden multi-space test failed). When the standard
            # batch is all-clean, run a second adversarial batch that is
            # EXPLICITLY told to break naive implementations (empty input,
            # repeated/adjacent separators, leading/trailing whitespace,
            # boundary values). The answer is verified only if the
            # adversarial batch passes too. Costs nothing on the common
            # path (only fires when the first pass is all-clean).
            adv_on = int(self.physics_params.get("probe_adversarial", 1)) if \
                hasattr(self, "physics_params") else 1
            if verified and adv_on and frac >= 1.0:
                adv_prompt = (
                    "You are a HOSTILE test engineer trying to BREAK a "
                    "candidate implementation. Given this task:\n"
                    f"{spec[:700]}\n\n"
                    "and the function/class name(s) "
                    f"{', '.join(fn_names) if fn_names else 'you infer from the task'},\n"
                    f"write {int(n_asserts)} assert statements that a NAIVE or "
                    "SUBTLY-WRONG implementation would FAIL: boundary inputs "
                    "(empty, single-element), repeated/adjacent separators, "
                    "leading/trailing whitespace, very large inputs, and every "
                    "edge case the task implies. Your asserts MUST call the "
                    "exact name(s) listed above and encode the CORRECT expected "
                    "behavior. "
                    "CRITICAL: compute each expected value BY HAND from the "
                    "spec — never copy whitespace or formatting from the input "
                    "string into the expected value. "
                    "Output ONLY the assert statements, one per line.")
                adv_body = {
                    "model": model,
                    "messages": [{"role": "user", "content": adv_prompt}],
                    "stream": False,
                    "options": {"temperature": gen_temp, "num_predict": 500},
                }
                adv_detail = "unknown"
                adv_failures = 0
                for _adv in range(2):
                    try:
                        advd = call_ollama_chat(adv_body, timeout=120.0)
                        adv_asserts = (advd.get("message", {}).get("content", "") or "").strip()
                        m3 = re.search(r"```(?:python|py)?\s*\n(.*?)```",
                                       adv_asserts, re.DOTALL)
                        if m3:
                            adv_asserts = m3.group(1)
                        adv_lines = [ln.strip() for ln in adv_asserts.splitlines()
                                     if ln.strip().startswith("assert")]
                        if not adv_lines:
                            adv_detail = "adversarial: no assert lines"
                            continue
                        adv_src = code + "\n\n" + "\n".join(adv_lines)
                        ap = subprocess.run(
                            [sys.executable, "-c", adv_src],
                            capture_output=True, text=True, timeout=10)
                        if ap.returncode == 0:
                            adv_detail = (f"adversarial pass "
                                          f"({len(adv_lines)} asserts)")
                            break  # one clean adversarial batch is enough
                        adv_detail = (ap.stderr or ap.stdout or "").strip()[:200]
                        if ("SyntaxError" in adv_detail
                                or "IndentationError" in adv_detail):
                            continue  # probe-side syntax error -> regenerate
                        # Candidate FAILED the adversarial batch. Reference-
                        # anchor it the same way as the standard batch: an
                        # adversarial assert only counts if the reference
                        # (independent t_low solution) passes it. Two
                        # reference-anchored failures = genuine rejection.
                        adv_rf = _reliable_frac(code, adv_lines)
                        if adv_rf >= float(agree_frac):
                            adv_detail = (f"adversarial reference-anchored "
                                          f"{adv_rf:.2f}")
                            break  # not actually failing on reliable asserts
                        adv_failures += 1
                        adv_detail = (f"adversarial anchored {adv_rf:.2f} "
                                      f"(raw {adv_detail})")
                    except subprocess.TimeoutExpired:
                        adv_detail = "adversarial: exec timeout"
                        break
                    except Exception as e:
                        adv_detail = f"adversarial: {e}"[:200]
                        break
                if adv_failures >= 2:
                    # Both adversarial batches reference-anchor-reject the
                    # answer: it genuinely fails the task's edge cases.
                    verified = False
                    detail = adv_detail
                else:
                    # ── MUTATION GATE (E applied to the probe itself) ──
                    # The adversarial batch passed, but verification still
                    # rests on the probe's own asserts — which can be VACUOUS
                    # even when they pass. The can_finish false-verify at 7b:
                    # same-model reference shared the candidate's blind spot,
                    # so reference-anchoring self-confirmed a WRONG answer.
                    # Mutate the reference (skip-last-iteration / skip-first-
                    # edge) and require the assert set to KILL a minimum
                    # fraction of mutants. Low kill-rate = the asserts don't
                    # discriminate real bugs = verification untrustworthy ->
                    # escalate instead of trusting. Fires only on this
                    # all-clean path (the risky one); darwin-evolvable.
                    mut_on = int(self.physics_params.get("probe_mut", 1)) if \
                        hasattr(self, "physics_params") else 1
                    if mut_on:
                        last_asserts = (locals().get("adv_lines") or
                                        locals().get("assert_lines") or [])
                        mk = _mut_kill_frac(last_asserts)
                        mut_thresh = float(
                            self.physics_params.get("probe_mut_kill", 0.5)) \
                            if hasattr(self, "physics_params") else 0.5
                        if last_asserts and mk < mut_thresh:
                            verified = False
                            detail = (f"mut-gate {mk:.2f} < {mut_thresh:.2f} "
                                      f"(vacuous asserts, "
                                      f"{len(last_asserts)} lines)")
                        else:
                            detail = f"{frac:.2f} + {adv_detail}"
                    else:
                        detail = f"{frac:.2f} + {adv_detail}"
            else:
                detail = f"{frac:.2f}" if verified else f"{first_err}"
            return verified, detail
        except Exception as e:
            print(f"[kai_bridge] verify_answer_exec warn: {e}",
                  file=sys.stderr, flush=True)
            return False, "verify error"

    def update_tau(self, response_novelty: float, vfe_tau_rate: float, tau_min: float, tau_max: float):
        """Post-response tau update — zero-centered on the novelty baseline.

        The old law τ ← τ·√(1 − rate·VFE) was a ONE-WAY ratchet (factor ≤ 1):
        tau only ever shrank and pinned at tau_min, so the controller looked
        dead to the physics bench (tau spread = 0.000 = liveness falsified).
        Mirror the phen law (phen_continuity._dilate, the verified law):
        gate = 0.5 − sigmoid((η − η₀)/0.25), neutral at baseline novelty η₀=0.5,
        dilates when the response is confidently low-novelty, shrinks when
        surprised. Tau then moves up AND down across queries → the bench can
        see it react. Persists to disk."""
        vfe = max(0.0, min(1.0, response_novelty))
        gate = 0.5 - _sigmoid((vfe - 0.5) / 0.25)
        self.tau = max(tau_min, min(tau_max, self.tau * (1.0 + vfe_tau_rate * gate)))
        self._save_tau()

    def compute_novelty(self, text: str) -> float:
        """Surrogate text novelty [0,1] — semantic diversity proxy.
        Only used as seed for native VFE sampler; actual physics is C++."""
        if not text:
            return 0.5  # neutral default
        # Use first ~100 chars, lower sensitivity
        sample = text[:100]
        if len(sample) < 3:
            return 0.5
        unique = len(set(sample.lower()))
        # Saturation at ~20 unique chars
        diversity = min(1.0, unique / 20.0)
        # Short texts aren't automatically novel; weight toward medium
        return 0.3 + 0.4 * diversity

    # ── Action → observation physics (Tier 1: close perceive-act-observe) ──
    # Every tool execution feeds its result back INTO the physics layer:
    # g_ij novelty (embedding distance from the recent observation window),
    # curvature (temporal second-difference of that novelty stream — Ricci
    # proxy), and a tau update driven by OBSERVATION novelty instead of
    # response-text novelty. This is the loop the physics bench can falsify:
    # action consequences must move the state, not just be echoed to the model.

    OBS_WINDOW = 8  # rolling window of recent observation embeddings

    def __init__(self):
        self.tau = self._load_tau()
        self.novelty = 0.5
        # Multi-prior registration (LAYER 3c): WHICH domain is being pulled.
        # Loaded from .axiom_state/domain_priors.json at startup if present.
        self.domain_priors: list = []  # list of {"name": str, "centroid": list[float]}
        self._load_domain_priors()
        # Observation-physics state (persisted so curvature survives restarts).
        self.obs_ema = 0.5      # running EMA of observation novelty
        self.obs_prev = 0.5     # previous novelty (for second-difference)
        self.obs_curvature = 0.0
        self.obs_window: list = []  # recent embeddings, bounded
        self._load_obs_state()

    def _load_obs_state(self):
        try:
            if os.path.exists(TAU_STATE_FILE):
                with open(TAU_STATE_FILE) as f:
                    d = json.load(f)
                self.obs_ema = float(d.get("obs_ema", 0.5))
                self.obs_prev = float(d.get("obs_prev", 0.5))
                self.obs_curvature = float(d.get("obs_curvature", 0.0))
        except Exception:
            pass

    def observe_action(self, tool_name: str, obs: str) -> dict:
        """Feed one executed tool's result back into the physics layer.

        Returns the physics readout {(g_novelty, curvature, tau)} so the
        bridge can log it — the action's world response becomes measurable
        state, exactly what kai_physics_bench's LIVENESS check demands.
        """
        text = obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False)
        emb = self._embed_query(f"{tool_name}: {text[:800]}")
        if emb is None:
            # Degrade: textual novelty proxy (no embedding service right now).
            nov = self.compute_novelty(text)
        else:
            # g_ij: geodesic distance in representation space between this
            # observation and the recent-window mean (novelty = world pushed
            # back with something NEW).
            if self.obs_window:
                dim = len(emb)
                mean = [sum(w[i] for w in self.obs_window) / len(self.obs_window)
                        for i in range(dim)]
                na = sum(x * x for x in emb) ** 0.5
                nb = sum(x * x for x in mean) ** 0.5
                if na > 1e-8 and nb > 1e-8:
                    cos = sum(emb[i] * mean[i] for i in range(dim)) / (na * nb)
                else:
                    cos = 0.0
                nov = max(0.0, min(1.0, 1.0 - cos))  # distance from known
            else:
                nov = 0.5
            self.obs_window = (self.obs_window + [emb])[-self.OBS_WINDOW:]

        # Curvature: temporal second-difference of novelty (Ricci proxy).
        curvature = nov - 2.0 * self.obs_ema + self.obs_prev
        self.obs_prev = self.obs_ema
        self.obs_ema = 0.8 * self.obs_ema + 0.2 * nov
        self.obs_curvature = 0.5 * self.obs_curvature + 0.5 * curvature

        # Tau: observation novelty drives the same law as response novelty —
        # confident (low novelty) dilates, surprising world-response freezes.
        old_tau = self.tau
        self.update_tau(nov, vfe_tau_rate=self.physics_params.get("vfe_tau_rate", 0.061),
                        tau_min=self.physics_params.get("tau_min", 0.855),
                        tau_max=self.physics_params.get("tau_max", 1.657))
        readout = {
            "g_novelty": round(nov, 4),
            "curvature": round(self.obs_curvature, 4),
            "ema": round(self.obs_ema, 4),
            "tau": round(self.tau, 4),
            "tau_delta": round(self.tau - old_tau, 4),
            "tool": tool_name,
        }
        self._last_obs_emb = emb  # reuse in store_obs (no second embed)
        # Persist curvature state alongside tau (survives restarts).
        with BRIDGE_LOCK:
            try:
                os.makedirs(os.path.dirname(TAU_STATE_FILE), exist_ok=True)
                state = {"tau": self.tau, "obs_ema": self.obs_ema,
                         "obs_prev": self.obs_prev, "obs_curvature": self.obs_curvature}
                with open(TAU_STATE_FILE, "w") as f:
                    json.dump(state, f)
            except Exception:
                pass
        return readout


# ── Ollama API call ───────────────────────────────────────────────────────

OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

def call_ollama_chat(body: dict, timeout: float = 120.0) -> dict:
    """Call Ollama /api/chat with the given body dict. Returns parsed JSON.
    timeout: per-request HTTP timeout. Cold model loads on the 6GB card can
    exceed 120s (gemma4:12b ~162s), so fuse teacher calls pass a larger
    budget; the default stays 120s for the interactive chat path."""
    # gemma4 is a "thinking" model by default: it fills `message.thinking`
    # and returns EMPTY `message.content`. For code generation we need the
    # final answer in content — disable thinking for gemma models.
    if "gemma" in body.get("model", "").lower():
        body = {**body, "think": False}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable: {e.reason}")


# ── OpenAI → Ollama message conversion ────────────────────────────────────

def convert_openai_to_ollama(openai_msg: dict) -> dict:
    """Convert an OpenAI message dict to Ollama message format.

    Handles tool_calls and tool_result roles.
    """
    role = openai_msg.get("role", "user")
    content = openai_msg.get("content", "")

    if isinstance(content, list):
        # Multimodal content — extract text
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
        content = " ".join(text_parts)

    ollama_msg = {"role": role, "content": content}

    # Tool calls from assistant
    if role == "assistant" and "tool_calls" in openai_msg:
        ollama_msg["tool_calls"] = []
        for tc in openai_msg["tool_calls"]:
            func = tc.get("function", {})
            ollama_msg["tool_calls"].append({
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                },
            })

    return ollama_msg


def convert_tools(tools: list) -> list:
    """Convert OpenAI tools array to Ollama tools format."""
    if not tools:
        return []
    ollama_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                },
            })
    return ollama_tools


# ── Live physics params (darwin-promoted, closed-loop) ────────────────────

PHYSICS_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   ".kai_physics_params.json")

class PhysicsParams:
    """Live physics params, overridable by .kai_physics_params.json.

    The darwin promotion path (kai darwin promote -> JSON param patch) writes
    this file; the bridge loads it at boot AND live-reloads it on every chat
    request. This closes the self-improvement loop end-to-end:
        evolve -> promote (safety-gated) -> params file -> live bridge ->
        bench delta -> next generation.
    """

    DEFAULTS = {
        "base_temperature": 0.33,
        "top_p": 0.997,
        "novelty_scale": 0.30,
        "vfe_tau_rate": 0.061,
        "tau_min": 0.855,
        "tau_max": 1.657,
        # Attempt-gated escalation controller (LAYER 2b). THREE tiers now:
        # t_low (confident first try) -> t_mid (first retry samples the
        # competence window that fixed-0.7 empirically hits on matrix_transpose
        # at 3b/7b/12b) -> t_high (exploration). The old 2-tier controller
        # (0.15 -> 1.6) jumped clean over the ~0.6 window, so matrix_transpose
        # was structurally unsolvable under escalation. The ramp is
        # darwin-evolvable.
        "t_low": 0.15,
        "t_mid": 0.60,
        "t_high": 1.60,
        # Auto self-recall knobs (LAYER 1), darwin-evolvable.
        "recall_thin_len": 40,
        "recall_sim_gate": 0.45,
        "recall_top_k": 3,
        "recall_ctx_chars": 300,
        # Autonomous closed loop (LAYER 2c): bridge self-verifies and
        # escalates WITHOUT an external attempt field. auto_loop=1 enables
        # it for production requests; probe knobs are darwin-evolvable.
        # probe_mode: 0 = reviewer (LLM judges its own code — falsified at
        # 3b, never escalates); 1 = execution (code is run against
        # self-generated asserts — deterministic, world-grounded signal).
        "auto_loop": 0,
        "auto_max_attempts": 3,
        "probe_mode": 1,
        "probe_verify_temp": 0.0,
        "probe_n_verify": 1,
        "probe_agree_frac": 1.0,
        "probe_gen_temp": 0.4,
        # Number of assert statements the exec probe generates per batch.
        # 3 missed the hidden-test failures (matrix_transpose false-verified
        # at 7b: self-asserts passed, 6 hidden tests failed). More asserts
        # per batch = stricter world-grounded signal, darwin-evolvable.
        "probe_n_asserts": 5,
        # Adversarial second-pass gate: when the standard assert batch passes
        # FULLY clean (the false-verification blind spot — reverse_words
        # verified at agree=1.00 with single-space asserts while the hidden
        # multi-space test failed), run a second batch explicitly told to
        # BREAK naive implementations. Answer verified only if it passes too.
        # Fires only on the all-clean path; darwin-evolvable.
        "probe_adversarial": 1,
        # Mutation gate (E applied to the probe itself): on the all-clean
        # pass path, mutate the reference implementation (off-by-one, nuked
        # return) and require the assert set to KILL at least this fraction
        # of mutants. A low kill-rate = vacuous/self-confirming asserts (the
        # can_finish false-verify at 7b: same-model reference shared the
        # candidate's blind spot, asserts verified a WRONG answer). Gate
        # failure = verification untrustworthy -> escalate. Darwin-evolvable.
        "probe_mut": 1,
        "probe_mut_kill": 0.5,
    }

    def __init__(self):
        self.values = dict(self.DEFAULTS)
        self._mtime = -1.0
        self.reload()

    def reload(self):
        try:
            mtime = os.path.getmtime(PHYSICS_PARAMS_PATH)
            if mtime == self._mtime:
                return
            with open(PHYSICS_PARAMS_PATH) as f:
                data = json.load(f)
            for k in self.DEFAULTS:
                if k in data and isinstance(data[k], (int, float)):
                    self.values[k] = float(data[k])
            self._mtime = mtime
        except FileNotFoundError:
            self._mtime = -1.0
        except Exception:
            pass

    def get(self, key, default=None):
        self.reload()
        return self.values.get(key, default)

    def snapshot(self):
        self.reload()
        return dict(self.values)


# ── HTTP handler ──────────────────────────────────────────────────────────

class KaiBridgeHandler(BaseHTTPRequestHandler):
    vfe_state = VFEState()  # shared across requests (class variable)
    corpus = CorpusAttractor()  # loaded once at startup
    physics_params = PhysicsParams()  # darwin-promoted params, live-reloadable

    def log_message(self, format, *args):
        sys.stderr.write("[kai_bridge] %s\n" % (format % args))

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream_chunk(self, data: dict):
        """Send one SSE chunk."""
        line = "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
        self.wfile.write(line.encode("utf-8"))

    def _send_stream_done(self):
        self.wfile.write(b"data: [DONE]\n\n")

    # ── Test-time self-recall loop (LAYER 3f / plan #4) ─────────────────

    def _refine_once(self, q: str, model: str, top_k: int, temperature: float) -> dict:
        """Draft → recall → re-answer, bounded to ONE retry. Actuates on thin
        drafts (smoke-then-recall) OR strong recall (top1_sim >= 0.70).
        Returns the loop trace so callers can audit the recall delta:
        {answer, retried, draft_thin, recalled_top1_sim, recall_delta, ...}"""
        om = model.split("/", 1)[-1] if "/" in model else model
        draft = self._llm_once(om, q, temperature)
        t0 = time.time()
        hits = []
        try:
            hits = self.corpus.recall(q, top_k=top_k, kind="all")
        except Exception:
            hits = []
        recall_time = time.time() - t0

        top1_sim = hits[0]["similarity"] if hits else 0.0
        thin = len(draft.strip()) < 40
        retried = False
        final = draft

        # Two actuation paths (bounded to ONE retry):
        #   A. Smoke-then-recall: draft is thin AND memory is meaningful
        #      (top-1 sim >= 0.40) -> inject context, re-answer.
        #   B. Grounding-recall: draft is substantive but recall is STRONG
        #      (top-1 sim >= 0.70). Inject the retrieved memory and re-answer —
        #      the loop actuates on high-similarity memory, not only on thin
        #      drafts. Turns passive recall into an active perceive->act loop.
        inject = (thin and hits and top1_sim >= 0.40) or (
            not thin and hits and top1_sim >= 0.70)
        if inject:
            ctx = "[retrieved via test-time self-recall]\n"
            for h in hits[:min(3, len(hits))]:
                ctx += f"  ({h['similarity']:.2f}) {h['path']}: {h['text'][:220]}\n"
            msgs = [
                {"role": "system", "content": "Answer using the recalled memory above. Be concise."},
                {"role": "user", "content": f"{ctx}\n\nQuestion: {q}"},
            ]
            try:
                pp = self.physics_params
                # Grounded re-answer samples COLD: the recalled memory is the
                # evidence, so the model should exploit it, not explore around
                # it. Falsified in recall_darwin_stack bench (+0.11): the
                # darwin base_temperature=1.40 (tuned for the exploratory raw
                # path) made the grounded re-answer creative-but-imprecise.
                # Cold grounding: T_eff = min(0.5, darwin base) * tau.
                grounded_base = min(pp.get("base_temperature", temperature), 0.5)
                body = {
                    "model": om, "messages": msgs, "stream": False,
                    "options": {
                        "temperature": round(self.vfe_state.compute_temperature(
                            grounded_base,
                            pp.get("novelty_scale", 0.30)), 4),
                        "top_p": pp.get("top_p", 0.997),
                        "num_predict": 400,
                        "kai_vfe": True,
                        "kai_tau": self.vfe_state.tau,
                        "kai_novelty_scale": pp.get("novelty_scale", 0.30),
                        "kai_vfe_tau_rate": pp.get("vfe_tau_rate", 0.061),
                        "kai_tau_min": pp.get("tau_min", 0.855),
                        "kai_tau_max": pp.get("tau_max", 1.657),
                        "kai_attractor_path": "/tmp/kai_data/attractor.bin",
                    },
                }
                resp = call_ollama_chat(body)
                r = resp.get("message", {}).get("content", "").strip()
                if r:
                    retried = True
                    final = r
            except Exception:
                pass  # keep draft

        # delta = how much the recalled memory raised the grounding signal
        delta = top1_sim
        # "log the delta" (plan #4)
        try:
            import os as _os
            ledger_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                        ".kai_refine_ledger.jsonl")
            rec = {
                "t": time.time(), "q": q[:200], "model": om,
                "draft_len": len(draft), "thin_and_retried": retried,
                "retried": retried, "top1_sim": round(top1_sim, 3),
                "actuated": "thin" if (thin and retried) else ("grounding" if (not thin and retried) else "none"),
                "recall_ms": round(recall_time * 1000, 1),
            }
            with open(ledger_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return {
            "query": q,
            "answer": final,
            "retried": retried,
            "draft_thin": thin,
            "recalled_top1_sim": round(top1_sim, 3),
            "recalled_hits": len(hits),
            "recall_ms": round(recall_time * 1000, 1),
            "recall_delta": round(delta, 3),
        }

    def _llm_once(self, model: str, q: str, temperature: float) -> str:
        """One unstreamed LLM call for a single question, with a short answer
        budget and a fail-fast timeout. Returns stripped text.

        Physics-wired: applies the live darwin-promoted params (base_temperature,
        novelty_scale, top_p, tau band) exactly like the chat path, so the
        refine/recall loop stacks the darwin lever on top of the recall lever."""
        om = model.split("/", 1)[-1] if "/" in model else model
        try:
            pp = self.physics_params
            novelty = self.vfe_state.compute_novelty(q)
            adapted_temp = self.vfe_state.compute_temperature(
                pp.get("base_temperature", temperature), pp.get("novelty_scale", 0.30))
            # Difficulty-gated temperature (same probe as the chat path): the
            # refine/recall loop's drafts are generated at THIS temperature, so
            # easy queries answer near-greedy and hard ones explore.
            use_temp = adapted_temp
            if os.environ.get("KAI_BRIDGE_PROBE", "on") != "off":
                p_temp, _ = self.vfe_state.probe_difficulty(om, q, max_probe_tokens=160)
                if p_temp is not None:
                    use_temp = p_temp
            body = {
                "model": om,
                "messages": [{"role": "user", "content": q}],
                "stream": False,
                "options": {
                    "temperature": round(use_temp, 4),
                    "top_p": pp.get("top_p", 0.997),
                    "num_predict": 300,
                    "kai_vfe": True,
                    "kai_tau": self.vfe_state.tau,
                    "kai_novelty_scale": pp.get("novelty_scale", 0.30),
                    "kai_vfe_tau_rate": pp.get("vfe_tau_rate", 0.061),
                    "kai_tau_min": pp.get("tau_min", 0.855),
                    "kai_tau_max": pp.get("tau_max", 1.657),
                    "kai_attractor_path": "/tmp/kai_data/attractor.bin",
                },
            }
            resp = call_ollama_chat(body)
            msg = resp.get("message", {}) or {}
            ans = (msg.get("content") or "").strip()
            if not ans:
                ans = (msg.get("thinking") or "").strip()
            return ans.strip()
        except Exception:
            return ""

    # ── LAYER 4: multi-teacher Kalman fusion (ceiling raiser) ─────────────────
    # Tiers >3B by routing each query region to the teacher whose epistemic
    # uncertainty is lowest. Each teacher answers via Ollama (no GGUF load →
    # no OOM); per-answer uncertainty is estimated proxy-style (char-entropy:
    # a confident, converged answer has low lexical entropy; a hedged/garbled
    # one has high entropy). The scalar quality proxy (length-normalized) is
    # inverse-variance Kalman-weighted across teachers — the bridge-side mirror
    # of vfe.rs::kalman_fuse_sources. The best-expert answer is returned.

    # Tier ceiling raised 2026-08-16: swap doubled (2G -> 16G), so the 12b
    # teacher now fits (4GB VRAM + RAM offload, ~9s warm). Replaces tinyllama
    # as the weakest teacher. 16b remains OOM-excluded on the 6GB card.
    FUSION_TEACHERS = ["gemma4:12b", "qwen2.5-coder:7b", "qwen2.5-coder:3b"]

    def _teacher_answer(self, om: str, q: str, temperature: float,
                        max_tokens: int) -> dict:
        """One teacher's answer + epistemic proxy stats. Returns
        {model, answer, uncertainty (variance), quality (scalar value)}."""
        try:
            # Ollama models are stored without the "kai/" alias prefix.
            # Strip it so e.g. "kai/qwen2.5-coder:7b" maps to the real tag.
            ollama_model = om[4:] if om.startswith("kai/") else om
            # Physics-wired adaptive temperature: read the live VFE tau
            # (time-dilation) from the phen-trace and modulate the sampling
            # temperature as T_eff = base * tau. Higher tau → more exploratory
            # sampling in high-uncertainty regions. Falls back to base if no
            # trace is available yet.
            tau = 1.0
            for _src in (".axiom_state/phen_trace.jsonl",
                         "rust/kai-fusion/.axiom_state/phen_trace.jsonl"):
                try:
                    if os.path.exists(_src):
                        with open(_src) as _f:
                            lines = _f.readlines()
                        if lines:
                            tau = json.loads(lines[-1]).get("tau", 1.0)
                except Exception:
                    pass
            eff_temp = temperature * (tau if tau > 0.0 else 1.0)
            body = {
                "model": ollama_model, "messages": [{"role": "user", "content": q}],
                "stream": False,
                "options": {"temperature": round(eff_temp, 4), "num_predict": max_tokens},
                # Keep the teacher resident for 5 min: serial 3-teacher dispatch
                # on the 6GB card otherwise cold-loads (and evicts) each model
                # per query, ~2-3 min each -> fuse becomes unusably slow.
                "keep_alive": "5m",
            }
            resp = call_ollama_chat(body, timeout=300.0)
            msg = resp.get("message", {}) or {}
            # Reasoning models (gemma4:12b) put the answer in `thinking` with
            # empty `content`; prefer content, fall back to thinking.
            ans = (msg.get("content") or "").strip()
            if not ans:
                ans = (msg.get("thinking") or "").strip()
            ans = ans.strip()
        except Exception as e:
            return {"model": om, "answer": "", "uncertainty": 1.0,
                    "quality": 0.0, "error": str(e)}
        if not ans:
            return {"model": om, "answer": "", "uncertainty": 1.0, "quality": 0.0}
        # Epistemic uncertainty proxy: Shannon entropy of the char distribution.
        # Low entropy => confident/converged answer; high => hedged or garbled.
        from collections import Counter
        counts = Counter(ans.lower())
        total = sum(counts.values())
        ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
        # normalize by theoretical max for this length
        n_unique = len(counts)
        max_ent = math.log2(n_unique) if n_unique > 1 else 1.0
        norm_ent = (ent / max_ent) if max_ent > 0 else 0.0
        uncertainty = max(1e-3, min(1.0, norm_ent))  # variance proxy (0,1]
        # Quality proxy: length (substance) × coherence (1-uncertainty)
        quality = min(1.0, len(ans) / 200.0) * (1.0 - uncertainty)
        return {"model": om, "answer": ans, "uncertainty": uncertainty,
                "quality": quality, "len": len(ans)}

    def _fuse_multi_teacher(self, q: str, temperature: float, max_tokens: int,
                            teachers: list) -> dict:
        """Dispatch q to N teachers, Kalman-fuse their scalar quality estimates
        (inverse-variance), and route to the best (lowest-uncertainty) answer.
        Mirrors vfe.rs::kalman_fuse_sources over the per-model Estimate stream."""
        estimates = []
        per_teacher = []
        for om in teachers:
            t = self._teacher_answer(om, q, temperature, max_tokens)
            per_teacher.append(t)
            estimates.append({"model": om, "value": t["quality"],
                              "variance": t["uncertainty"]})
        # Inverse-variance (Kalman) fusion of the scalar quality estimates
        # f = Σ(v_i/var_i) / Σ(1/var_i), var_f = 1/Σ(1/var_i)
        w_sum = sum(1.0 / e["variance"] for e in estimates)
        fused_value = sum(e["value"] / e["variance"] for e in estimates) / w_sum
        fused_var = 1.0 / w_sum
        # Route to the teacher with lowest uncertainty (best expert for region)
        ranked = sorted(per_teacher, key=lambda x: x["uncertainty"])
        best = ranked[0]
        return {
            "query": q,
            "answer": best["answer"],
            "routed_to": best["model"],
            "fused_quality": round(fused_value, 4),
            "fused_uncertainty": round(fused_var, 4),
            "per_teacher": [
                {"model": t["model"], "uncertainty": round(t["uncertainty"], 4),
                 "quality": round(t["quality"], 4), "len": t.get("len", 0)}
                for t in per_teacher
            ],
        }


    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/models":
            models = [
                {"id": "kai/tinyllama", "object": "model", "created": 1720000000, "owned_by": "kai"},
                {"id": "kai/qwen2.5-coder:3b", "object": "model", "created": 1720000000, "owned_by": "kai"},
                {"id": "kai/qwen2.5-coder:7b", "object": "model", "created": 1720000000, "owned_by": "kai"},
                {"id": "kai/qwen2.5:7b", "object": "model", "created": 1720000000, "owned_by": "kai"},
                {"id": "kai/qwen3.5:9b", "object": "model", "created": 1720000000, "owned_by": "kai"},
                {"id": "kai/deepseek-coder-v2:16b", "object": "model", "created": 1720000000, "owned_by": "kai"},
                {"id": "kai/gemma4:12b", "object": "model", "created": 1720000000, "owned_by": "kai"},
            ]
            self._send_json(200, {"object": "list", "data": models})
        elif parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif parsed.path == "/v1/recall":
            # Explicit memory recall: ?q=<query>&top_k=<n> — like a human
            # deliberately recalling relevant past experiences. Scoped by
            # optional ?kind=chat|doc|all (default all).
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0]
            kind = (qs.get("kind") or ["all"])[0]
            top_k = min(20, int((qs.get("top_k") or ["5"])[0]))
            if not q.strip():
                self._send_json(400, {"error": "missing q parameter"})
                return
            results = self.corpus.recall(q, top_k=top_k, kind=kind)
            self._send_json(200, {"query": q, "kind": kind, "recalled": results})
        elif parsed.path == "/v1/refine":
            # LAYER 3f: test-time self-recall loop (plan #4). Smoke-then-recall:
            # draft a raw answer, deliberately recall relevant memory, and only
            # re-answer with the recalled context when the draft is thin or the
            # recalled top-1 similarity clears the injection threshold. Bounded
            # to ONE retry by default (extra inference cost per turn).
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0]
            model = (qs.get("model") or ["kai/qwen2.5-coder:3b"])[0]
            top_k = min(20, int((qs.get("top_k") or ["3"])[0]))
            kai_temperature = float((qs.get("temperature") or ["0.7"])[0])
            if not q.strip():
                self._send_json(400, {"error": "missing q parameter"})
                return
            result = self._refine_once(q, model, top_k, kai_temperature)
            self._send_json(200, result)
        elif parsed.path == "/v1/fuse":
            # LAYER 4: multi-teacher Kalman fusion. Route q to the teacher with
            # lowest epistemic uncertainty and return the fused quality stats.
            # ?q=<query>&teachers=a,b,c&temperature=0.7&max_tokens=80
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            q = (qs.get("q") or [""])[0]
            temperature = float((qs.get("temperature") or ["0.7"])[0])
            max_tokens = int((qs.get("max_tokens") or ["80"])[0])
            teachers = (qs.get("teachers") or [",".join(self.FUSION_TEACHERS)])[0]
            teachers = [t.strip() for t in teachers.split(",") if t.strip()]
            if not q.strip():
                self._send_json(400, {"error": "missing q parameter"})
                return
            with BRIDGE_LOCK:
                res = self._fuse_multi_teacher(q, temperature, max_tokens, teachers)
            self._send_json(200, res)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/v1/recall", "/recall"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json(400, {"error": "empty request"})
                return
            try:
                req = json.loads(self.rfile.read(content_length))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return
            q = req.get("query") or req.get("q") or ""
            kind = req.get("kind", "all")
            top_k = min(20, int(req.get("top_k", 5)))
            if not q.strip():
                self._send_json(400, {"error": "missing query"})
                return
            results = self.corpus.recall(q, top_k=top_k, kind=kind)
            self._send_json(200, {"query": q, "kind": kind, "recalled": results})
            return
        if parsed.path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "empty request"})
            return

        body = self.rfile.read(content_length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        # Parse request
        model_raw = req.get("model", "kai/tinyllama")
        # Strip "kai/" prefix to get the Ollama model name
        ollama_model = model_raw.split("/", 1)[-1] if "/" in model_raw else model_raw
        messages = req.get("messages", [])
        stream = req.get("stream", False)
        max_tokens = req.get("max_tokens", 256)
        temperature = req.get("temperature", 0.7)
        attempt = int(req.get("attempt", 0) or 0)  # retry index (0 = first try)
        tools = req.get("tools", [])
        tool_choice = req.get("tool_choice", "auto")

        # ── SOUL injection (W1): unconditional identity prepend ──────
        # Every kai/ model answers as the Axiom Alien no matter what the
        # caller sent. Closes the "raw path yields an un-Kai'ed model" hole.
        # Override with env KAI_BRIDGE_SOUL=off only for raw-op testing.
        if SOUL_INJECT_ENV != "off":
            messages.insert(0, {"role": "system", "content": SOUL_SYSTEM})

        # ── Corpus context injection ──────────────────────────────────
        # Search attractor for relevant directories, inject as system context.
        # Knobs are darwin-evolvable via the live params file: recall_top_k
        # (how many hits), recall_ctx_chars (per-hit context budget), and
        # recall_sim_gate (min similarity to inject — the FUSE gate, so a
        # tuned gate prevents irrelevant memory from entering the prompt).
        # This is the fuse path that the memory-grounded ruler grades.
        user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if user_text.strip():
            rk = max(1, int(self.physics_params.get("recall_top_k", 3)))
            rc = max(50, int(self.physics_params.get("recall_ctx_chars", 300)))
            rg = float(self.physics_params.get("recall_sim_gate", 0.45))
            hits = self.corpus.search(user_text, top_k=rk)
            hits = [h for h in hits if h["similarity"] >= rg]
            if hits:
                context_lines = ["[Corpus context from attractor memory]"]
                for h in hits[:rk]:
                    context_lines.append(f"  {h['path']} (sim={h['similarity']}): {h['text'][:rc]}")
                context_str = "\n".join(context_lines[:5])
                print(f"[kai_bridge] attractor hits: {[h['path'] for h in hits]}", file=sys.stderr)
                # Inject as a system message before other messages
                messages.insert(0, {"role": "system", "content": context_str})

        # ── Convert to Ollama format ───────────────────────────────────
        ollama_messages = [convert_openai_to_ollama(m) for m in messages]
        ollama_tools = convert_tools(tools)

        # ── LAYER 3c: multi-prior introspection (WHICH memory is pulled) ─
        # Compute the dominant domain attractor for this input BEFORE
        # sending, so the reply can report which memory bounded the answer.
        dom_name, dom_resp = self.vfe_state.dominant_domain(user_text)

        # ── LAYER 2: driver VFE controller (compute_temperature is used) ─
        # novelty for THIS request drives temperature; native sampler gets
        # the same physics knobs. This is the fix: previously the outgoing
        # temperature was passed through unmodified while compute_temperature
        # sat unused.
        req_novelty = self.vfe_state.compute_novelty(user_text)
        self.vfe_state.novelty = req_novelty
        pp = self.physics_params
        adapted_temp = self.vfe_state.compute_temperature(
            pp.get("base_temperature", temperature), pp.get("novelty_scale", 0.30))

        # ── LAYER 2b: attempt-gated temperature (escalation policy) ────────
        # Replication (code_rep29) falsified the uniform-temp controller:
        # compute_temperature pinned ~1.95 on every code prompt, collapsing
        # physics pass@1 to 0.76 while greedy held 0.90. Attempt-gated
        # escalation needs NO difficulty oracle (both probes — embedding
        # agreement and self-verification — measured too noisy on 3b):
        #   attempt 0 (first try): CONFIDENT temperature (t_low) — near-greedy
        #     quality, preserving pass@1.
        #   attempt > 0 (retry): EXPLORATORY temperature (t_high) — find the
        #     solution greedy cannot reach, preserving pass@K.
        # The retry signal IS the perceive->act loop: only escalate when the
        # first action failed. KAI_BRIDGE_PROBE=on re-enables the optional
        # attempt-0 self-verification probe (measured unreliable, default off).
        pp = self.physics_params
        t_low = float(pp.get("t_low", 0.15))
        t_mid = float(pp.get("t_mid", 0.60))
        t_high = float(pp.get("t_high", 1.60))
        p_agree = None
        if attempt > 0:
            # RAMP, not jump: first retry samples the mid competence window
            # (where fixed-0.7 solves tasks the 0.15/1.6 tiers both miss),
            # later retries go fully exploratory.
            probe_temp = t_mid if attempt == 1 else t_high
            print(f"[kai_bridge] attempt={attempt} -> exploratory T={probe_temp}",
                  file=sys.stderr, flush=True)
        elif os.environ.get("KAI_BRIDGE_PROBE", "off") == "on":
            p_temp, p_agree = self.vfe_state.probe_difficulty(
                ollama_model, user_text, max_probe_tokens=min(max_tokens, 160))
            if p_temp is not None:
                probe_temp = p_temp
                print(f"[kai_bridge] probe: verify={p_agree} -> T={probe_temp} "
                      f"(was {adapted_temp:.3f})", file=sys.stderr, flush=True)
        else:
            probe_temp = t_low  # attempt 0, probe off: confident first try

        # Build options (native VFE sampler handles adaptive physics)
        # Params come from .kai_physics_params.json (darwin-promoted) or the
        # Darwin-evolved defaults: base_temp=0.33, top_p=0.997, tau_rate=0.061,
        # tau_min=0.855, tau_max=1.657
        options = {
            "temperature": probe_temp,
            "top_p": pp.get("top_p", 0.997),
            "num_predict": max_tokens,
            "kai_vfe": True,
            "kai_tau": self.vfe_state.tau,
            "kai_novelty_scale": pp.get("novelty_scale", 0.30),
            "kai_vfe_tau_rate": pp.get("vfe_tau_rate", 0.061),
            "kai_tau_min": pp.get("tau_min", 0.855),
            "kai_tau_max": pp.get("tau_max", 1.657),
            "kai_attractor_path": "/tmp/kai_data/attractor.bin",
        }

        ollama_body = {
            "model": ollama_model,
            "messages": ollama_messages,
            "stream": stream,
            "options": options,
        }
        if ollama_tools:
            ollama_body["tools"] = ollama_tools
            ollama_body["tool_choice"] = tool_choice

        # ── Call Ollama with the LAYER 1 action loop ─────────────────
        # Loop: model emits tool_calls -> we EXECUTE them via the agency,
        # append the observation, re-ask the model (bounded). The agent thus
        # sees its own action's consequence -> closes perceive-act-observe.
        #
        # LAYER 2c AUTONOMOUS LOOP: when auto_loop=1 and the caller did NOT
        # send an external `attempt` retry index, the bridge closes the loop
        # ITSELF: generate at t_low -> self-verify (majority vote) -> if
        # unverified, escalate to t_high and regenerate, bounded by
        # auto_max_attempts. This is the production-closed loop: no harness
        # tells the bridge when it failed; the bridge decides via its own
        # epistemic self-assessment (probe knobs are darwin-evolvable).
        auto_loop_on = (int(req.get("auto", 0) or 0) == 1
                        and float(pp.get("auto_loop", 0)) == 1.0)
        auto_max = int(pp.get("auto_max_attempts", 3))
        auto_attempts = 0
        auto_verified = False
        auto_agreement = 0.0
        # Escalation level: incremented ONLY on real temperature escalations.
        # A fused-recall re-ask (t_low, attempt 1) does not consume a level,
        # so the t_mid competence window is never skipped when recall fires.
        esc_level = 0
        t_start = time.time()
        action_turns = 0
        action_observations = []  # (tool_name, obs)
        max_turns = ACTION_LOOP_TURNS if (ACTION_LOOP_ENABLED and AGENCY is not None) else 0
        while True:
            try:
                # 7b on 6GB VRAM + auto-recall + absorb can exceed the 120s
                # default; a bridge-side timeout kills the client connection
                # mid-request (RemoteDisconnected on the bench). Use the
                # generous 900s ceiling that matches the bench client (the
                # action loop can take ~10 min/task at 7b).
                ollama_resp = call_ollama_chat(ollama_body, timeout=900.0)
            except RuntimeError as e:
                self._send_json(502, {"error": str(e)})
                return
            ollama_msg = ollama_resp.get("message", {})
            resp_content = ollama_msg.get("content", "")
            resp_tool_calls = ollama_msg.get("tool_calls", [])

            # If we've executed no action (or reached the cap / no tool), now
            # we have a final response to hand back: break with this answer.
            if not resp_tool_calls or action_turns >= max_turns:
                # LAYER 2c: autonomous self-verification. If we generated a
                # final answer and the loop is on (and we still have attempts
                # left), verify it; on failure, escalate temperature and
                # regenerate. The bridge decides — not the caller.
                if (auto_loop_on and resp_content.strip()
                        and auto_attempts < auto_max):
                    if int(pp.get("probe_mode", 1)) == 0:
                        auto_verified, auto_agreement = self.vfe_state.verify_answer(
                            resp_content, ollama_model,
                            verify_temp=float(pp.get("probe_verify_temp", 0.0)),
                            n_samples=int(pp.get("probe_n_verify", 1)),
                            agree_frac=float(pp.get("probe_agree_frac", 1.0)))
                    else:
                        auto_verified, auto_agreement = self.vfe_state.verify_answer_exec(
                            resp_content, ollama_model,
                            n_samples=int(pp.get("probe_n_verify", 1)),
                            agree_frac=float(pp.get("probe_agree_frac", 1.0)),
                            gen_temp=float(pp.get("probe_gen_temp", 0.4)),
                            question=user_text)
                    auto_attempts += 1
                    if auto_verified:
                        print(f"[kai_bridge] auto-verify PASSED attempt {auto_attempts} "
                              f"(agree={auto_agreement}) — returning at T={probe_temp}",
                              file=sys.stderr, flush=True)
                        break
                    # ── FUSED CLOSED LOOP (recall BEFORE escalation) ──
                    # On the FIRST verification failure, compose the two
                    # biggest measured edges: inject corpus memory for the
                    # question and re-ask at the confident temperature before
                    # overheating. If memory can supply the missing fact/pattern
                    # (memory ruler: raw 0.429 -> fused 1.000), the loop closes
                    # at t_low instead of gambling at t_mid/t_high. The recall
                    # re-ask does NOT consume an escalation level (it is still
                    # a confident-temp attempt) — so the t_mid competence
                    # window is never skipped when recall fires.
                    if auto_attempts == 1 and self.corpus is not None:
                        try:
                            hits = self.corpus.search(user_text,
                                                      top_k=int(pp.get("recall_top_k", 3)))
                            best = hits[0]["similarity"] if hits else 0.0
                            if hits and best >= float(pp.get("recall_sim_gate", 0.45)):
                                ctx = "[retrieved via fused auto self-recall]\n"
                                ctx_chars = int(pp.get("recall_ctx_chars", 300))
                                for h in hits[: int(pp.get("recall_top_k", 3))]:
                                    ctx += f"  {h['path']} (sim={h['similarity']:.2f}): {h['text'][:ctx_chars]}\n"
                                reask_msgs = [{"role": "system", "content": ctx}] + \
                                    [convert_openai_to_ollama(m) for m in ollama_messages]
                                options["temperature"] = float(pp.get("t_low", 0.15))
                                probe_temp = float(pp.get("t_low", 0.15))
                                ollama_body["options"] = options
                                ollama_body["messages"] = reask_msgs
                                print(f"[kai_bridge] auto-verify FAILED (agree={auto_agreement}) "
                                      f"attempt {auto_attempts}/{auto_max} -> fused recall "
                                      f"(best={best:.2f}) re-ask at T={probe_temp}",
                                      file=sys.stderr, flush=True)
                                action_turns = 0
                                action_observations = []
                                continue
                        except Exception as e:
                            print(f"[kai_bridge] fused recall warn: {e}",
                                  file=sys.stderr, flush=True)
                    # Escalate on the RAMP: escalation level 0 -> t_mid
                    # (the competence window), level 1+ -> t_high. esc_level
                    # is incremented ONLY here, so a fused-recall re-ask
                    # (t_low) never steals the t_mid tier.
                    esc_level += 1
                    nxt = (float(pp.get("t_mid", 0.60))
                           if esc_level == 1
                           else float(pp.get("t_high", 1.60)))
                    options["temperature"] = nxt
                    probe_temp = nxt
                    ollama_body["options"] = options
                    # Re-arm messages from the ORIGINAL conversation (the
                    # failed draft is NOT fed back — exploration is fresh,
                    # not self-reinforcing).
                    ollama_body["messages"] = ollama_messages
                    print(f"[kai_bridge] auto-verify FAILED (agree={auto_agreement}) "
                          f"attempt {auto_attempts}/{auto_max} -> escalate to T={probe_temp} "
                          f"(esc_level={esc_level})",
                          file=sys.stderr, flush=True)
                    action_turns = 0
                    action_observations = []
                    continue
                break

            # --- Action phase: execute requested tools in-process ---
            for tc in resp_tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    args = {}
                obs = AGENCY.dispatch(name, args, corpus=self.corpus)
                action_observations.append({"tool": name, "result": obs})
                print(f"[kai_bridge][act] {name} -> {json.dumps(obs, ensure_ascii=False)[:200]}", file=sys.stderr, flush=True)
                # Tier 1: feed the action's consequence back INTO physics —
                # g_ij novelty + curvature + tau all react to the observation,
                # closing perceive-act-observe past the model surface.
                try:
                    phys = self.vfe_state.observe_action(name, obs)
                    print(f"[kai_bridge][obs] {name} -> g_ij={phys['g_novelty']} "
                          f"curv={phys['curvature']} tau={phys['tau']}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[kai_bridge][obs] WARN observe_action failed: {e}", file=sys.stderr, flush=True)
                # Persist the observation as memory (__chat__/obs_<ts>) so the
                # next session can retrieve what the world actually returned.
                try:
                    if obs:
                        self.corpus.store_obs(obs, tool_name=name,
                                              embedding=getattr(self.vfe_state, "_last_obs_emb", None))
                except Exception as e:
                    print(f"[kai_bridge][obs] WARN obs memory persist failed: {e}", file=sys.stderr, flush=True)
                # Feed the observation back to the model as a tool message.
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": json.dumps(obs, ensure_ascii=False)})
            # Convert the augmented messages back for Ollama and re-ask.
            ollama_messages = [convert_openai_to_ollama(m) for m in messages]
            ollama_body["messages"] = ollama_messages
            action_turns += 1
        elapsed = time.time() - t_start

        # ── Auto self-recall fallback (LAYER 1) ───────────────────────
        # Models too weak to emit a tool_call still get the ACTION LOOP via
        # adaptive retrieval: if we have a user question, no action ran, and
        # the model's answer is thin, AND recall finds something relevant,
        # inject it as context and re-ask once. Bounded to 1 retry — never
        # loops unbounded on user traffic.
        self_recall_used = False
        if (ACTION_LOOP_ENABLED and AGENCY is not None
                and not action_observations
                and user_text.strip()):
            # Recall knobs are darwin-evolvable via the live params file:
            #   recall_thin_len  — answer "thin" threshold (default 40 chars)
            #   recall_sim_gate  — min corpus similarity to inject (default 0.45)
            #   recall_top_k     — how many hits to inject (default 3)
            #   recall_ctx_chars — per-hit context budget (default 300)
            thin = not resp_content.strip() or len(resp_content.strip()) < \
                int(self.physics_params.get("recall_thin_len", 40))
            try:
                hits = self.corpus.search(user_text, top_k=int(self.physics_params.get("recall_top_k", 3)))
                best = hits[0]["similarity"] if hits else 0.0
            except Exception:
                hits, best = [], 0.0
            if thin and hits and best >= float(self.physics_params.get("recall_sim_gate", 0.45)):
                self_recall_used = True
                ctx = "[retrieved via auto self-recall]\n"
                ctx_chars = int(self.physics_params.get("recall_ctx_chars", 300))
                for h in hits[: int(self.physics_params.get("recall_top_k", 3))]:
                    ctx += f"  {h['path']} (sim={h['similarity']:.2f}): {h['text'][:ctx_chars]}\n"
                print(f"[kai_bridge][act] auto-recall injected (best={best:.2f})", file=sys.stderr, flush=True)
                messages.insert(0, {"role": "system", "content": ctx})
                ollama_body["messages"] = [convert_openai_to_ollama(m) for m in messages]
                try:
                    ollama_resp = call_ollama_chat(ollama_body)
                    ollama_msg = ollama_resp.get("message", {})
                    resp_content = ollama_msg.get("content", "")
                    resp_tool_calls = ollama_msg.get("tool_calls", [])
                except RuntimeError:
                    pass
        elapsed += time.time() - t_start

        # Process response (after optional action loop)

        # Sync tau from native VFE sampler (if available in response),
        # otherwise use Python-side proxy tracking.
        native_tau = ollama_resp.get("generation_settings", {}).get("kai_tau")
        if native_tau is not None and isinstance(native_tau, (int, float)):
            self.vfe_state.tau = float(native_tau)
            self.vfe_state._save_tau()
        else:
            # Python-side proxy: estimate tau from response text novelty
            resp_novelty = self.vfe_state.compute_novelty(resp_content)
            self.vfe_state.update_tau(resp_novelty,
                                  vfe_tau_rate=self.physics_params.get("vfe_tau_rate", 0.061),
                                  tau_min=self.physics_params.get("tau_min", 0.855),
                                  tau_max=self.physics_params.get("tau_max", 1.657))

        # ── Chat absorption ───────────────────────────────────────────
        # Absorb (user query + assistant response) into persistent attractor
        # memory so future similar questions retrieve this conversation.
        try:
            print(f"[kai_bridge] absorb check: user={len(user_text)} resp={len(resp_content)}", file=sys.stderr, flush=True)
            if user_text.strip() and resp_content.strip():
                self.corpus.store_chat(user_text, resp_content)
        except Exception as e:
            print(f"[kai_bridge] WARN: chat absorption failed: {e}", file=sys.stderr, flush=True)

        # Token usage
        prompt_tokens = ollama_resp.get("prompt_eval_count", 0)
        completion_tokens = ollama_resp.get("eval_count", 0)

        # ── Build OpenAI response ──────────────────────────────────────
        # After the action loop, determine whether tool calls should still be
        # surfaced to the caller. If we executed actions in-process and the
        # model landed on a final answer, HIDE the intermediate tool_calls
        # (they were already acted on). Only surface them if we hit the cap
        # without resolving (caller should take over).
        hidden_actions = (bool(action_observations) or self_recall_used) and resp_content.strip()
        surfacing_calls = resp_tool_calls and not hidden_actions
        choice = {
            "index": 0,
            "finish_reason": "stop" if not surfacing_calls else "tool_calls",
        }

        # Convert Ollama tool_calls to OpenAI format (only if surfacing)
        openai_tool_calls = []
        if surfacing_calls:
            for tc in resp_tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                # OpenAI expects arguments as a JSON string, Ollama may return dict
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                openai_tool_calls.append({
                    "id": tc.get("id", f"call_{int(time.time())}_{len(openai_tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    },
                })

        if stream:
            # Streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            # First chunk with role
            first = {
                "id": f"kai-chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_raw,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            self._send_stream_chunk(first)

            if resp_tool_calls:
                # Tool calls in streaming
                for tci, tc in enumerate(openai_tool_calls):
                    chunk = {
                        "id": first["id"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_raw,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [tc]},
                            "finish_reason": None,
                        }],
                    }
                    self._send_stream_chunk(chunk)
                # Final chunk
                final = {
                    "id": first["id"],
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_raw,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
                self._send_stream_chunk(final)
            else:
                # Content streaming
                if resp_content:
                    content_chunk = {
                        "id": first["id"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_raw,
                        "choices": [{"index": 0, "delta": {"content": resp_content}, "finish_reason": None}],
                    }
                    self._send_stream_chunk(content_chunk)
                # Final chunk
                final = {
                    "id": first["id"],
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_raw,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self._send_stream_chunk(final)

            self._send_stream_done()
        else:
            # Non-streaming
            msg_out = {"role": "assistant", "content": resp_content}
            if openai_tool_calls:
                msg_out["tool_calls"] = openai_tool_calls

            resp = {
                "id": f"kai-chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_raw,
                "choices": [{
                    "index": 0,
                    "message": msg_out,
                    "finish_reason": "stop" if not openai_tool_calls else "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                # ── LAYER 3c introspection: which memory bounded the answer ─
                # Physics observables so callers can see the VFE trace:
                #   novelty (this request), tau (post-response), whether the
                #   native sampler synced tau, and which domain attractor was
                #   dominant (multi-prior introspection).
                "kai_physics": {
                    "novelty": round(self.vfe_state.novelty, 4),
                    "temperature": round(probe_temp, 4),
                    "tau": round(self.vfe_state.tau, 4),
                    "dominant_domain": dom_name,
                    "dominant_responsibility": round(dom_resp, 4),
                    "probe_agreement": p_agree,
                    # LAYER 2c autonomous loop readout: how the bridge
                    # self-escalated on this request (no external attempt).
                    "auto_attempts": auto_attempts,
                    "auto_verified": int(auto_verified),
                    "auto_agreement": (round(auto_agreement, 4)
                                       if isinstance(auto_agreement, (int, float))
                                       else auto_agreement),
                },
            }
            self._send_json(200, resp)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kai AGI OpenAI-compatible bridge")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()

    print(f"Kai bridge — calling native VFE Ollama at {OLLAMA_BASE}", file=sys.stderr)
    print(f"Listening on http://0.0.0.0:{args.port}", file=sys.stderr)
    print("Models:", file=sys.stderr)
    for m in ["kai/tinyllama", "kai/qwen2.5-coder:3b", "kai/qwen2.5-coder:7b",
              "kai/qwen2.5:7b", "kai/qwen3.5:9b", "kai/deepseek-coder-v2:16b",
              "kai/gemma4:12b"]:
        print(f"  {m}", file=sys.stderr)
    print("VFE: native C++ sampler (kai-vfe in chain, attractor file)", file=sys.stderr)
    print("Tool calling: pass-through to Ollama", file=sys.stderr)

    # ── Auto re-absorption thread ────────────────────────────────────
    # Periodically re-scans the AxiomTree for new/changed .md/.txt files
    # and feeds them into doc memory (mtime-aware, resume-safe).
    import threading
    import subprocess

    def _reabsorb_once():
        # Single-flight guard so concurrent runs can't double-embed
        if os.path.exists(REABSORB_LOCK):
            return
        try:
            open(REABSORB_LOCK, "w").close()
            tool = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "kai_absorb_docs.py")
            subprocess.run(
                [sys.executable, "-u", tool, "--root", REABSORB_ROOT, "--resume"],
                timeout=3600,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Refresh the in-memory index so new docs are searchable.
            # Respect the doc shard cap so a re-absorb pass never balloons
            # memory above the safe peak (LAYER 0 OOM safety). Merge is by key,
            # so re-reading the same capped set is idempotent and bounded.
            import glob
            doc_shards = sorted(glob.glob(DOC_MEMORY_PATH.replace(".json", ".*.json")))
            if MAX_DOC_SHARDS is not None:
                doc_shards = doc_shards[:MAX_DOC_SHARDS]
            for shard_path in doc_shards:
                KaiBridgeHandler.corpus._load_shard(shard_path, "Doc memory")
            KaiBridgeHandler.corpus._build_matrix()
            print(f"[kai_bridge] re-absorption pass complete "
                  f"({len(KaiBridgeHandler.corpus.entries)} entries)", file=sys.stderr)
        except Exception as e:
            print(f"[kai_bridge] re-absorption failed: {e}", file=sys.stderr)
        finally:
            try:
                os.remove(REABSORB_LOCK)
            except OSError:
                pass

    def _reabsorb_loop():
        while True:
            time.sleep(REABSORB_INTERVAL)
            try:
                _reabsorb_once()
            except Exception as e:
                print(f"[kai_bridge] re-absorb loop error: {e}", file=sys.stderr)

    t = threading.Thread(target=_reabsorb_loop, daemon=True)
    t.start()
    print(f"Auto re-absorption: every {REABSORB_INTERVAL // 60} min from "
          f"{REABSORB_ROOT}", file=sys.stderr)

    # ── Continuous phenomenology thread ───────────────────────────────
    # Ticks the lived subjective clock every second and appends to
    # .axiom_state/phen_trace.jsonl (same law as the python daemon's
    # bracket-line) so Kai and the bridge share ONE continuous lived curve.
    # Rust side reads it via `kai phen` / tau-prior generation seeding.
    def _phen_loop():
        import phen_continuity
        pc = phen_continuity.PhenContinuity()
        # Close any offline gap since the last trace row (bounded backfill)
        # so subjective time is gapless across bridge restarts.
        pc.close_gap()
        last_tick = 0.0
        while True:
            now = time.monotonic()
            dt = now - last_tick
            if last_tick == 0.0:
                dt = 1.0
            try:
                pc.tick(dt)
            except Exception as e:
                print(f"[kai_bridge] phen tick error: {e}", file=sys.stderr)
            last_tick = now
            time.sleep(1.0)

    _pt = threading.Thread(target=_phen_loop, daemon=True)
    _pt.start()
    print("Continuous phenomenology: .axiom_state/phen_trace.jsonl "
          "(1 tick/s, gap-closed on boot)", file=sys.stderr)

    # ThreadingHTTPServer: one thread per request, so a slow model turn never
    # blocks the phen tick, the re-absorption loop, or parallel client calls.
    server = ThreadingHTTPServer(("0.0.0.0", args.port), KaiBridgeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.server_close()


if __name__ == "__main__":
    main()
