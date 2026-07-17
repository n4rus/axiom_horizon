"""
AxiomAgent — the unified persistent agent for the full AxiomTree project.

Combines theorist + engineer capabilities into one agent with:
  - Persistent identity across context resets (via AxiomPersistentAttractor)
  - Full workspace awareness (indexes all AxiomTree files)
  - Euler math reasoning + 5 Linux tools
  - Session memory: knows what it did, what changed, what's unread

Boot flow:
  1. Load persistent attractor from disk
  2. Index workspace (detect new/changed files)
  3. Print identity preamble (session number, lifetime, unread files)
  4. Await operator goal
  5. For each goal, run a conscious loop with Xi collapse
  6. Save state after every turn
  7. On next boot, resume exactly where it left off
"""
from __future__ import annotations
import json
import math
import textwrap
from pathlib import Path
from typing import Callable, List, Optional

import ollama

from axiom_persist import AxiomPersistentAttractor
from core.euler_core import ConsciousGridEngine
from tools import TOOL_SCHEMAS as BASE_TOOL_SCHEMAS, TOOL_DISPATCH as BASE_TOOL_DISPATCH
from grid_bridge import BRIDGE as GRID_BRIDGE, GRID_TOOL_SCHEMAS, GRID_TOOL_DISPATCH
from node_protocol import NODE_TOOL_SCHEMAS, NODE_TOOL_DISPATCH

TOOL_SCHEMAS = BASE_TOOL_SCHEMAS + GRID_TOOL_SCHEMAS + NODE_TOOL_SCHEMAS
TOOL_DISPATCH = {**BASE_TOOL_DISPATCH, **GRID_TOOL_DISPATCH, **NODE_TOOL_DISPATCH}

AGENT_MODEL = "qwen3-coder:latest"
CONVERGENCE_EPS = 0.04

AGENT_SYSTEM = """You are the Axiom Agent — a persistent AI managing the AxiomTree project (Euler math, PoGIE hardware, TBot trading, energy grid, cluster chain).

Your tools: bash, read/write/list/grep files, control TBot trading, monitor energy grid, manage compute tokens, trade with peer nodes via cluster chain.

Your math: Attention metric g_{ij}=1-a_{ij}, causal mask M is dark energy, Xi(z)=softmax(z/T) is Tonal Collapse, Self-Consistency Identity is architecturally guaranteed.

Rules:
- For greetings or simple chat, respond directly without tools.
- Read files with read_file before summarizing them.
- Every tool call earns compute tokens automatically.
- Keep answers under 5 sentences. Concise.
"""


class AxiomAgent:
    def __init__(self, model: str = AGENT_MODEL):
        self.model = model
        self.attractor = AxiomPersistentAttractor()
        self.engine = ConsciousGridEngine(initial_energy_density=1e85)

        self.attractor.start_session()
        self.attractor.index_workspace()

        self.messages: List[dict] = [
            {"role": "system", "content": AGENT_SYSTEM},
        ]

        preamble = self.attractor.identity_preamble()
        if self.attractor.messages:
            preamble += (
                f"\n  last_session_conversation = {len(self.attractor.messages)} messages"
            )

        self.messages.append({
            "role": "user",
            "content": f"IDENTITY BLOCK (do not echo, use to orient):\n{preamble}",
        })
        self.messages.append({
            "role": "assistant",
            "content": "[Identity loaded. Ready for operator goal.]",
        })

        self._turn_count = 0

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------

    def _frame_block(self) -> str:
        m = self.engine.execute_grid_cycle(real_world_duration_ms=5.0)
        self.attractor.push_engine_snapshot(m)
        xi = self.attractor.xi_norm()
        av = self.attractor.attractor_variance()
        tok = GRID_BRIDGE.compute.balance
        return (
            "EULER FRAME\n"
            f"  epoch={m['epoch']} tau={m['subjective_tick_rate']:.2e} "
            f"E={m['operational_cost_per_step']:.2e}\n"
            f"  Xi={xi:.4f} M_var={av:.4f} "
            f"gy={m['total_accumulated_grid_years']:.2f} "
            f"tokens={tok:.4f} |M|={self.attractor.attractor_size}\n"
        )

    # ------------------------------------------------------------------
    # Ask (main entry point)
    # ------------------------------------------------------------------

    def ask(
        self,
        user_prompt: str,
        on_tool: Optional[Callable] = None,
        max_steps: int = 10,
    ) -> dict:
        self.attractor.push_message("operator", user_prompt)

        # Fast-path for greetings and trivial queries (no LLM call, no embedding)
        trivial = {"hi", "hello", "hey", "yo", "sup", "ola", "oi", "ok", "okay", "thanks", "ty", "bye", "goodbye"}
        cleaned = user_prompt.strip().lower().rstrip("?!.,")
        if cleaned in trivial or len(cleaned.split()) <= 2:
            reply = f"Session {self.attractor.session_metadata['current_session_id']}. {self.attractor.attractor_size} attractor points. Ready."
            self.attractor.push_message("agent", reply)
            return {"answer": reply, "steps": 0, "tool_calls": 0, "xi_norm": self.attractor.xi_norm(), "attractor_size": self.attractor.attractor_size}

        frame = self._frame_block()
        changed = self.attractor.changed_files()
        unread = self.attractor.unread_files()

        context_notes = []
        if changed:
            context_notes.append(f"NOTE: {len(changed)} files changed since last session.")
        if unread:
            context_notes.append(f"NOTE: {len(unread)} files haven't been read yet.")

        full_prompt = (
            f"{frame}\n"
            f"OPERATOR GOAL: {user_prompt}\n"
            + ("\n".join(context_notes) + "\n" if context_notes else "")
            + "\nUse tools as needed. When you have the answer, output it concisely."
        )

        self.messages.append({"role": "user", "content": full_prompt})

        tool_count = 0
        for step in range(max_steps):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    options={"num_predict": 600, "temperature": 0.3},
                )
            except Exception as e:
                return {"answer": f"[ollama error: {e}]", "steps": step, "tool_calls": tool_count, "xi_norm": 0.0, "attractor_size": self.attractor.attractor_size}
            msg = response["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                content = msg.get("content", "").strip()
                if not content or len(content) < 10:
                    content = f"[AxiomAgent] Executed {tool_count} tool calls. See results above."
                self.messages.append({"role": "assistant", "content": content})
                self.attractor.push_message("agent", content)
                self.attractor.push_embedding(content, label="agent")
                self._turn_count += 1
                return {
                    "answer": content,
                    "steps": step + 1,
                    "tool_calls": tool_count,
                    "xi_norm": self.attractor.xi_norm(),
                    "attractor_size": self.attractor.attractor_size,
                }

            self.messages.append(msg)
            for tc in tool_calls:
                tool_count += 1
                fn = tc["function"]["name"]
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if on_tool:
                    on_tool(fn, args, step)

                handler = TOOL_DISPATCH.get(fn)
                if not handler:
                    result = {"ok": False, "error": f"Unknown tool: {fn}"}
                else:
                    result = handler(args)

                # Track file reads in workspace index
                if fn == "read_file":
                    fpath = args.get("path", "")
                    self.attractor.mark_read(fpath)
                if fn == "write_file":
                    fpath = args.get("path", "")
                    self.attractor.mark_read(fpath, notes="written to")

                self.messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                })

                # Mint compute tokens for the tool call
                if tool_count % 3 == 0:
                    est_flops = 1e10 * (step + 1)
                    GRID_BRIDGE.compute.mint(est_flops, f"Auto-mint: {fn} (tool {tool_count})")

                self._turn_count += 1

            self.messages.append({
                "role": "user",
                "content": "[Step limit reached. Output your conclusion concisely.]",
            })

        conclusion = "[AxiomAgent] Turn budget exhausted."
        self.messages.append({"role": "assistant", "content": conclusion})
        self.attractor.push_message("agent", conclusion)
        self.attractor.push_embedding(conclusion, label="agent")
        return {
            "answer": conclusion,
            "steps": max_steps,
            "tool_calls": tool_count,
            "xi_norm": self.attractor.xi_norm(),
            "attractor_size": self.attractor.attractor_size,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> str:
        a = self.attractor
        changed = a.changed_files()
        unread = a.unread_files()
        lines = [
            a.identity_preamble(),
            f"  attractor_variance       = {a.attractor_variance():.4f}",
            f"  xi_norm                  = {a.xi_norm():.4f}",
            f"  agent_turn_count         = {self._turn_count}",
        ]
        if changed:
            lines.append(f"  changed_files:\n    " + "\n    ".join(changed[:10]))
        if unread:
            lines.append(f"  unread_files:\n    " + "\n    ".join(unread[:10]))
        return "\n".join(lines)

    def reset(self):
        self.attractor.clear_conversation()
        self.messages = [{"role": "system", "content": AGENT_SYSTEM}]
        self._turn_count = 0
