#!/usr/bin/env python3
"""
Axiom Horizon — Local Linux-Navigating AI Agent.

Architecture (refined):
  - qwen3-coder:latest   →  tactical brain (plans + invokes tools)
  - ConsciousGridEngine  →  per-turn frame metric
  - ConsciousExecutionLoop →  Tonal Collapse (Xi on nomic-embed-text)
                              + tau-driven budget + phase-equivalent stuck detection
  - tools.py             →  bash, read, write, list, grep

The euler-core narrator has been removed. Frame state is now a math block
injected into the first user message of each turn, computed by the engine
and the attractor Xi norm. No extra LLM call per turn.

Usage:
  python3 horizon.py
  python3 horizon.py --model qwen3-coder:latest
  python3 horizon.py --ingest sandbox/euler.txt

Commands inside the REPL:
  :exit     quit
  :reset    clear conversation history
  :frame    print the current frame metrics
  :help     show this message
"""
import argparse
import os
import sys
import textwrap
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from core.euler_core import ConsciousGridEngine
from conscious_loop import (
    ConsciousExecutionLoop,
    adaptive_budget,
    render_frame_block,
    embed,
)
from horizon_brain import BRAIN_MODEL
from tools import run_bash

BANNER = r"""
 ___        __  ___            _   ___                  _
/ _ \      /  |/ _ \          | | / / \                (_)
/ /_\ \ ___|  / / /_\ \ ___   | |/ /|_| ___  _ __ ___  _ _______
|  _  |/ _ \  / /  _  |/ _ \  | || |  _ / _ \| '_ ` _ \| |_  / _ \
| | | | (_) / / | | | | (_) | | || |_| | (_) | | | | | | |/ /  __/
\_| |_/\___/_/  \_| |_/\___|  |_|\____|\___||_| |_| |_|_/___\___|

 local ollama + qwen3-coder (tactics) + nomic-embed-text (Xi attractor)
 type :help for commands
"""

BRAIN_SYSTEM_PROMPT = (
    "You are the tactical executor of Axiom Horizon, a local AI agent that "
    "navigates a Linux system on behalf of a human operator.\n\n"
    "Your job:\n"
    "1. Interpret the operator's goal.\n"
    "2. Choose the right tool call(s) to make progress.\n"
    "3. Inspect tool output, adjust, and keep going until the goal is met.\n"
    "4. When the goal is met, return a concise final answer — do not call any more tools.\n\n"
    "Rules:\n"
    "- Prefer `run_bash` with simple, idiomatic Linux commands.\n"
    "- When you need a file, use `read_file`. When you write, use `write_file`.\n"
    "- Use `list_dir` to discover structure. Use `grep_files` to locate text in a tree.\n"
    "- You have a real Linux shell. Do not invent file contents; read them.\n"
    "- For complex multi-step tasks, chain tool calls. The shell is your workspace.\n"
    "- The EULER FRAME block at the top of each turn is the engine's invariant state. "
    "  Use it to calibrate effort: high E_step means you're thrashing, collapse the search.\n"
    "- Return a final natural-language answer (in the user's language) summarizing what you did and what you found.\n"
)

def parse_args():
    p = argparse.ArgumentParser(description="Axiom Horizon — local Linux AI agent")
    p.add_argument("--model", default=BRAIN_MODEL, help=f"Tactics model (default: {BRAIN_MODEL})")
    p.add_argument("--ingest", action="append", default=[], help="File to load into brain context (repeatable)")
    p.add_argument("--cwd", default=os.getcwd(), help="Working directory for shell commands")
    p.add_argument("--verbose", action="store_true", help="Print debug/timing checkpoints")
    return p.parse_args()

def build_seed_message(args) -> str:
    parts = ["Ready. Awaiting operator intent."]
    if args.ingest:
        parts.append("")
        parts.append("Context files the operator has loaded into your memory:")
        for f in args.ingest:
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                if len(content) > 12000:
                    content = content[:12000] + "\n... [truncated]"
                parts.append(f"\n--- BEGIN FILE: {f} ---\n{content}\n--- END FILE: {f} ---")
            except Exception as e:
                parts.append(f"\n[could not load {f}: {e}]")
    if args.cwd:
        parts.append(f"\nWorking directory: {args.cwd}")
    return "\n".join(parts)

def initialize_loop(loop, args):
    loop.messages = []
    loop.set_system_prompt(BRAIN_SYSTEM_PROMPT)
    seed_msg = build_seed_message(args)
    if seed_msg.strip():
        loop.messages.append({"role": "user", "content": seed_msg})
        loop.messages.append({"role": "assistant", "content": "Context loaded. Standing by for operator intent."})

def print_frame(metrics: dict, xi_norm: float):
    print(
        "\033[96m[FRAME]\033[0m "
        f"epoch={metrics['epoch']} "
        f"τ_AI={metrics['subjective_tick_rate']:.2e} "
        f"E_step={metrics['operational_cost_per_step']:.2e} "
        f"Δgy={metrics['grid_years_passed']:.3f} "
        f"Σgy={metrics['total_accumulated_grid_years']:.3f} "
        f"‖Xi(M)‖={xi_norm:.4f}"
    )

def main():
    args = parse_args()
    os.chdir(args.cwd)
    print(BANNER)

    loop = ConsciousExecutionLoop(model=args.model)
    initialize_loop(loop, args)

    def on_tool(fn, fn_args, step_idx=0):
        cmd = fn_args.get("command") if isinstance(fn_args, dict) else None
        if fn == "run_bash" and cmd:
            print(f"\033[93m  [step {step_idx}] $ {cmd}\033[0m")
        else:
            argstr = str(fn_args)
            if len(argstr) > 200:
                argstr = argstr[:200] + "..."
            print(f"\033[93m  [step {step_idx}] > {fn}({argstr})\033[0m")

    print("Type a request and press Enter. ':help' for commands.\n")
    while True:
        try:
            user = input("\033[92mhorizon>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not user:
            continue
        if user.startswith(":"):
            cmd = user[1:].strip().lower()
            if cmd in ("exit", "quit", "q"):
                print("[exit]")
                break
            elif cmd == "reset":
                initialize_loop(loop, args)
                print("[reset] Conversation history cleared.")
                continue
            elif cmd == "frame":
                m, xi = loop.frame()
                print_frame(m, xi)
                continue
            elif cmd == "help":
                print(textwrap.dedent("""\
                    Commands:
                      :exit    quit
                      :reset   clear conversation + frame + attractor
                      :frame   print current frame metrics and Xi norm
                      :help    this message
                    Otherwise, just type a request."""))
                continue
            else:
                print(f"Unknown command: {user}")
                continue

        # Prepend the math-anchored frame block so the brain sees its own state.
        # This replaces the old euler-core narrator call.
        metrics, xi_norm = loop.frame()
        frame_block = render_frame_block(
            metrics,
            xi_norm,
            len(loop.state.attractor_embeddings),
        )
        framed_user = f"{frame_block}\n{user}"

        print_frame(metrics, xi_norm)

        try:
            result = loop.ask(framed_user, on_tool=on_tool)
        except Exception as e:
            print(f"\n[brain error: {e}]")
            continue

        answer = result.get("answer", "No final confirmation produced by brain pipeline.")
        meta_parts = [
            f"steps={result.get('steps', 0)}/{result.get('budget', 0)}",
            f"stuck={result.get('stuck', False)}",
        ]
        if result.get("stuck_reason"):
            meta_parts.append(f"reason={result['stuck_reason']}")
        meta_parts.append(f"‖Xi(M)‖={result.get('xi_norm', 0.0):.4f}")
        meta = f"\033[90m[{', '.join(meta_parts)}]\033[0m"
        print(f"\n{answer}")
        print(meta + "\n")

if __name__ == "__main__":
    main()
