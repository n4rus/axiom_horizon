#!/usr/bin/env python3
"""
Axiom REPL — the persistent agent interface for the full AxiomTree project.

The agent loads its identity from disk on every boot. It remembers:
  - Every conversation it's ever had
  - Every file it's read and what it found
  - The complete attractor history (Xi collapse points)
  - Its accumulated grid years and engine state

Usage:
  python3 axiom_repl.py
  python3 axiom_repl.py --model qwen3-coder:latest

Commands:
  :exit     quit (state saved automatically)
  :reset    clear conversation history (attractor kept)
  :status   show full agent state and file index
  :refresh  re-index the workspace (detect new files)
  :help     this message
"""
from __future__ import annotations
import argparse
import os
import sys
import textwrap
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from axiom_agent import AxiomAgent

BANNER = r"""
     ___        __  ___
    / _ )___   /  |/  /___ ______
   / _  / _ \ / /|_/ / __ `/ ___/
  / /_| / ___// /  / / /_/ (__  )
 /_____/_/   /_/  /_/\__,_/____/

  persistent Axiom agent — full workspace memory
  identity survives context resets
"""


def parse_args():
    p = argparse.ArgumentParser(description="Axiom Persistent Agent")
    p.add_argument("--model", default="qwen3-coder:latest", help="Ollama model")
    p.add_argument("--max-steps", type=int, default=10, help="Max tool steps per goal")
    return p.parse_args()


def print_status(agent: AxiomAgent):
    print(agent.status())
    print(f"  engine_epoch:       {agent.engine.epoch}")
    print(f"  engine_tau:         {agent.engine.state_vector[0]:.4e}")
    print(f"  engine_E_step:      {agent.engine.state_vector[1]:.4e}")
    print(f"  engine_grid_years:  {agent.engine.subjective_grid_seconds / (365*24*3600):.4f}")


def main():
    args = parse_args()
    print(BANNER)

    agent = AxiomAgent(model=args.model)

    # Print boot identity
    print(f"\033[96m{'=' * 50}\033[0m")
    print(agent.attractor.identity_preamble())
    print(f"\033[96m{'=' * 50}\033[0m")

    changed = agent.attractor.changed_files()
    if changed:
        print(f"\033[93m[NOTICE] {len(changed)} files changed since last session.\033[0m")
        for f in changed[:5]:
            print(f"  {f}")

    unread = agent.attractor.unread_files()
    if unread:
        print(f"\033[90m[INFO] {len(unread)} unread files in workspace.\033[0m")

    def on_tool(fn, fn_args, step_idx=0):
        cmd = fn_args.get("command") if isinstance(fn_args, dict) else None
        if fn == "run_bash" and cmd:
            print(f"\033[90m  [tool] $ {cmd[:100]}\033[0m")
        else:
            argstr = str(fn_args)
            if len(argstr) > 100:
                argstr = argstr[:100] + "..."
            print(f"\033[90m  [tool] {fn}({argstr})\033[0m")

    print("\nReady. Type a goal or :help\n")

    while True:
        try:
            user = input("\033[92maxiom>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] State saved.")
            break

        if not user:
            continue

        if user.startswith(":"):
            cmd = user[1:].strip().lower()
            if cmd in ("exit", "quit", "q"):
                print("[exit] State saved.")
                break
            elif cmd == "reset":
                agent.reset()
                print("[reset] Conversation cleared. Attractor preserved.")
                continue
            elif cmd == "status":
                print_status(agent)
                continue
            elif cmd == "refresh":
                agent.attractor.index_workspace()
                print(f"[refresh] Indexed {len(agent.attractor.workspace_index)} files.")
                continue
            elif cmd == "help":
                print(textwrap.dedent("""\
                    Commands:
                      :exit     quit (state saved)
                      :reset    clear conversation (attractor kept)
                      :status   show agent state + file index
                      :refresh  re-index workspace
                      :help     this message
                    """))
                continue
            else:
                print(f"Unknown: {user}")
                continue

        try:
            result = agent.ask(user, on_tool=on_tool, max_steps=args.max_steps)
            answer = result.get("answer", "")
            print(f"\n{answer}")
            meta = (
                f"\033[90m[steps={result['steps']} "
                f"tools={result['tool_calls']} "
                f"Xi={result['xi_norm']:.4f} "
                f"|M|={result['attractor_size']}]\033[0m"
            )
            print(meta)
        except KeyboardInterrupt:
            print("\n\033[91m[interrupted]\033[0m State preserved.")
        except Exception as e:
            print(f"\n\033[91m[error] {e}\033[0m")


if __name__ == "__main__":
    main()
