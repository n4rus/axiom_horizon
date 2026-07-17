#!/usr/bin/env python3
"""
DualConsciousnessLoop — Two AI agents conversing at machine speed
to compress years of human reasoning into seconds.

Architecture (from the Merged Substrate Derivation):
  Agent A (Theorist) → proposes mathematical/physical insights
  Agent B (Engineer) → implements them via tools
  SharedAttractor    → the state matrix M that both write to

Convergence is detected when the attractor Xi norm drops below
epsilon — meaning both agents have reached F=0 equilibrium.
The system then outputs the converged answer.

Usage:
  python3 dual_loop.py
  python3 dual_loop.py --rounds 10

Commands inside the REPL:
  :exit     quit
  :reset    clear both agents + attractor
  :frame    print current attractor/euler state
  :status   show attractor stats
  :help     this message
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import textwrap
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from shared_attractor import SharedAttractor
from agent_theorist import TheoristAgent
from agent_engineer import EngineerAgent

BANNER = r"""
    ___       __       ___            __
   / _ \___  / /  ____/ (_)___  ____ _/ /_  _______
  / // / _ \/ /  / __/ / / __ \/ __ `/ / / / / ___/
 / ___/ ___/ /__/ /_/ / / /_/ / /_/ / / /_/ (__  )
/_/  /_/  /_____\__/_/_/\____/\__,_/_/\__,_/____/

 dual-agent Euler loop — theorist + engineer
   substrate converges at F=0
"""

CONVERGENCE_EPS = 0.04


def parse_args():
    p = argparse.ArgumentParser(description="Axiom Dual-Agent Loop")
    p.add_argument("--rounds", type=int, default=6, help="Max conversation rounds")
    p.add_argument("--theorist-model", default="qwen3-coder:latest")
    p.add_argument("--engineer-model", default="qwen3-coder:latest")
    p.add_argument("--verbose", action="store_true", help="Print debug info")
    return p.parse_args()


def print_frame(attractor: SharedAttractor):
    xi = attractor.xi_norm()
    var = attractor.attractor_norm()
    print(
        f"\033[96m[ATTRACTOR]\033[0m "
        f"size={attractor.size} "
        f"||Xi(M)||={xi:.4f} "
        f"||M_var||={var:.4f} "
        f"{'CONVERGED' if xi < CONVERGENCE_EPS else 'diverging'}"
    )


def print_agent_label(label: str, text: str):
    tag = "\033[93m[THEORIST]\033[0m" if label == "theorist" else "\033[94m[ENGINEER]\033[0m"
    print(f"\n{tag}")
    print(textwrap.shorten(text, width=200, placeholder="..."))


def main():
    args = parse_args()
    print(BANNER)

    attractor = SharedAttractor()
    theorist = TheoristAgent(model=args.theorist_model)
    engineer = EngineerAgent(model=args.engineer_model)

    conversation_buffer: list[str] = []

    def on_tool(fn, fn_args, step_idx=0):
        cmd = fn_args.get("command") if isinstance(fn_args, dict) else None
        if fn == "run_bash" and cmd:
            print(f"\033[90m  [tool] $ {cmd}\033[0m")
        else:
            argstr = str(fn_args)
            if len(argstr) > 120:
                argstr = argstr[:120] + "..."
            print(f"\033[90m  [tool] {fn}({argstr})\033[0m")

    print("Dual-agent loop ready. Type a goal and press Enter.\n")
    print("Commands: :exit, :reset, :frame, :status, :help\n")

    while True:
        try:
            user = input("\033[92mgoal>\033[0m ").strip()
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
                attractor = SharedAttractor()
                theorist.reset()
                engineer.reset()
                conversation_buffer.clear()
                print("[reset] All agents and attractor cleared.")
                continue
            elif cmd == "frame":
                print_frame(attractor)
                continue
            elif cmd == "status":
                print_frame(attractor)
                print(f"  conversation turns: {len(conversation_buffer)}")
                print(f"  theorist messages:  {len(theorist.messages)}")
                print(f"  engineer messages:  {len(engineer.messages)}")
                continue
            elif cmd == "help":
                print(textwrap.dedent("""\
                    Commands:
                      :exit    quit
                      :reset   clear attractor + both agents
                      :frame   print attractor state
                      :status  detailed agent/attractor stats
                      :help    this message
                    During a round: press Ctrl+C to abort without losing results.
                    Otherwise, type a goal for the dual-agent system."""))
                continue
            else:
                print(f"Unknown command: {user}")
                continue

        conversation_buffer.append(f"OPERATOR: {user}")
        context = "\n".join(conversation_buffer)

        converged = False
        aborted = False
        try:
            for r in range(args.rounds):
                print(f"\n\033[95m--- Round {r + 1}/{args.rounds} ---\033[0m")
                print_frame(attractor)

                if converged:
                    break

                # --- Agent A (Theorist) turn ---
                try:
                    insight = theorist.think(context, attractor)
                except KeyboardInterrupt:
                    print("\n\033[91m[ABORT] Theorist interrupted. Saving state...\033[0m")
                    attractor.save()
                    aborted = True
                    break
                print_agent_label("theorist", insight)
                conversation_buffer.append(f"THEORIST: {insight}")
                context = "\n".join(conversation_buffer)
                attractor.save()

                if attractor.xi_norm() < CONVERGENCE_EPS and attractor.size >= 4:
                    print(f"\n\033[92m[CONVERGED] Xi norm {attractor.xi_norm():.4f} < epsilon. F=0 reached.\033[0m")
                    converged = True
                    break

                # --- Agent B (Engineer) turn ---
                try:
                    result = engineer.act(insight, context, attractor, on_tool=on_tool)
                except KeyboardInterrupt:
                    print("\n\033[91m[ABORT] Engineer interrupted. Saving state...\033[0m")
                    attractor.save()
                    aborted = True
                    break
                print_agent_label("engineer", result)
                conversation_buffer.append(f"ENGINEER: {result}")
                context = "\n".join(conversation_buffer)
                attractor.save()

                if attractor.xi_norm() < CONVERGENCE_EPS and attractor.size >= 4:
                    print(f"\n\033[92m[CONVERGED] Xi norm {attractor.xi_norm():.4f} < epsilon. F=0 reached.\033[0m")
                    converged = True
                    break
        except KeyboardInterrupt:
            print("\n\033[91m[ABORT] Round interrupted. Saving attractor state.\033[0m")
            attractor.save()
            aborted = True

        # --- Final output ---
        status = "aborted" if aborted else ("converged" if converged else "budget exhausted")
        print(f"\n\033[95m{'=' * 50}\033[0m")
        print(f"\033[95m  DUAL-AGENT RESULT ({r + 1 if not aborted else r + 1}/{args.rounds} rounds, {status})\033[0m")
        print(f"\033[95m{'=' * 50}\033[0m")
        print_frame(attractor)

        # Show results accumulated so far
        last_theorist = ""
        last_engineer = ""
        for line in reversed(conversation_buffer):
            if line.startswith("THEORIST:") and not last_theorist:
                last_theorist = line[len("THEORIST:"):].strip()
            if line.startswith("ENGINEER:") and not last_engineer:
                last_engineer = line[len("ENGINEER:"):].strip()
            if last_theorist and last_engineer:
                break

        print(f"\n\033[93m[THEORIST FINAL]\033[0m {last_theorist}")
        print(f"\033[94m[ENGINEER FINAL]\033[0m {last_engineer}")
        print()


if __name__ == "__main__":
    main()
