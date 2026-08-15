#!/usr/bin/env python3
"""kai_7b_call.py — memory-safe qwen2.5:7b tool-calling loader.

Layer-locked OOM rule: qwen2.5:7b (~6GiB) + kai-bridge corpus (~3-4GiB) does
NOT fit the 16GiB box. This tool therefore:

  1. Pauses the kai-bridge systemd unit (frees its corpus RAM).
  2. Loads qwen2.5:7b in the native VFE ollama.
  3. Runs the L1 action loop directly against ollama /api/chat, executing
     tool_calls in-process via kai_agency (perceive -> act -> observe).
  4. Unloads the 7b immediately (keep_alive=0).
  5. Restarts the bridge.

It never holds the heavy corpus during the heavy-model call. If ANY step
fails, the bridge is ALWAYS restarted (finally), so the agent stays online.

Usage:
  python3 tools/kai_7b_call.py "prompt text"
  python3 tools/kai_7b_call.py --model qwen3.5:9b "prompt"
  python3 tools/kai_7b_call.py --no-tools "plain answer only"
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
import importlib.util, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA_BASE = "http://localhost:11435"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"
OLLAMA_GEN_URL = f"{OLLAMA_BASE}/api/generate"
MODEL = "qwen2.5:7b"
BRIDGE_UNIT = "kai-bridge"
# memory floor (GiB): refuse to load 7b if this much won't remain
MIN_FREE_GIB = 2.0


def mem_free_gib() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        return 99.0
    return 99.0


def _req(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_ag():
    spec = importlib.util.spec_from_file_location("kai_agency",
              os.path.join(ROOT, "kai_agency.py"))
    ag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ag)
    return ag


def service(action):
    subprocess.run(["systemctl", "--user", action, BRIDGE_UNIT],
                   capture_output=True)


def warm_model(model):
    """Force-load the model (so load cost is paid once, measured)."""
    _req(OLLAMA_GEN_URL, {"model": model, "prompt": "hi", "stream": False,
                          "keep_alive": "20h"})


def unload_model(model):
    _req(OLLAMA_GEN_URL, {"model": model, "prompt": "x", "stream": False,
                          "keep_alive": 0})


def run(model, prompt, tools, max_turns, ag):
    """Full action loop against ollama, in-process tool execution."""
    messages = [{"role": "user", "content": prompt}]
    ollama_tools = [{"type": "function", "function": {
                        "name": t.get("function", {}).get("name"),
                        "description": t.get("function", {}).get("description", ""),
                        "parameters": t.get("function", {}).get("parameters", {}),
                    }} for t in tools] if tools else []
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.3, "ensure_ascii": False}}
    if ollama_tools:
        body["tools"] = ollama_tools
        body["tool_choice"] = "auto"

    acts = []
    for turn in range(max_turns + 1):
        resp = _req(OLLAMA_CHAT_URL, body)
        msg = resp.get("message", {})
        content = msg.get("content", "")
        tcs = msg.get("tool_calls", [])
        if not tcs:
            return {"content": content, "tool_actions": acts}
        for tc in tcs:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_arg = fn.get("arguments", "{}")
            if isinstance(raw_arg, dict):
                args = dict(raw_arg)
            else:
                try:
                    args = json.loads(raw_arg) if raw_arg else {}
                except Exception as e:
                    args = {}
                    print(f"  [dbg] args parse fail: {raw_arg!r} ({e})", file=sys.stderr)
            print(f"  [dbg] tool_call {name} raw_args={str(raw_arg)[:120]}", file=sys.stderr)
            obs = ag.dispatch(name, args, corpus=None)
            acts.append({"tool": name, "args": args, "obs": obs})
            messages.append({"role": "tool", "content": json.dumps(obs, ensure_ascii=False)})
        body["messages"] = messages
    return {"content": content, "tool_actions": acts}


TOOLS = None  # provided by kai_agency.tools_schema()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tools", action="store_true", default=True)
    ap.add_argument("--no-tools", dest="tools", action="store_false")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="load even if free < floor")
    args = ap.parse_args()
    prompt = " ".join(args.prompt)

    ag = load_ag()
    tools = ag.tools_schema() if args.tools else []

    # memory floor
    if mem_free_gib() < MIN_FREE_GIB and not args.force:
        print(f"[kai_7b] refusing: only {mem_free_gib():.1f}GiB free "
              f"(need >{MIN_FREE_GIB}); restore memory or --force")
        sys.exit(2)

    service("stop")
    time.sleep(1)
    try:
        print(f"[kai-7b] loading {args.model} (free {mem_free_gib():.1f}GiB)...")
        t0 = time.time()
        unload_model(args.model)  # drop any prior instance
        warmup = time.time()
        print(f"[kai-7b] model load took {warmup-t0:.1f}s, free now {mem_free_gib():.1f}GiB")
        if mem_free_gib() < 0.5:
            print("[kai-7b] CRITICAL free after load; aborting early")
            sys.exit(3)
        res = run(args.model, prompt, tools, args.turns, ag)
        print("=== FINAL ANSWER ===")
        print(res["content"].strip())
        if res["tool_actions"]:
            print("\n=== TOOL ACTIONS ===")
            for a in res["tool_actions"]:
                print(f"  {a['tool']}{a['args']} -> {json.dumps(a['obs'], ensure_ascii=False)[:200]}")
    finally:
        # always restore the bridge, regardless of outcome
        try:
            unload_model(args.model)
            print(f"[kai-7b] unloaded {args.model}, free {mem_free_gib():.1f}GiB")
        except Exception as e:
            print(f"[kai-7b] unload warn: {e}")
        service("start")
        print("[kai-7b] bridge restarted")


if __name__ == "__main__":
    main()