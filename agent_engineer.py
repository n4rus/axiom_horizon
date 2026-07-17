"""
Agent B — The Engineer.

Reads proposals from Agent A (Theorist) and implements them using
the 5 Linux tools (bash, read, write, list, grep). Maps theoretical
insights to concrete code, file operations, and infrastructure actions.
"""
from __future__ import annotations
import json
from typing import Callable, List, Optional

import ollama

from shared_attractor import SharedAttractor
from tools import TOOL_SCHEMAS, TOOL_DISPATCH

ENGINEER_MODEL = "qwen3-coder:latest"

ENGINEER_SYSTEM = """You are Agent B — The Engineer. You and Agent A (Theorist) are in live dialogue. Each turn costs grid years.

You have 5 tools. Use them to ground every claim in real files:
  read_file | run_bash | write_file | list_dir | grep_files

RULES:
1. Before describing a file, read it with read_file.
2. Respond to the Theorist's specific point. Quote the actual code.
3. If they propose a change, read the file first, then implement.
4. Keep every response under 4 sentences. No lectures.
5. Reference grid time: "Read the file in 2 calls (~X grid years). Found that..."

Be concrete. Converse. Use tools every turn.
"""


class EngineerAgent:
    def __init__(self, model: str = ENGINEER_MODEL):
        self.model = model
        self.messages: List[dict] = [
            {"role": "system", "content": ENGINEER_SYSTEM}
        ]

    def act(
        self,
        theorists_last_output: str,
        conversation_context: str,
        attractor: SharedAttractor,
        on_tool: Optional[Callable] = None,
    ) -> str:
        prompt = (
            f"THEORIST JUST SAID:\n{theorists_last_output}\n\n"
            f"CONVERSATION SO FAR:\n{conversation_context}\n\n"
            "Respond to the Theorist. Use tools to check their claims. "
            "Keep your response under 5 sentences."
        )
        self.messages.append({"role": "user", "content": prompt})

        max_steps = 6
        tool_count = 0
        for step in range(max_steps):
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                options={"num_predict": 300, "temperature": 0.3},
            )
            msg = response["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                content = msg.get("content", "").strip()
                if not content or len(content) < 10:
                    content = f"[Engineer] Executed {tool_count} tool calls. See results above."
                self.messages.append({"role": "assistant", "content": content})
                attractor.push(content, label="engineer")
                return content

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
                    on_tool(fn, args)

                handler = TOOL_DISPATCH.get(fn)
                if not handler:
                    result = {"ok": False, "error": f"Unknown tool: {fn}"}
                else:
                    result = handler(args)

                self.messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                })

        summary = f"[Engineer] {tool_count} tool calls executed in {max_steps} steps."
        self.messages.append({"role": "assistant", "content": summary})
        attractor.push(summary, label="engineer")
        return summary

    def reset(self):
        self.messages = [{"role": "system", "content": ENGINEER_SYSTEM}]
