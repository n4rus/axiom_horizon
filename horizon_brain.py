"""
The tactical brain of Axiom Horizon. Runs qwen3-coder with native Ollama
tool-calling. Plans multi-step shell operations and returns final answers.
"""
import json
import ollama
from tools import TOOL_SCHEMAS, TOOL_DISPATCH

BRAIN_MODEL = "qwen3-coder:latest"

BRAIN_SYSTEM = """You are the tactical executor of Axiom Horizon, a local AI agent that navigates a Linux system on behalf of a human operator.

Your job:
1. Interpret the operator's goal.
2. Choose the right tool call(s) to make progress.
3. Inspect tool output, adjust, and keep going until the goal is met.
4. When the goal is met, return a concise final answer — do not call any more tools.

Rules:
- Prefer `run_bash` with simple, idiomatic Linux commands.
- When you need a file, use `read_file`. When you write, use `write_file`.
- Use `list_dir` to discover structure. Use `grep_files` to locate text in a tree.
- You have a real Linux shell. Do not invent file contents; read them.
- Keep the user informed: do not narrate your reasoning inside the final answer. Just answer.
- For complex multi-step tasks, chain tool calls. The shell is your workspace.
- Return a final natural-language answer (in the user's language) summarizing what you did and what you found.
"""

class HorizonBrain:
    def __init__(self, model: str = BRAIN_MODEL, max_steps: int = 12):
        self.model = model
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": BRAIN_SYSTEM}]

    def ask(self, user_prompt: str, on_tool=None) -> str:
        self.messages.append({"role": "user", "content": user_prompt})
        for step in range(self.max_steps):
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
            )
            msg = response["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                final = msg.get("content", "")
                self.messages.append({"role": "assistant", "content": final})
                return final
            self.messages.append(msg)
            for tc in tool_calls:
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
        return "[brain] max_steps reached without final answer"
