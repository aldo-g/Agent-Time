"""Entry point for running the prompt-driven Polymarket agent."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agents.prompt_agent import PromptAgent, PromptAgentError
from agents.tool_agent import ToolAgent, ToolAgentError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Polymarket agent (LLM tool loop or heuristic).")
    parser.add_argument(
        "--agent",
        choices=("tool", "prompt"),
        default="tool",
        help="Agent implementation to run (default: tool)",
    )
    parser.add_argument("--limit", type=int, default=40, help="Number of markets to consider")
    parser.add_argument(
        "--sort",
        type=str,
        default="last-bet-time",
        help="Sort order for Polymarket markets (last-bet-time, updated-time, volume24h, liquidity)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model name for tool agent (default: env OPENAI_MODEL or gpt-4o-mini)",
    )
    parser.add_argument("--max-steps", type=int, default=12, help="Max LLM/tool iterations (tool agent only)")
    parser.add_argument(
        "--debug-tools",
        action="store_true",
        help="Print tool invocation logs (tool agent only)",
    )
    args = parser.parse_args(argv)

    _load_env_defaults()
    if args.agent == "prompt":
        agent = PromptAgent()
        try:
            plan = agent.run(limit=args.limit, sort=args.sort)
        except PromptAgentError as exc:
            print(f"Agent failed: {exc}", file=sys.stderr)
            return 1
        print(plan.as_text())
    else:
        agent = ToolAgent(model=args.model, max_steps=args.max_steps, debug_tools=args.debug_tools)
        try:
            transcript = agent.run(limit=args.limit, sort=args.sort)
        except ToolAgentError as exc:
            print(f"Tool agent failed: {exc}", file=sys.stderr)
            return 1
        print(transcript)
    return 0


def _load_env_defaults(env_filename: str = ".env") -> None:
    """Populate os.environ from KEY=VALUE .env files without overriding existing vars."""
    search_paths = [
        Path(env_filename),
        Path(__file__).resolve().parent / env_filename,
    ]
    seen: set[Path] = set()
    for path in search_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _apply_env_file(resolved)


def _apply_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        contents = path.read_text()
    except Exception:
        return
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
