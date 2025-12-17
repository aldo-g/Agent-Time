"""LLM-driven tool agent that decides which Polymarket functions to call."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import requests

from connectors.polymarket import PolymarketClient, PolymarketAPIError

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class ToolAgentError(RuntimeError):
    pass


@dataclass
class ToolAction:
    action: str
    tool: str | None = None
    tool_input: Dict[str, Any] | None = None
    output: str | None = None


class ToolAgent:
    """Simple ReAct-style loop where the LLM chooses which tool to run next."""

    def __init__(
        self,
        model: str | None = None,
        client: PolymarketClient | None = None,
        max_steps: int = 12,
        temperature: float = 0.2,
        debug_tools: bool = False,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ToolAgentError("OPENAI_API_KEY is required for tool agent.")
        self._api_key = api_key
        self._model = model or DEFAULT_MODEL
        self._client = client or PolymarketClient()
        self._max_steps = max_steps
        self._temperature = temperature
        self._debug_tools = debug_tools or bool(os.environ.get("AGENT_DEBUG"))
        self._tools: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "list_markets": self._tool_list_markets,
            "get_market": self._tool_get_market,
            "get_market_bets": self._tool_get_market_bets,
        }
        if getattr(self._client, "supports_portfolio", False):
            self._tools["get_portfolio"] = self._tool_get_portfolio

    def run(self, limit: int = 40, sort: str = "last-bet-time") -> str:
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    "You are operating without prior state. Begin by fetching markets "
                    f"(suggested limit {limit}, sort {sort}) and determine profitable trades. "
                    "Finish by summarizing the plan and explicit bet instructions."
                ),
            },
        ]
        for _ in range(self._max_steps):
            reply = self._chat(messages)
            action = self._parse_action(reply)
            messages.append({"role": "assistant", "content": reply})
            if action.action == "call_tool":
                observation = self._execute_tool(action)
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_OUTPUT:\n{observation}",
                    }
                )
                continue
            if action.action == "finish" and action.output:
                return action.output
        raise ToolAgentError("LLM failed to produce a final answer within step limit.")

    def _execute_tool(self, action: ToolAction) -> str:
        if not action.tool or not action.tool_input:
            return "ERROR: tool request missing name or input."
        handler = self._tools.get(action.tool)
        if handler is None:
            return f"ERROR: unknown tool {action.tool}."
        self._log(f"Calling tool {action.tool} with input {action.tool_input}")
        try:
            result = handler(action.tool_input)
        except PolymarketAPIError as exc:
            detail = self._client.last_error
            extra = f" | detail: {detail}" if detail and detail not in str(exc) else ""
            self._log(f"Tool {action.tool} failed: {exc}{extra}")
            return f"ERROR: Polymarket API error: {exc}{extra}"
        except Exception as exc:
            self._log(f"Tool {action.tool} raised unexpected error: {exc}")
            return f"ERROR: tool execution failed: {exc}"
        preview = json.dumps(result, default=str)[:5000]
        self._log(f"Tool {action.tool} success; preview: {preview[:2000]}")
        return preview

    def _tool_list_markets(self, tool_input: Dict[str, Any]) -> Any:
        params = {
            "limit": int(tool_input.get("limit", 20)),
            "sort": tool_input.get("sort", "last-bet-time"),
        }
        return self._client.list_markets(**params)

    def _tool_get_market(self, tool_input: Dict[str, Any]) -> Any:
        market_id = tool_input.get("market_id") or tool_input.get("id")
        if not market_id:
            raise ToolAgentError("get_market requires 'market_id'.")
        return self._client.get_market(str(market_id))

    def _tool_get_market_bets(self, tool_input: Dict[str, Any]) -> Any:
        market_id = tool_input.get("market_id") or tool_input.get("id")
        if not market_id:
            raise ToolAgentError("get_market_bets requires 'market_id'.")
        limit = int(tool_input.get("limit", 100))
        return self._client.get_market_bets(str(market_id), limit=limit)

    def _tool_get_portfolio(self, _: Dict[str, Any]) -> Any:
        if not getattr(self._client, "supports_portfolio", False):
            raise ToolAgentError("Portfolio tool is not supported by the current connector.")
        return self._client.get_portfolio()

    def _chat(self, messages: Sequence[Dict[str, str]]) -> str:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": self._temperature,
        }
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise ToolAgentError(f"OpenAI API error: {response.status_code} {response.text}")
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ToolAgentError(f"Malformed OpenAI response: {data}") from exc
        if not content:
            raise ToolAgentError("OpenAI returned empty content.")
        return content

    def _parse_action(self, content: str) -> ToolAction:
        try:
            payload = json.loads(content.strip())
        except json.JSONDecodeError as exc:
            raise ToolAgentError(f"Assistant reply was not valid JSON: {content}") from exc
        action = payload.get("action")
        if action == "call_tool":
            return ToolAction(
                action="call_tool",
                tool=payload.get("tool"),
                tool_input=payload.get("input") or {},
            )
        if action == "finish":
            return ToolAction(action="finish", output=payload.get("output", ""))
        raise ToolAgentError(f"Unknown action in assistant reply: {content}")

    def _system_prompt(self) -> str:
        base_tools = [
            ("list_markets", "list_markets(limit:int=20, sort:str='last-bet-time') -> recent markets."),
            ("get_market", "get_market(market_id:str) -> market metadata."),
            ("get_market_bets", "get_market_bets(market_id:str, limit:int=100) -> recent fills/prob changes."),
        ]
        if "get_portfolio" in self._tools:
            base_tools.append(("get_portfolio", "get_portfolio() -> current cash/positions."))
        tool_lines = "\n".join(f"{idx}. {desc}" for idx, (_, desc) in enumerate(base_tools, start=1))
        return (
            "You are a Polymarket trading analyst. Decide which tools to call to gather data and "
            "craft profitable bets. Return your reasoning in JSON instructions so the runtime can "
            f"execute them. Available tools:\n{tool_lines}\n\n"
            "Respond only with JSON in one of two formats:\n"
            "{\"action\":\"call_tool\",\"tool\":\"tool_name\",\"input\":{...}}\n"
            "or\n"
            "{\"action\":\"finish\",\"output\":\"Final multi-paragraph plan with explicit bet instructions\"}\n"
            "Use the tools iteratively, cite crowd trends if possible, weigh feasibility and "
            "liquidity, and only finish when you have concrete recommendations."
        )

    def _log(self, message: str) -> None:
        if not self._debug_tools:
            return
        print(f"[tool-agent] {message}", file=sys.stderr)
