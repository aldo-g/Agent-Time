"""Callback helpers for tracking tool usage."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Any, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


def _extract_wallet_from_text(output: str) -> Optional[str]:
    match = re.search(r"\bWallet:\s*([\w\-]+)", output)
    if match:
        return match.group(1)
    return None


class ToolCallTracker(BaseCallbackHandler):
    """Track successful and failed tool invocations during a run."""

    def __init__(self) -> None:
        self.successful_tools: List[str] = []
        self.failed_tools: List[str] = []
        self.trade_outputs: List[str] = []
        self.failed_tool_errors: List[str] = []
        self.wallets_seen: set[str] = set()
        self.wallets_by_tool: dict[str, str] = {}
        self.wallet_mismatch_error: Optional[str] = None

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ANN401
        name = self._extract_tool_name(kwargs)
        if name:
            self.successful_tools.append(name)
            if name in {"manifold_place_bet", "manifold_sell_position"} and output is not None:
                self.trade_outputs.append(str(output))
            if (
                name in {"manifold_place_bet", "manifold_sell_position"}
                and isinstance(output, str)
                and (output.startswith("Bet skipped") or output.startswith("Sell skipped"))
            ):
                self.failed_tools.append(name)
                self.failed_tool_errors.append(f"{name}: {output}")
            if name in {"manifold_portfolio", "portfolio_analytics"} and isinstance(output, str):
                wallet = self._extract_wallet(output)
                if wallet:
                    self.wallets_seen.add(wallet)
                    self.wallets_by_tool[name] = wallet
                    if len(self.wallets_seen) > 1 and self.wallet_mismatch_error is None:
                        self.wallet_mismatch_error = (
                            "Wallet mismatch across tools: "
                            + ", ".join(sorted(self.wallets_seen))
                        )

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        name = self._extract_tool_name(kwargs)
        if name:
            self.failed_tools.append(name)
            err_text = str(error).strip() or error.__class__.__name__
            self.failed_tool_errors.append(f"{name}: {err_text}")

    @staticmethod
    def _extract_tool_name(kwargs: dict[str, Any]) -> Optional[str]:
        if "name" in kwargs and isinstance(kwargs["name"], str):
            return kwargs["name"]
        if "run_name" in kwargs and isinstance(kwargs["run_name"], str):
            return kwargs["run_name"]
        tool = kwargs.get("tool")
        if isinstance(tool, str):
            return tool
        if tool is not None:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                return name
        serialized = kwargs.get("serialized")
        if isinstance(serialized, dict):
            name = serialized.get("name") or serialized.get("id")
            if isinstance(name, str):
                return name
        if isinstance(serialized, str):
            return serialized
        if serialized is not None:
            name = getattr(serialized, "name", None) or getattr(serialized, "id", None)
            if isinstance(name, str):
                return name
        return None

    @staticmethod
    def _extract_wallet(output: str) -> Optional[str]:
        return _extract_wallet_from_text(output)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenUsageTracker(BaseCallbackHandler):
    """Track token usage from LLM responses."""

    def __init__(self) -> None:
        self.usage = TokenUsage()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        usage = _extract_token_usage(response)
        if usage is None:
            return
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        self.usage.total_tokens += usage.total_tokens


class StepDelay(BaseCallbackHandler):
    """Sleep briefly before each LLM call to throttle request rate."""

    def __init__(self, delay_sec: float) -> None:
        try:
            delay = float(delay_sec)
        except Exception:
            delay = 0.0
        self.delay_sec = max(0.0, delay)

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if self.delay_sec:
            time.sleep(self.delay_sec)


def _extract_token_usage(response: Any) -> TokenUsage | None:  # noqa: ANN401
    if response is None:
        return None
    llm_output = getattr(response, "llm_output", None)
    usage = _parse_usage_dict(llm_output)
    if usage:
        return usage
    generations = getattr(response, "generations", None)
    if generations:
        for generation_group in generations:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                if message is not None:
                    usage = _parse_usage_dict(getattr(message, "usage_metadata", None))
                    if usage:
                        return usage
                    usage = _parse_usage_dict(getattr(message, "response_metadata", None))
                    if usage:
                        return usage
                usage = _parse_usage_dict(getattr(generation, "generation_info", None))
                if usage:
                    return usage
    return None


def _parse_usage_dict(payload: Any) -> TokenUsage | None:  # noqa: ANN401
    if not isinstance(payload, dict):
        return None
    nested = (
        payload.get("token_usage")
        or payload.get("usage")
        or payload.get("usage_metadata")
        or payload
    )
    if not isinstance(nested, dict):
        return None
    prompt = nested.get("prompt_tokens") or nested.get("input_tokens")
    completion = nested.get("completion_tokens") or nested.get("output_tokens")
    total = nested.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    total = int(total or (prompt + completion))
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

class ConsoleLogger(BaseCallbackHandler):
    """Lightweight stdout logger that avoids LangChain's StdOut handler quirks."""

    def __init__(self, show_inputs: bool = True, show_outputs: bool = True) -> None:
        self.show_inputs = show_inputs
        self.show_outputs = show_outputs
        self.expected_wallet = (os.environ.get("AGENT_EXPECTED_WALLET") or "").strip()
        self.current_wallet = self.expected_wallet

    def _preview(self, value: Any) -> str:
        try:
            preview = str(value)
        except Exception:
            preview = "<unprintable>"
        # Keep callback logs single-line so external log parsers don't assign
        # fallback timestamps to continuation lines.
        preview = preview.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
        if len(preview) > 200:
            preview = preview[:197] + "..."
        return preview

    def _name(self, kwargs: dict[str, Any]) -> str:
        if "name" in kwargs and isinstance(kwargs["name"], str):
            return kwargs["name"]
        if "run_name" in kwargs and isinstance(kwargs["run_name"], str):
            return kwargs["run_name"]
        tool = kwargs.get("tool")
        if isinstance(tool, str):
            return tool
        if tool is not None:
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                return name
        serialized = kwargs.get("serialized")
        if isinstance(serialized, dict):
            name = serialized.get("name") or serialized.get("id")
            if isinstance(name, str):
                return name
        elif serialized is not None:
            name = getattr(serialized, "name", None) or getattr(serialized, "id", None)
            if isinstance(name, str):
                return name
        return "unknown"

    def on_tool_start(self, serialized: Any, input_str: Any, **kwargs: Any) -> None:  # noqa: ANN401
        name = self._name({**kwargs, "serialized": serialized})
        wallet = self.current_wallet or self.expected_wallet or "unknown"
        if not self.show_inputs:
            print(f"[tool:start] calling tool '{name}' in wallet '{wallet}'")
            return
        payload = input_str
        if payload in (None, ""):
            payload = kwargs.get("input") or kwargs.get("inputs")
        print(
            f"[tool:start] calling tool '{name}' in wallet '{wallet}' "
            f"with input {self._preview(payload)}"
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ANN401
        name = self._name(kwargs)
        if isinstance(output, str):
            observed_wallet = _extract_wallet_from_text(output)
            if observed_wallet:
                self.current_wallet = observed_wallet
        wallet = self.current_wallet or self.expected_wallet or "unknown"
        if not self.show_outputs:
            print(f"[tool:end] tool '{name}' ran successfully in wallet '{wallet}'")
            return
        print(
            f"[tool:end] tool '{name}' ran successfully in wallet '{wallet}' -> "
            f"{self._preview(output)}"
        )

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        name = self._name(kwargs)
        wallet = self.current_wallet or self.expected_wallet or "unknown"
        print(f"[tool:error] tool '{name}' failed in wallet '{wallet}': {error}")
