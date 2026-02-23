"""Callback helpers for tracking tool usage."""

from __future__ import annotations

import time
from typing import Any, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


class ToolCallTracker(BaseCallbackHandler):
    """Track successful and failed tool invocations during a run."""

    def __init__(self) -> None:
        self.successful_tools: List[str] = []
        self.failed_tools: List[str] = []
        self.trade_outputs: List[str] = []
        self.failed_tool_errors: List[str] = []

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


class ConsoleLogger(BaseCallbackHandler):
    """Lightweight stdout logger that avoids LangChain's StdOut handler quirks."""

    def __init__(self, show_inputs: bool = True, show_outputs: bool = True) -> None:
        self.show_inputs = show_inputs
        self.show_outputs = show_outputs

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
        if not self.show_inputs:
            print(f"[tool:start] calling tool '{name}'")
            return
        payload = input_str
        if payload in (None, ""):
            payload = kwargs.get("input") or kwargs.get("inputs")
        print(
            f"[tool:start] calling tool '{name}' "
            f"with input {self._preview(payload)}"
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ANN401
        name = self._name(kwargs)
        if not self.show_outputs:
            print(f"[tool:end] tool '{name}' ran successfully")
            return
        print(
            f"[tool:end] tool '{name}' ran successfully -> "
            f"{self._preview(output)}"
        )

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        name = self._name(kwargs)
        print(f"[tool:error] tool '{name}' failed: {error}")
