"""Callback helpers for tracking tool usage."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


class ToolCallTracker(BaseCallbackHandler):
    """Track successful and failed tool invocations during a run."""

    def __init__(self) -> None:
        self.successful_tools: List[str] = []
        self.failed_tools: List[str] = []

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # noqa: ANN401
        name = self._extract_tool_name(kwargs)
        if name:
            self.successful_tools.append(name)

    def on_tool_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> None:
        name = self._extract_tool_name(kwargs)
        if name:
            self.failed_tools.append(name)

    @staticmethod
    def _extract_tool_name(kwargs: dict[str, Any]) -> Optional[str]:
        if "name" in kwargs and isinstance(kwargs["name"], str):
            return kwargs["name"]
        if "run_name" in kwargs and isinstance(kwargs["run_name"], str):
            return kwargs["run_name"]
        serialized = kwargs.get("serialized")
        if isinstance(serialized, dict):
            name = serialized.get("name")
            if isinstance(name, str):
                return name
        return None
