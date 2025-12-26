#!/usr/bin/env python3
"""Orchestrate multiple Agent-Time competitors in a single run."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent.runner import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MAX_STEPS,
    DEFAULT_TEMPERATURE,
    run_daily_session,
)

DEFAULT_CONFIG_PATH = os.environ.get("AGENT_CONFIG_PATH", "agents.example.json")
DEFAULT_RESULTS_PATH = os.environ.get("AGENT_RESULTS_PATH", "results/multi_agent_runs.jsonl")


@dataclass
class AgentConfig:
    """Configuration for a single competitor."""

    name: str
    model_provider: str
    model: str
    manifold_key: str | None = None
    manifold_key_env: str | None = None
    instruction_override: str | None = None
    temperature: float | None = None
    max_steps: int | None = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentConfig":
        required = {"name", "model_provider", "model"}
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Agent entry is missing required fields: {', '.join(missing)}")
        return cls(
            name=str(payload["name"]),
            model_provider=str(payload["model_provider"]),
            model=str(payload["model"]),
            manifold_key=payload.get("manifold_key"),
            manifold_key_env=payload.get("manifold_key_env"),
            instruction_override=payload.get("instruction_override"),
            temperature=payload.get("temperature"),
            max_steps=payload.get("max_steps"),
        )

    def resolve_instruction(self) -> str:
        return self.instruction_override or DEFAULT_INSTRUCTION

    def resolve_temperature(self) -> float:
        return float(self.temperature if self.temperature is not None else DEFAULT_TEMPERATURE)

    def resolve_max_steps(self) -> int:
        return int(self.max_steps if self.max_steps is not None else DEFAULT_MAX_STEPS)

    def resolve_manifold_key(self) -> str:
        if self.manifold_key:
            return self.manifold_key
        if self.manifold_key_env:
            key = os.environ.get(self.manifold_key_env)
            if key:
                return key
            raise RuntimeError(
                f"Environment variable {self.manifold_key_env} referenced by agent '{self.name}' is empty."
            )
        raise RuntimeError(f"No Manifold API key configured for agent '{self.name}'.")


def load_agent_configs(path: str) -> List[AgentConfig]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Agent config file must contain a JSON list.")
    configs = [AgentConfig.from_dict(item) for item in data]
    if not configs:
        raise ValueError("Agent config file is empty. Add at least one agent entry.")
    return configs


@contextlib.contextmanager
def _temporary_env(var_name: str, value: str):
    previous = os.environ.get(var_name)
    os.environ[var_name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = previous


def _persist_result(record: Dict[str, Any], path: str) -> None:
    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")


def run_multi_agent(configs: List[AgentConfig], *, results_path: str) -> None:
    for cfg in configs:
        timestamp = datetime.now(timezone.utc).isoformat()
        manifold_key = cfg.resolve_manifold_key()
        instruction = cfg.resolve_instruction()
        print(f"\n=== Running agent '{cfg.name}' ({cfg.model_provider}:{cfg.model}) ===")
        success = False
        output: str | Dict[str, Any] | None = None
        error: str | None = None
        with _temporary_env("MANIFOLD_API_KEY", manifold_key):
            try:
                result = run_daily_session(
                    instruction,
                    model=cfg.model,
                    provider=cfg.model_provider,
                    temperature=cfg.resolve_temperature(),
                    max_steps=cfg.resolve_max_steps(),
                )
                output = result.get("output") if isinstance(result, dict) else result
                success = True
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                print(f"Agent '{cfg.name}' failed: {error}")
        record = {
            "timestamp": timestamp,
            "agent": cfg.name,
            "model_provider": cfg.model_provider,
            "model": cfg.model,
            "instruction": instruction,
            "success": success,
            "output": output,
            "error": error,
        }
        _persist_result(record, results_path)
        if success:
            print(f"Agent '{cfg.name}' completed.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple Agent-Time competitors sequentially.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file describing each agent.",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help="Where to append JSONL run records.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    configs = load_agent_configs(args.config)
    run_multi_agent(configs, results_path=args.results)


if __name__ == "__main__":
    main()
