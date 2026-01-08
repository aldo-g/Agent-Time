#!/usr/bin/env python3
"""Orchestrate multiple Agent-Time competitors in a single run."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent.manifold.data import EventSummary, events_to_dicts, load_open_markets
from agent.runner import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MAX_STEPS,
    DEFAULT_TEMPERATURE,
    run_daily_session,
)

DEFAULT_CONFIG_PATH = os.environ.get("AGENT_CONFIG_PATH", "agents.json")
DEFAULT_RESULTS_PATH = os.environ.get("AGENT_RESULTS_PATH", "results/multi_agent_runs.jsonl")
MARKET_CACHE_ENV = "PREDICT_ARENA_MARKET_CACHE"
DEFAULT_MARKET_CACHE_PATH = Path(os.environ.get(MARKET_CACHE_ENV, "data/shared_markets.json"))
DEFAULT_MARKET_LIMIT = int(os.environ.get("AGENT_MARKET_CACHE_LIMIT", "40"))
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Claude",
    "claude": "Claude",
    "gemini": "Gemini",
    "google": "Gemini",
}


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


def _prepare_market_cache(limit: int, cache_path: Path) -> List[EventSummary] | None:
    try:
        events = load_open_markets(limit, 0)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: unable to fetch shared market snapshot ({exc}). Agents will fetch individually.")
        return None
    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "events": events_to_dicts(events),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle)
    os.environ[MARKET_CACHE_ENV] = str(cache_path)
    os.environ["PREDICT_ARENA_MARKET_FETCHED_AT"] = snapshot["fetched_at"]
    os.environ["PREDICT_ARENA_MARKET_COUNT"] = str(len(snapshot["events"]))
    print(f"Shared market snapshot saved to {cache_path} ({len(snapshot['events'])} markets).")
    return events


def _parse_trades_from_output(output: object) -> List[Dict[str, str]]:
    if not isinstance(output, str):
        return []
    trades: List[Dict[str, str]] = []
    last_trade = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        clean_line = line.strip("*_- ")
        match_trade = re.search(r"trade\s*-\s*(.+)", clean_line, flags=re.IGNORECASE)
        if match_trade:
            last_trade = {"trade": match_trade.group(1).strip("* "), "reason": ""}
            trades.append(last_trade)
            continue
        match_reason = re.search(r"reason\s*-\s*(.+)", clean_line, flags=re.IGNORECASE)
        if match_reason and last_trade is not None:
            last_trade["reason"] = match_reason.group(1).strip()
    filtered = [
        {"trade": entry["trade"], "reason": entry["reason"]}
        for entry in trades
        if entry.get("trade")
    ]
    return filtered


def _print_session_summary(
    market_events: List[EventSummary] | None,
    agent_records: List[Dict[str, Any]],
) -> None:
    print("\n=== Session Snapshot ===")
    if market_events:
        print("Markets provided:")
        for event in market_events:
            label = event.title or event.event_id
            url_note = f" ({event.url})" if event.url else ""
            print(f"- {label}{url_note}")
    else:
        print("Markets provided: unavailable (live fetch failed).")
    print("\nTrades:")
    for record in agent_records:
        label = PROVIDER_LABELS.get(record["provider"].lower(), record["agent"])
        trades = record.get("trades") or []
        tool_calls = record.get("tool_calls") or []
        if not trades:
            status = "success" if record.get("success") else "failed"
            reason = record.get("error") or "No trades recorded."
            print(f"{label}: ({status}) {reason}")
            if tool_calls:
                print(f"{label} tools: {', '.join(tool_calls)}")
            continue
        for trade in trades:
            reason = trade.get("reason", "").strip()
            reason_note = f" - {reason}" if reason else ""
            print(f"{label}: {trade['trade']}{reason_note}")
        if tool_calls:
            print(f"{label} tools: {', '.join(tool_calls)}")


def run_multi_agent(
    configs: List[AgentConfig],
    *,
    results_path: str,
    market_limit: int,
    market_cache_path: Path | None = None,
) -> None:
    cache_path = market_cache_path or DEFAULT_MARKET_CACHE_PATH
    market_events = _prepare_market_cache(market_limit, cache_path)
    session_records: List[Dict[str, Any]] = []
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
                tool_calls = result.get("tool_calls_unique") if isinstance(result, dict) else None
                success = True
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if cfg.model_provider.lower() in {"gemini", "google"}:
                    lowered = error.lower()
                    if "resourceexhausted" in lowered or "quota" in lowered or "429" in lowered:
                        error = (
                            "Gemini quota hit (429 ResourceExhausted). "
                            "Check your plan/usage limits and retry later."
                        )
                print(f"Agent '{cfg.name}' failed: {error}")
                tool_calls = None
        record = {
            "timestamp": timestamp,
            "agent": cfg.name,
            "model_provider": cfg.model_provider,
            "model": cfg.model,
            "instruction": instruction,
            "success": success,
            "output": output,
            "error": error,
            "tool_calls": tool_calls,
        }
        _persist_result(record, results_path)
        record_summary = {
            "agent": cfg.name,
            "provider": cfg.model_provider,
            "success": success,
            "error": error,
            "trades": _parse_trades_from_output(output),
            "tool_calls": tool_calls,
        }
        session_records.append(record_summary)
        if success:
            print(f"Agent '{cfg.name}' completed.")
    _print_session_summary(market_events, session_records)


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
    parser.add_argument(
        "--market-limit",
        type=int,
        default=DEFAULT_MARKET_LIMIT,
        help="Number of markets to fetch once and share with all agents.",
    )
    parser.add_argument(
        "--market-cache",
        default=str(DEFAULT_MARKET_CACHE_PATH),
        help="File path for the shared market snapshot JSON.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    configs = load_agent_configs(args.config)
    run_multi_agent(
        configs,
        results_path=args.results,
        market_limit=args.market_limit,
        market_cache_path=Path(args.market_cache),
    )


if __name__ == "__main__":
    main()
