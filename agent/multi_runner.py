#!/usr/bin/env python3
"""Orchestrate multiple Agent-Time competitors in a single run."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent.manifold.data import EventSummary, events_to_dicts, load_open_markets
from agent.manifold.portfolio import PortfolioSnapshot, fetch_portfolio_snapshot
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
DEFAULT_MARKET_LIMIT = int(os.environ.get("AGENT_MARKET_CACHE_LIMIT", "25"))
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


def _estimate_bankroll(snapshot: PortfolioSnapshot) -> tuple[float, float]:
    cash = snapshot.cash_balance or 0.0
    positions_value = snapshot.investment_value
    if positions_value is None and snapshot.unrealized_pnl is not None:
        positions_value = float(snapshot.unrealized_pnl)
    if positions_value is not None:
        positions_value = float(positions_value)
        return cash + positions_value, abs(positions_value)
    net_value = 0.0
    gross_exposure = 0.0
    for position in snapshot.positions:
        value = position.estimated_value()
        if value is None:
            continue
        net_value += value
        gross_exposure += abs(value)
    return cash + net_value, gross_exposure


def _snapshot_to_dict(snapshot: PortfolioSnapshot) -> Dict[str, Any]:
    bankroll, gross_exposure = _estimate_bankroll(snapshot)
    positions_value = snapshot.investment_value
    if positions_value is None and snapshot.unrealized_pnl is not None:
        positions_value = float(snapshot.unrealized_pnl)
    if positions_value is None:
        positions_value = 0.0
        for position in snapshot.positions:
            value = position.estimated_value()
            if value is None:
                continue
            positions_value += value
    positions = [
        {
            "market_id": position.market_id,
            "question": position.question,
            "outcome": position.outcome,
            "shares": position.shares,
            "avg_price": position.avg_price,
            "mark_price": position.mark_price,
            "pnl": position.pnl,
        }
        for position in snapshot.positions[:5]
    ]
    return {
        "wallet": snapshot.wallet,
        "cash_balance": snapshot.cash_balance,
        "realized_pnl": snapshot.realized_pnl,
        "unrealized_pnl": snapshot.unrealized_pnl,
        "investment_value": snapshot.investment_value,
        "cash_investment_value": snapshot.cash_investment_value,
        "positions_value": positions_value,
        "bankroll": bankroll,
        "gross_exposure": gross_exposure,
        "open_positions": len(snapshot.positions),
        "positions": positions,
    }


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
        wager_match = re.search(
            r"wagered\s+([\d.,]+)\s+mana\s+on\s+'([^']+)'(?:\s+in\s+market\s+([A-Za-z0-9_-]+))?"
            r"(?:.*?bet id:?\s*([A-Za-z0-9]+))?",
            clean_line,
            flags=re.IGNORECASE,
        )
        if wager_match:
            amount, outcome, market_id, bet_id = wager_match.groups()
            summary_parts = [
                f"Wagered {amount} MANA on '{outcome}'",
            ]
            if market_id:
                summary_parts.append(f"market {market_id}")
            if bet_id:
                summary_parts.append(f"(Bet ID {bet_id})")
            trades.append({"trade": " ".join(summary_parts).strip(), "reason": ""})
            last_trade = trades[-1]
            continue
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


def _extract_no_trade_reason(output: object) -> str | None:
    """Return a short reason for no trades if one can be inferred from the output text."""
    if not isinstance(output, str):
        return None
    for raw_line in output.splitlines():
        line = raw_line.strip().strip("#*- ")
        if not line:
            continue
        snippet = line
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        return snippet
    return None


def _print_session_summary(
    market_events: List[EventSummary] | None,
    agent_records: List[Dict[str, Any]],
) -> None:
    print("\n=== Session Snapshot ===")
    if market_events:
        print(f"Markets provided: {len(market_events)} shared events.")
    else:
        print("Markets provided: unavailable (live fetch failed).")
    print("\nAgent results:")
    for record in agent_records:
        label = PROVIDER_LABELS.get(record["provider"].lower(), record["agent"])
        trades = record.get("trades") or []
        tool_calls = record.get("tool_calls") or []
        tool_errors = record.get("tool_errors") or []
        status = "success" if record.get("success") else "failed"
        error = record.get("error")
        no_trade_reason = record.get("no_trade_reason")
        if trades:
            headline = trades[0].get("trade", "").strip()
            more = f" (+{len(trades)-1} more)" if len(trades) > 1 else ""
            print(f"{label}: {status}; trades={len(trades)} {more} {headline}")
        else:
            reason_note = ""
            if no_trade_reason:
                reason_note = f" Reason: {no_trade_reason}"
            print(f"{label}: {status}; {error or 'No trades recorded.'}{reason_note}")
        if error and not trades:
            continue
        if tool_calls:
            print(f"{label} tools used: {', '.join(tool_calls)}")
        if tool_errors:
            print(f"{label} tool errors: {', '.join(tool_errors)}")


def run_multi_agent(
    configs: List[AgentConfig],
    *,
    results_path: str,
    market_limit: int,
    market_cache_path: Path | None = None,
    verbose: bool = False,
) -> None:
    # Ensure a shared MANIFOLD_API_KEY cannot leak across agents.
    os.environ.pop("MANIFOLD_API_KEY", None)
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
        tool_calls = None
        captured_trades = None
        tool_errors = None
        portfolio_snapshot = None
        portfolio_error = None
        with (
            _temporary_env("MANIFOLD_API_KEY", manifold_key),
            _temporary_env("AGENT_NAME", cfg.name),
            _temporary_env("AGENT_PROVIDER", cfg.model_provider),
            _temporary_env("AGENT_MODEL", cfg.model),
        ):
            for attempt in range(2):
                try:
                    result = run_daily_session(
                        instruction,
                        model=cfg.model,
                        provider=cfg.model_provider,
                        temperature=cfg.resolve_temperature(),
                        max_steps=cfg.resolve_max_steps(),
                        verbose=verbose,
                    )
                    output = result.get("output") if isinstance(result, dict) else result
                    tool_calls = result.get("tool_calls_unique") if isinstance(result, dict) else None
                    captured_trades = result.get("captured_trades") if isinstance(result, dict) else None
                    tool_errors = result.get("tool_call_errors") if isinstance(result, dict) else None
                    success = True
                    error = None  # clear any previous attempt errors
                    break
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).strip()
                    error = msg or f"{exc.__class__.__name__}"
                    if attempt == 0:
                        print(f"Retrying agent '{cfg.name}' after error: {error}")
                        time.sleep(2)
                        continue
                    if cfg.model_provider.lower() in {"gemini", "google"}:
                        lowered = error.lower()
                        if "resourceexhausted" in lowered or "quota" in lowered or "429" in lowered:
                            error = (
                                "Gemini quota hit (429 ResourceExhausted). "
                                "Check your plan/usage limits and retry later."
                            )
                    print(f"Agent '{cfg.name}' failed: {error}")
            if success:
                for attempt in range(2):
                    try:
                        snapshot = fetch_portfolio_snapshot(None)
                        portfolio_snapshot = _snapshot_to_dict(snapshot)
                        if tool_calls and any(
                            call in tool_calls
                            for call in ("manifold_place_bet", "manifold_sell_position")
                        ):
                            time.sleep(2)
                            snapshot = fetch_portfolio_snapshot(None)
                            portfolio_snapshot = _snapshot_to_dict(snapshot)
                        break
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc).strip()
                        portfolio_error = msg or f"{exc.__class__.__name__}"
                        if attempt == 0:
                            time.sleep(2)
                            continue
        record = {
            "timestamp": timestamp,
            "agent": cfg.name,
            "model_provider": cfg.model_provider,
            "model": cfg.model,
            "instruction": instruction,
            "success": success,
            "output": output,
            "error": None if success else error,
            "tool_calls": tool_calls,
            "captured_trades": captured_trades,
            "tool_errors": tool_errors,
            "portfolio": portfolio_snapshot,
            "portfolio_error": portfolio_error,
        }
        _persist_result(record, results_path)
        trades = _parse_trades_from_output(output)
        if captured_trades:
            for entry in captured_trades:
                trades.extend(_parse_trades_from_output(entry))
        no_trade_reason = _extract_no_trade_reason(output) if success and not trades else None
        if success and not trades and not no_trade_reason:
            no_trade_reason = "Agent did not provide a reason for no trades."
        record_summary = {
            "agent": cfg.name,
            "provider": cfg.model_provider,
            "success": success,
            "error": error,
            "trades": trades,
            "no_trade_reason": no_trade_reason,
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
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
    parser.add_argument(
        "--agent",
        dest="agents",
        action="append",
        help=(
            "Run only the named agent(s) from the config. "
            "Repeat the flag or separate names with commas (e.g. --agent gpt-runner --agent claude-runner)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable LangChain verbose output inside each agent run.",
    )
    return parser


def _parse_agent_filters(raw_agents: list[str] | None) -> list[str]:
    if not raw_agents:
        return []
    agents: list[str] = []
    for raw in raw_agents:
        for name in raw.split(","):
            trimmed = name.strip()
            if trimmed:
                agents.append(trimmed)
    return agents


def _select_agents(configs: List[AgentConfig], requested: list[str]) -> List[AgentConfig]:
    if not requested:
        return configs
    normalized = {cfg.name.lower(): cfg for cfg in configs}
    selected: list[AgentConfig] = []
    missing: list[str] = []
    for name in requested:
        match = normalized.get(name.lower())
        if match:
            if match not in selected:
                selected.append(match)
        else:
            missing.append(name)
    if missing:
        available = ", ".join(cfg.name for cfg in configs)
        raise ValueError(f"Requested agent(s) not found: {', '.join(missing)}. Available: {available}")
    return selected


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    configs = load_agent_configs(args.config)
    requested_agents = _parse_agent_filters(args.agents)
    if requested_agents:
        configs = _select_agents(configs, requested_agents)
    run_multi_agent(
        configs,
        results_path=args.results,
        market_limit=args.market_limit,
        market_cache_path=Path(args.market_cache),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
