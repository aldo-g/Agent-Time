#!/usr/bin/env python3
"""Run Agent-Time in single-agent mode."""

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
from agent.db import DbWriter, EquitySnapshotRecord, OpenPositionRecord, RunActionRecord, TradeExecutionRecord
from agent.tools.manifold.config import DEFAULT_TRADE_LOG_PATH, TRADE_LOG_ENV

DEFAULT_CONFIG_PATH = os.environ.get("AGENT_CONFIG_PATH", "agent.json")
DEFAULT_RESULTS_PATH = os.environ.get("AGENT_RESULTS_PATH", "results/gpt_runs.jsonl")
MARKET_CACHE_ENV = "PREDICT_ARENA_MARKET_CACHE"
DEFAULT_MARKET_CACHE_PATH = Path(os.environ.get(MARKET_CACHE_ENV, "data/shared_markets.json"))
DEFAULT_MARKET_LIMIT = int(os.environ.get("AGENT_MARKET_CACHE_LIMIT", "25"))
PROVIDER_LABELS = {
    "openai": "OpenAI",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _init_db_writer_with_session_retry() -> tuple[DbWriter, int, datetime]:
    attempts = max(1, _env_int("AGENT_DB_CONNECT_RETRIES", 30))
    delay_seconds = max(0.1, _env_float("AGENT_DB_CONNECT_RETRY_DELAY_SECONDS", 2.0))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            db_writer = DbWriter.from_env()
            db_writer.ping()
            db_writer.ensure_schema()
            latest_session = db_writer.get_latest_session()
            if not latest_session:
                raise RuntimeError("No session found. Run market-fetcher first.")
            return db_writer, latest_session["id"], latest_session["started_at"]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                print(
                    f"Database/session not ready yet (attempt {attempt}/{attempts}): {exc}. "
                    f"Retrying in {delay_seconds:.1f}s..."
                )
                time.sleep(delay_seconds)
    assert last_error is not None
    raise RuntimeError(last_error)


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

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
        provider = str(payload["model_provider"]).lower()
        if provider != "openai":
            raise ValueError(
                f"Agent '{payload['name']}' has unsupported model_provider '{provider}'. "
                "Only 'openai' is supported."
            )
        return cls(
            name=str(payload["name"]),
            model_provider="openai",
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
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Agent config file must contain a JSON object or a JSON list.")
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


def _load_market_cache(cache_path: Path) -> List[EventSummary] | None:
    if not cache_path.exists():
        print(f"Warning: market cache not found at {cache_path}.")
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: unable to read market cache ({exc}).")
        return None
    events = snapshot.get("events")
    if not isinstance(events, list):
        return None
    os.environ[MARKET_CACHE_ENV] = str(cache_path)
    fetched_at = snapshot.get("fetched_at")
    if isinstance(fetched_at, str):
        os.environ["PREDICT_ARENA_MARKET_FETCHED_AT"] = fetched_at
    os.environ["PREDICT_ARENA_MARKET_COUNT"] = str(len(events))
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
            trades.append(
                {
                    "trade": " ".join(summary_parts).strip(),
                    "reason": "",
                    "amount": amount.replace(",", ""),
                }
            )
            last_trade = trades[-1]
            continue
        match_trade = re.search(r"trade\s*-\s*(.+)", clean_line, flags=re.IGNORECASE)
        if match_trade:
            last_trade = {"trade": match_trade.group(1).strip("* "), "reason": "", "amount": None}
            trades.append(last_trade)
            continue
        match_reason = re.search(r"reason\s*-\s*(.+)", clean_line, flags=re.IGNORECASE)
        if match_reason and last_trade is not None:
            last_trade["reason"] = match_reason.group(1).strip()
    filtered = [
        {"trade": entry["trade"], "reason": entry.get("reason") or "", "amount": entry.get("amount")}
        for entry in trades
        if entry.get("trade")
    ]
    return filtered


def _parse_executed_tool_trades(captured_trades: object) -> List[Dict[str, str]]:
    """Parse only executed trade tool outputs (exclude skipped/failed intents)."""
    if not isinstance(captured_trades, list):
        return []
    trades: List[Dict[str, str]] = []
    seen: set[str] = set()
    for entry in captured_trades:
        text = str(entry or "").strip()
        if not text:
            continue
        lower = text.lower()
        if "bet skipped" in lower or "sell skipped" in lower:
            continue
        wager_match = re.search(
            r"wagered\s+([\d.,]+)\s+mana\s+on\s+'([^']+)'\s+in\s+market\s+([A-Za-z0-9_-]+)"
            r"(?:.*?bet id:?\s*([A-Za-z0-9_-]+))?",
            text,
            flags=re.IGNORECASE,
        )
        if wager_match:
            amount, outcome, market_id, bet_id = wager_match.groups()
            parts = [f"Wagered {amount} MANA on '{outcome}' in market {market_id}"]
            if bet_id:
                parts.append(f"(Bet ID {bet_id})")
            trade_text = " ".join(parts)
            signature = f"buy:{trade_text}"
            if signature not in seen:
                seen.add(signature)
                trades.append(
                    {
                        "trade": trade_text,
                        "reason": "",
                        "amount": amount.replace(",", ""),
                    }
                )
            continue
        sell_match = re.search(
            r"sold\s+([\d.,]+)\s+shares\s+of\s+'([^']+)'\s+in\s+market\s+([A-Za-z0-9_-]+)"
            r"(?:.*?bet id:?\s*([A-Za-z0-9_-]+))?",
            text,
            flags=re.IGNORECASE,
        )
        if sell_match:
            shares, outcome, market_id, bet_id = sell_match.groups()
            parts = [f"Sold {shares} shares of '{outcome}' in market {market_id}"]
            if bet_id:
                parts.append(f"(Bet ID {bet_id})")
            trade_text = " ".join(parts)
            signature = f"sell:{trade_text}"
            if signature not in seen:
                seen.add(signature)
                trades.append(
                    {
                        "trade": trade_text,
                        "reason": "",
                        "amount": None,
                    }
                )
            continue
    return trades


def _parse_plan_payload(plan_output: object) -> dict[str, Any] | None:
    if isinstance(plan_output, dict):
        return plan_output
    if not isinstance(plan_output, str):
        return None
    text = plan_output.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_action_type(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"place_bet", "buy", "bet"}:
        return "place_bet"
    if raw in {"sell_position", "sell"}:
        return "sell_position"
    return raw or "unknown"


def _plan_actions_from_output(plan_output: object) -> list[dict[str, Any]]:
    payload = _parse_plan_payload(plan_output)
    if not payload:
        return []
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        normalized.append(
            {
                "action_index": idx,
                "action_type": _normalize_action_type(action.get("action") or action.get("action_type")),
                "market_id": action.get("market_id"),
                "outcome": action.get("outcome"),
                "amount": _coerce_float(action.get("amount")),
                "shares": _coerce_float(action.get("shares")),
                "belief_prob": _coerce_float(action.get("belief_prob")),
                "market_prob": _coerce_float(action.get("market_prob")),
                "edge_at_plan": (
                    _coerce_float(action.get("edge"))
                    if _coerce_float(action.get("edge")) is not None
                    else (
                        _coerce_float(action.get("belief_prob")) - _coerce_float(action.get("market_prob"))
                        if _coerce_float(action.get("belief_prob")) is not None
                        and _coerce_float(action.get("market_prob")) is not None
                        else None
                    )
                ),
                "limit_prob": _coerce_float(action.get("limit_prob")),
                "answer": action.get("answer"),
                "requires_news_catalyst": (
                    bool(action.get("requires_news_catalyst"))
                    if action.get("requires_news_catalyst") is not None
                    else None
                ),
                "catalyst_urls": (
                    [str(url) for url in action.get("catalyst_urls", []) if str(url).strip()]
                    if isinstance(action.get("catalyst_urls"), list)
                    else None
                ),
                "reason": str(action.get("reason") or "").strip() or None,
            }
        )
    return normalized


def _match_plan_action_for_log(
    *,
    action: str | None,
    market_id: str | None,
    outcome: str | None,
    run_actions: list[RunActionRecord],
    used_indices: set[int],
) -> int | None:
    desired_type = "place_bet" if (action or "").upper() == "BUY" else "sell_position"
    market_id_norm = str(market_id or "").strip()
    outcome_norm = str(outcome or "").strip().lower()
    for idx, planned in enumerate(run_actions):
        if idx in used_indices:
            continue
        if planned.action_type != desired_type:
            continue
        if market_id_norm and str(planned.market_id or "").strip() != market_id_norm:
            continue
        if outcome_norm and str(planned.outcome or "").strip().lower() != outcome_norm:
            continue
        return idx
    for idx, planned in enumerate(run_actions):
        if idx in used_indices:
            continue
        if planned.action_type == desired_type:
            return idx
    return None


def _has_declared_trade_lines(output: object) -> bool:
    if not isinstance(output, str):
        return False
    if _parse_trades_from_output(output):
        return True
    lowered = output.lower()
    if "concrete trade" in lowered:
        return True
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_lower = line.lower()
        if re.search(r"^\s*trade\s*-\s*", line, flags=re.IGNORECASE):
            return True
        if line_lower.startswith("action:") and any(
            token in line_lower for token in ("bet", "buy", "sell", "placed", "execute", "wager")
        ):
            return True
        if re.search(r"\bplaced\s+(a\s+)?bet\b", line_lower):
            return True
        if re.search(r"\bwagered\s+[\d.,]+\s*(mana|\$)", line_lower):
            return True
        if re.search(r"\bexecuted\s+trade\b", line_lower):
            return True
        if re.search(r"\bbet\s+\$?\d", line_lower):
            return True
        if re.search(r"\bbuy\s+\$?\d", line_lower):
            return True
        if re.search(r"\bsold\s+(a\s+)?position\b", line_lower):
            return True
        if re.search(r"\bopened\s+(a\s+)?new\s+position\b", line_lower):
            return True
        if re.search(r"\bopened\s+(a\s+)?position\b", line_lower):
            return True
        if re.search(r"\bclosed\s+(a\s+)?position\b", line_lower):
            return True
    return False


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


def _load_trade_log_entries(
    path: Path,
    *,
    agent_name: str,
    start: datetime,
    end: datetime | None,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("agent") != agent_name:
                continue
            ts_raw = payload.get("timestamp")
            if not isinstance(ts_raw, str):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            if ts < start:
                continue
            if end is not None and ts > end:
                continue
            entries.append(payload)
    return entries


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
    skip_market_fetch: bool = False,
    max_attempts: int = 2,
    verbose: bool = False,
) -> None:
    if len(configs) != 1:
        names = ", ".join(cfg.name for cfg in configs)
        raise ValueError(
            "Single-agent mode expects exactly one selected agent config. "
            f"Selected {len(configs)}: {names}"
        )
    # Ensure MANIFOLD_API_KEY is always set explicitly for the configured runner.
    os.environ.pop("MANIFOLD_API_KEY", None)
    cache_path = market_cache_path or DEFAULT_MARKET_CACHE_PATH
    market_events = (
        _load_market_cache(cache_path)
        if skip_market_fetch
        else _prepare_market_cache(market_limit, cache_path)
    )
    session_records: List[Dict[str, Any]] = []
    db_writer: DbWriter | None = None
    session_id: int | None = None
    session_started_at = datetime.now(timezone.utc)
    run_ids: Dict[str, int] = {}
    try:
        db_writer, session_id, session_started_at = _init_db_writer_with_session_retry()
        for cfg in configs:
            run_id = db_writer.get_run_id(session_id=session_id)
            if run_id is None:
                run_id = db_writer.create_run_placeholder(
                    session_id=session_id,
                    model_provider=cfg.model_provider,
                    model=cfg.model,
                    started_at=session_started_at,
                )
            run_ids[cfg.name] = run_id
    except Exception as exc:  # noqa: BLE001
        print(f"Database not configured or unavailable. Skipping DB writes. ({exc})")
        db_writer = None
    trade_log_path = Path(os.environ.get(TRADE_LOG_ENV, str(DEFAULT_TRADE_LOG_PATH)))
    for cfg in configs:
        timestamp = datetime.now(timezone.utc).isoformat()
        manifold_key = cfg.resolve_manifold_key()
        instruction = cfg.resolve_instruction()
        print(f"\n=== Running agent '{cfg.name}' ({cfg.model_provider}:{cfg.model}) ===")
        success = False
        output: str | Dict[str, Any] | None = None
        plan_output: str | Dict[str, Any] | None = None
        error: str | None = None
        tool_calls = None
        tool_calls_all = None
        captured_trades = None
        tool_errors = None
        portfolio_snapshot = None
        portfolio_error = None
        pre_portfolio_snapshot = None
        post_open_positions: list[OpenPositionRecord] = []
        pre_cash_balance: float | None = None
        cash_netted: float | None = None
        run_started_at: datetime | None = None
        run_finished_at: datetime | None = None
        run_duration_ms: int | None = None
        run_id = run_ids.get(cfg.name) if db_writer is not None else None
        with (
            _temporary_env("MANIFOLD_API_KEY", manifold_key),
            _temporary_env("AGENT_NAME", cfg.name),
            _temporary_env("AGENT_PROVIDER", cfg.model_provider),
            _temporary_env("AGENT_MODEL", cfg.model),
        ):
            try:
                pre_snapshot = fetch_portfolio_snapshot(None, api_key=manifold_key)
                pre_cash_balance = pre_snapshot.cash_balance
                pre_portfolio_snapshot = _snapshot_to_dict(pre_snapshot)
            except Exception:  # noqa: BLE001
                pre_cash_balance = None
                pre_portfolio_snapshot = None
            attempts = max(1, int(max_attempts))
            attempt = 0
            while attempt < attempts:
                if attempt > 0 and db_writer is not None and session_id is not None:
                    run_id = db_writer.create_run_placeholder(
                        session_id=session_id,
                        model_provider=cfg.model_provider,
                        model=cfg.model,
                        started_at=datetime.now(timezone.utc),
                    )
                run_started_at = datetime.now(timezone.utc)
                run_finished_at = None
                run_duration_ms = None
                output = None
                plan_output = None
                tool_calls = None
                tool_calls_all = None
                captured_trades = None
                tool_errors = None
                try:
                    result = run_daily_session(
                        instruction,
                        model=cfg.model,
                        temperature=cfg.resolve_temperature(),
                        max_steps=cfg.resolve_max_steps(),
                        verbose=verbose,
                    )
                    output = result.get("output") if isinstance(result, dict) else result
                    plan_output = result.get("plan_output") if isinstance(result, dict) else None
                    tool_calls = result.get("tool_calls_unique") if isinstance(result, dict) else None
                    tool_calls_all = result.get("tool_calls") if isinstance(result, dict) else None
                    captured_trades = result.get("captured_trades") if isinstance(result, dict) else None
                    tool_errors = result.get("tool_call_errors") if isinstance(result, dict) else None
                    if isinstance(result, dict):
                        started_raw = result.get("run_started_at")
                        finished_raw = result.get("run_finished_at")
                        duration_raw = result.get("run_duration_ms")
                        if isinstance(started_raw, str):
                            try:
                                run_started_at = datetime.fromisoformat(started_raw)
                            except ValueError:
                                pass
                        if isinstance(finished_raw, str):
                            try:
                                run_finished_at = datetime.fromisoformat(finished_raw)
                            except ValueError:
                                run_finished_at = None
                        if isinstance(duration_raw, int):
                            run_duration_ms = duration_raw
                    declared_trade = _has_declared_trade_lines(output)
                    trade_tool_called = any(
                        tool in (tool_calls_all or [])
                        for tool in ("manifold_place_bet", "manifold_sell_position")
                    )
                    if declared_trade and not trade_tool_called:
                        raise RuntimeError(
                            "Trade guard: model reported Trade - lines without calling a trade tool."
                        )
                    success = True
                    error = None  # clear any previous attempt errors
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).strip()
                    error = msg or f"{exc.__class__.__name__}"
                    run_finished_at = datetime.now(timezone.utc)
                    run_duration_ms = int((run_finished_at - run_started_at).total_seconds() * 1000)
                    if attempt == 0:
                        print(f"Retrying agent '{cfg.name}' after error: {error}")
                        time.sleep(2)
                    elif attempt < attempts - 1:
                        print(f"Retrying agent '{cfg.name}' after error: {error}")
                        time.sleep(2)
                    print(f"Agent '{cfg.name}' failed: {error}")
                attempt += 1
                if run_finished_at is None:
                    run_finished_at = datetime.now(timezone.utc)
                if run_duration_ms is None and run_started_at is not None:
                    run_duration_ms = int((run_finished_at - run_started_at).total_seconds() * 1000)
                if db_writer is not None and run_id is not None and run_started_at is not None:
                    try:
                        db_writer.update_run(
                            run_id=run_id,
                            started_at=run_started_at,
                            finished_at=run_finished_at,
                            run_duration_ms=run_duration_ms,
                            success=success,
                            error=None if success else error,
                            no_trade_reason=None,
                            tool_calls_count=len(tool_calls_all or []),
                            cash_netted=None,
                            current_balance=None,
                            cash_balance=None,
                            position_balance=None,
                            bankroll=None,
                            plan_output_json=_parse_plan_payload(plan_output),
                            execution_output=output if isinstance(output, str) else None,
                            metadata={
                                "tool_calls": tool_calls_all or [],
                                "tool_errors": tool_errors or [],
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to update run {run_id}: {exc}")
                if success:
                    break
            if success:
                for attempt in range(2):
                    try:
                        snapshot = fetch_portfolio_snapshot(None, api_key=manifold_key)
                        portfolio_snapshot = _snapshot_to_dict(snapshot)
                        post_open_positions = [
                            OpenPositionRecord(
                                market_id=position.market_id,
                                market_slug=position.slug,
                                question=position.question,
                                outcome=position.outcome,
                                shares=position.shares,
                                avg_price=position.avg_price,
                                mark_price=position.mark_price,
                                position_value=position.estimated_value(),
                                pnl=position.pnl,
                            )
                            for position in snapshot.positions
                        ]
                        if tool_calls and any(
                            call in tool_calls
                            for call in ("manifold_place_bet", "manifold_sell_position")
                        ):
                            time.sleep(2)
                            snapshot = fetch_portfolio_snapshot(None, api_key=manifold_key)
                            portfolio_snapshot = _snapshot_to_dict(snapshot)
                            post_open_positions = [
                                OpenPositionRecord(
                                    market_id=position.market_id,
                                    market_slug=position.slug,
                                    question=position.question,
                                    outcome=position.outcome,
                                    shares=position.shares,
                                    avg_price=position.avg_price,
                                    mark_price=position.mark_price,
                                    position_value=position.estimated_value(),
                                    pnl=position.pnl,
                                )
                                for position in snapshot.positions
                            ]
                        break
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc).strip()
                        portfolio_error = msg or f"{exc.__class__.__name__}"
                        if attempt == 0:
                            time.sleep(2)
                            continue
            if isinstance(portfolio_snapshot, dict):
                post_cash_balance = portfolio_snapshot.get("cash_balance")
                if pre_cash_balance is not None and post_cash_balance is not None:
                    cash_netted = float(post_cash_balance) - float(pre_cash_balance)
        if run_finished_at is None:
            run_finished_at = datetime.now(timezone.utc)
        record = {
            "timestamp": timestamp,
            "agent": cfg.name,
            "model_provider": cfg.model_provider,
            "model": cfg.model,
            "instruction": instruction,
            "success": success,
            "plan_output": plan_output,
            "output": output,
            "error": None if success else error,
            "tool_calls": tool_calls,
            "captured_trades": captured_trades,
            "tool_errors": tool_errors,
            "portfolio": portfolio_snapshot,
            "portfolio_error": portfolio_error,
            "run_started_at": run_started_at.isoformat(),
            "run_finished_at": run_finished_at.isoformat() if run_finished_at else None,
            "run_duration_ms": run_duration_ms,
            "cash_netted": cash_netted,
        }
        _persist_result(record, results_path)
        if verbose:
            print("\n--- Agent Output ---")
            if isinstance(output, str):
                print(output)
            else:
                print(output)
            if tool_errors:
                print("\n--- Tool Errors ---")
                for tool_error in tool_errors:
                    print(f"- {tool_error}")
            if not success and error:
                print(f"\n--- Agent Error ---\n{error}")
        summary_trades = _parse_executed_tool_trades(captured_trades)
        no_trade_reason = _extract_no_trade_reason(output) if success and not summary_trades else None
        if success and not summary_trades and not no_trade_reason:
            no_trade_reason = "Agent did not provide a reason for no trades."
        if db_writer is not None:
            try:
                bankroll = None
                if isinstance(portfolio_snapshot, dict):
                    bankroll = portfolio_snapshot.get("bankroll")
                    cash_balance = portfolio_snapshot.get("cash_balance")
                    position_balance = portfolio_snapshot.get("positions_value")
                else:
                    cash_balance = None
                    position_balance = None
                if run_id is not None and run_started_at is not None:
                    db_writer.update_run(
                        run_id=run_id,
                        started_at=run_started_at,
                        finished_at=run_finished_at,
                        run_duration_ms=run_duration_ms,
                        success=success,
                        error=None if success else error,
                        no_trade_reason=no_trade_reason,
                        tool_calls_count=len(tool_calls_all or []),
                        cash_netted=float(cash_netted) if cash_netted is not None else None,
                        current_balance=float(bankroll) if bankroll is not None else None,
                        cash_balance=float(cash_balance) if cash_balance is not None else None,
                        position_balance=float(position_balance) if position_balance is not None else None,
                        bankroll=float(bankroll) if bankroll is not None else None,
                        plan_output_json=_parse_plan_payload(plan_output),
                        execution_output=output if isinstance(output, str) else json.dumps(output),
                        metadata={
                            "tool_calls": tool_calls_all or [],
                            "tool_errors": tool_errors or [],
                            "captured_trades": captured_trades or [],
                        },
                    )
                if run_id is not None:
                    planned_payloads = _plan_actions_from_output(plan_output)
                    run_actions: list[RunActionRecord] = [
                        RunActionRecord(
                            action_index=int(action.get("action_index") or idx),
                            action_type=str(action.get("action_type") or "unknown"),
                            market_id=str(action.get("market_id") or "") or None,
                            outcome=str(action.get("outcome") or "") or None,
                            amount=_coerce_float(action.get("amount")),
                            shares=_coerce_float(action.get("shares")),
                            belief_prob=_coerce_float(action.get("belief_prob")),
                            market_prob=_coerce_float(action.get("market_prob")),
                            edge_at_plan=_coerce_float(action.get("edge_at_plan")),
                            limit_prob=_coerce_float(action.get("limit_prob")),
                            answer=str(action.get("answer") or "") or None,
                            requires_news_catalyst=action.get("requires_news_catalyst"),
                            catalyst_urls=action.get("catalyst_urls"),
                            reason=str(action.get("reason") or "") or None,
                            status="planned",
                        )
                        for idx, action in enumerate(planned_payloads)
                    ]

                    log_entries = _load_trade_log_entries(
                        trade_log_path,
                        agent_name=cfg.name,
                        start=run_started_at,
                        end=run_finished_at,
                    )
                    used_action_indices: set[int] = set()
                    matched_log_entries: list[tuple[int | None, Dict[str, Any]]] = []
                    for entry in log_entries:
                        matched_idx = _match_plan_action_for_log(
                            action=str(entry.get("action") or ""),
                            market_id=str(entry.get("market_id") or ""),
                            outcome=str(entry.get("outcome") or ""),
                            run_actions=run_actions,
                            used_indices=used_action_indices,
                        )
                        if matched_idx is not None:
                            used_action_indices.add(matched_idx)
                            run_actions[matched_idx].status = "executed"
                        matched_log_entries.append((matched_idx, entry))

                    trade_tool_errors = [
                        str(tool_error)
                        for tool_error in (tool_errors or [])
                        if "manifold_place_bet" in str(tool_error) or "manifold_sell_position" in str(tool_error)
                    ]
                    error_matches: list[tuple[int | None, str]] = []
                    used_error_action_indices: set[int] = set()
                    for tool_error_text in trade_tool_errors:
                        desired_type = (
                            "place_bet"
                            if "manifold_place_bet" in tool_error_text
                            else "sell_position"
                            if "manifold_sell_position" in tool_error_text
                            else "unknown"
                        )
                        matched_idx = None
                        for idx, action in enumerate(run_actions):
                            if idx in used_error_action_indices:
                                continue
                            if action.action_type != desired_type:
                                continue
                            if action.status == "executed":
                                continue
                            matched_idx = idx
                            used_error_action_indices.add(idx)
                            break
                        if matched_idx is not None:
                            lowered = tool_error_text.lower()
                            if " skipped" in lowered:
                                run_actions[matched_idx].status = "skipped"
                                run_actions[matched_idx].skip_reason = tool_error_text
                            else:
                                run_actions[matched_idx].status = "failed"
                                run_actions[matched_idx].failure_reason = tool_error_text
                        error_matches.append((matched_idx, tool_error_text))

                    for action in run_actions:
                        if action.status == "executed":
                            continue
                        if action.status in {"skipped", "failed"}:
                            continue
                        if success:
                            action.status = "skipped"
                            action.skip_reason = "Not executed by execution phase."
                        else:
                            action.status = "failed"
                            action.failure_reason = error or "Run failed before execution."

                    action_ids = db_writer.insert_run_actions(run_id=run_id, actions=run_actions)
                    action_id_by_index = {idx: action_ids[idx] for idx in range(min(len(action_ids), len(run_actions)))}

                    executions: list[TradeExecutionRecord] = []
                    for matched_idx, entry in matched_log_entries:
                        run_action_id = action_id_by_index.get(matched_idx) if matched_idx is not None else None
                        reason = run_actions[matched_idx].reason if matched_idx is not None else None
                        action = str(entry.get("action") or "")
                        market = str(entry.get("market") or entry.get("market_id") or "unknown market")
                        outcome = str(entry.get("outcome") or "")
                        summary = f"{action} {market}" + (f" [{outcome}]" if outcome else "")
                        executions.append(
                            TradeExecutionRecord(
                                run_action_id=run_action_id,
                                market_id=str(entry.get("market_id") or "") or None,
                                market_slug=str(entry.get("market_slug") or "") or None,
                                action=action or None,
                                outcome=outcome or None,
                                amount=_coerce_float(entry.get("amount")),
                                shares=_coerce_float(entry.get("shares")),
                                prob_before=_coerce_float(entry.get("prob_before")),
                                prob_after=_coerce_float(entry.get("prob_after")),
                                bet_id=str(entry.get("bet_id") or "") or None,
                                status=str(entry.get("status") or "executed"),
                                error=None,
                                reason=reason,
                                summary=summary,
                                raw_response=entry,
                            )
                        )

                    for matched_idx, tool_error_text in error_matches:
                        run_action_id = action_id_by_index.get(matched_idx) if matched_idx is not None else None
                        planned = run_actions[matched_idx] if matched_idx is not None else None
                        action_label = "BUY" if "manifold_place_bet" in tool_error_text else "SELL"
                        status = "skipped" if " skipped" in tool_error_text.lower() else "failed"
                        executions.append(
                            TradeExecutionRecord(
                                run_action_id=run_action_id,
                                market_id=planned.market_id if planned else None,
                                market_slug=None,
                                action=action_label,
                                outcome=planned.outcome if planned else None,
                                amount=planned.amount if planned else None,
                                shares=planned.shares if planned else None,
                                prob_before=planned.market_prob if planned else None,
                                prob_after=None,
                                bet_id=None,
                                status=status,
                                error=tool_error_text,
                                reason=planned.reason if planned else None,
                                summary=tool_error_text,
                                raw_response={"tool_error": tool_error_text},
                            )
                        )

                    db_writer.insert_trade_executions(run_id=run_id, executions=executions)

                    equity_snapshots: list[EquitySnapshotRecord] = []
                    if isinstance(portfolio_snapshot, dict):
                        equity_snapshots.append(
                            EquitySnapshotRecord(
                                cash_balance=_coerce_float(portfolio_snapshot.get("cash_balance")),
                                positions_value=_coerce_float(portfolio_snapshot.get("positions_value")),
                                bankroll=_coerce_float(portfolio_snapshot.get("bankroll")),
                                gross_exposure=_coerce_float(portfolio_snapshot.get("gross_exposure")),
                                open_positions=_coerce_int(portfolio_snapshot.get("open_positions")),
                                snapshot_json=portfolio_snapshot,
                            )
                        )
                    db_writer.insert_equity_snapshots(run_id=run_id, snapshots=equity_snapshots)
                    db_writer.insert_open_positions(run_id=run_id, positions=post_open_positions)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to write run to DB: {exc}")
        record_summary = {
            "agent": cfg.name,
            "provider": cfg.model_provider,
            "success": success,
            "error": error,
            "trades": summary_trades,
            "no_trade_reason": no_trade_reason,
            "tool_calls": tool_calls,
            "tool_errors": tool_errors,
        }
        session_records.append(record_summary)
        if success:
            print(f"Agent '{cfg.name}' completed.")
    if db_writer is not None and session_id is not None:
        try:
            db_writer.finish_session(session_id=session_id, finished_at=datetime.now(timezone.utc))
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to finalize session {session_id}: {exc}")
    _print_session_summary(market_events, session_records)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agent-Time single-agent runner.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file (single selected agent in single-agent mode).",
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
        help="Number of markets to fetch once and share with the runner.",
    )
    parser.add_argument(
        "--market-cache",
        default=str(DEFAULT_MARKET_CACHE_PATH),
        help="File path for the shared market snapshot JSON.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("AGENT_MAX_ATTEMPTS", "2")),
        help="Number of attempts per agent run before giving up (default: 2).",
    )
    parser.add_argument(
        "--skip-market-fetch",
        action="store_true",
        help="Use the shared market cache file instead of fetching markets live.",
    )
    parser.add_argument(
        "--agent",
        dest="agents",
        action="append",
        help=(
            "Select the agent entry from config. Repeat the flag or separate names with commas; "
            "single-agent mode requires exactly one final selection."
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
        skip_market_fetch=args.skip_market_fetch,
        max_attempts=args.max_attempts,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
