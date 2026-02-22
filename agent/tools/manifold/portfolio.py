"""Portfolio inspection and risk analysis tools."""

from __future__ import annotations

import ast
import os
import re
from typing import List, Optional, Tuple

from agent.manifold.portfolio import PortfolioSnapshot, fetch_portfolio_snapshot
from agent.manifold.trading import MarketDetails, fetch_market_details

from . import config
from .errors import _is_not_found_error
from .limits import _enforce_market_limit
from .summaries import _summarize_portfolio

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except Exception:
        return default


DEFAULT_MANIFOLD_PORTFOLIO_MAX_CALLS = _env_int("AGENT_MANIFOLD_PORTFOLIO_MAX_CALLS", 2)
DEFAULT_PORTFOLIO_ANALYTICS_MAX_CALLS = _env_int("AGENT_PORTFOLIO_ANALYTICS_MAX_CALLS", 2)

_portfolio_call_count = 0
_portfolio_analytics_call_count = 0


def reset_portfolio_tool_state() -> None:
    """Reset per-run counters used to prevent runaway portfolio tool loops."""
    global _portfolio_call_count, _portfolio_analytics_call_count
    _portfolio_call_count = 0
    _portfolio_analytics_call_count = 0


def _resolve_expected_wallet() -> str | None:
    expected = (os.environ.get("AGENT_EXPECTED_WALLET") or "").strip()
    return expected or None


def _assert_expected_wallet(snapshot: PortfolioSnapshot, *, tool_name: str) -> None:
    expected_wallet = _resolve_expected_wallet()
    if not expected_wallet:
        return
    observed_wallet = (snapshot.wallet or "").strip()
    if observed_wallet.lower() != expected_wallet.lower():
        raise RuntimeError(
            f"Expected wallet '{expected_wallet}' but saw '{observed_wallet}' in {tool_name}."
        )


def _enforce_tool_call_limit(tool_name: str, call_count: int, max_calls: int) -> None:
    if call_count <= max_calls:
        return
    raise RuntimeError(
        f"Tool call limit exceeded for {tool_name}: {call_count} calls this run "
        f"(max {max_calls})."
    )


def _estimate_bankroll(snapshot: PortfolioSnapshot) -> Tuple[float, float]:
    cash = snapshot.cash_balance or 0.0
    positions_value = snapshot.investment_value
    if positions_value is None and snapshot.unrealized_pnl is not None:
        positions_value = snapshot.unrealized_pnl
    if positions_value is not None:
        positions_value = float(positions_value)
        return cash + positions_value, abs(positions_value)
    gross_exposure = 0.0
    net_value = 0.0
    for position in snapshot.positions:
        value = position.estimated_value()
        if value is None:
            continue
        net_value += value
        gross_exposure += abs(value)
    bankroll = cash + net_value
    return bankroll, gross_exposure


def _run_portfolio(wallet: str | None = None, required: bool = False) -> str:
    global _portfolio_call_count
    _portfolio_call_count += 1
    _enforce_tool_call_limit(
        "manifold_portfolio",
        _portfolio_call_count,
        max(1, int(DEFAULT_MANIFOLD_PORTFOLIO_MAX_CALLS)),
    )
    try:
        snapshot = fetch_portfolio_snapshot(wallet, api_key=os.environ.get("MANIFOLD_API_KEY"))
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        return f"Unable to fetch Manifold portfolio: {exc}"
    _assert_expected_wallet(snapshot, tool_name="manifold_portfolio")
    return _summarize_portfolio(snapshot)


def _run_portfolio_analytics(max_positions: int = 5) -> str:
    global _portfolio_analytics_call_count
    _portfolio_analytics_call_count += 1
    _enforce_tool_call_limit(
        "portfolio_analytics",
        _portfolio_analytics_call_count,
        max(1, int(DEFAULT_PORTFOLIO_ANALYTICS_MAX_CALLS)),
    )
    try:
        max_positions = int(max_positions)
    except Exception:
        max_positions = 5
    if max_positions < 1:
        max_positions = 5
    try:
        snapshot = fetch_portfolio_snapshot(None, api_key=os.environ.get("MANIFOLD_API_KEY"))
    except Exception as exc:  # noqa: BLE001
        return f"Unable to fetch portfolio analytics: {exc}"
    _assert_expected_wallet(snapshot, tool_name="portfolio_analytics")
    bankroll, gross_exposure = _estimate_bankroll(snapshot)
    cash = snapshot.cash_balance or 0.0
    lines = [
        f"Wallet: {snapshot.wallet}",
        f"Estimated bankroll: ${bankroll:,.2f} (cash ${cash:,.2f})",
        f"Gross exposure: ${gross_exposure:,.2f}",
    ]
    warnings: List[str] = []
    top_positions = snapshot.positions[:max_positions]
    if not top_positions:
        lines.append("No open positions to analyze.")
    else:
        lines.append("Top positions:")
        for position in top_positions:
            value = position.estimated_value()
            if value is None:
                value_note = "value unknown"
            else:
                value_note = f"${value:,.2f}"
                if bankroll > 0 and abs(value) / bankroll > config.RISK_MAX_SINGLE_POSITION_PCT:
                    warnings.append(
                        f"Position '{position.question}' exceeds {config.RISK_MAX_SINGLE_POSITION_PCT:.0%} of bankroll."
                    )
            meta_bits = [f"id={position.market_id}"]
            if position.slug:
                meta_bits.append(f"slug={position.slug}")
            meta = " ".join(meta_bits)
            lines.append(
                f"- {position.question} [{position.outcome}] {position.shares:.2f} shares ({value_note}) {meta}"
            )
    if bankroll > 0 and gross_exposure / bankroll > config.RISK_MAX_GROSS_EXPOSURE_PCT:
        warnings.append(
            f"Gross exposure exceeds {config.RISK_MAX_GROSS_EXPOSURE_PCT:.0%} of bankroll."
        )
    if warnings:
        lines.append("Risk alerts:")
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("Risk alerts: none.")
    return "\n".join(lines)


def _run_risk_gate(
    raw: object | None = None,
    *,
    market_id: str | None = None,
    outcome: str | None = None,
    amount: float | None = None,
    belief_prob: float | None = None,
    market_prob: Optional[float] = None,
    bankroll: Optional[float] = None,
) -> str:
    if raw is not None:
        parsed = _parse_legacy_risk_gate_payload(raw)
        market_id = market_id or _coerce_optional_str(parsed.get("market_id") or parsed.get("marketId"))
        outcome = outcome or _coerce_optional_str(parsed.get("outcome"))
        amount = amount if amount is not None else _coerce_optional_float(parsed.get("amount"))
        belief_prob = (
            belief_prob if belief_prob is not None else _coerce_optional_float(parsed.get("belief_prob"))
        )
        market_prob = (
            market_prob if market_prob is not None else _coerce_optional_float(parsed.get("market_prob"))
        )
        bankroll = bankroll if bankroll is not None else _coerce_optional_float(parsed.get("bankroll"))
    if not market_id or not outcome or amount is None or belief_prob is None:
        return (
            "Risk gate input missing required fields. "
            "Provide market_id, outcome, amount, and belief_prob."
        )
    try:
        amount = float(amount)
    except Exception:
        return "Risk gate input invalid: amount must be a number."
    try:
        belief_prob = float(belief_prob)
    except Exception:
        return "Risk gate input invalid: belief_prob must be a number."
    if market_prob is not None:
        try:
            market_prob = float(market_prob)
        except Exception:
            market_prob = None
    if bankroll is not None:
        try:
            bankroll = float(bankroll)
        except Exception:
            bankroll = None
    snapshot = None
    if bankroll is None:
        snapshot = fetch_portfolio_snapshot(None, api_key=os.environ.get("MANIFOLD_API_KEY"))
        bankroll, gross_exposure = _estimate_bankroll(snapshot)
    else:
        gross_exposure = 0.0
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Risk gate skipped: market not found ({normalized})."
        raise
    if market_prob is None:
        if details.outcome_type.upper() in {"BINARY", "PSEUDO_NUMERIC"}:
            for option in details.answers:
                if option.label.upper() == outcome.strip().upper():
                    market_prob = option.probability
                    break
    lines = [
        f"Market: {details.question}",
        f"Proposed bet: {amount:.2f} on {outcome}",
    ]
    warnings: List[str] = []
    if bankroll and amount / bankroll > config.RISK_MAX_BET_PCT:
        warnings.append(
            f"Bet size exceeds {config.RISK_MAX_BET_PCT:.0%} of bankroll (${bankroll:,.2f})."
        )
    if gross_exposure and bankroll and gross_exposure / bankroll > config.RISK_MAX_GROSS_EXPOSURE_PCT:
        warnings.append(
            f"Gross exposure exceeds {config.RISK_MAX_GROSS_EXPOSURE_PCT:.0%} of bankroll."
        )
    if market_prob is not None:
        edge = belief_prob - market_prob
        suggested_fraction = max(0.0, edge * config.KELLY_MULTIPLIER)
        suggested_amount = bankroll * suggested_fraction if bankroll else None
        lines.append(f"Belief prob: {belief_prob:.2%}; market prob: {market_prob:.2%}; edge: {edge:.2%}.")
        if edge <= 0:
            warnings.append("Edge is non-positive; skip this bet unless new information improves the edge.")
        if suggested_amount is not None:
            lines.append(f"Kelly-style cap: ${suggested_amount:,.2f} (multiplier {config.KELLY_MULTIPLIER:.2f}).")
    else:
        lines.append("Market prob unavailable; Kelly sizing skipped.")
    if warnings:
        lines.append("Risk gate: FAIL")
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("Risk gate: PASS")
    return "\n".join(lines)


def _parse_legacy_risk_gate_payload(raw: object) -> dict[str, object]:
    """Accept legacy ReAct-style tool input such as \"a=1, b='x'\"."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    out: dict[str, object] = {}
    # key='value', key=12.34, key=true formats
    for key, _, single_q, double_q, bare in re.findall(
        r"(\w+)\s*=\s*('([^']*)'|\"([^\"]*)\"|([^,]+))",
        text,
    ):
        token = single_q or double_q or (bare or "").strip()
        lowered = token.lower()
        if lowered == "none":
            value: object = None
        elif lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                if "." in token:
                    value = float(token)
                else:
                    value = int(token)
            except ValueError:
                value = token
        out[key] = value
    return out


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
