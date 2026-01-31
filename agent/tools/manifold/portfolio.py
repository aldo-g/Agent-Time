"""Portfolio inspection and risk analysis tools."""

from __future__ import annotations

from typing import List, Optional, Tuple

from agent.manifold.portfolio import PortfolioSnapshot, fetch_portfolio_snapshot
from agent.manifold.trading import MarketDetails, fetch_market_details

from . import config
from .errors import _is_not_found_error
from .limits import _enforce_market_limit
from .summaries import _summarize_portfolio


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
    try:
        snapshot = fetch_portfolio_snapshot(wallet)
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        return f"Unable to fetch Manifold portfolio: {exc}"
    return _summarize_portfolio(snapshot)


def _run_portfolio_analytics(max_positions: int = 5) -> str:
    snapshot = fetch_portfolio_snapshot(None)
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
            lines.append(
                f"- {position.question} [{position.outcome}] {position.shares:.2f} shares ({value_note})"
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
    *,
    market_id: str,
    outcome: str,
    amount: float,
    belief_prob: float,
    market_prob: Optional[float] = None,
    bankroll: Optional[float] = None,
) -> str:
    snapshot = None
    if bankroll is None:
        snapshot = fetch_portfolio_snapshot(None)
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

