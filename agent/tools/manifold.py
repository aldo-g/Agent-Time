"""Manifold-related tool implementations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import urllib.parse

from agent.manifold.constants import RESOLUTION_CUTOFF_MS
from agent.manifold.data import EventSummary, MarketSummary, events_from_dicts, load_open_markets
from agent.manifold.history import fetch_market_history
from agent.manifold.portfolio import PortfolioSnapshot, PortfolioPosition, fetch_portfolio_snapshot
from agent.manifold.trading import (
    MarketDetails,
    fetch_market_details,
    lookup_answer_id,
    place_bet,
    sell_position,
)


CUTOFF_ISO = datetime.fromtimestamp(RESOLUTION_CUTOFF_MS / 1000, tz=timezone.utc).date().isoformat()
MARKET_CACHE_ENV = "PREDICT_ARENA_MARKET_CACHE"
_MARKET_CACHE: List[EventSummary] | None = None
TRADE_LOG_ENV = "AGENT_TRADE_LOG_PATH"
DEFAULT_TRADE_LOG_PATH = Path(os.environ.get(TRADE_LOG_ENV, "results/trades.jsonl"))
RISK_MAX_BET_PCT = float(os.environ.get("RISK_MAX_BET_PCT", "0.05"))
RISK_MAX_SINGLE_POSITION_PCT = float(os.environ.get("RISK_MAX_SINGLE_POSITION_PCT", "0.2"))
RISK_MAX_GROSS_EXPOSURE_PCT = float(os.environ.get("RISK_MAX_GROSS_EXPOSURE_PCT", "0.7"))
KELLY_MULTIPLIER = float(os.environ.get("RISK_KELLY_MULTIPLIER", "0.5"))


def _summarize_event(event: EventSummary) -> str:
    """Return a single-line synopsis of an event's key markets."""
    markets: List[MarketSummary] = event.markets[:5]
    snippets: List[str] = []
    for market in markets:
        odds = ", ".join(
            f"{outcome.name} {outcome.price * 100:.1f}%"
            for outcome in market.outcomes[:4]
        )
        if len(market.outcomes) > 4:
            odds += ", ..."
        id_note = f"(id: {market.market_id})" if market.market_id else ""
        snippets.append(f"{market.question} {id_note}: {odds}")
    extra = len(event.markets) - len(markets)
    extra_note = f" (+{extra} more markets)" if extra > 0 else ""
    tag_note = f" Tags: {', '.join(event.tags)}." if event.tags else ""
    url_note = f" URL: {event.url}." if event.url else ""
    return f"{event.title}{extra_note}{tag_note}{url_note}\n" + "\n".join(f"  - {line}" for line in snippets)


def _summarize_events(events: Iterable[EventSummary]) -> str:
    descriptions = [_summarize_event(event) for event in events]
    return "\n\n".join(descriptions) if descriptions else "No open markets were returned."


def _normalize_market_identifier(market_id: str) -> str:
    if not market_id:
        return market_id
    cleaned = market_id.strip().strip(".,;:!?)\"]'")
    if "manifold.markets" in cleaned:
        parsed = urllib.parse.urlparse(cleaned)
        path = parsed.path.strip("/")
        if path:
            return path.split("/")[-1].strip(".,;:!?)\"]'")
    if any(char.isspace() for char in cleaned):
        cleaned = cleaned.split()[0]
    return cleaned


def _is_not_found_error(error: Exception) -> bool:
    message = str(error).lower()
    return "404" in message or "not found" in message or "contract not found" in message


def _load_cached_markets() -> List[EventSummary] | None:
    global _MARKET_CACHE
    if _MARKET_CACHE is not None:
        return _MARKET_CACHE
    cache_path = os.environ.get(MARKET_CACHE_ENV)
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    records = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return None
    _MARKET_CACHE = events_from_dicts(records)
    return _MARKET_CACHE


def _summarize_position(position: PortfolioPosition) -> str:
    details = f"{position.shares:.2f} shares"
    mark_price = position.mark_price if position.mark_price is not None else position.avg_price
    if mark_price is not None:
        details += f" @ {mark_price * 100:.2f}%"
    value = position.estimated_value()
    if value is not None:
        details += f" (~${value:,.2f})"
    deltas = []
    if position.avg_price is not None and position.mark_price is not None:
        delta = (position.mark_price - position.avg_price) * 100
        deltas.append(f"Δpx {delta:+.2f}pp")
    if position.pnl is not None:
        deltas.append(f"PnL ${position.pnl:+,.2f}")
    if deltas:
        details += " (" + ", ".join(deltas) + ")"
    return f"- {position.question} [{position.outcome}] {details}"


def _summarize_portfolio(snapshot: PortfolioSnapshot) -> str:
    lines = [f"Wallet: {snapshot.wallet}"]
    ledger_bits = []
    if snapshot.cash_balance is not None:
        ledger_bits.append(f"cash ${snapshot.cash_balance:,.2f}")
    if snapshot.realized_pnl is not None:
        ledger_bits.append(f"realized PnL ${snapshot.realized_pnl:,.2f}")
    if snapshot.unrealized_pnl is not None:
        ledger_bits.append(f"unrealized PnL ${snapshot.unrealized_pnl:,.2f}")
    if ledger_bits:
        lines.append("Ledger: " + ", ".join(ledger_bits))
    else:
        lines.append("Ledger: cash/exposure data unavailable from current endpoint.")
    positions = snapshot.positions[:5]
    if not positions:
        lines.append("No open positions.")
    else:
        lines.append("Top positions:")
        for position in positions:
            lines.append(f"  {_summarize_position(position)}")
        extra = len(snapshot.positions) - len(positions)
        if extra > 0:
            lines.append(f"  ... plus {extra} additional positions.")
    return "\n".join(lines)


def _append_trade_log(entry: Dict[str, object]) -> None:
    log_path = Path(os.environ.get(TRADE_LOG_ENV, str(DEFAULT_TRADE_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")


def _run_fetch_markets(limit: int = 20, offset: int = 0) -> str:
    cached = _load_cached_markets()
    if cached:
        subset = cached[offset : offset + limit] if offset < len(cached) else []
        if subset:
            return _summarize_events(subset)
    events = load_open_markets(limit, offset)
    return _summarize_events(events)


def _run_portfolio(wallet: str | None = None, required: bool = False) -> str:
    try:
        snapshot = fetch_portfolio_snapshot(wallet)
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        return f"Unable to fetch Manifold portfolio: {exc}"
    return _summarize_portfolio(snapshot)


def _run_market_details(market_id: str) -> str:
    normalized = _normalize_market_identifier(market_id)
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Market not found: {normalized}. It may be deleted or merged."
        raise
    lines = [
        f"Market {details.market_id} details:",
        f"Question: {details.question}",
    ]
    if details.url:
        lines.append(f"URL: {details.url}")
    if details.close_time is not None:
        close_dt = datetime.fromtimestamp(details.close_time / 1000, tz=timezone.utc)
        lines.append(f"Closes: {close_dt.isoformat()}")
    lines.append(f"Outcome type: {details.outcome_type}")
    lines.append("Available outcomes:")
    for option in details.answers:
        prob_note = ""
        if option.probability is not None:
            prob_note = f" ({option.probability * 100:.2f}% implied)"
        answer_note = f" [answerId {option.answer_id}]" if option.answer_id else ""
        lines.append(f"- {option.label}{prob_note}{answer_note}")
    lines.append("Use these labels when placing bets.")
    return "\n".join(lines)


def _run_place_bet(
    *,
    market_id: str,
    outcome: str,
    amount: float,
    limit_prob: Optional[float] = None,
    answer: Optional[str] = None,
) -> str:
    if amount <= 0:
        raise RuntimeError("amount must be positive.")
    normalized = _normalize_market_identifier(market_id)
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Bet skipped: market not found ({normalized})."
        raise
    if details.close_time is None:
        raise RuntimeError("Cannot trade markets without a close date.")
    if details.close_time > RESOLUTION_CUTOFF_MS:
        raise RuntimeError(f"This market resolves after {CUTOFF_ISO}; choose an earlier market.")
    snapshot = fetch_portfolio_snapshot(None)
    if snapshot.cash_balance is not None and amount > snapshot.cash_balance + 1e-6:
        raise RuntimeError(
            f"Bet amount {amount:.2f} exceeds available balance {snapshot.cash_balance:.2f}."
        )
    target_label = outcome.strip()
    answer_id = None
    outcome_type = details.outcome_type.upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        normalized = target_label.upper()
        if normalized not in {"YES", "NO"}:
            raise RuntimeError("Binary markets only accept YES or NO outcomes.")
        target_label = normalized
    else:
        lookup_label = answer or target_label
        if not lookup_label:
            raise RuntimeError("Provide answer=<label> when betting on multi-answer markets.")
        answer_id = lookup_answer_id(details, lookup_label)
        if not answer_id and lookup_label.strip().lower().startswith("top outcome"):
            best_option = None
            best_prob = -1.0
            for option in details.answers:
                if option.answer_id is None:
                    continue
                probability = option.probability if option.probability is not None else 0.0
                if probability > best_prob:
                    best_prob = probability
                    best_option = option
            if best_option:
                answer_id = best_option.answer_id
                target_label = best_option.label
        if not answer_id:
            raise RuntimeError(f"Unable to resolve answer '{lookup_label}'. Call manifold_market_details first.")
        target_label = lookup_label
    prob_before = _resolve_market_prob(details, target_label, answer)
    try:
        receipt = place_bet(
            market_id=details.market_id,
            outcome=target_label,
            amount=amount,
            limit_prob=limit_prob,
            answer_id=answer_id,
        )
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Bet skipped: market not found ({details.market_id})."
        raise
    try:
        _append_trade_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": os.environ.get("AGENT_NAME"),
                "provider": os.environ.get("AGENT_PROVIDER"),
                "model": os.environ.get("AGENT_MODEL"),
                "wallet": snapshot.wallet,
                "action": "BUY",
                "market_id": details.market_id,
                "market_slug": details.slug,
                "market": details.question,
                "market_url": details.url,
                "outcome": target_label,
                "amount": receipt.amount,
                "shares": receipt.shares,
                "prob_before": prob_before,
                "prob_after": receipt.probability,
                "limit_prob": limit_prob,
                "bet_id": receipt.bet_id,
                "status": "OPEN",
            }
        )
    except Exception:
        pass
    limit_note = f" with limit {limit_prob * 100:.2f}%" if limit_prob is not None else ""
    summary = (
        f"Wagered {amount:.2f} MANA on '{target_label}' in market {details.market_id}{limit_note}. "
        f"Bet ID: {receipt.bet_id or 'unknown'}."
    )
    return summary


def _run_sell_position(
    *,
    market_id: str,
    outcome: str,
    shares: float,
    answer: Optional[str] = None,
) -> str:
    if shares <= 0:
        raise RuntimeError("shares must be positive.")
    normalized = _normalize_market_identifier(market_id)
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Sell skipped: market not found ({normalized})."
        raise
    target_label = outcome.strip()
    answer_id = None
    outcome_type = details.outcome_type.upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        normalized = target_label.upper()
        if normalized not in {"YES", "NO"}:
            raise RuntimeError("Binary markets only accept YES or NO outcomes.")
        target_label = normalized
    else:
        lookup_label = answer or target_label
        if not lookup_label:
            raise RuntimeError("Provide answer=<label> when selling multi-answer markets.")
        answer_id = lookup_answer_id(details, lookup_label)
        if not answer_id:
            raise RuntimeError(f"Unable to resolve answer '{lookup_label}'. Call manifold_market_details first.")
        target_label = lookup_label
    snapshot = fetch_portfolio_snapshot(None)
    holding = None
    for position in snapshot.positions:
        if position.market_id != details.market_id:
            continue
        if position.outcome.strip().lower() == target_label.strip().lower():
            holding = position
            break
    if holding and abs(holding.shares) + 1e-6 < shares:
        raise RuntimeError(
            f"Sell shares {shares:.2f} exceeds holding {abs(holding.shares):.2f} for {target_label}."
        )
    prob_before = _resolve_market_prob(details, target_label, answer)
    try:
        receipt = sell_position(
            market_id=details.market_id,
            outcome=target_label,
            shares=shares,
            answer_id=answer_id,
        )
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Sell skipped: market not found ({details.market_id})."
        raise
    try:
        _append_trade_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": os.environ.get("AGENT_NAME"),
                "provider": os.environ.get("AGENT_PROVIDER"),
                "model": os.environ.get("AGENT_MODEL"),
                "wallet": snapshot.wallet,
                "action": "SELL",
                "market_id": details.market_id,
                "market_slug": details.slug,
                "market": details.question,
                "market_url": details.url,
                "outcome": target_label,
                "amount": receipt.amount,
                "shares": receipt.shares,
                "prob_before": prob_before,
                "prob_after": receipt.probability,
                "limit_prob": None,
                "bet_id": receipt.bet_id,
                "status": "OPEN",
            }
        )
    except Exception:
        pass
    summary = (
        f"Sold {shares:.2f} shares of '{target_label}' in market {details.market_id}. "
        f"Bet ID: {receipt.bet_id or 'unknown'}."
    )
    return summary


def _resolve_market_prob(details: MarketDetails, outcome: str, answer: Optional[str]) -> Optional[float]:
    outcome_type = details.outcome_type.upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        normalized = outcome.strip().upper()
        for option in details.answers:
            if option.label.upper() == normalized:
                return option.probability
        return None
    lookup_label = answer or outcome
    if not lookup_label:
        return None
    for option in details.answers:
        if option.label.strip().lower() == lookup_label.strip().lower():
            return option.probability
    return None


def _run_limit_order_preview(
    *,
    market_id: str,
    outcome: str,
    amount: float,
    limit_prob: Optional[float] = None,
    answer: Optional[str] = None,
) -> str:
    normalized = _normalize_market_identifier(market_id)
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Preview unavailable: market not found ({normalized})."
        raise
    market_prob = _resolve_market_prob(details, outcome, answer)
    lines = [
        f"Market: {details.question}",
        f"Order: {amount:.2f} on {outcome}",
    ]
    if market_prob is None:
        lines.append("Market probability unavailable; preview is approximate.")
        return "\n".join(lines)
    lines.append(f"Current implied probability: {market_prob:.2%}")
    if limit_prob is None:
        lines.append("Limit: none (market order); expected fill near current probability.")
        return "\n".join(lines)
    slippage = abs(limit_prob - market_prob)
    lines.append(f"Limit probability: {limit_prob:.2%}")
    lines.append(f"Estimated slippage vs market: {slippage:.2%}")
    if limit_prob > market_prob:
        lines.append("Limit above market: likely immediate fill.")
    elif limit_prob < market_prob:
        lines.append("Limit below market: may rest on the book until price moves.")
    else:
        lines.append("Limit at market: likely immediate fill.")
    return "\n".join(lines)


def _estimate_bankroll(snapshot: PortfolioSnapshot) -> Tuple[float, float]:
    cash = snapshot.cash_balance or 0.0
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
                if bankroll > 0 and abs(value) / bankroll > RISK_MAX_SINGLE_POSITION_PCT:
                    warnings.append(
                        f"Position '{position.question}' exceeds {RISK_MAX_SINGLE_POSITION_PCT:.0%} of bankroll."
                    )
            lines.append(
                f"- {position.question} [{position.outcome}] {position.shares:.2f} shares ({value_note})"
            )
    if bankroll > 0 and gross_exposure / bankroll > RISK_MAX_GROSS_EXPOSURE_PCT:
        warnings.append(
            f"Gross exposure exceeds {RISK_MAX_GROSS_EXPOSURE_PCT:.0%} of bankroll."
        )
    if warnings:
        lines.append("Risk alerts:")
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("Risk alerts: none.")
    return "\n".join(lines)


def _run_event_timer(market_id: str) -> str:
    normalized = _normalize_market_identifier(market_id)
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Timer unavailable: market not found ({normalized})."
        raise
    if details.close_time is None:
        return f"Market {details.market_id} has no close time on record."
    close_dt = datetime.fromtimestamp(details.close_time / 1000, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = close_dt - now
    hours = delta.total_seconds() / 3600
    status = "OPEN" if delta.total_seconds() > 0 else "CLOSED"
    lines = [
        f"Market {details.market_id} closes at {close_dt.isoformat()} ({status}).",
        f"Time until close: {delta.days}d {abs(delta.seconds) // 3600}h.",
    ]
    if details.close_time > RESOLUTION_CUTOFF_MS:
        lines.append(f"Warning: closes after cutoff {CUTOFF_ISO}.")
    if hours < 24 and delta.total_seconds() > 0:
        lines.append("Note: closes within 24 hours; liquidity may be thin.")
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
    normalized = _normalize_market_identifier(market_id)
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
    if bankroll and amount / bankroll > RISK_MAX_BET_PCT:
        warnings.append(
            f"Bet size exceeds {RISK_MAX_BET_PCT:.0%} of bankroll (${bankroll:,.2f})."
        )
    if gross_exposure and bankroll and gross_exposure / bankroll > RISK_MAX_GROSS_EXPOSURE_PCT:
        warnings.append(
            f"Gross exposure exceeds {RISK_MAX_GROSS_EXPOSURE_PCT:.0%} of bankroll."
        )
    if market_prob is not None:
        edge = belief_prob - market_prob
        suggested_fraction = max(0.0, edge * KELLY_MULTIPLIER)
        suggested_amount = bankroll * suggested_fraction if bankroll else None
        lines.append(f"Belief prob: {belief_prob:.2%}; market prob: {market_prob:.2%}; edge: {edge:.2%}.")
        if suggested_amount is not None:
            lines.append(f"Kelly-style cap: ${suggested_amount:,.2f} (multiplier {KELLY_MULTIPLIER:.2f}).")
    else:
        lines.append("Market prob unavailable; Kelly sizing skipped.")
    if warnings:
        lines.append("Risk gate: FAIL")
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("Risk gate: PASS")
    return "\n".join(lines)


def _run_market_history(market_id: str, limit: int = 200) -> str:
    normalized = _normalize_market_identifier(market_id)
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"History unavailable: market not found ({normalized})."
        raise
    bets = fetch_market_history(details.market_id, limit=limit)
    if not bets:
        return f"No recent bets found for market {details.market_id}."
    latest = bets[0]
    latest_time = datetime.fromtimestamp(latest.timestamp / 1000, tz=timezone.utc).isoformat()
    total_volume = sum(abs(bet.amount) for bet in bets)
    lines = [
        f"Market: {details.question}",
        f"Recent bets analyzed: {len(bets)}",
        f"Latest bet: {latest_time} ({latest.outcome}, amount {latest.amount:.2f})",
        f"Total volume (sample): ${total_volume:,.2f}",
    ]
    last_probs = [bet.prob_after for bet in bets[:5] if bet.prob_after is not None]
    if last_probs:
        lines.append("Recent probAfter:")
        for bet, prob in zip(bets[:5], last_probs):
            bet_time = datetime.fromtimestamp(bet.timestamp / 1000, tz=timezone.utc).isoformat()
            lines.append(f"- {bet_time}: {prob:.2%}")
    return "\n".join(lines)


__all__ = [
    "_run_event_timer",
    "_run_fetch_markets",
    "_run_limit_order_preview",
    "_run_market_details",
    "_run_market_history",
    "_run_place_bet",
    "_run_portfolio",
    "_run_portfolio_analytics",
    "_run_risk_gate",
    "_run_sell_position",
]
