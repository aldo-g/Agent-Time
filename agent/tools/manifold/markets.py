"""Market discovery and metadata tooling."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from agent.manifold.data import EventSummary, load_open_markets
from agent.manifold.history import fetch_market_history
from agent.manifold.trading import MarketDetails, fetch_market_details

from .cache import _load_cached_markets
from .config import CUTOFF_ISO, RESOLUTION_CUTOFF_MS
from .errors import _is_not_found_error
from .limits import _enforce_market_limit
from .summaries import _summarize_events


def _run_fetch_markets(limit: int = 10, offset: int = 0) -> str:
    try:
        limit = int(limit)
    except Exception:
        limit = 25
    try:
        offset = int(offset)
    except Exception:
        offset = 0
    cached = _load_cached_markets()
    if cached:
        subset = cached[offset : offset + limit] if offset < len(cached) else []
        if subset:
            return _summarize_events(subset)
    events = load_open_markets(limit, offset)
    return _summarize_events(events)


def _run_market_details(market_id: str) -> str:
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
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


def _run_event_timer(market_id: str) -> str:
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
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


def _run_market_history(market_id: str, limit: int = 200) -> str:
    try:
        limit = int(limit)
    except Exception:
        limit = 200
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
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
