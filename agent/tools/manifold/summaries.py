"""Formatting helpers for Manifold responses."""

from __future__ import annotations

from typing import Iterable, List

from agent.manifold.data import EventSummary
from agent.manifold.portfolio import PortfolioPosition, PortfolioSnapshot


def _summarize_event(event: EventSummary) -> str:
    """Return a compact synopsis of a single event."""
    first_market = event.markets[0] if event.markets else None
    parts: List[str] = [event.title or (event.event_id or "Untitled event")]
    if first_market:
        top_outcome = first_market.outcomes[0] if first_market.outcomes else None
        odds_note = f"{top_outcome.name} {top_outcome.price * 100:.1f}%" if top_outcome else "odds n/a"
        parts.append(f"{first_market.question} ({odds_note})")
        parts.append(f"id={first_market.market_id}")
    if event.url:
        parts.append(event.url)
    extra_markets = len(event.markets) - 1 if event.markets else 0
    if extra_markets > 0:
        parts.append(f"+{extra_markets} more markets")
    return " | ".join(parts)


def _summarize_events(events: Iterable[EventSummary]) -> str:
    descriptions = [_summarize_event(event) for event in events]
    if not descriptions:
        return "No open markets were returned."
    shown = descriptions[:5]
    lines = shown
    extra = len(descriptions) - len(shown)
    if extra > 0:
        lines.append(f"... {extra} more events.")
    return "\n".join(lines)


def _summarize_position(position: PortfolioPosition) -> str:
    mark_price = position.mark_price if position.mark_price is not None else position.avg_price
    value = position.estimated_value()
    bits = []
    if mark_price is not None:
        bits.append(f"{mark_price * 100:.1f}%")
    if value is not None:
        bits.append(f"${value:,.0f}")
    meta_bits = [f"id={position.market_id}"]
    if position.slug:
        meta_bits.append(f"slug={position.slug}")
    meta = " ".join(meta_bits)
    return f"{position.question} [{position.outcome}] ({', '.join(bits)}) {meta}"


def _summarize_portfolio(snapshot: PortfolioSnapshot) -> str:
    cash = snapshot.cash_balance if snapshot.cash_balance is not None else 0.0
    invested = snapshot.investment_value
    if invested is None and snapshot.unrealized_pnl is not None:
        invested = snapshot.unrealized_pnl
    invested = invested or 0.0
    lines = [
        f"Wallet: {snapshot.wallet}",
        f"Cash ${cash:,.0f} | Invested ${invested:,.0f} | Positions {len(snapshot.positions)}",
    ]
    positions = snapshot.positions[:2]
    if positions:
        lines.append("Top positions:")
        for position in positions:
            lines.append(f"- {_summarize_position(position)}")
        extra = len(snapshot.positions) - len(positions)
        if extra > 0:
            lines.append(f"... plus {extra} more.")
    return "\n".join(lines)
