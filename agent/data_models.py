"""Shared Polymarket data models and helpers used by the LangChain agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, List

from fetch_gamma import fetch_events

BASE_MARKET_URL = "https://polymarket.com/market"


@dataclass
class OutcomeQuote:
    name: str
    price: float


@dataclass
class MarketSummary:
    event_id: str
    event_title: str
    market_id: str
    question: str
    url: str | None
    outcomes: List[OutcomeQuote]
    tags: List[str]


@dataclass
class EventSummary:
    event_id: str
    title: str
    url: str | None
    tags: List[str] = field(default_factory=list)
    markets: List[MarketSummary] = field(default_factory=list)


def _parse_sequence(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _parse_tags(raw_tags: object) -> List[str]:
    tags = []
    for tag in _parse_sequence(raw_tags):
        if isinstance(tag, dict):
            label = tag.get("label")
        else:
            label = tag
        if label:
            tags.append(str(label))
    return tags


def load_open_markets(limit: int, offset: int) -> List[EventSummary]:
    """Fetch open Polymarket events and normalize markets grouped by event."""
    payload = fetch_events(limit, offset)
    if isinstance(payload, dict):
        events: Iterable[dict] = payload.get("events") or payload.get("data") or []
    elif isinstance(payload, list):
        events = payload
    else:
        events = []

    summaries: List[EventSummary] = []
    for event in events:
        event_id = str(event.get("id", ""))
        event_title = event.get("title") or event.get("ticker") or "Untitled event"
        slug = event.get("slug")
        url = f"{BASE_MARKET_URL}/{slug}" if slug else None
        tags = _parse_tags(event.get("tags"))
        event_summary = EventSummary(event_id=event_id, title=event_title, url=url, tags=tags)

        for market in event.get("markets") or []:
            names = _parse_sequence(market.get("outcomes"))
            prices = _parse_sequence(market.get("outcomePrices"))
            if not names or not prices:
                continue
            outcomes = []
            for name, price in zip(names, prices):
                try:
                    price_value = float(price)
                except (TypeError, ValueError):
                    continue
                outcomes.append(OutcomeQuote(name=str(name), price=price_value))
            if not outcomes:
                continue
            market_id = str(market.get("id") or market.get("conditionId") or "unknown")
            question = market.get("question") or event_title
            event_summary.markets.append(
                MarketSummary(
                    event_id=event_id,
                    event_title=event_title,
                    market_id=market_id,
                    question=question,
                    url=url,
                    outcomes=outcomes,
                    tags=tags,
                )
            )

        if event_summary.markets:
            summaries.append(event_summary)

    return summaries


__all__ = [
    "EventSummary",
    "MarketSummary",
    "OutcomeQuote",
    "load_open_markets",
]
