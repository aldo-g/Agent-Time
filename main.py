#!/usr/bin/env python3
"""Entry point for the Agent-Time trading agent."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, List

import env_loader  # noqa: F401
from fetch_gamma import fetch_events
from polymarket_portfolio import PortfolioSnapshot, fetch_portfolio_snapshot

try:
    from openai_client import summarize_markets
except Exception:  # pragma: no cover - optional dependency
    summarize_markets = None  # type: ignore[var-annotated]
try:
    from web_search import WebSearchUnavailable, search_web
except Exception:  # pragma: no cover - optional dependency
    WebSearchUnavailable = None  # type: ignore[assignment]
    search_web = None  # type: ignore[assignment]

BASE_MARKET_URL = "https://polymarket.com/market"
MAX_MARKETS_PER_EVENT = 5
MAX_OUTCOMES_PER_MARKET = 4
MAX_POSITIONS_PER_SUMMARY = 5
SEARCH_PREFIX = "SEARCH:"


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
    """Fetch open events and normalize markets grouped by event."""
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


def print_events(events: List[EventSummary]) -> None:
    if not events:
        print("No markets available.")
        return
    for idx, event in enumerate(events, 1):
        print(f"{idx}. {event.title} (event {event.event_id})")
        if event.url:
            print(f"   URL: {event.url}")
        markets_to_show = event.markets[:MAX_MARKETS_PER_EVENT]
        for market in markets_to_show:
            odds = ", ".join(
                f"{outcome.name} {outcome.price * 100:.1f}%"
                for outcome in market.outcomes[:MAX_OUTCOMES_PER_MARKET]
            )
            if len(market.outcomes) > MAX_OUTCOMES_PER_MARKET:
                odds += ", ..."
            print(f"   - {market.question}: {odds}")
        extra = len(event.markets) - len(markets_to_show)
        if extra > 0:
            print(f"   ... {extra} more markets")
        print()


def _format_money(value: float | None) -> str:
    if value is None:
        return "?"
    return f"${value:,.2f}"


def load_portfolio_snapshot(*, required: bool = False) -> PortfolioSnapshot | None:
    wallet = os.environ.get("POLYMARKET_WALLET_ADDRESS")
    if not wallet:
        if required:
            raise RuntimeError("POLYMARKET_WALLET_ADDRESS is required to inspect portfolio.")
        return None
    try:
        return fetch_portfolio_snapshot(wallet)
    except Exception as exc:  # noqa: BLE001
        if required:
            raise RuntimeError(f"Unable to fetch Polymarket portfolio for {wallet}: {exc}") from exc
        print(f"Unable to fetch Polymarket portfolio for {wallet}: {exc}")
        return None


def print_portfolio(snapshot: PortfolioSnapshot) -> None:
    print(f"Portfolio for wallet {snapshot.wallet}:")
    ledger_bits = []
    if snapshot.cash_balance is not None:
        ledger_bits.append(f"cash {_format_money(snapshot.cash_balance)}")
    if snapshot.realized_pnl is not None:
        ledger_bits.append(f"realized PnL {_format_money(snapshot.realized_pnl)}")
    if snapshot.unrealized_pnl is not None:
        ledger_bits.append(f"unrealized PnL {_format_money(snapshot.unrealized_pnl)}")
    if ledger_bits:
        print("   " + ", ".join(ledger_bits))
    positions_to_show = snapshot.positions[:MAX_POSITIONS_PER_SUMMARY]
    if not positions_to_show:
        print("   No open positions.")
        print()
        return
    for position in positions_to_show:
        mark_price = position.mark_price if position.mark_price is not None else position.avg_price
        value = position.estimated_value()
        detail = f"{position.shares:.2f} shares"
        if mark_price is not None:
            detail += f" @ {mark_price * 100:.2f}%"
        if position.avg_price is not None:
            detail += f" (avg {position.avg_price * 100:.2f}%)"
        value_text = f" (~{_format_money(value)})" if value is not None else ""
        print(f"   - {position.question} [{position.outcome}] {detail}{value_text}")
    extra_positions = len(snapshot.positions) - len(positions_to_show)
    if extra_positions > 0:
        print(f"   ... {extra_positions} more positions")
    print()


def _run_web_search(query: str, limit: int) -> None:
    query = (query or "").strip()
    if not query:
        print("Provide a non-empty search query.")
        return
    if search_web is None:
        print("Web search unavailable. Install duckduckgo_search to enable this feature.")
        return
    try:
        results = search_web(query, max_results=limit)
    except WebSearchUnavailable as exc:
        print(exc)
        return
    count = len(results)
    heading = f"Top {count} search result{'s' if count != 1 else ''} for \"{query}\":"
    print(heading)
    if count == 0:
        print("No results.")
        return
    for idx, result in enumerate(results, 1):
        title = getattr(result, "title", None) or "Untitled result"
        url = getattr(result, "url", "") or ""
        snippet = getattr(result, "snippet", "") or ""
        print(f"{idx}. {title}")
        if url:
            print(f"   {url}")
        if snippet:
            print(f"   {snippet}")
        print()


def _extract_search_queries(text: str) -> List[str]:
    queries: List[str] = []
    if not text:
        return queries
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if not raw.upper().startswith(SEARCH_PREFIX):
            continue
        _, _, remainder = raw.partition(":")
        query = remainder.strip()
        if query and query not in queries:
            queries.append(query)
    return queries


def _run_requested_searches(queries: List[str], limit: int) -> None:
    if not queries:
        return
    if search_web is None:
        print(
            "Search queries requested by the agent, but web search support is unavailable. "
            "Install duckduckgo_search to enable automatic research."
        )
        return
    print("Executing requested research queries...")
    for query in queries:
        print(f"==> SEARCH: {query}")
        _run_web_search(query, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-Time market discovery agent")
    parser.add_argument("--limit", type=int, default=20, help="Number of events to inspect (default: 20)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset in multiples of limit")
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Ask OpenAI for a quick recommendation on which markets to investigate",
    )
    parser.add_argument("--search", metavar="QUERY", help="Run a quick DuckDuckGo search and exit")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=5,
        help="Maximum number of search results to show per query (default: 5)",
    )
    args = parser.parse_args()
    if args.search:
        _run_web_search(args.search, args.search_limit)
        return
    portfolio_required = bool(os.environ.get("POLYMARKET_WALLET_ADDRESS"))
    try:
        portfolio = load_portfolio_snapshot(required=portfolio_required)
    except RuntimeError as exc:
        print(exc)
        return
    events = load_open_markets(args.limit, args.offset)
    if portfolio:
        print_portfolio(portfolio)
    elif portfolio_required:
        print("No portfolio data available; aborting before recommendations.")
        return
    print_events(events)

    if args.recommend:
        if summarize_markets is None:
            print("OpenAI client not available. Install openai and set OPENAI_API_KEY.")
            return
        flat_markets = [market for event in events for market in event.markets]
        try:
            summary = summarize_markets(flat_markets, portfolio=portfolio)
        except Exception as exc:  # noqa: BLE001
            print(f"OpenAI recommendation failed: {exc}")
            return
        queries = _extract_search_queries(summary)
        if queries:
            print("Agent requested research before finalizing recommendation.")
            _run_requested_searches(queries, args.search_limit)
        print("OpenAI Recommendation:")
        print(summary)


if __name__ == "__main__":
    main()
