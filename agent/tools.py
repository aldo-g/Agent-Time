"""LangChain tool definitions that expose Agent-Time capabilities to an LLM."""

from __future__ import annotations

import os
from typing import Iterable, List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from agent.data_models import EventSummary, MarketSummary, load_open_markets
from polymarket_portfolio import PortfolioSnapshot, PortfolioPosition, fetch_portfolio_snapshot

try:  # pragma: no cover - optional dependency
    from web_search import WebSearchUnavailable, search_web
except Exception:  # pragma: no cover - optional dependency
    WebSearchUnavailable = None  # type: ignore[assignment]
    search_web = None  # type: ignore[assignment]


class FetchMarketsInput(BaseModel):
    """Inputs for the market discovery tool."""

    limit: int = Field(20, ge=1, le=200, description="Number of events to inspect (max 200).")
    offset: int = Field(0, ge=0, description="Pagination offset (multiples of limit).")


class PortfolioInput(BaseModel):
    """Inputs for the portfolio snapshot tool."""

    wallet: str | None = Field(
        default=None,
        description="Polymarket wallet address. Defaults to POLYMARKET_WALLET_ADDRESS env variable.",
    )
    required: bool = Field(
        default=False,
        description="If true, raise an error when the wallet cannot be resolved.",
    )


class SearchInput(BaseModel):
    """Inputs for the DuckDuckGo search tool."""

    query: str = Field(..., description="Keywords to search for.")
    limit: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of results to return (1-25).",
    )


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
        snippets.append(f"{market.question}: {odds}")
    extra = len(event.markets) - len(markets)
    extra_note = f" (+{extra} more markets)" if extra > 0 else ""
    tag_note = f" Tags: {', '.join(event.tags)}." if event.tags else ""
    url_note = f" URL: {event.url}." if event.url else ""
    return f"{event.title}{extra_note}{tag_note}{url_note}\n" + "\n".join(f"  - {line}" for line in snippets)


def _summarize_events(events: Iterable[EventSummary]) -> str:
    descriptions = [_summarize_event(event) for event in events]
    return "\n\n".join(descriptions) if descriptions else "No open markets were returned."


def _summarize_position(position: PortfolioPosition) -> str:
    details = f"{position.shares:.2f} shares"
    mark_price = position.mark_price if position.mark_price is not None else position.avg_price
    if mark_price is not None:
        details += f" @ {mark_price * 100:.2f}%"
    value = position.estimated_value()
    if value is not None:
        details += f" (~${value:,.2f})"
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


def _summarize_search_results(results: List[object]) -> str:
    if not results:
        return "No results."
    lines = []
    for idx, result in enumerate(results, 1):
        title = getattr(result, "title", "Untitled result")
        url = getattr(result, "url", "")
        snippet = getattr(result, "snippet", "")
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _run_fetch_markets(limit: int = 20, offset: int = 0) -> str:
    events = load_open_markets(limit, offset)
    return _summarize_events(events)


def _run_portfolio(wallet: str | None = None, required: bool = False) -> str:
    target_wallet = wallet or os.environ.get("POLYMARKET_WALLET_ADDRESS")
    if not target_wallet:
        if required:
            raise RuntimeError("Portfolio inspection requested but no wallet was provided.")
        return "No wallet address available. Provide wallet=<address> or set POLYMARKET_WALLET_ADDRESS."
    snapshot = fetch_portfolio_snapshot(target_wallet)
    return _summarize_portfolio(snapshot)


def _run_search(query: str, limit: int = 5) -> str:
    if search_web is None:
        raise RuntimeError("Web search tool unavailable. Install duckduckgo_search to enable it.")
    try:
        results = search_web(query, max_results=limit)
    except WebSearchUnavailable as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(str(exc)) from exc
    return _summarize_search_results(results)


def build_agent_tools() -> List[StructuredTool]:
    """Return the list of LangChain tools exposed to the trading agent."""
    fetch_tool = StructuredTool.from_function(
        name="polymarket_events",
        func=_run_fetch_markets,
        description=(
            "Inspect live Polymarket events and markets sorted by 24h volume. "
            "Use this to discover actionable opportunities."
        ),
        args_schema=FetchMarketsInput,
    )
    portfolio_tool = StructuredTool.from_function(
        name="portfolio_snapshot",
        func=_run_portfolio,
        description=(
            "Retrieve the latest Polymarket portfolio for the configured wallet. "
            "Call this before sizing trades to respect exposure and risk."
        ),
        args_schema=PortfolioInput,
    )
    tools = [fetch_tool, portfolio_tool]
    if search_web is not None:
        search_tool = StructuredTool.from_function(
            name="duckduckgo_search",
            func=_run_search,
            description="Run a DuckDuckGo search to gather fresh news or data.",
            args_schema=SearchInput,
        )
        tools.append(search_tool)
    return tools


__all__ = [
    "build_agent_tools",
]
