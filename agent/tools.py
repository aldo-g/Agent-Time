"""LangChain tool definitions that expose Agent-Time capabilities to an LLM."""

from __future__ import annotations

import os
from typing import Iterable, List, Literal, Optional, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from agent.polymarket.data import EventSummary, MarketSummary, load_open_markets
from agent.polymarket.portfolio import PortfolioSnapshot, PortfolioPosition, fetch_portfolio_snapshot
from agent.polymarket.trading import fetch_market_tokens, find_token_id, place_limit_order

try:  # pragma: no cover - optional dependency
    from agent.web.web_search import WebSearchUnavailable, search_web
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


class MarketDetailsInput(BaseModel):
    """Inputs for the Polymarket market lookup tool."""

    market_id: str = Field(..., description="Polymarket condition/market ID (e.g., 0xabc...).")


class PlaceOrderInput(BaseModel):
    """Inputs for the order placement tool."""

    market_id: str = Field(..., description="Polymarket market/condition ID.")
    outcome: str = Field(..., description="Outcome label exactly as shown in the market (e.g., 'Yes', 'No').")
    token_id: str | None = Field(
        default=None,
        description="Optional Polymarket token ID. If omitted the tool will attempt to resolve it automatically.",
    )
    side: Literal["BUY", "SELL"] = Field(..., description="BUY to go long, SELL to close/short.")
    price: float = Field(..., gt=0.0, lt=1.0, description="Limit price between 0 and 1.")
    shares: float | None = Field(
        default=None,
        gt=0.0,
        description="Number of shares to trade. For BUY orders you can omit this and use stake_usd instead.",
    )
    stake_usd: float | None = Field(
        default=None,
        gt=0.0,
        description="Dollar spend for BUY orders (shares computed as stake_usd / price).",
    )
    order_type: Literal["GTC", "FAK", "FOK", "GTD"] = Field(
        default="GTC",
        description="Order time-in-force. Default GTC.",
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


def _run_market_details(market_id: str) -> str:
    tokens = fetch_market_tokens(market_id)
    lines = [f"Market {market_id} token map:"]
    title = tokens.raw.get("question") or tokens.raw.get("title")
    if isinstance(title, str):
        lines.append(f"Question: {title}")
    for outcome, token_id in tokens.tokens.items():
        lines.append(f"- {outcome}: {token_id}")
    lines.append("Use these token IDs when submitting orders.")
    return "\n".join(lines)


def _run_place_order(
    *,
    market_id: str,
    outcome: str,
    token_id: Optional[str] = None,
    side: Literal["BUY", "SELL"],
    price: float,
    shares: Optional[float] = None,
    stake_usd: Optional[float] = None,
    order_type: Literal["GTC", "FAK", "FOK", "GTD"] = "GTC",
) -> str:
    wallet = os.environ.get("POLYMARKET_WALLET_ADDRESS")
    if not wallet:
        raise RuntimeError("Set POLYMARKET_WALLET_ADDRESS before placing orders.")
    normalized_market = market_id.strip()
    if not normalized_market.startswith("0x"):
        raise RuntimeError(
            "market_id must be the Polymarket condition ID (starts with 0x). "
            "Call polymarket_market_details first to translate from a slug URL to a market_id."
        )
    normalized_side = side.upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise RuntimeError("side must be BUY or SELL.")
    if price <= 0 or price >= 1:
        raise RuntimeError("price must be between 0 and 1.")
    snapshot = fetch_portfolio_snapshot(wallet)
    shares, notional = _determine_order_size(
        side=normalized_side,
        price=price,
        shares=shares,
        stake_usd=stake_usd,
        snapshot=snapshot,
        market_id=market_id,
        outcome=outcome,
    )
    resolved_token_id = token_id
    if not resolved_token_id:
        market_tokens = fetch_market_tokens(market_id)
        resolved_token_id = find_token_id(market_tokens.raw, outcome)
        if not resolved_token_id:
            # Try case-insensitive lookup from the prepared mapping
            desired = outcome.strip().lower()
            for label, tok in market_tokens.tokens.items():
                if label.strip().lower() == desired:
                    resolved_token_id = tok
                    break
    if not resolved_token_id:
        raise RuntimeError(f"Unable to resolve token ID for outcome '{outcome}'.")
    receipt = place_limit_order(
        market_id=market_id,
        token_id=resolved_token_id,
        side=normalized_side,
        price=price,
        shares=shares,
        order_type=order_type,
    )
    summary = (
        f"Submitted {normalized_side} order for {shares:.2f} shares of '{outcome}' "
        f"@ {price * 100:.2f}% (token {resolved_token_id}). "
        f"Notional ${receipt.usd_notional:,.2f}. "
        f"Order ID: {receipt.order_id or 'unknown'}."
    )
    return summary


def _determine_order_size(
    *,
    side: str,
    price: float,
    shares: Optional[float],
    stake_usd: Optional[float],
    snapshot: PortfolioSnapshot,
    market_id: str,
    outcome: str,
) -> tuple[float, float]:
    """Return (shares, usd_notional) with guardrails for bankroll/sells."""
    computed_shares: Optional[float] = shares
    spend = stake_usd
    if side == "BUY":
        if computed_shares is None and spend is None:
            raise RuntimeError("Provide shares or stake_usd for BUY orders.")
        if computed_shares is None:
            computed_shares = spend / price  # type: ignore[operator]
        spend = computed_shares * price
        cash_available = snapshot.cash_balance or 0.0
        if cash_available <= 0:
            raise RuntimeError("No cash available to buy.")
        max_fraction = float(os.environ.get("AGENT_MAX_ORDER_FRACTION", "0.5"))
        max_allowed = cash_available * max_fraction
        if spend > max_allowed:
            raise RuntimeError(
                f"Order notional ${spend:,.2f} exceeds allowed spend ${max_allowed:,.2f} "
                f"(max fraction {max_fraction:.0%})."
            )
    else:  # SELL
        if computed_shares is None and spend is not None:
            computed_shares = spend
        if computed_shares is None:
            raise RuntimeError("Provide shares when selling.")
        owned = 0.0
        target = outcome.strip().lower()
        for position in snapshot.positions:
            if position.market_id == market_id and position.outcome.strip().lower() == target:
                owned += position.shares
        if owned <= 0:
            raise RuntimeError("No matching position to sell.")
        if computed_shares - owned > 1e-6:
            raise RuntimeError(
                f"Attempting to sell {computed_shares:.2f} shares but only {owned:.2f} are available."
            )
        spend = computed_shares * price
    return computed_shares, spend
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
    market_details_tool = StructuredTool.from_function(
        name="polymarket_market_details",
        func=_run_market_details,
        description="Look up token IDs and metadata for a Polymarket market before trading it.",
        args_schema=MarketDetailsInput,
    )
    place_order_tool = StructuredTool.from_function(
        name="polymarket_place_order",
        func=_run_place_order,
        description=(
            "Submit a signed Polymarket limit order. Provide the market_id, outcome name, price, "
            "and either shares or stake_usd. This tool enforces bankroll limits and will fail if "
            "you try to exceed available cash or sell more shares than owned."
        ),
        args_schema=PlaceOrderInput,
    )
    tools = [fetch_tool, portfolio_tool, market_details_tool, place_order_tool]
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
