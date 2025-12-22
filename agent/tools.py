"""LangChain tool definitions that expose Agent-Time capabilities to an LLM."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from agent.manifold.constants import RESOLUTION_CUTOFF_MS
from agent.manifold.data import EventSummary, MarketSummary, load_open_markets
from agent.manifold.portfolio import PortfolioSnapshot, PortfolioPosition, fetch_portfolio_snapshot
from agent.manifold.trading import MarketDetails, fetch_market_details, lookup_answer_id, place_bet

try:  # pragma: no cover - optional dependency
    from agent.web.web_search import WebSearchUnavailable, search_web
except Exception:  # pragma: no cover - optional dependency
    WebSearchUnavailable = None  # type: ignore[assignment]
    search_web = None  # type: ignore[assignment]


CUTOFF_ISO = datetime.fromtimestamp(RESOLUTION_CUTOFF_MS / 1000, tz=timezone.utc).date().isoformat()


class FetchMarketsInput(BaseModel):
    """Inputs for the market discovery tool."""

    limit: int = Field(20, ge=1, le=200, description="Number of events to inspect (max 200).")
    offset: int = Field(0, ge=0, description="Pagination offset (multiples of limit).")


class PortfolioInput(BaseModel):
    """Inputs for the portfolio snapshot tool."""

    wallet: str | None = Field(
        default=None,
        description="Deprecated. Manifold accounts are inferred from MANIFOLD_API_KEY.",
    )
    required: bool = Field(
        default=False,
        description="If true, raise an error when the wallet cannot be resolved.",
    )


class MarketDetailsInput(BaseModel):
    """Inputs for the Manifold market lookup tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")


class PlaceBetInput(BaseModel):
    """Inputs for the Manifold bet placement tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="Desired outcome (YES/NO or answer label).")
    amount: float = Field(..., gt=0.0, description="Mana to wager on the outcome.")
    limit_prob: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Optional limit probability (0-1). Leave empty for a market order.",
    )
    answer: str | None = Field(
        default=None,
        description="Optional answer label for multi-choice markets when outcome alone is ambiguous.",
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
    try:
        snapshot = fetch_portfolio_snapshot(wallet)
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        return f"Unable to fetch Manifold portfolio: {exc}"
    return _summarize_portfolio(snapshot)


def _run_market_details(market_id: str) -> str:
    details = fetch_market_details(market_id)
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
    details = fetch_market_details(market_id)
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
    receipt = place_bet(
        market_id=details.market_id,
        outcome=target_label,
        amount=amount,
        limit_prob=limit_prob,
        answer_id=answer_id,
    )
    limit_note = f" with limit {limit_prob * 100:.2f}%" if limit_prob is not None else ""
    summary = (
        f"Wagered {amount:.2f} MANA on '{target_label}' in market {details.market_id}{limit_note}. "
        f"Bet ID: {receipt.bet_id or 'unknown'}."
    )
    return summary


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
        name="manifold_markets",
        func=_run_fetch_markets,
        description=(
            "Inspect live Manifold markets sorted by 24h volume. "
            "Use this to discover actionable opportunities."
        ),
        args_schema=FetchMarketsInput,
    )
    portfolio_tool = StructuredTool.from_function(
        name="manifold_portfolio",
        func=_run_portfolio,
        description=(
            "Retrieve the latest Manifold account snapshot for the authenticated API key. "
            "Call this before sizing trades to respect exposure and risk."
        ),
        args_schema=PortfolioInput,
    )
    market_details_tool = StructuredTool.from_function(
        name="manifold_market_details",
        func=_run_market_details,
        description="Look up metadata, answers, and URLs for a Manifold market before trading it.",
        args_schema=MarketDetailsInput,
    )
    place_bet_tool = StructuredTool.from_function(
        name="manifold_place_bet",
        func=_run_place_bet,
        description=(
            "Submit a Manifold bet using play-money Mana. Provide the market_id, desired outcome or answer "
            "label, optional limit probability, and Mana amount. The tool will fail if you try to wager more "
            "than the available balance."
        ),
        args_schema=PlaceBetInput,
    )
    tools = [fetch_tool, portfolio_tool, market_details_tool, place_bet_tool]
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
