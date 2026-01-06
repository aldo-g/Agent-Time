"""LangChain tool definitions that expose Agent-Time capabilities to an LLM."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool

from agent.manifold.constants import RESOLUTION_CUTOFF_MS
from agent.manifold.data import EventSummary, MarketSummary, events_from_dicts, load_open_markets
from agent.manifold.history import fetch_market_history
from agent.manifold.portfolio import PortfolioSnapshot, PortfolioPosition, fetch_portfolio_snapshot
from agent.manifold.trading import MarketDetails, fetch_market_details, lookup_answer_id, place_bet

try:  # pragma: no cover - optional dependency
    from agent.web.web_search import WebSearchUnavailable, search_web
except Exception:  # pragma: no cover - optional dependency
    WebSearchUnavailable = None  # type: ignore[assignment]
    search_web = None  # type: ignore[assignment]


CUTOFF_ISO = datetime.fromtimestamp(RESOLUTION_CUTOFF_MS / 1000, tz=timezone.utc).date().isoformat()
MARKET_CACHE_ENV = "PREDICT_ARENA_MARKET_CACHE"
_MARKET_CACHE: List[EventSummary] | None = None
RISK_MAX_BET_PCT = float(os.environ.get("RISK_MAX_BET_PCT", "0.05"))
RISK_MAX_SINGLE_POSITION_PCT = float(os.environ.get("RISK_MAX_SINGLE_POSITION_PCT", "0.2"))
RISK_MAX_GROSS_EXPOSURE_PCT = float(os.environ.get("RISK_MAX_GROSS_EXPOSURE_PCT", "0.7"))
KELLY_MULTIPLIER = float(os.environ.get("RISK_KELLY_MULTIPLIER", "0.5"))
RSS_URLS_ENV = "NEWS_RSS_URLS"
DEFAULT_RSS_URLS = ["https://feeds.reuters.com/reuters/topNews"]
BLUESKY_API_URL = os.environ.get("BLUESKY_API_URL", "https://public.api.bsky.app")


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


class RssFetchInput(BaseModel):
    """Inputs for the RSS/news fetch tool."""

    query: str | None = Field(
        default=None,
        description="Optional keyword filter applied to titles/snippets.",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum items to return.")
    sources: List[str] | None = Field(
        default=None,
        description="Optional list of RSS URLs to override the default NEWS_RSS_URLS set.",
    )


class BlueskySearchInput(BaseModel):
    """Inputs for the Bluesky search tool."""

    query: str = Field(..., description="Search terms, hashtags, or keywords.")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum posts to return.")


class MarketHistoryInput(BaseModel):
    """Inputs for the market history tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    limit: int = Field(default=200, ge=1, le=500, description="Number of recent bets to analyze.")


class PortfolioAnalyticsInput(BaseModel):
    """Inputs for the portfolio analytics tool."""

    max_positions: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of largest positions to surface.",
    )


class EventTimerInput(BaseModel):
    """Inputs for the event timer tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")


class RiskGateInput(BaseModel):
    """Inputs for the risk gate tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="YES/NO or answer label.")
    amount: float = Field(..., gt=0.0, description="Proposed Mana to wager.")
    belief_prob: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Agent's subjective probability (0-1).",
    )
    market_prob: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Current market probability (0-1).",
    )
    bankroll: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional bankroll override; uses cash + positions if omitted.",
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
    details = fetch_market_details(market_id)
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
    details = fetch_market_details(market_id)
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


def _run_search(query: str, limit: int = 5) -> str:
    if search_web is None:
        raise RuntimeError("Web search tool unavailable. Install duckduckgo_search to enable it.")
    try:
        results = search_web(query, max_results=limit)
    except WebSearchUnavailable as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(str(exc)) from exc
    return _summarize_search_results(results)


def _load_rss_sources(sources: Optional[List[str]]) -> List[str]:
    if sources:
        return [source for source in sources if isinstance(source, str)]
    env_value = os.environ.get(RSS_URLS_ENV, "")
    sources_from_env = [entry.strip() for entry in env_value.split(",") if entry.strip()]
    return sources_from_env or DEFAULT_RSS_URLS


def _parse_rss_feed(xml_bytes: bytes) -> List[dict]:
    root = ET.fromstring(xml_bytes)
    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or "Untitled"
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        description = item.findtext("description") or ""
        items.append(
            {
                "title": title.strip(),
                "link": link.strip(),
                "pub_date": pub_date.strip(),
                "description": description.strip(),
            }
        )
    return items


def _run_rss_fetch(
    query: Optional[str] = None,
    limit: int = 10,
    sources: Optional[List[str]] = None,
) -> str:
    feeds = _load_rss_sources(sources)
    if not feeds:
        raise RuntimeError("No RSS feeds configured. Set NEWS_RSS_URLS or pass sources=[].")
    items: List[dict] = []
    for feed in feeds:
        try:
            with urllib.request.urlopen(feed, timeout=10) as response:
                xml_bytes = response.read()
            items.extend(_parse_rss_feed(xml_bytes))
        except Exception:
            continue
    if query:
        needle = query.lower()
        items = [
            item
            for item in items
            if needle in item.get("title", "").lower() or needle in item.get("description", "").lower()
        ]
    if not items:
        return "No results."
    lines = []
    for idx, item in enumerate(items[:limit], 1):
        lines.append(f"{idx}. {item.get('title')}")
        if item.get("link"):
            lines.append(f"   {item.get('link')}")
        if item.get("pub_date"):
            lines.append(f"   {item.get('pub_date')}")
    return "\n".join(lines)


def _run_bluesky_search(query: str, limit: int = 10) -> str:
    endpoint = f"{BLUESKY_API_URL.rstrip('/')}/xrpc/app.bsky.feed.searchPosts"
    params = {"q": query, "limit": limit}
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.load(response)
    posts = payload.get("posts") if isinstance(payload, dict) else []
    if not isinstance(posts, list) or not posts:
        return "No results."
    lines = []
    for idx, post in enumerate(posts[:limit], 1):
        if not isinstance(post, dict):
            continue
        author = post.get("author", {})
        handle = ""
        if isinstance(author, dict):
            handle = author.get("handle") or ""
        record = post.get("record", {})
        text = ""
        if isinstance(record, dict):
            text = record.get("text") or ""
        line = f"{idx}. {text.strip()}"
        if handle:
            line += f" (@{handle})"
        lines.append(line)
        uri = post.get("uri")
        if uri:
            lines.append(f"   {uri}")
    return "\n".join(lines) if lines else "No results."


def _run_market_history(market_id: str, limit: int = 200) -> str:
    details = fetch_market_details(market_id)
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
    portfolio_analytics_tool = StructuredTool.from_function(
        name="portfolio_analytics",
        func=_run_portfolio_analytics,
        description=(
            "Summarize portfolio exposure, concentration risk, and top positions. "
            "Use before sizing trades to avoid overexposure."
        ),
        args_schema=PortfolioAnalyticsInput,
    )
    event_timer_tool = StructuredTool.from_function(
        name="event_timer",
        func=_run_event_timer,
        description="Report how long until a market closes and flag cutoff violations.",
        args_schema=EventTimerInput,
    )
    risk_gate_tool = StructuredTool.from_function(
        name="risk_gate",
        func=_run_risk_gate,
        description=(
            "Check a proposed bet against bankroll and concentration limits, "
            "and return a Kelly-style sizing hint."
        ),
        args_schema=RiskGateInput,
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
    tools = [
        fetch_tool,
        portfolio_tool,
        portfolio_analytics_tool,
        event_timer_tool,
        risk_gate_tool,
        market_details_tool,
        place_bet_tool,
    ]
    if search_web is not None:
        search_tool = StructuredTool.from_function(
            name="duckduckgo_search",
            func=_run_search,
            description="Run a DuckDuckGo search to gather fresh news or data.",
            args_schema=SearchInput,
        )
        tools.append(search_tool)
    rss_tool = StructuredTool.from_function(
        name="rss_fetch",
        func=_run_rss_fetch,
        description="Pull headlines from configured RSS feeds, optionally filtered by keyword.",
        args_schema=RssFetchInput,
    )
    news_tool = StructuredTool.from_function(
        name="news_api",
        func=_run_rss_fetch,
        description="Alias for rss_fetch to pull curated headlines.",
        args_schema=RssFetchInput,
    )
    history_tool = StructuredTool.from_function(
        name="manifold_market_history",
        func=_run_market_history,
        description="Summarize recent bet activity for a Manifold market.",
        args_schema=MarketHistoryInput,
    )
    bluesky_tool = StructuredTool.from_function(
        name="bluesky_search",
        func=_run_bluesky_search,
        description="Search public Bluesky posts for recent sentiment or catalysts.",
        args_schema=BlueskySearchInput,
    )
    tools.extend([rss_tool, news_tool, history_tool, bluesky_tool])
    return tools


__all__ = [
    "build_agent_tools",
]
