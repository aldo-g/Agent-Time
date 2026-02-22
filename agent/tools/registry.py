"""Tool registry builder."""

from __future__ import annotations

from typing import List

from langchain_core.tools import StructuredTool

from agent.tools.inputs import (
    BlueskySearchInput,
    EventTimerInput,
    FetchMarketsInput,
    LimitOrderPreviewInput,
    MarketDetailsInput,
    MarketHistoryInput,
    NotebookEvalInput,
    PlaceBetInput,
    PortfolioAnalyticsInput,
    PortfolioInput,
    RiskGateInput,
    RssFetchInput,
    SearchInput,
    SellPositionInput,
    WebScrapeInput,
)
from agent.tools.manifold import (
    _run_event_timer,
    _run_fetch_markets,
    _run_limit_order_preview,
    _run_market_details,
    _run_market_history,
    _run_place_bet,
    _run_portfolio,
    _run_portfolio_analytics,
    _run_risk_gate,
    _run_sell_position,
)
from agent.tools.web import (
    _run_bluesky_search,
    _run_notebook_eval,
    _run_rss_fetch,
    _run_search,
    _run_web_scrape,
    web_search_available,
)


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
        description="Report how long until a market closes.",
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
    sell_tool = StructuredTool.from_function(
        name="manifold_sell_position",
        func=_run_sell_position,
        description=(
            "Sell an existing Manifold position for the selected outcome. "
            "This tool always attempts to liquidate the full position and ignores partial share inputs."
        ),
        args_schema=SellPositionInput,
    )
    limit_preview_tool = StructuredTool.from_function(
        name="limit_order_preview",
        func=_run_limit_order_preview,
        description="Preview expected fill vs limit probability for a proposed order.",
        args_schema=LimitOrderPreviewInput,
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
    if web_search_available():
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
    web_scrape_tool = StructuredTool.from_function(
        name="web_scrape",
        func=_run_web_scrape,
        description="Fetch and summarize a specific URL for evidence.",
        args_schema=WebScrapeInput,
    )
    notebook_eval_tool = StructuredTool.from_function(
        name="notebook_eval",
        func=_run_notebook_eval,
        description="Execute a small Python snippet for quick analysis; set result=<value> to return a value.",
        args_schema=NotebookEvalInput,
    )
    bluesky_tool = StructuredTool.from_function(
        name="bluesky_search",
        func=_run_bluesky_search,
        description="Search public Bluesky posts for recent sentiment or catalysts.",
        args_schema=BlueskySearchInput,
    )
    tools.extend(
        [
            sell_tool,
            limit_preview_tool,
            rss_tool,
            news_tool,
            history_tool,
            web_scrape_tool,
            notebook_eval_tool,
            bluesky_tool,
        ]
    )
    return tools
