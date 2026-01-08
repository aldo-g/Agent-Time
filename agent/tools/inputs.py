"""Pydantic input models for tool schemas."""

from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


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


class SellPositionInput(BaseModel):
    """Inputs for the Manifold sell tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="Desired outcome (YES/NO or answer label).")
    shares: float = Field(..., gt=0.0, description="Number of shares to sell.")
    answer: str | None = Field(
        default=None,
        description="Optional answer label for multi-choice markets when outcome alone is ambiguous.",
    )


class LimitOrderPreviewInput(BaseModel):
    """Inputs for limit order preview tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="Desired outcome (YES/NO or answer label).")
    amount: float = Field(..., gt=0.0, description="Mana to wager on the outcome.")
    limit_prob: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Optional limit probability (0-1).",
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


class WebScrapeInput(BaseModel):
    """Inputs for web page scraping."""

    url: str = Field(..., description="URL to fetch and summarize.")
    max_chars: int = Field(
        default=2000,
        ge=200,
        le=10000,
        description="Maximum characters of text to return.",
    )


class NotebookEvalInput(BaseModel):
    """Inputs for notebook-style Python evaluation."""

    code: str = Field(..., description="Python code snippet to execute.")


class RssFetchInput(BaseModel):
    """Inputs for the RSS/news fetch tool."""

    query: str | None = Field(
        default=None,
        description="Optional keyword filter applied to titles/snippets.",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum items to return.")
    sources: str | None = Field(
        default=None,
        description="Optional comma-separated RSS URLs to override the default NEWS_RSS_URLS set.",
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
