"""Pydantic input models for tool schemas."""

from __future__ import annotations

from typing import List

try:
    # Prefer LangChain's compatibility shim when available (Pydantic v1-style API).
    from langchain_core.pydantic_v1 import BaseModel, Field, validator, conint, confloat
except ImportError:
    try:
        from pydantic.v1 import BaseModel, Field, validator, conint, confloat  # type: ignore
    except ImportError:
        from pydantic import BaseModel, Field, validator, conint, confloat


Int1To200 = conint(ge=1, le=200)
NonNegativeInt = conint(ge=0)
SearchLimit = conint(ge=1, le=25)
MediumListLimit = conint(ge=1, le=50)
HistoryLimit = conint(ge=1, le=500)
MaxPositionsLimit = conint(ge=1, le=50)
WebScrapeChars = conint(ge=200, le=10000)
PositiveAmount = confloat(gt=0.0)
ProbBounded = confloat(gt=0.0, lt=1.0)


class FetchMarketsInput(BaseModel):
    """Inputs for the market discovery tool."""

    limit: Int1To200 = Field(20, description="Number of events to inspect (max 200).")
    offset: NonNegativeInt = Field(0, description="Pagination offset (multiples of limit).")

    @validator("limit", pre=True)
    def _coerce_limit(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty."""
        if value is None or value == "":
            return 20
        try:
            return int(value)
        except (TypeError, ValueError):
            return 20

    @validator("offset", pre=True)
    def _coerce_offset(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty."""
        if value is None or value == "":
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


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
    amount: PositiveAmount = Field(..., description="Mana to wager on the outcome.")
    limit_prob: ProbBounded | None = Field(
        default=None, description="Optional limit probability (0-1). Leave empty for a market order."
    )
    answer: str | None = Field(
        default=None,
        description="Optional answer label for multi-choice markets when outcome alone is ambiguous.",
    )


class SellPositionInput(BaseModel):
    """Inputs for the Manifold sell tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="Desired outcome (YES/NO or answer label).")
    shares: PositiveAmount = Field(..., description="Number of shares to sell.")
    answer: str | None = Field(
        default=None,
        description="Optional answer label for multi-choice markets when outcome alone is ambiguous.",
    )


class LimitOrderPreviewInput(BaseModel):
    """Inputs for limit order preview tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="Desired outcome (YES/NO or answer label).")
    amount: PositiveAmount = Field(..., description="Mana to wager on the outcome.")
    limit_prob: ProbBounded | None = Field(default=None, description="Optional limit probability (0-1).")
    answer: str | None = Field(
        default=None,
        description="Optional answer label for multi-choice markets when outcome alone is ambiguous.",
    )


class SearchInput(BaseModel):
    """Inputs for the DuckDuckGo search tool."""

    query: str = Field(..., description="Keywords to search for.")
    limit: SearchLimit = Field(default=5, description="Maximum number of results to return (1-25).")

    @validator("limit", pre=True)
    def _coerce_limit(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty."""
        if value is None or value == "":
            return 5
        try:
            return int(value)
        except (TypeError, ValueError):
            return 5


class WebScrapeInput(BaseModel):
    """Inputs for web page scraping."""

    url: str = Field(..., description="URL to fetch and summarize.")
    max_chars: WebScrapeChars = Field(default=2000, description="Maximum characters of text to return.")


class NotebookEvalInput(BaseModel):
    """Inputs for notebook-style Python evaluation."""

    code: str = Field(..., description="Python code snippet to execute.")


class RssFetchInput(BaseModel):
    """Inputs for the RSS/news fetch tool."""

    query: str | None = Field(
        default=None,
        description="Optional keyword filter applied to titles/snippets.",
    )
    limit: MediumListLimit = Field(default=10, description="Maximum items to return.")
    sources: str | None = Field(
        default=None,
        description="Optional comma-separated RSS URLs to override the default NEWS_RSS_URLS set.",
    )

    @validator("limit", pre=True)
    def _coerce_limit(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty."""
        if value is None or value == "":
            return 10
        try:
            return int(value)
        except (TypeError, ValueError):
            return 10


class BlueskySearchInput(BaseModel):
    """Inputs for the Bluesky search tool."""

    query: str = Field(..., description="Search terms, hashtags, or keywords.")
    limit: MediumListLimit = Field(default=10, description="Maximum posts to return.")

    @validator("limit", pre=True)
    def _coerce_limit(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty."""
        if value is None or value == "":
            return 10
        try:
            return int(value)
        except (TypeError, ValueError):
            return 10


class MarketHistoryInput(BaseModel):
    """Inputs for the market history tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    limit: HistoryLimit = Field(default=200, description="Number of recent bets to analyze.")

    @validator("limit", pre=True)
    def _coerce_limit(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty."""
        if value is None or value == "":
            return 200
        try:
            return int(value)
        except (TypeError, ValueError):
            return 200


class PortfolioAnalyticsInput(BaseModel):
    """Inputs for the portfolio analytics tool."""

    max_positions: MaxPositionsLimit = Field(default=5, description="Number of largest positions to surface.")

    @validator("max_positions", pre=True)
    def _coerce_max_positions(cls, value: object) -> int:
        """Accept ints or numeric strings; fall back to default on null/empty/invalid."""
        if value is None or value == "":
            return 5
        try:
            return int(value)
        except (TypeError, ValueError):
            return 5


class EventTimerInput(BaseModel):
    """Inputs for the event timer tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")


class RiskGateInput(BaseModel):
    """Inputs for the risk gate tool."""

    market_id: str = Field(..., description="Manifold market id or slug.")
    outcome: str = Field(..., description="YES/NO or answer label.")
    amount: PositiveAmount = Field(..., description="Proposed Mana to wager.")
    belief_prob: ProbBounded = Field(..., description="Agent's subjective probability (0-1).")
    market_prob: ProbBounded | None = Field(default=None, description="Current market probability (0-1).")
    bankroll: PositiveAmount | None = Field(
        default=None, description="Optional bankroll override; uses cash + positions if omitted."
    )
