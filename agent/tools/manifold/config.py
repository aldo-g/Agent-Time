"""Configuration and constants for Manifold tools."""

from __future__ import annotations

import os
from pathlib import Path

MARKET_CACHE_ENV = "PREDICT_ARENA_MARKET_CACHE"
TRADE_LOG_ENV = "AGENT_TRADE_LOG_PATH"
DEFAULT_TRADE_LOG_PATH = Path(os.environ.get(TRADE_LOG_ENV, "results/trades.jsonl"))
RISK_MAX_BET_PCT = float(os.environ.get("RISK_MAX_BET_PCT", "0.05"))
RISK_MAX_SINGLE_POSITION_PCT = float(os.environ.get("RISK_MAX_SINGLE_POSITION_PCT", "0.2"))
RISK_MAX_GROSS_EXPOSURE_PCT = float(os.environ.get("RISK_MAX_GROSS_EXPOSURE_PCT", "0.7"))
KELLY_MULTIPLIER = float(os.environ.get("RISK_KELLY_MULTIPLIER", "0.5"))
MARKET_INSPECTION_LIMIT = int(os.environ.get("AGENT_MARKET_INSPECTION_LIMIT", "5"))
REQUIRE_TRUSTED_CATALYST_FOR_NEWS_TRADES = (
    (os.environ.get("REQUIRE_TRUSTED_CATALYST_FOR_NEWS_TRADES", "1").strip().lower())
    not in {"0", "false", "no", "off"}
)
_DEFAULT_TRUSTED_CATALYST_DOMAINS = (
    "apnews.com,"
    "reuters.com,"
    "ft.com,"
    "bloomberg.com,"
    "wsj.com,"
    "nytimes.com,"
    "washingtonpost.com,"
    "economist.com,"
    "axios.com"
)
TRUSTED_CATALYST_DOMAINS = tuple(
    domain.strip().lower()
    for domain in os.environ.get("TRUSTED_CATALYST_DOMAINS", _DEFAULT_TRUSTED_CATALYST_DOMAINS).split(",")
    if domain.strip()
)
