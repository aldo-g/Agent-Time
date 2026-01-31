"""Configuration and constants for Manifold tools."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from agent.manifold.constants import RESOLUTION_CUTOFF_MS

CUTOFF_ISO = datetime.fromtimestamp(RESOLUTION_CUTOFF_MS / 1000, tz=timezone.utc).date().isoformat()
MARKET_CACHE_ENV = "PREDICT_ARENA_MARKET_CACHE"
TRADE_LOG_ENV = "AGENT_TRADE_LOG_PATH"
DEFAULT_TRADE_LOG_PATH = Path(os.environ.get(TRADE_LOG_ENV, "results/trades.jsonl"))
RISK_MAX_BET_PCT = float(os.environ.get("RISK_MAX_BET_PCT", "0.05"))
RISK_MAX_SINGLE_POSITION_PCT = float(os.environ.get("RISK_MAX_SINGLE_POSITION_PCT", "0.2"))
RISK_MAX_GROSS_EXPOSURE_PCT = float(os.environ.get("RISK_MAX_GROSS_EXPOSURE_PCT", "0.7"))
KELLY_MULTIPLIER = float(os.environ.get("RISK_KELLY_MULTIPLIER", "0.5"))
MARKET_INSPECTION_LIMIT = int(os.environ.get("AGENT_MARKET_INSPECTION_LIMIT", "5"))

