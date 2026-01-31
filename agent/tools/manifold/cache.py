"""Helpers for reading cached Manifold market payloads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from agent.manifold.data import EventSummary, events_from_dicts

from .config import MARKET_CACHE_ENV

_MARKET_CACHE: List[EventSummary] | None = None


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

