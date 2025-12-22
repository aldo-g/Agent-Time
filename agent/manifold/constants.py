"""Shared configuration constants for Manifold integrations."""

from __future__ import annotations

from datetime import datetime, timezone
import os

DEFAULT_API_LIMIT = 500
MAX_API_LIMIT = int(os.environ.get("MANIFOLD_API_LIMIT", str(DEFAULT_API_LIMIT)))

_now = datetime.now(timezone.utc)
_year_end = datetime(_now.year + 1, 1, 1, tzinfo=timezone.utc)
DEFAULT_CUTOFF_MS = int(_year_end.timestamp() * 1000)
RESOLUTION_CUTOFF_MS = int(os.environ.get("MANIFOLD_MAX_CLOSE_MS", str(DEFAULT_CUTOFF_MS)))

MANIFOLD_API_ROOT = os.environ.get("MANIFOLD_API_ROOT", "https://api.manifold.markets/v0").rstrip("/")

__all__ = [
    "MANIFOLD_API_ROOT",
    "MAX_API_LIMIT",
    "RESOLUTION_CUTOFF_MS",
]
