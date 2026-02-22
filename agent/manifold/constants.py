"""Shared configuration constants for Manifold integrations."""

from __future__ import annotations

import os

DEFAULT_API_LIMIT = 500
MAX_API_LIMIT = int(os.environ.get("MANIFOLD_API_LIMIT", str(DEFAULT_API_LIMIT)))

MANIFOLD_API_ROOT = os.environ.get("MANIFOLD_API_ROOT", "https://api.manifold.markets/v0").rstrip("/")

__all__ = [
    "MANIFOLD_API_ROOT",
    "MAX_API_LIMIT",
]
