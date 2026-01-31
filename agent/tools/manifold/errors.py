"""Error helpers for Manifold tools."""

from __future__ import annotations


def _is_not_found_error(error: Exception) -> bool:
    message = str(error).lower()
    return "404" in message or "not found" in message or "contract not found" in message

