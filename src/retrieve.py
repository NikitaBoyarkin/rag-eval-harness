"""Thin retrieval facade re-exported for clarity (index.search does the work)."""
from __future__ import annotations

from .index import Index, search

__all__ = ["Index", "search"]
