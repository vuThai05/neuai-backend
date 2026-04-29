"""Helpers to clean up web result lists before they hit the synthesizer."""

from __future__ import annotations

from typing import List

from .client import WebResult


def dedupe_by_url(results: List[WebResult]) -> List[WebResult]:
    """Drop duplicate URLs, keeping the first occurrence."""
    seen: set[str] = set()
    deduped: List[WebResult] = []
    for r in results:
        url = (r.url or "").strip().rstrip("/")
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped
