"""Heuristic query builder for web search fallback."""

from __future__ import annotations

from typing import List, Optional


_SITE_HINTS = {
    # If the user asks about NEU policies/news, bias the second query
    # toward the official domain.
    "RAG_INSUFFICIENT_LOW_TOP_SCORE": "neu.edu.vn",
    "RAG_INSUFFICIENT_LOW_COVERAGE": "neu.edu.vn",
    "BORDERLINE_ESCALATE_TO_WEB": "neu.edu.vn",
    "QUERY_REQUIRES_FRESHNESS": "neu.edu.vn",
}


def build_web_queries(question: str, reason_code: Optional[str] = None) -> List[str]:
    """Return up to two queries derived from the user question."""
    base = (question or "").strip()
    if not base:
        return []

    queries: List[str] = [base]
    site_hint = _SITE_HINTS.get(reason_code or "")
    if site_hint:
        queries.append(f"{base} site:{site_hint}")
    return queries
