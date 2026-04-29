"""Retrieval scoring helpers and signal computation.

Pure functions used by both `RAGRetriever` (already produces dense/hybrid
scores) and `decision_engine.decide` (consumes a derived `RetrievalSignals`).
Kept here, in the `rag` layer, so the retrieval concept stays close to its
data source while still being reusable from orchestration.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from src.orchestration.schemas import RetrievalSignals


__all__ = [
    "to_unit",
    "tokens",
    "detect_freshness",
    "has_direct_match",
    "compute_signals",
    "FRESHNESS_PATTERNS",
]


# Vietnamese (diacritic-tolerant) + English keywords that hint a question
# requires data more recent than what the internal index typically holds.
FRESHNESS_PATTERNS = [
    r"m[oơô]i\s*nh[aâ]t",
    r"\bm[oơô]i\b",
    r"h[oôơ]m\s*nay",
    r"h[oôơ]m\s*qua",
    r"g[aâ]n\s*[dđ][aâ]y",
    r"c[aâ]p\s*nh[aâ]t",
    r"th[oôơ]ng\s*b[aá]o\s*m[oơô]i",
    r"\bdeadline\b",
    r"\blatest\b",
    r"\bnewest\b",
    r"\brecent\b",
    r"\btoday\b",
    r"\bnow\b",
]

_FRESHNESS_RE = re.compile("|".join(FRESHNESS_PATTERNS), re.IGNORECASE)


def to_unit(score: float) -> float:
    """Clamp a score into the [0, 1] range without raising."""
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return float(score)


def tokens(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase content tokens (>=3 chars)."""
    return {tok for tok in re.findall(r"\w+", (text or "").lower()) if len(tok) > 2}


def detect_freshness(question: str) -> bool:
    """Return True if the question text matches any freshness keyword."""
    return bool(_FRESHNESS_RE.search(question or ""))


def has_direct_match(question: str, docs: List[Dict[str, Any]]) -> bool:
    """Heuristic: top doc shares at least 2 distinct tokens with the question."""
    q_tokens = tokens(question)
    if not q_tokens or not docs:
        return False
    top_text = docs[0].get("text", "")
    overlap = q_tokens.intersection(tokens(top_text))
    return len(overlap) >= 2


def compute_signals(docs: List[Dict[str, Any]], question: str) -> RetrievalSignals:
    """Derive routing signals from retriever output and the user question."""
    if not docs:
        return RetrievalSignals(
            top_score=0.0,
            avg_top3=0.0,
            score_gap=0.0,
            source_diversity=0,
            has_direct_match=False,
            needs_freshness=detect_freshness(question),
        )

    scores = [to_unit(float(d.get("score", 0.0))) for d in docs]
    top_score = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    score_gap = max(0.0, top_score - second)
    avg_top3 = sum(scores[:3]) / min(3, len(scores))

    permalinks: Set[str] = set()
    for d in docs:
        src = d.get("source") or {}
        url = src.get("permalink_url") or src.get("url")
        if url:
            permalinks.add(str(url))

    return RetrievalSignals(
        top_score=top_score,
        avg_top3=avg_top3,
        score_gap=score_gap,
        source_diversity=len(permalinks),
        has_direct_match=has_direct_match(question, docs),
        needs_freshness=detect_freshness(question),
    )
