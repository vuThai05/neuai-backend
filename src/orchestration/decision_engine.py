"""Rule-based routing between RAG_ONLY and RAG_PLUS_WEB.

Thresholds and reason codes follow `docs/decision-policy.md`. The engine is
intentionally pure: it consumes retriever output and the user question and
returns a `Decision` plus the underlying `RetrievalSignals` so callers can log
or override behaviour later.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .schemas import Decision, RetrievalSignals


HIGH_TOP_SCORE = 0.62
HIGH_AVG_TOP3 = 0.55
LOW_TOP_SCORE = 0.45
LOW_AVG_TOP3 = 0.40
MIN_SOURCE_DIVERSITY_FOR_RAG = 2

# Keywords (Vietnamese diacritic-stripped + English) that hint at freshness
# requirements. We keep the regex tolerant of common spelling variants seen
# in the dataset (e.g. "moi nhat", "mới nhất").
_FRESHNESS_PATTERNS = [
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
_FRESHNESS_RE = re.compile("|".join(_FRESHNESS_PATTERNS), re.IGNORECASE)


def _to_unit(score: float) -> float:
    """Clamp a score into the [0, 1] range without raising."""
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return float(score)


def _detect_freshness(question: str) -> bool:
    return bool(_FRESHNESS_RE.search(question or ""))


def _tokens(text: str) -> Set[str]:
    return {tok for tok in re.findall(r"\w+", (text or "").lower()) if len(tok) > 2}


def _has_direct_match(question: str, docs: List[Dict[str, Any]]) -> bool:
    """Heuristic: top doc shares at least 2 distinct content tokens with the question."""
    q_tokens = _tokens(question)
    if not q_tokens or not docs:
        return False
    top_text = docs[0].get("text", "")
    overlap = q_tokens.intersection(_tokens(top_text))
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
            needs_freshness=_detect_freshness(question),
        )

    scores = [_to_unit(float(d.get("score", 0.0))) for d in docs]
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
        has_direct_match=_has_direct_match(question, docs),
        needs_freshness=_detect_freshness(question),
    )


def decide(signals: RetrievalSignals) -> Decision:
    """Apply the rule-based decision policy and return a `Decision`."""
    confidence = round(_to_unit(signals.top_score), 3)

    if signals.needs_freshness:
        return Decision(
            route="RAG_PLUS_WEB",
            reason_code="QUERY_REQUIRES_FRESHNESS",
            reason="Cau hoi co tin hieu can du lieu cap nhat moi.",
            confidence=confidence,
        )

    if signals.top_score < LOW_TOP_SCORE:
        return Decision(
            route="RAG_PLUS_WEB",
            reason_code="RAG_INSUFFICIENT_LOW_TOP_SCORE",
            reason="Diem retrieval cao nhat thap, can mo rong tim kiem.",
            confidence=confidence,
        )

    if signals.avg_top3 < LOW_AVG_TOP3 or signals.source_diversity == 0:
        return Decision(
            route="RAG_PLUS_WEB",
            reason_code="RAG_INSUFFICIENT_LOW_COVERAGE",
            reason="Bang chung noi bo it hoac thieu da dang nguon.",
            confidence=confidence,
        )

    if (
        signals.top_score >= HIGH_TOP_SCORE
        and signals.avg_top3 >= HIGH_AVG_TOP3
        and signals.source_diversity >= MIN_SOURCE_DIVERSITY_FOR_RAG
    ):
        return Decision(
            route="RAG_ONLY",
            reason_code="RAG_SUFFICIENT_HIGH_CONFIDENCE",
            reason="Du bang chung noi bo de tra loi truc tiep.",
            confidence=confidence,
        )

    # Vung xam: 0.45 <= top_score < 0.62.
    if signals.has_direct_match:
        return Decision(
            route="RAG_ONLY",
            reason_code="BORDERLINE_KEEP_INTERNAL",
            reason="Diem retrieval o vung xam nhung co dau hieu khop noi bo.",
            confidence=confidence,
        )

    return Decision(
        route="RAG_PLUS_WEB",
        reason_code="BORDERLINE_ESCALATE_TO_WEB",
        reason="Diem retrieval o vung xam, can xac thuc them tu nguon ngoai.",
        confidence=confidence,
    )
