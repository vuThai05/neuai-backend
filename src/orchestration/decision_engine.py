"""Rule-based routing between RAG_ONLY and RAG_PLUS_WEB.

Thresholds and reason codes follow `docs/decision-policy.md`. Signal
extraction lives in `src.rag.scoring`; this module is intentionally focused
on the policy itself so it can be evolved (or replaced by an LLM planner)
without touching retrieval logic.
"""

from __future__ import annotations

from .schemas import Decision, RetrievalSignals


__all__ = ["decide"]


HIGH_TOP_SCORE = 0.62
HIGH_AVG_TOP3 = 0.55
LOW_TOP_SCORE = 0.45
LOW_AVG_TOP3 = 0.40
MIN_SOURCE_DIVERSITY_FOR_RAG = 2


def _to_unit(score: float) -> float:
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return float(score)


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
