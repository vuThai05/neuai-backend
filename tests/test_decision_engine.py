"""Decision engine policy tests covering all 6 reason codes."""

from __future__ import annotations

from src.orchestration.decision_engine import decide
from src.rag.scoring import compute_signals

from .conftest import make_signals


def test_high_confidence_keeps_internal():
    signals = make_signals(top_score=0.85, avg_top3=0.7, source_diversity=3)
    decision = decide(signals)
    assert decision.route == "RAG_ONLY"
    assert decision.reason_code == "RAG_SUFFICIENT_HIGH_CONFIDENCE"


def test_low_top_score_escalates_to_web():
    signals = make_signals(top_score=0.2, avg_top3=0.1, source_diversity=1)
    decision = decide(signals)
    assert decision.route == "RAG_PLUS_WEB"
    assert decision.reason_code == "RAG_INSUFFICIENT_LOW_TOP_SCORE"


def test_low_coverage_escalates_to_web():
    # Top score is acceptable but avg_top3 too low and diversity weak.
    signals = make_signals(top_score=0.5, avg_top3=0.2, source_diversity=0)
    decision = decide(signals)
    assert decision.route == "RAG_PLUS_WEB"
    assert decision.reason_code == "RAG_INSUFFICIENT_LOW_COVERAGE"


def test_freshness_keyword_forces_web():
    signals = make_signals(top_score=0.95, avg_top3=0.9, source_diversity=4, needs_freshness=True)
    decision = decide(signals)
    assert decision.route == "RAG_PLUS_WEB"
    assert decision.reason_code == "QUERY_REQUIRES_FRESHNESS"


def test_borderline_escalate_to_web_when_no_direct_match():
    signals = make_signals(top_score=0.55, avg_top3=0.5, source_diversity=2, has_direct_match=False)
    decision = decide(signals)
    assert decision.route == "RAG_PLUS_WEB"
    assert decision.reason_code == "BORDERLINE_ESCALATE_TO_WEB"


def test_borderline_keeps_internal_when_direct_match():
    signals = make_signals(top_score=0.55, avg_top3=0.5, source_diversity=2, has_direct_match=True)
    decision = decide(signals)
    assert decision.route == "RAG_ONLY"
    assert decision.reason_code == "BORDERLINE_KEEP_INTERNAL"


def test_compute_signals_detects_freshness_keywords():
    signals = compute_signals([], "lich thi moi nhat la khi nao")
    assert signals.needs_freshness is True


def test_compute_signals_does_not_flag_lich_alone():
    # Regression: 'lich hoc' (study schedule) must not trigger freshness.
    signals = compute_signals([], "lich hoc tuan toi")
    assert signals.needs_freshness is False


def test_compute_signals_handles_empty_docs():
    signals = compute_signals([], "cau hoi binh thuong")
    assert signals.top_score == 0.0
    assert signals.source_diversity == 0
    assert signals.has_direct_match is False


def test_compute_signals_counts_distinct_permalinks():
    docs = [
        {"score": 0.9, "text": "abc xyz quan trong", "source": {"permalink_url": "https://a.com/1"}},
        {"score": 0.8, "text": "abc khac", "source": {"permalink_url": "https://a.com/1"}},  # duplicate URL
        {"score": 0.7, "text": "abc them", "source": {"permalink_url": "https://b.com/2"}},
    ]
    signals = compute_signals(docs, "abc xyz")
    assert signals.source_diversity == 2
    assert signals.top_score > 0.85
