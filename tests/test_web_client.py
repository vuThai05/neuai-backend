"""Tests for the web search service and DuckDuckGo provider wrapper."""

from __future__ import annotations

import time
from typing import List

import pytest

from src.web import WebResult, WebSearchService
from src.web.client import DuckDuckGoProvider
from src.web.query_builder import build_web_queries
from src.web.reranker import rerank_lexical
from src.web.result_parser import dedupe_by_url

from .conftest import FakeWebProvider


def test_build_web_queries_no_site_for_internal_decision():
    queries = build_web_queries("hoi gi do", "RAG_SUFFICIENT_HIGH_CONFIDENCE")
    assert queries == ["hoi gi do"]


def test_build_web_queries_adds_site_filter_for_low_score():
    queries = build_web_queries("hoi gi do", "RAG_INSUFFICIENT_LOW_TOP_SCORE")
    assert queries == ["hoi gi do", "hoi gi do site:neu.edu.vn"]


def test_build_web_queries_handles_empty_input():
    assert build_web_queries("   ", "RAG_INSUFFICIENT_LOW_TOP_SCORE") == []


def test_dedupe_by_url_removes_duplicates_case_insensitive():
    results = [
        WebResult(title="A", url="https://x.com/1", snippet=""),
        WebResult(title="B", url="https://X.com/1/", snippet=""),
        WebResult(title="C", url="https://x.com/2", snippet=""),
        WebResult(title="D", url="", snippet=""),
    ]
    deduped = dedupe_by_url(results)
    assert [r.title for r in deduped] == ["A", "C"]


def test_rerank_prefers_overlap_with_question():
    question = "lich thi cuoi ky neu"
    results = [
        WebResult(title="Tin chung", url="https://x/1", snippet="lap trinh"),
        WebResult(title="Lich thi neu cuoi ky", url="https://x/2", snippet="thong tin chi tiet"),
    ]
    ranked = rerank_lexical(question, results)
    assert ranked[0].title == "Lich thi neu cuoi ky"


def test_service_respects_max_queries():
    provider = FakeWebProvider(results_by_query={
        "q": [WebResult(title="t", url="https://a", snippet="")],
        "q site:neu.edu.vn": [WebResult(title="t2", url="https://b", snippet="")],
    })
    service = WebSearchService(provider, max_queries=1)
    results, metrics = service.search_for_question(
        "q", type("D", (), {"reason_code": "RAG_INSUFFICIENT_LOW_TOP_SCORE"})()
    )
    assert metrics.web_calls == 1
    assert len(provider.calls) == 1


def test_service_counts_timeouts_and_continues():
    provider = FakeWebProvider(
        results_by_query={
            "q site:neu.edu.vn": [WebResult(title="t", url="https://b", snippet="")],
        },
        timeout_for={"q"},
    )
    service = WebSearchService(provider, max_queries=2)
    results, metrics = service.search_for_question(
        "q", type("D", (), {"reason_code": "RAG_INSUFFICIENT_LOW_TOP_SCORE"})()
    )
    assert metrics.timeouts == 1
    assert metrics.web_calls == 1
    assert any(r.url == "https://b" for r in results)


def test_service_counts_errors():
    provider = FakeWebProvider(
        raise_for={"q", "q site:neu.edu.vn"},
    )
    service = WebSearchService(provider, max_queries=2)
    results, metrics = service.search_for_question(
        "q", type("D", (), {"reason_code": "RAG_INSUFFICIENT_LOW_TOP_SCORE"})()
    )
    assert metrics.errors == 2
    assert metrics.web_calls == 0
    assert results == []


def test_service_breaks_when_budget_exhausted():
    class SlowProvider:
        def search(self, query, max_results, timeout_ms):
            time.sleep(0.3)
            return [WebResult(title="t", url=f"https://{query}", snippet="")]

    # Budget allows roughly one provider call (300ms) before hitting the
    # 200ms-minimum guardrail and breaking out.
    service = WebSearchService(
        SlowProvider(),
        max_queries=5,
        web_timeout_ms=2500,
        total_budget_ms=400,
    )
    results, metrics = service.search_for_question(
        "q", type("D", (), {"reason_code": "RAG_INSUFFICIENT_LOW_TOP_SCORE"})()
    )
    assert metrics.web_calls == 1
    # Sleep granularity on Windows can shave a few ms off; 250ms is safe.
    assert metrics.web_latency_ms >= 250
    assert len(results) == 1


def test_duckduckgo_provider_enforces_per_query_timeout(monkeypatch):
    """Stub DDGS so we don't hit the network and verify timeout handling."""
    import src.web.client as client_mod

    class HangingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, *a, **k):
            time.sleep(2)
            return []

    fake_module = type("Mod", (), {"DDGS": HangingDDGS})

    def fake_import(name, *a, **k):
        if name == "duckduckgo_search":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr(client_mod, "__import__", fake_import, raising=False)
    # Patch the lazy import inside the method by injecting the module directly.
    import sys
    sys.modules["duckduckgo_search"] = fake_module

    provider = DuckDuckGoProvider()
    with pytest.raises(TimeoutError):
        provider.search("anything", max_results=1, timeout_ms=200)
