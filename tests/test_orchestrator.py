"""Orchestrator integration tests using in-memory fakes."""

from __future__ import annotations

import pytest

from src.llm import Synthesizer
from src.orchestration import ChatOrchestrator
from src.orchestration.schemas import ChatRequest
from src.web import WebResult, WebSearchService

from .conftest import FakeWebProvider


HIGH_CONFIDENCE_DOCS = [
    {
        "_id": "post::1",
        "score": 0.92,
        "text": "noi dung khop voi cau hoi",
        "source": {"permalink_url": "https://x.com/1", "post_id": "1"},
    },
    {
        "_id": "post::2",
        "score": 0.78,
        "text": "noi dung phu",
        "source": {"permalink_url": "https://x.com/2", "post_id": "2"},
    },
    {
        "_id": "post::3",
        "score": 0.7,
        "text": "noi dung khac",
        "source": {"permalink_url": "https://x.com/3", "post_id": "3"},
    },
]


def _make_orchestrator(fake_retriever, fake_ollama, fake_repo, web_service=None):
    return ChatOrchestrator(
        retriever=fake_retriever,
        ollama_client=fake_ollama,
        chatlog_repo=fake_repo,
        synthesizer=Synthesizer(fake_ollama),
        web_service=web_service,
    )


def test_rag_only_happy_path(fake_retriever, fake_ollama, fake_repo):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo)

    response = orch.run(ChatRequest(question="vong loai cua cau hoi binh thuong"))

    assert response.decision.route == "RAG_ONLY"
    assert response.decision.reason_code == "RAG_SUFFICIENT_HIGH_CONFIDENCE"
    assert all(s.type == "internal" for s in response.sources)
    assert response.answer  # synthesizer was called
    assert len(fake_ollama.prompts) == 1
    assert response.conversation_id.startswith("conv-")


def test_rag_plus_web_happy_path(fake_retriever, fake_ollama, fake_repo):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    provider = FakeWebProvider(
        results_by_query={
            "lich thi moi nhat 2025": [
                WebResult(title="Lich thi NEU", url="https://neu.edu.vn/lich", snippet="ban cong bo"),
                WebResult(title="Tin tuyen sinh", url="https://news.vn/abc", snippet="mua thi sap toi"),
            ],
            "lich thi moi nhat 2025 site:neu.edu.vn": [
                WebResult(title="Lich thi NEU", url="https://neu.edu.vn/lich", snippet="trung lap"),
                WebResult(title="Lich thi NEU 2", url="https://neu.edu.vn/2", snippet="them mot"),
            ],
        }
    )
    web_service = WebSearchService(provider, max_queries=2, total_budget_ms=4000)
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo, web_service)

    response = orch.run(ChatRequest(question="lich thi moi nhat 2025"))

    assert response.decision.route == "RAG_PLUS_WEB"
    types = {s.type for s in response.sources}
    assert "web" in types and "internal" in types
    web_sources = [s for s in response.sources if s.type == "web"]
    assert len(web_sources) == 3  # 2 unique URLs from query 1 + 1 new from query 2 (one dup)


def test_rag_plus_web_falls_back_when_provider_errors(
    fake_retriever, fake_ollama, fake_repo,
):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    provider = FakeWebProvider(raise_for={"lich moi nhat?", "lich moi nhat? site:neu.edu.vn"})
    web_service = WebSearchService(provider, max_queries=2)
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo, web_service)

    response = orch.run(ChatRequest(question="lich moi nhat?"))

    assert response.decision.route == "RAG_PLUS_WEB"
    assert response.answer.startswith("[He thong")
    # Only internal sources because web layer never produced anything.
    assert all(s.type == "internal" for s in response.sources)


def test_rag_plus_web_falls_back_when_provider_times_out(
    fake_retriever, fake_ollama, fake_repo,
):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    provider = FakeWebProvider(timeout_for={"lich moi nhat?", "lich moi nhat? site:neu.edu.vn"})
    web_service = WebSearchService(provider, max_queries=2)
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo, web_service)

    response = orch.run(ChatRequest(question="lich moi nhat?"))

    assert response.decision.route == "RAG_PLUS_WEB"
    assert response.answer.startswith("[He thong")


def test_empty_question_raises(fake_retriever, fake_ollama, fake_repo):
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo)
    with pytest.raises(ValueError):
        orch.run(ChatRequest(question="   "))


def test_orchestrator_persists_assistant_message(fake_retriever, fake_ollama, fake_repo):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo)

    response = orch.run(ChatRequest(question="cau hoi a"))

    convo = fake_repo.conversations[response.conversation_id]
    assistant_msgs = [m for m in convo["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["decision"]["route"] == "RAG_ONLY"
    assert assistant_msgs[0]["sources"]


def test_orchestrator_logs_chat_completed_event(
    fake_retriever, fake_ollama, fake_repo, caplog,
):
    import logging

    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    orch = _make_orchestrator(fake_retriever, fake_ollama, fake_repo)

    with caplog.at_level(logging.INFO, logger="src.orchestration.chat_orchestrator"):
        orch.run(ChatRequest(question="cau hoi giam sat"))

    matched = [r for r in caplog.records if "chat_completed" in r.getMessage()]
    assert matched, "expected at least one chat_completed log line"
    assert "route" in matched[0].getMessage()
