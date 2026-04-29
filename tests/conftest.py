"""Pytest fixtures shared across the test suite.

Heavy backend dependencies (BGE-M3, Qdrant client, MongoDB) are stubbed out
at import time so unit tests can run on a developer laptop without spinning
up real services. Anything that needs real infra belongs to scripts/.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest


def _install_stubs() -> None:
    """Install lightweight module stubs before src.* imports are resolved."""
    # numpy: only the ndarray symbol is needed at import time. Real numpy ops
    # never run because we stub RAGRetriever before instantiation in fixtures.
    if "numpy" not in sys.modules:
        np_stub = types.ModuleType("numpy")
        np_stub.ndarray = type("NDArray", (), {})
        np_stub.linalg = types.ModuleType("numpy.linalg")
        sys.modules["numpy"] = np_stub
        sys.modules["numpy.linalg"] = np_stub.linalg

    if "FlagEmbedding" not in sys.modules:
        flag_stub = types.ModuleType("FlagEmbedding")
        flag_stub.BGEM3FlagModel = type("BGEM3FlagModel", (), {})
        sys.modules["FlagEmbedding"] = flag_stub

    if "qdrant_client" not in sys.modules:
        qd_stub = types.ModuleType("qdrant_client")
        qd_stub.QdrantClient = type("QdrantClient", (), {})
        sys.modules["qdrant_client"] = qd_stub

    if "src.utils.config" not in sys.modules:
        cfg_stub = types.ModuleType("src.utils.config")
        cfg_stub.MONGO_URI = "mongodb://stub"
        cfg_stub.MONGO_DB_NAME = "stub_db"
        cfg_stub.QDRANT_COLLECTION_NAME = "stub_collection"
        cfg_stub.get_mongo_client = lambda: None
        cfg_stub.get_qdrant_client = lambda: None
        sys.modules["src.utils.config"] = cfg_stub


_install_stubs()


# Now safe to import backend modules.
from src.orchestration.schemas import ChatRequest, Decision, RetrievalSignals  # noqa: E402
from src.web import WebResult, WebSearchService  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRetriever:
    """In-memory retriever returning whatever docs the test seeded."""

    def __init__(self, docs: Optional[List[Dict[str, Any]]] = None) -> None:
        self.docs = docs or []
        self.calls: List[Dict[str, Any]] = []

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k})
        return list(self.docs[:top_k])


class FakeOllama:
    """Records prompts and returns a canned answer."""

    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeChatlogRepo:
    """In-memory ChatLogRepository compatible with the orchestrator."""

    def __init__(self) -> None:
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.next_id = 0

    def _new_id(self) -> str:
        self.next_id += 1
        return f"conv-{self.next_id:03d}"

    def create_conversation(self, first_user_message: str):
        cid = self._new_id()
        self.conversations[cid] = {
            "id": cid,
            "title": first_user_message[:30],
            "messages": [{"role": "user", "content": first_user_message}],
        }
        return types.SimpleNamespace(id=cid)

    def add_user_message(self, conversation_id: str, content: str) -> None:
        self.conversations.setdefault(
            conversation_id,
            {"id": conversation_id, "title": content[:30], "messages": []},
        )["messages"].append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        decision: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conversations.setdefault(
            conversation_id,
            {"id": conversation_id, "title": "", "messages": []},
        )["messages"].append(
            {
                "role": "assistant",
                "content": content,
                "sources": sources or [],
                "decision": decision,
            }
        )


class FakeWebProvider:
    """Web search provider returning canned results per query."""

    def __init__(self, results_by_query: Optional[Dict[str, List[WebResult]]] = None,
                 raise_for: Optional[set] = None,
                 timeout_for: Optional[set] = None) -> None:
        self.results_by_query = results_by_query or {}
        self.raise_for = raise_for or set()
        self.timeout_for = timeout_for or set()
        self.calls: List[str] = []

    def search(self, query: str, max_results: int, timeout_ms: int) -> List[WebResult]:
        self.calls.append(query)
        if query in self.raise_for:
            raise RuntimeError(f"boom on {query}")
        if query in self.timeout_for:
            raise TimeoutError(f"timeout on {query}")
        return list(self.results_by_query.get(query, []))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def fake_repo() -> FakeChatlogRepo:
    return FakeChatlogRepo()


@pytest.fixture
def fake_web_provider() -> FakeWebProvider:
    return FakeWebProvider()


@pytest.fixture
def fake_web_service(fake_web_provider: FakeWebProvider) -> WebSearchService:
    return WebSearchService(
        provider=fake_web_provider,
        max_queries=2,
        max_results_per_query=5,
        web_timeout_ms=2500,
        total_budget_ms=4000,
    )


@pytest.fixture
def chat_request_factory():
    def _make(question: str, conversation_id: Optional[str] = None) -> ChatRequest:
        return ChatRequest(question=question, conversation_id=conversation_id)
    return _make


@pytest.fixture
def app(fake_retriever, fake_ollama, fake_repo, fake_web_service):
    """Build a FastAPI app wired with the in-memory fakes."""
    from fastapi import FastAPI
    from src.api.chat import router as chat_router
    from src.api.health import router as health_router
    from src.llm import Synthesizer
    from src.orchestration import ChatOrchestrator

    application = FastAPI()
    application.include_router(chat_router)
    application.include_router(health_router)

    application.state.retriever = fake_retriever
    application.state.ollama_client = fake_ollama
    application.state.chatlog_repo = fake_repo
    application.state.synthesizer = Synthesizer(fake_ollama)
    application.state.web_service = fake_web_service
    application.state.orchestrator = ChatOrchestrator(
        retriever=fake_retriever,
        ollama_client=fake_ollama,
        chatlog_repo=fake_repo,
        synthesizer=Synthesizer(fake_ollama),
        web_service=fake_web_service,
    )
    return application


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_signals(**overrides: Any) -> RetrievalSignals:
    base = dict(
        top_score=0.5,
        avg_top3=0.45,
        score_gap=0.05,
        source_diversity=2,
        has_direct_match=False,
        needs_freshness=False,
    )
    base.update(overrides)
    return RetrievalSignals(**base)
