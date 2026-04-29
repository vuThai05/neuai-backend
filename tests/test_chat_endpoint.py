"""HTTP-level tests using FastAPI's TestClient against the wired app fixture."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.web import WebResult


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
]


def test_health_endpoint(app):
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["components"]["orchestrator"] is True
    assert data["components"]["web_service"] is True


def test_chat_endpoint_shape(app, fake_retriever, fake_web_provider):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "cau hoi a"})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {"answer", "conversation_id", "decision", "sources"}
    assert body["decision"]["route"] in {"RAG_ONLY", "RAG_PLUS_WEB"}
    assert isinstance(body["sources"], list)


def test_chat_endpoint_freshness_triggers_web(app, fake_retriever, fake_web_provider):
    fake_retriever.docs = HIGH_CONFIDENCE_DOCS
    fake_web_provider.results_by_query = {
        "lich thi moi nhat 2025": [WebResult(title="Lich", url="https://neu.edu.vn/x", snippet="snip")],
        "lich thi moi nhat 2025 site:neu.edu.vn": [],
    }
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "lich thi moi nhat 2025"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["route"] == "RAG_PLUS_WEB"
    assert any(s["type"] == "web" for s in body["sources"])


def test_chat_endpoint_rejects_empty_question(app):
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": ""})
    assert response.status_code in (400, 422, 500)
