"""End-to-end orchestrator for `/chat` requests.

P0 only routes through `RAG_ONLY`: even when the decision engine prefers
`RAG_PLUS_WEB`, the orchestrator answers using the internal RAG context and
passes the routing metadata to the frontend so the user is informed that
external evidence would help. The web tool path is added in P1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.chatlog import ChatLogRepository
from src.llm import OllamaClient
from src.rag import RAGRetriever, build_context, build_prompt

from .decision_engine import compute_signals, decide
from .schemas import ChatRequest, ChatResponse, Source


SNIPPET_MAX_LEN = 240


class ChatOrchestrator:
    """Coordinate retrieval, decision, generation and persistence for one turn."""

    def __init__(
        self,
        retriever: RAGRetriever,
        ollama_client: OllamaClient,
        chatlog_repo: ChatLogRepository,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.ollama_client = ollama_client
        self.chatlog_repo = chatlog_repo
        self.top_k = top_k

    def run(self, payload: ChatRequest) -> ChatResponse:
        question = payload.question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        conversation_id = self._ensure_conversation(payload.conversation_id, question)

        docs: List[Dict[str, Any]] = self.retriever.retrieve(question, top_k=self.top_k) or []

        signals = compute_signals(docs, question)
        decision = decide(signals)

        context = build_context(docs) if docs else ""
        prompt = build_prompt(question, context)
        answer = self.ollama_client.generate(prompt)

        sources = [self._map_source(d) for d in docs]

        self.chatlog_repo.add_assistant_message(
            conversation_id,
            answer,
            sources=[s.model_dump() for s in sources],
            decision=decision.model_dump(),
        )

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            decision=decision,
            sources=sources,
        )

    def _ensure_conversation(self, conversation_id: Optional[str], question: str) -> str:
        if conversation_id:
            self.chatlog_repo.add_user_message(conversation_id, question)
            return conversation_id

        chat_log = self.chatlog_repo.create_conversation(question)
        return chat_log.id

    @staticmethod
    def _map_source(doc: Dict[str, Any]) -> Source:
        src_meta: Dict[str, Any] = doc.get("source") or {}
        text = (doc.get("text") or "").strip()

        snippet = text[:SNIPPET_MAX_LEN]
        if len(text) > SNIPPET_MAX_LEN:
            snippet = snippet.rstrip() + "..."

        title_raw = (
            src_meta.get("title")
            or src_meta.get("author")
            or src_meta.get("post_id")
            or "Internal source"
        )

        return Source(
            type="internal",
            title=str(title_raw),
            url=src_meta.get("permalink_url") or src_meta.get("url"),
            snippet=snippet,
            score=float(doc.get("score", 0.0)),
        )
