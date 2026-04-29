"""End-to-end orchestrator for `/chat` requests.

Pipeline:
  ensure conversation -> retrieve -> compute signals -> decide
  -> (optionally) web search -> synthesize -> persist -> log structured event.

The web service and the synthesizer are injected so the orchestrator stays
trivially testable; observability is built in via `Timer` / `StepTimers`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.chatlog import ChatLogRepository
from src.llm import OllamaClient, Synthesizer
from src.rag import RAGRetriever
from src.rag.scoring import compute_signals
from src.utils.logging import get_logger, log_event
from src.utils.timers import StepTimers, Timer
from src.web import WebMetrics, WebResult, WebSearchService

from .decision_engine import decide
from .schemas import ChatRequest, ChatResponse, Decision, Source


SNIPPET_MAX_LEN = 240

# Note prepended when web search was supposed to happen but failed/yielded nothing.
WEB_FALLBACK_NOTE = (
    "[He thong tam thoi chua truy cap duoc nguon web. "
    "Day la cau tra loi dua tren du lieu noi bo hien co.]\n\n"
)

logger = get_logger(__name__)


class ChatOrchestrator:
    """Coordinate retrieval, decision, generation, web fallback and persistence."""

    def __init__(
        self,
        retriever: RAGRetriever,
        ollama_client: OllamaClient,
        chatlog_repo: ChatLogRepository,
        synthesizer: Optional[Synthesizer] = None,
        web_service: Optional[WebSearchService] = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.ollama_client = ollama_client
        self.chatlog_repo = chatlog_repo
        self.synthesizer = synthesizer or Synthesizer(ollama_client)
        self.web_service = web_service
        self.top_k = top_k

    def run(self, payload: ChatRequest) -> ChatResponse:
        question = payload.question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        timers = StepTimers()
        web_metrics = WebMetrics()

        with Timer() as total_timer:
            conversation_id = self._ensure_conversation(payload.conversation_id, question)

            with Timer() as retrieve_timer:
                docs: List[Dict[str, Any]] = self.retriever.retrieve(question, top_k=self.top_k) or []
            timers.retrieve_ms = retrieve_timer.elapsed_ms

            signals = compute_signals(docs, question)
            decision = decide(signals)

            web_results: List[WebResult] = []
            if decision.route == "RAG_PLUS_WEB" and self.web_service is not None:
                with Timer() as web_timer:
                    web_results, web_metrics = self.web_service.search_for_question(
                        question=question, decision=decision
                    )
                timers.web_ms = web_timer.elapsed_ms

            with Timer() as llm_timer:
                answer, sources = self._produce_answer(
                    question=question,
                    decision=decision,
                    docs=docs,
                    web_results=web_results,
                )
            timers.llm_ms = llm_timer.elapsed_ms

            self.chatlog_repo.add_assistant_message(
                conversation_id,
                answer,
                sources=[s.model_dump() for s in sources],
                decision=decision.model_dump(),
            )

        timers.total_ms = total_timer.elapsed_ms

        log_event(
            logger,
            "chat_completed",
            conversation_id=conversation_id,
            route=decision.route,
            reason_code=decision.reason_code,
            top_score=signals.top_score,
            avg_top3=signals.avg_top3,
            source_diversity=signals.source_diversity,
            needs_freshness=signals.needs_freshness,
            web_calls=web_metrics.web_calls,
            web_latency_ms=timers.web_ms,
            web_timeouts=web_metrics.timeouts,
            web_errors=web_metrics.errors,
            retrieve_ms=timers.retrieve_ms,
            llm_latency_ms=timers.llm_ms,
            total_latency_ms=timers.total_ms,
            answer_has_citation=bool(sources),
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

    def _produce_answer(
        self,
        question: str,
        decision: Decision,
        docs: List[Dict[str, Any]],
        web_results: List[WebResult],
    ) -> Tuple[str, List[Source]]:
        """Pick a synthesis strategy based on the decision and available evidence."""
        internal_sources = [self._map_internal_source(d) for d in docs]

        if decision.route == "RAG_ONLY":
            answer = self.synthesizer.synthesize_internal(question, docs)
            return answer, internal_sources

        # RAG_PLUS_WEB path.
        if not web_results:
            # Either web is disabled, errored, or returned nothing. Fall back
            # to internal answer with the fallback note from decision-policy.md.
            answer = self.synthesizer.synthesize_internal(question, docs)
            return WEB_FALLBACK_NOTE + answer, internal_sources

        web_sources = [self._map_web_source(r, rank=i) for i, r in enumerate(web_results)]
        answer = self.synthesizer.synthesize_with_web(question, docs, web_results)
        return answer, internal_sources + web_sources

    @staticmethod
    def _map_internal_source(doc: Dict[str, Any]) -> Source:
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

    @staticmethod
    def _map_web_source(result: WebResult, rank: int) -> Source:
        snippet = (result.snippet or "")[:SNIPPET_MAX_LEN]
        score = result.score if result.score else 1.0 / (rank + 1)
        return Source(
            type="web",
            title=result.title or "Web result",
            url=result.url or None,
            snippet=snippet,
            score=float(score),
        )
