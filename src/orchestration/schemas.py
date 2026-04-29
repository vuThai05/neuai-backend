"""Shared Pydantic models and types used across the orchestration layer.

These models define the public contract returned by the `/chat` endpoint and
the typed payloads exchanged between retriever, decision engine, and
synthesizer steps.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Route = Literal["RAG_ONLY", "RAG_PLUS_WEB"]

ReasonCode = Literal[
    "RAG_SUFFICIENT_HIGH_CONFIDENCE",
    "RAG_INSUFFICIENT_LOW_TOP_SCORE",
    "RAG_INSUFFICIENT_LOW_COVERAGE",
    "QUERY_REQUIRES_FRESHNESS",
    "BORDERLINE_ESCALATE_TO_WEB",
    "BORDERLINE_KEEP_INTERNAL",
]

SourceType = Literal["internal", "web"]


class Decision(BaseModel):
    """Routing decision metadata returned to the frontend."""

    route: Route
    reason_code: ReasonCode
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class Source(BaseModel):
    """Normalized source descriptor used by the frontend."""

    type: SourceType
    title: str
    url: Optional[str] = None
    snippet: str
    score: float


class RetrievalSignals(BaseModel):
    """Numerical signals computed from retriever output for routing."""

    top_score: float
    avg_top3: float
    score_gap: float
    source_diversity: int
    has_direct_match: bool
    needs_freshness: bool


class ChatRequest(BaseModel):
    """Shape of the JSON request body sent from the frontend."""

    question: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Shape of the JSON response consumed by the frontend."""

    answer: str
    conversation_id: str
    decision: Decision
    sources: List[Source]
