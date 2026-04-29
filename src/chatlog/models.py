"""Models for chat logging."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Reference to a source document used in an assistant response.

    Stored alongside the assistant message so a conversation can be replayed
    later with its citations intact. The schema mirrors `Source` in
    `src/orchestration/schemas.py`, with all fields optional/defaulted to keep
    legacy records readable.
    """

    type: str = "internal"
    title: str = ""
    url: Optional[str] = None
    snippet: str = ""
    score: Optional[float] = None


class DecisionLog(BaseModel):
    """Routing decision metadata recorded with an assistant message."""

    route: str
    reason_code: str
    reason: str
    confidence: float


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    id: str = Field(default_factory=lambda: f"msg-{datetime.now().timestamp()}")
    type: str  # "user" or "assistant"
    content: str
    timestamp: str
    refs: Optional[List[SourceRef]] = None
    decision: Optional[DecisionLog] = None


class ChatLog(BaseModel):
    """A complete conversation/chat session."""

    id: str = Field(default_factory=lambda: f"conv-{datetime.now().timestamp()}")
    title: Optional[str] = None
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
