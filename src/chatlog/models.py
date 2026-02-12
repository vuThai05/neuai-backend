"""Models for chat logging."""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Reference to a source document used in the response."""
    link: str
    text: str
    score: Optional[float] = None
    dense_score: Optional[float] = None


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    id: str = Field(default_factory=lambda: f"msg-{datetime.now().timestamp()}")
    type: str  # "user" or "assistant"
    content: str
    timestamp: str
    refs: Optional[List[SourceRef]] = None  # Only for assistant messages


class ChatLog(BaseModel):
    """A complete conversation/chat session."""
    id: str = Field(default_factory=lambda: f"conv-{datetime.now().timestamp()}")
    title: Optional[str] = None  # Auto-generated from first user message
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
