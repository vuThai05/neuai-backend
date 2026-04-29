"""Orchestration layer: schemas, decision engine, chat orchestrator.

`ChatOrchestrator` is exposed lazily via PEP 562 `__getattr__` to avoid a
circular import: the orchestrator depends on `src.rag.scoring`, which in
turn imports schemas from this package.
"""

from .decision_engine import decide
from .schemas import (
    ChatRequest,
    ChatResponse,
    Decision,
    RetrievalSignals,
    Source,
)

__all__ = [
    "ChatOrchestrator",
    "ChatRequest",
    "ChatResponse",
    "Decision",
    "RetrievalSignals",
    "Source",
    "decide",
]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "ChatOrchestrator":
        from .chat_orchestrator import ChatOrchestrator
        return ChatOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
