"""FastAPI dependency helpers that read shared services from `app.state`."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.chatlog import ChatLogRepository
from src.orchestration.chat_orchestrator import ChatOrchestrator


def get_orchestrator(request: Request) -> ChatOrchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Backend services are not ready yet. Please try again shortly.",
        )
    return orchestrator


def get_chatlog_repo(request: Request) -> ChatLogRepository:
    repo = getattr(request.app.state, "chatlog_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Chat logging service not ready.",
        )
    return repo
