"""HTTP routes for chat-related endpoints (`/chat`, `/conversations`, `/conversations/{id}`)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.chatlog import ChatLogRepository
from src.orchestration.chat_orchestrator import ChatOrchestrator
from src.orchestration.schemas import ChatRequest, ChatResponse

from .dependencies import get_chatlog_repo, get_orchestrator

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        return orchestrator.run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        print(f"/chat endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@router.get("/conversations")
async def list_conversations(
    repo: ChatLogRepository = Depends(get_chatlog_repo),
) -> Dict[str, Any]:
    try:
        conversations = repo.list_conversations(limit=50)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        print(f"/conversations endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return {
        "conversations": [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": len(conv.messages),
            }
            for conv in conversations
        ]
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    repo: ChatLogRepository = Depends(get_chatlog_repo),
) -> Dict[str, Any]:
    try:
        chat_log = repo.get_conversation(conversation_id)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        print(f"/conversations/{conversation_id} endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    if not chat_log:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "id": chat_log.id,
        "title": chat_log.title,
        "created_at": chat_log.created_at,
        "updated_at": chat_log.updated_at,
        "messages": [
            {
                "id": msg.id,
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "decision": _decision_to_dict(msg.decision),
                "refs": _refs_to_list(msg.refs),
            }
            for msg in chat_log.messages
        ],
    }


def _decision_to_dict(decision: Optional[Any]) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    return {
        "route": decision.route,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "confidence": decision.confidence,
    }


def _refs_to_list(refs: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    if not refs:
        return None
    return [
        {
            "type": ref.type,
            "title": ref.title,
            "url": ref.url,
            "snippet": ref.snippet,
            "score": ref.score,
        }
        for ref in refs
    ]
