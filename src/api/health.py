"""Health check route used by infra/monitoring."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    components = {
        "retriever": getattr(state, "retriever", None) is not None,
        "ollama": getattr(state, "ollama_client", None) is not None,
        "chatlog": getattr(state, "chatlog_repo", None) is not None,
        "orchestrator": getattr(state, "orchestrator", None) is not None,
        "synthesizer": getattr(state, "synthesizer", None) is not None,
        "web_service": getattr(state, "web_service", None) is not None,
    }
    # web_service is optional; treat its absence as fine, not degraded.
    required = {k: v for k, v in components.items() if k != "web_service"}
    return {
        "status": "ok" if all(required.values()) else "degraded",
        "components": components,
    }
