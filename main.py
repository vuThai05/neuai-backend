"""FastAPI bootstrap for the NeuAI Agent backend.

Stays thin: load env, configure logging, build long-lived services and wire
them into a `ChatOrchestrator`. All HTTP routes live under `src.api.*` and
all business logic lives under `src.orchestration.*`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.chat import router as chat_router
from src.api.health import router as health_router
from src.chatlog import ChatLogRepository
from src.llm import OllamaClient, OllamaConfig, Synthesizer
from src.orchestration import ChatOrchestrator
from src.rag import RAGRetriever
from src.utils.config import MONGO_DB_NAME, get_mongo_client
from src.utils.logging import configure_logging, get_logger
from src.web import build_default_service


configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


def _build_ollama_client() -> OllamaClient:
    base_url = os.environ.get("OLLAMA_BASE_URL")
    model = os.environ.get("OLLAMA_MODEL")

    if not base_url:
        raise RuntimeError(
            "OLLAMA_BASE_URL not found. Please set it in your environment or in the backend `.env` file."
        )
    if not model:
        raise RuntimeError(
            "OLLAMA_MODEL not found. Please set it in your environment or in the backend `.env` file."
        )

    timeout_seconds = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60"))
    return OllamaClient(
        OllamaConfig(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    )


def _init_retriever() -> RAGRetriever:
    # Touch Mongo at startup so we fail fast if credentials are wrong even
    # though retrieval itself reads from Qdrant.
    client = get_mongo_client()
    client[MONGO_DB_NAME].list_collection_names()
    return RAGRetriever(use_hybrid=True)


def create_app() -> FastAPI:
    app = FastAPI(title="NeuAI Agent Backend")

    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(health_router)

    @app.on_event("startup")
    async def on_startup() -> None:  # pragma: no cover - startup wiring
        try:
            retriever = _init_retriever()
            ollama_client = _build_ollama_client()
            chatlog_repo = ChatLogRepository()
            synthesizer = Synthesizer(ollama_client)
            web_service = build_default_service(os.environ)
            orchestrator = ChatOrchestrator(
                retriever=retriever,
                ollama_client=ollama_client,
                chatlog_repo=chatlog_repo,
                synthesizer=synthesizer,
                web_service=web_service,
            )

            app.state.retriever = retriever
            app.state.ollama_client = ollama_client
            app.state.chatlog_repo = chatlog_repo
            app.state.synthesizer = synthesizer
            app.state.web_service = web_service
            app.state.orchestrator = orchestrator

            logger.info(
                "Backend ready: web_enabled=%s top_k=%d",
                web_service is not None,
                orchestrator.top_k,
            )
        except Exception as exc:
            logger.exception("Failed to initialize backend services: %s", exc)
            raise

    return app


app = create_app()
