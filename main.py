from __future__ import annotations

import os
from typing import Any, List, Optional

# Load environment variables FIRST, before any other imports that need them
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from src.rag import RAGRetriever, build_context, build_prompt
from src.utils.config import get_mongo_client, MONGO_DB_NAME
from src.chatlog import ChatLogRepository

class ChatRequest(BaseModel):
    """Shape of the JSON request body sent from the frontend."""
    question: str
    conversation_id: Optional[str] = None  # For existing conversations


class SourceModel(BaseModel):
    """Shape of a single source document returned to the frontend."""
    text: str
    score: float
    dense_score: float
    link: Optional[str]


class ChatResponse(BaseModel):
    """Shape of the JSON response consumed by the frontend."""
    answer: str
    sources: List[SourceModel]
    conversation_id: str  # Return conversation ID for client tracking


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Please set it in your environment or in the backend `.env` file."
        )

    return genai.Client(
        api_key=api_key,
        )

def init_retriever() -> RAGRetriever:
    _client = get_mongo_client()
    _client[MONGO_DB_NAME].list_collection_names()

    return RAGRetriever(use_hybrid=True)

app = FastAPI(title="RAG + Gemini Chatbot API")
# --- CORS configuration ---

frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")

origins = [
    frontend_url,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily initialized global instances so that startup failures are explicit.
retriever: Optional[RAGRetriever] = None
gemini_client: Optional[genai.Client] = None
chatlog_repo: Optional[ChatLogRepository] = None

@app.on_event("startup")
async def on_startup() -> None:
    """Initialize long-lived resources on application startup."""
    global retriever, gemini_client, chatlog_repo
    try:
        retriever = init_retriever()
        gemini_client = get_gemini_client()
        chatlog_repo = ChatLogRepository()
    except Exception as exc:  # pragma: no cover - startup failure path
        # Log and re-raise so the process fails fast instead of serving broken endpoints
        print(f"Failed to initialize backend services: {exc}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if retriever is None or gemini_client is None or chatlog_repo is None:
        raise HTTPException(
            status_code=503,
            detail="Backend services are not ready yet. Please try again shortly.",
        )

    conversation_id: Optional[str] = None

    try:
        # 0) Create or get conversation
        if payload.conversation_id:
            conversation_id = payload.conversation_id
            # Add user message to existing conversation
            chatlog_repo.add_user_message(conversation_id, payload.question)
        else:
            # Create new conversation with first user message
            chat_log = chatlog_repo.create_conversation(payload.question)
            conversation_id = chat_log.id

        # 1) Retrieve relevant documents from MongoDB-backed knowledge base
        docs = retriever.retrieve(payload.question, top_k=5)
        if not docs:
            context = ""
        else:
            context = build_context(docs)

        # 2) Build prompt and send to Gemini
        prompt = build_prompt(payload.question, context)
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        answer_text = (
            resp.text.strip()
            if getattr(resp, "text", None)
            else "No response from Gemini."
        )

        # 3) Map internal docs into frontend-friendly `sources`
        sources: List[SourceModel] = []
        sources_for_log: List[Dict[str, Any]] = []
        for d in docs:
            src_meta: dict[str, Any] = d.get("source", {}) or {}
            link = src_meta.get("permalink_url")
            sources.append(
                SourceModel(
                    text=d.get("text", ""),
                    score=float(d.get("score", 0.0)),
                    dense_score=float(d.get("dense_score", 0.0)),
                    link=link,
                )
            )
            sources_for_log.append({
                "link": link,
                "text": d.get("text", ""),
                "score": float(d.get("score", 0.0)),
                "dense_score": float(d.get("dense_score", 0.0)),
            })

        # 4) Save assistant response to chat log
        chatlog_repo.add_assistant_message(
            conversation_id,
            answer_text,
            sources_for_log,
        )

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            conversation_id=conversation_id,
        )

    except HTTPException:
        # Re-raise explicit HTTP errors unchanged
        raise
    except Exception as exc:
        # Catch-all for unexpected backend errors
        print(f"/chat endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/conversations")
async def list_conversations() -> dict:
    """Get list of recent conversations."""
    if chatlog_repo is None:
        raise HTTPException(status_code=503, detail="Chat logging service not ready.")
    
    try:
        conversations = chatlog_repo.list_conversations(limit=50)
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
    except Exception as exc:
        print(f"/conversations endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    """Get a specific conversation by ID."""
    if chatlog_repo is None:
        raise HTTPException(status_code=503, detail="Chat logging service not ready.")
    
    try:
        chat_log = chatlog_repo.get_conversation(conversation_id)
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
                    "refs": [
                        {
                            "link": ref.link,
                            "text": ref.text,
                            "score": ref.score,
                            "dense_score": ref.dense_score,
                        }
                        for ref in (msg.refs or [])
                    ] if msg.refs else None,
                }
                for msg in chat_log.messages
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"/conversations/{conversation_id} endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")
