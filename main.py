from __future__ import annotations

import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

from src.rag import RAGRetriever, build_context, build_prompt
from src.utils.config import get_mongo_client, MONGO_DB_NAME


# Load environment variables from `.env` for local development.
# In production, you typically rely on real environment variables instead.
load_dotenv()


class ChatRequest(BaseModel):
    """Shape of the JSON request body sent from the frontend."""
    question: str


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

@app.on_event("startup")
async def on_startup() -> None:
    """Initialize long-lived resources on application startup."""
    global retriever, gemini_client
    try:
        retriever = init_retriever()
        gemini_client = get_gemini_client()
    except Exception as exc:  # pragma: no cover - startup failure path
        # Log and re-raise so the process fails fast instead of serving broken endpoints
        print(f"Failed to initialize backend services: {exc}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint consumed by the Next.js frontend.
    Request:
        POST /chat
        {
          "question": "Your question here"
        }
    Response:
        {
          "answer": "...",
          "sources": [
            {
              "text": "...",
              "score": 0.9,
              "dense_score": 0.8,
              "link": "https://..."
            },
            ...
          ]
        }
    """

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if retriever is None or gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Backend services are not ready yet. Please try again shortly.",
        )

    try:
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

        return ChatResponse(answer=answer_text, sources=sources)

    except HTTPException:
        # Re-raise explicit HTTP errors unchanged
        raise
    except Exception as exc:
        # Catch-all for unexpected backend errors
        print(f"/chat endpoint error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error.")

