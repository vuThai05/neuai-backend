"""RAG (Retrieval-Augmented Generation) module: retrieval + context shaping."""

from .context_builder import (
    build_combined_context,
    build_internal_context,
    build_post_with_comments_context,
)
from .retriever import RAGRetriever, build_context, build_prompt, build_single_context
from .scoring import (
    compute_signals,
    detect_freshness,
    has_direct_match,
    to_unit,
    tokens,
)

__all__ = [
    "RAGRetriever",
    "build_context",
    "build_prompt",
    "build_single_context",
    "build_combined_context",
    "build_internal_context",
    "build_post_with_comments_context",
    "compute_signals",
    "detect_freshness",
    "has_direct_match",
    "to_unit",
    "tokens",
]
