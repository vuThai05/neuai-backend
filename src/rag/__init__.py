"""RAG (Retrieval-Augmented Generation) module for document retrieval and context building."""

from .retriever import RAGRetriever, build_context, build_prompt

__all__ = ["RAGRetriever", "build_context", "build_prompt"]

