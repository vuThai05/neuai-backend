"""LLM-side answer synthesis.

Encapsulates prompt assembly for the three answer modes the orchestrator
supports (internal-only, internal+web, no evidence). Prompt templates live as
plain text files under `prompts/` so non-developers can iterate on wording
without touching Python.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.rag.context_builder import build_combined_context, build_internal_context

from .ollama_client import OllamaClient


_PROMPTS_DIR = Path(__file__).parent / "prompts"


__all__ = [
    "Synthesizer",
    "build_internal_prompt",
    "build_web_prompt",
    "load_no_evidence_message",
]


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def build_internal_prompt(question: str, context: str) -> str:
    """Render the internal-only RAG prompt."""
    template = _load_template("rag_answer_prompt.txt")
    return template.format(question=question, context=context)


def build_web_prompt(question: str, context: str) -> str:
    """Render the prompt that mixes internal + web evidence."""
    template = _load_template("rag_web_answer_prompt.txt")
    return template.format(question=question, context=context)


def load_no_evidence_message() -> str:
    """Static message returned when neither internal nor web evidence is good enough."""
    return _load_template("no_evidence_prompt.txt").strip()


class Synthesizer:
    """Single entry point for assembling prompts and calling the LLM."""

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.ollama_client = ollama_client

    def synthesize_internal(self, question: str, internal_docs: List[Dict[str, Any]]) -> str:
        if not internal_docs:
            return self.synthesize_no_evidence(question)
        context = build_internal_context(internal_docs)
        prompt = build_internal_prompt(question=question, context=context)
        return self.ollama_client.generate(prompt)

    def synthesize_with_web(
        self,
        question: str,
        internal_docs: List[Dict[str, Any]],
        web_results: Iterable[Any],
    ) -> str:
        web_list = list(web_results)
        if not internal_docs and not web_list:
            return self.synthesize_no_evidence(question)
        context = build_combined_context(internal_docs, web_list)
        prompt = build_web_prompt(question=question, context=context)
        return self.ollama_client.generate(prompt)

    def synthesize_no_evidence(self, question: str) -> str:  # noqa: ARG002 - kept for API symmetry
        return load_no_evidence_message()
