"""Lexical reranker for web results.

P1 keeps this intentionally simple: combine original rank position with token
overlap between question and result text. A heavier reranker (e.g. BGE
cross-encoder) can replace this without touching the orchestrator.
"""

from __future__ import annotations

from typing import List

from src.rag.scoring import tokens

from .client import WebResult


def rerank_lexical(question: str, results: List[WebResult]) -> List[WebResult]:
    if not results:
        return []

    q_tokens = tokens(question)
    q_size = len(q_tokens) or 1
    rescored: List[WebResult] = []
    for i, r in enumerate(results):
        position_score = 1.0 / (i + 1)
        overlap = len(q_tokens.intersection(tokens(f"{r.title} {r.snippet}"))) if q_tokens else 0
        # Overlap is weighted higher than raw position so that a result far down
        # the SERP can still win if it strongly matches the query terms.
        score = round(0.3 * position_score + 0.7 * (overlap / q_size), 4)
        rescored.append(
            WebResult(
                title=r.title,
                url=r.url,
                snippet=r.snippet,
                score=score,
            )
        )

    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored
