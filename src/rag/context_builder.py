"""Builders that turn raw retrieval results into prompt-ready context strings.

The orchestrator should never assemble prompts directly from doc dicts; it
calls into one of these builders so the formatting (and the implicit ordering
of internal vs. web evidence) lives in a single place.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


__all__ = [
    "build_internal_context",
    "build_combined_context",
    "build_post_with_comments_context",
]


def build_internal_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format internal retriever output for the LLM prompt."""
    parts: List[str] = []
    for i, d in enumerate(retrieved_docs, start=1):
        meta = d.get("source") or {}
        link = meta.get("permalink_url") or ""
        dense_score = d.get("dense_score")
        score = float(d.get("score", 0.0))
        text = d.get("text", "")

        if dense_score is not None:
            parts.append(
                f"[DOC {i}] final_score={score:.3f} (dense={float(dense_score):.3f})\n"
                f"text: {text}\n"
                f"source: {link}\n"
            )
        else:
            parts.append(
                f"[DOC {i}] score={score:.3f}\n"
                f"text: {text}\n"
                f"source: {link}\n"
            )
    return "\n\n".join(parts)


def build_combined_context(
    internal_docs: List[Dict[str, Any]],
    web_results: Iterable[Any],
) -> str:
    """Render internal docs and web results in two clearly separated blocks.

    `web_results` is expected to be an iterable of objects/dicts exposing
    `title`, `url`, and `snippet` (matches both `src.web.client.WebResult`
    and the `Source` schema with `type="web"`).
    """
    blocks: List[str] = []

    if internal_docs:
        blocks.append("=== NOI BO ===\n" + build_internal_context(internal_docs))

    web_list = list(web_results)
    if web_list:
        web_parts: List[str] = []
        for i, r in enumerate(web_list, start=1):
            title = _attr(r, "title", "(khong co tieu de)")
            url = _attr(r, "url", "")
            snippet = _attr(r, "snippet", "")
            web_parts.append(
                f"[WEB {i}] title: {title}\n"
                f"url: {url}\n"
                f"snippet: {snippet}\n"
            )
        blocks.append("=== WEB ===\n" + "\n\n".join(web_parts))

    return "\n\n".join(blocks)


def build_post_with_comments_context(
    doc: Dict[str, Any],
    comments_by_post: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    """Format a single post with its comment thread (used by chat_cli)."""
    meta = doc.get("source") or {}
    link = meta.get("permalink_url") or ""
    post_id = meta.get("post_id")

    parts: List[str] = [
        "=== BAI VIET ===",
        f"text: {doc.get('text', '')}",
        f"source: {link}",
    ]

    if comments_by_post and post_id:
        comments = comments_by_post.get(post_id, [])
        if comments:
            parts.append("\n=== COMMENTS ===")
            for i, cmt in enumerate(comments, start=1):
                comment_text = (cmt.get("text") or "").strip()
                if comment_text and comment_text != "[NO_MESSAGE]":
                    parts.append(f"Comment {i}: {comment_text}")

    return "\n".join(parts)


def _attr(obj: Any, name: str, default: str) -> str:
    if isinstance(obj, dict):
        value = obj.get(name)
    else:
        value = getattr(obj, name, None)
    if value is None or value == "":
        return default
    return str(value)
