"""Web search client and orchestrator-facing service.

Exposes:
- `WebResult` data class.
- `WebSearchProvider` protocol that any concrete provider must satisfy.
- `DuckDuckGoProvider` that wraps `duckduckgo_search.DDGS` and enforces a
  per-query timeout via a worker thread.
- `WebSearchService` that builds queries, calls the provider within a total
  time budget, deduplicates and reranks the results.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple


logger = logging.getLogger(__name__)


DEFAULT_MAX_QUERIES = 2
DEFAULT_MAX_RESULTS_PER_QUERY = 5
DEFAULT_WEB_TIMEOUT_MS = 2500
DEFAULT_TOTAL_BUDGET_MS = 4000


@dataclass
class WebResult:
    """Normalized web search result."""

    title: str
    url: str
    snippet: str
    score: float = 0.0


@dataclass
class WebMetrics:
    """Per-request web search metrics surfaced to observability."""

    web_calls: int = 0
    web_latency_ms: int = 0
    timeouts: int = 0
    errors: int = 0


class WebSearchProvider(Protocol):
    def search(self, query: str, max_results: int, timeout_ms: int) -> List[WebResult]:
        ...


class DuckDuckGoProvider:
    """DuckDuckGo provider via the `duckduckgo-search` package.

    DDG is rate-limited and can sporadically fail with parsing errors; the
    service layer handles retries/fallback so this class can stay thin.
    """

    def __init__(self, region: str = "vn-vi", safesearch: str = "moderate") -> None:
        self.region = region
        self.safesearch = safesearch

    def search(self, query: str, max_results: int, timeout_ms: int) -> List[WebResult]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._raw_search, query, max_results)
            try:
                raw = future.result(timeout=timeout_ms / 1000.0)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"DuckDuckGo search timed out after {timeout_ms}ms for query={query!r}"
                ) from exc

        results: List[WebResult] = []
        for item in raw or []:
            results.append(
                WebResult(
                    title=str(item.get("title") or "").strip(),
                    url=str(item.get("href") or item.get("url") or "").strip(),
                    snippet=str(item.get("body") or item.get("snippet") or "").strip(),
                )
            )
        return results

    def _raw_search(self, query: str, max_results: int) -> List[Any]:
        from duckduckgo_search import DDGS  # imported lazily to keep startup cheap

        with DDGS() as ddgs:
            return list(
                ddgs.text(
                    query,
                    max_results=max_results,
                    region=self.region,
                    safesearch=self.safesearch,
                )
            )


class WebSearchService:
    """Compose query building, provider calls and reranking under a time budget."""

    def __init__(
        self,
        provider: WebSearchProvider,
        max_queries: int = DEFAULT_MAX_QUERIES,
        max_results_per_query: int = DEFAULT_MAX_RESULTS_PER_QUERY,
        web_timeout_ms: int = DEFAULT_WEB_TIMEOUT_MS,
        total_budget_ms: int = DEFAULT_TOTAL_BUDGET_MS,
    ) -> None:
        self.provider = provider
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query
        self.web_timeout_ms = web_timeout_ms
        self.total_budget_ms = total_budget_ms

    def search_for_question(
        self,
        question: str,
        decision: Any,
    ) -> Tuple[List[WebResult], WebMetrics]:
        """Run the full web-fallback pipeline and return results + metrics."""
        from .query_builder import build_web_queries
        from .reranker import rerank_lexical
        from .result_parser import dedupe_by_url

        reason_code = getattr(decision, "reason_code", None)
        queries = build_web_queries(question, reason_code)[: self.max_queries]

        metrics = WebMetrics()
        start = time.monotonic()
        gathered: List[WebResult] = []

        for query in queries:
            elapsed_ms = (time.monotonic() - start) * 1000
            remaining_ms = self.total_budget_ms - elapsed_ms
            if remaining_ms <= 0:
                break

            per_query_timeout = int(min(self.web_timeout_ms, remaining_ms))
            if per_query_timeout < 200:
                # Not enough budget left for a meaningful call.
                break

            try:
                results = self.provider.search(
                    query,
                    max_results=self.max_results_per_query,
                    timeout_ms=per_query_timeout,
                )
                metrics.web_calls += 1
                gathered.extend(results)
            except TimeoutError as exc:
                metrics.timeouts += 1
                logger.warning("Web search timeout: %s", exc)
            except Exception as exc:  # noqa: BLE001 - DDG can raise many shapes
                metrics.errors += 1
                logger.warning("Web search error for %r: %s", query, exc)

        metrics.web_latency_ms = int((time.monotonic() - start) * 1000)
        deduped = dedupe_by_url(gathered)
        ranked = rerank_lexical(question, deduped)
        return ranked, metrics


def build_default_service(env: Optional[dict] = None) -> Optional[WebSearchService]:
    """Construct a `WebSearchService` from environment variables.

    Returns None when `WEB_ENABLED=false`, so callers can skip the entire
    web layer without conditional logic in their own code.
    """
    env = env if env is not None else os.environ

    if str(env.get("WEB_ENABLED", "true")).lower() in {"false", "0", "no"}:
        return None

    provider = DuckDuckGoProvider(
        region=env.get("WEB_REGION", "vn-vi") or "vn-vi",
    )
    return WebSearchService(
        provider=provider,
        max_queries=int(env.get("WEB_MAX_QUERIES", DEFAULT_MAX_QUERIES)),
        max_results_per_query=int(env.get("WEB_MAX_RESULTS_PER_QUERY", DEFAULT_MAX_RESULTS_PER_QUERY)),
        web_timeout_ms=int(env.get("WEB_TIMEOUT_MS", DEFAULT_WEB_TIMEOUT_MS)),
        total_budget_ms=int(env.get("WEB_TOTAL_BUDGET_MS", DEFAULT_TOTAL_BUDGET_MS)),
    )
