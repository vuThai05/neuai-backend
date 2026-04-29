"""Web search fallback layer used by the orchestrator when retrieval is weak.

The default provider is DuckDuckGo (`DuckDuckGoProvider`), but
`WebSearchService` accepts any provider that conforms to
`WebSearchProvider` so we can swap it for Tavily/Serper later without
touching orchestration.
"""

from .client import (
    DEFAULT_MAX_QUERIES,
    DEFAULT_MAX_RESULTS_PER_QUERY,
    DEFAULT_TOTAL_BUDGET_MS,
    DEFAULT_WEB_TIMEOUT_MS,
    DuckDuckGoProvider,
    WebMetrics,
    WebResult,
    WebSearchProvider,
    WebSearchService,
    build_default_service,
)

__all__ = [
    "DEFAULT_MAX_QUERIES",
    "DEFAULT_MAX_RESULTS_PER_QUERY",
    "DEFAULT_TOTAL_BUDGET_MS",
    "DEFAULT_WEB_TIMEOUT_MS",
    "DuckDuckGoProvider",
    "WebMetrics",
    "WebResult",
    "WebSearchProvider",
    "WebSearchService",
    "build_default_service",
]
