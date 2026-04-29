"""Lightweight timing helpers used to attach latency to log events."""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    """Context manager that records elapsed wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.elapsed_ms = int((time.monotonic() - self._start) * 1000)


class StepTimers:
    """Container for per-step latencies surfaced in `chat_completed` log events."""

    def __init__(self) -> None:
        self.retrieve_ms: int = 0
        self.web_ms: int = 0
        self.llm_ms: int = 0
        self.total_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "retrieve_ms": self.retrieve_ms,
            "web_ms": self.web_ms,
            "llm_ms": self.llm_ms,
            "total_ms": self.total_ms,
        }
