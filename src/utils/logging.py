"""Structured logging helpers.

`configure_logging` is idempotent and should be called once at app startup.
`log_event` emits a single JSON line so the output can be tailed/grepped or
shipped to a log collector without extra parsing.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit plain text on stdout.

    Calling this multiple times is safe; only the first call installs handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    handler.setFormatter(formatter)
    root.handlers = [handler]
    root.setLevel(level.upper() if isinstance(level, str) else level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """Emit a single structured JSON event line at INFO level.

    The payload is a JSON dict with keys `event`, `ts`, plus the supplied
    fields. Non-serializable values fall back to `repr()`.
    """
    payload = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    try:
        line = json.dumps(payload, ensure_ascii=False, default=repr)
    except Exception:  # pragma: no cover - defensive
        line = json.dumps({"event": event, "error": "serialization_failed"})
    logger.info(line)
