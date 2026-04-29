"""Lightweight runtime validation for the crawler -> backend contract.

See `docs/crawler-contract.md` for the canonical description. The goal of
this module is *visibility*, not enforcement: it never raises so that a
malformed payload can't take the backend down. Instead it returns warning
strings that the caller logs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


CONTRACT_VERSION = 1

REQUIRED_PAYLOAD_FIELDS = ("doc_id", "type", "text", "source", "created_at")
RECOMMENDED_SOURCE_FIELDS = ("permalink_url",)
ALLOWED_TYPES = {"post", "comment"}


__all__ = [
    "CONTRACT_VERSION",
    "REQUIRED_PAYLOAD_FIELDS",
    "validate_qdrant_payload",
    "summarize_validation",
]


def validate_qdrant_payload(payload: Mapping[str, Any]) -> List[str]:
    """Return a list of human-readable warnings for one payload."""
    warnings: List[str] = []
    if not isinstance(payload, Mapping):
        return ["payload is not a dict"]

    for field in REQUIRED_PAYLOAD_FIELDS:
        if field not in payload or payload[field] in (None, ""):
            warnings.append(f"missing required field '{field}'")

    payload_type = payload.get("type")
    if payload_type and payload_type not in ALLOWED_TYPES:
        warnings.append(f"unknown type '{payload_type}' (expected one of {sorted(ALLOWED_TYPES)})")

    version = payload.get("version", CONTRACT_VERSION)
    if isinstance(version, int) and version > CONTRACT_VERSION:
        warnings.append(f"payload version {version} > backend version {CONTRACT_VERSION}")

    source = payload.get("source")
    if isinstance(source, Mapping):
        for field in RECOMMENDED_SOURCE_FIELDS:
            if not source.get(field):
                warnings.append(f"missing recommended source field '{field}'")
        if payload_type == "comment":
            if not source.get("post_id"):
                warnings.append("comment payload is missing 'source.post_id'")
            if not source.get("comment_id"):
                warnings.append("comment payload is missing 'source.comment_id'")
        elif payload_type == "post":
            if not source.get("post_id"):
                warnings.append("post payload is missing 'source.post_id'")
    elif source is not None:
        warnings.append("'source' is not a dict")

    return warnings


def summarize_validation(payloads: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-payload warnings into a single summary dict."""
    counts: Dict[str, int] = {}
    sample = 0
    for payload in payloads:
        sample += 1
        for warn in validate_qdrant_payload(payload):
            counts[warn] = counts.get(warn, 0) + 1
    return {
        "samples": sample,
        "warnings": counts,
        "ok": sample > 0 and not counts,
    }
