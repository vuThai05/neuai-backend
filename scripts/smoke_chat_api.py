"""Smoke test the deployed `/chat` endpoint with one or two canned questions.

Usage:
    python scripts/smoke_chat_api.py [--base-url http://localhost:8000]

Exits non-zero if `/health` is degraded or `/chat` doesn't return a
well-formed payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import httpx


DEFAULT_QUESTIONS = [
    "Truong NEU dao tao nganh marketing khong?",
    "Lich thi cuoi ky moi nhat 2025?",  # should trigger web fallback
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        health = client.get("/health")
        print("health:", health.status_code, health.json())
        if health.status_code != 200 or health.json().get("status") != "ok":
            print("health is not ok, aborting", file=sys.stderr)
            return 1

        for question in DEFAULT_QUESTIONS:
            print("\n>>>", question)
            resp = client.post("/chat", json={"question": question})
            print("status:", resp.status_code)
            if resp.status_code != 200:
                print("ERROR body:", resp.text)
                return 2
            body: Dict[str, Any] = resp.json()
            print(json.dumps({
                "answer": body["answer"][:280],
                "decision": body.get("decision"),
                "sources": [
                    {"type": s["type"], "title": s["title"][:80], "score": s.get("score")}
                    for s in body.get("sources", [])
                ],
            }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
