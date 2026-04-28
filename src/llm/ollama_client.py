"""Ollama HTTP client utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


@dataclass(frozen=True)
class OllamaConfig:
    """Runtime configuration for calling Ollama API."""

    base_url: str
    model: str
    timeout_seconds: int = 60


class OllamaClient:
    """Simple wrapper around Ollama's /api/generate endpoint."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    @property
    def model(self) -> str:
        return self._config.model

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
        }

        endpoint = self._config.base_url.rstrip("/") + "/api/generate"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON response.") from exc

        text = str(result.get("response", "")).strip()
        if text:
            return text

        # Fallback for unexpected payloads.
        return body.strip() or "No response from Ollama."
