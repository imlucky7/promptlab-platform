"""HTTP client for the local Ollama ``/api/generate`` endpoint."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaError(Exception):
    """Raised when an Ollama request fails."""


class OllamaClient:
    """Calls Ollama's generate API (non-streaming)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.ollama_timeout_seconds
        self._stub_mode = settings.ollama_preview_stub_mode

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: dict[str, Any] | str | None = None,
    ) -> str:
        """Generate text via ``POST /api/generate``.

        Args:
            prompt: User prompt text.
            system: Optional system prompt.
            response_format: Optional Ollama ``format`` (``"json"`` or JSON schema).

        Returns:
            The model's ``response`` field text.

        Raises:
            OllamaError: On network, HTTP, or parse failures.
        """
        if self._stub_mode:
            return self._stub_response(prompt, response_format)

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if response_format is not None:
            payload["format"] = response_format

        url = f"{self._base_url}/api/generate"
        start = time.perf_counter()
        logger.info("Ollama API request url=%s payload=%s", url, json.dumps(payload))
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            detail = exc.response.text.strip() or str(exc)
            logger.warning(
                "Ollama HTTP error url=%s latency_ms=%s payload=%s response=%s",
                url,
                latency_ms,
                json.dumps(payload),
                detail[:500],
            )
            raise OllamaError(f"Ollama returned HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "Ollama request failed url=%s latency_ms=%s payload=%s error=%s",
                url,
                latency_ms,
                json.dumps(payload),
                exc,
            )
            raise OllamaError(
                f"Could not reach Ollama at {self._base_url}. "
                "Ensure Ollama is running and the model is pulled."
            ) from exc

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "Ollama non-JSON response url=%s latency_ms=%s payload=%s response=%s",
                url,
                latency_ms,
                json.dumps(payload),
                response.text[:500],
            )
            raise OllamaError("Ollama returned a non-JSON response.") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "Ollama API response url=%s latency_ms=%s body=%s",
            url,
            latency_ms,
            json.dumps(body),
        )

        text = body.get("response")
        if not isinstance(text, str):
            raise OllamaError("Ollama response missing the 'response' field.")
        return text

    def _stub_response(
        self, prompt: str, response_format: dict[str, Any] | str | None
    ) -> str:
        """Return deterministic stub output for offline tests."""
        if response_format is not None:
            return json.dumps(
                {
                    "optimizedPrompt": (
                        "You are an expert travel planner.\n\n"
                        "Plan a detailed itinerary using the trip details provided.\n\n"
                        "[STUB optimized prompt]"
                    ),
                    "suggestions": [
                        {
                            "suggestionType": "fill_missing_detail",
                            "description": "Add dietary restrictions or mobility needs.",
                            "field": "constraints",
                        }
                    ],
                }
            )
        if "trip context" in prompt.lower() or "destination" in prompt.lower():
            return (
                "You are an expert travel planner.\n\n"
                "Create a day-by-day itinerary for the trip described below.\n\n"
                "[STUB generated prompt from trip context]"
            )
        return "[STUB Ollama response]"
