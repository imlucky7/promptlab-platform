"""Client for the external LLM gateway (LiteLLM or equivalent).

The gateway exposes an OpenAI-compatible ``/chat/completions`` API over multiple
providers. This client:

* maps a logical ``model_key`` to provider/gateway identifiers via configuration,
* performs the async HTTP call with basic retry/backoff,
* normalises results into an internal :class:`LLMResult`,
* supports a deterministic **stub mode** so the backend runs fully offline for
  local development (no API keys required).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResult:
    """Normalised result of a single model invocation.

    Attributes:
        model_key: Logical model key requested.
        provider_model_name: Concrete provider model name.
        gateway_model_identifier: Gateway-specific model identifier.
        text: Assistant text content.
        input_tokens: Prompt tokens reported by the gateway (0 if unknown).
        output_tokens: Completion tokens reported by the gateway (0 if unknown).
        latency_ms: Measured call latency in milliseconds.
        status: ``"success"`` or ``"error"``.
        error_message: Error description when ``status == "error"``.
        prompt_payload: The payload sent to the gateway.
        raw_response: The raw response body from the gateway.
    """

    model_key: str
    provider_model_name: str
    gateway_model_identifier: str
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    status: str = "success"
    error_message: str | None = None
    prompt_payload: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class LLMGatewayClient:
    """Async client wrapping the LLM gateway with retry and stub support."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the client.

        Args:
            settings: Application settings (gateway URL/key, model catalogue).
        """
        self._settings = settings

    async def chat_completion(
        self,
        model_key: str,
        prompt_text: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> LLMResult:
        """Execute a chat completion for a single logical model.

        Args:
            model_key: Logical model key (e.g. ``"anthropic"``).
            prompt_text: The full prompt text to send.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.

        Returns:
            A normalised :class:`LLMResult`. Errors are captured on the result
            (``status == "error"``) rather than raised, so one provider's
            failure never blocks the others.
        """
        model_cfg = self._settings.get_model_config(model_key)
        provider_model_name = model_cfg.provider_model_name if model_cfg else model_key
        gateway_identifier = model_cfg.gateway_model_identifier if model_cfg else model_key

        # Build an OpenAI-style chat payload.
        payload: dict[str, Any] = {
            "model": gateway_identifier,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self._settings.is_gateway_stubbed:
            return self._stub_completion(
                model_key, provider_model_name, gateway_identifier, prompt_text, payload
            )

        return await self._remote_completion(
            model_key, provider_model_name, gateway_identifier, payload
        )

    async def estimate_tokens(self, text: str, *, model: str) -> int | None:
        """Estimate input tokens for ``text`` via the gateway when supported.

        Args:
            text: The text to estimate.
            model: Gateway model name to use for estimation.

        Returns:
            The estimated token count, or ``None`` when the gateway cannot
            provide an estimate (e.g. stub mode or unsupported endpoint).
        """
        if self._settings.is_gateway_stubbed:
            # Stub mode cannot truly estimate; signal "unavailable" so callers
            # fall back to the local tokenizer.
            return None

        url = f"{self._settings.llm_gateway_base_url.rstrip('/')}/tokenize"  # type: ignore[union-attr]
        headers = self._auth_headers()
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_gateway_timeout_seconds) as client:
                resp = await client.post(url, json={"model": model, "input": text}, headers=headers)
                resp.raise_for_status()
                body = resp.json()
            # LiteLLM-style responses commonly expose token counts here.
            for key in ("input_tokens", "n_tokens", "tokens"):
                if isinstance(body.get(key), int):
                    return int(body[key])
            return None
        except Exception as exc:
            logger.warning("Gateway /tokenize call failed: %s", exc)
            return None

    async def _remote_completion(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        payload: dict[str, Any],
    ) -> LLMResult:
        """Perform the real HTTP call with retry/backoff.

        Args:
            model_key: Logical model key.
            provider_model_name: Concrete provider model name.
            gateway_identifier: Gateway-specific identifier.
            payload: The OpenAI-style request payload.

        Returns:
            A normalised :class:`LLMResult` (errors captured on the result).
        """
        url = f"{self._settings.llm_gateway_base_url.rstrip('/')}/chat/completions"  # type: ignore[union-attr]
        headers = self._auth_headers()
        attempts = self._settings.llm_gateway_max_retries + 1
        last_error: str | None = None
        start = time.perf_counter()

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self._settings.llm_gateway_timeout_seconds
                ) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    body = resp.json()
                latency_ms = int((time.perf_counter() - start) * 1000)
                return self._parse_openai_response(
                    model_key, provider_model_name, gateway_identifier, payload, body, latency_ms
                )
            except Exception as exc:  # Network / HTTP / parsing failures.
                last_error = str(exc)
                logger.warning(
                    "Gateway call failed for model '%s' (attempt %d/%d): %s",
                    model_key,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    # Exponential backoff: 0.5s, 1s, 2s, ...
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResult(
            model_key=model_key,
            provider_model_name=provider_model_name,
            gateway_model_identifier=gateway_identifier,
            status="error",
            error_message=last_error or "Unknown gateway error",
            latency_ms=latency_ms,
            prompt_payload=payload,
        )

    def _parse_openai_response(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        payload: dict[str, Any],
        body: dict[str, Any],
        latency_ms: int,
    ) -> LLMResult:
        """Parse an OpenAI-style response body into an :class:`LLMResult`.

        Args:
            model_key: Logical model key.
            provider_model_name: Concrete provider model name.
            gateway_identifier: Gateway-specific identifier.
            payload: The request payload (stored for audit).
            body: The response JSON body.
            latency_ms: Measured latency.

        Returns:
            The normalised result.
        """
        choices = body.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
        usage = body.get("usage") or {}
        return LLMResult(
            model_key=model_key,
            provider_model_name=provider_model_name,
            gateway_model_identifier=gateway_identifier,
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            status="success",
            prompt_payload=payload,
            raw_response=body,
        )

    def _stub_completion(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        prompt_text: str,
        payload: dict[str, Any],
    ) -> LLMResult:
        """Produce a deterministic stub response for offline development.

        Args:
            model_key: Logical model key.
            provider_model_name: Concrete provider model name.
            gateway_identifier: Gateway-specific identifier.
            prompt_text: The prompt text (used to size the stub usage numbers).
            payload: The request payload (stored for audit).

        Returns:
            A successful :class:`LLMResult` containing canned text.
        """
        text = (
            f"[STUB:{model_key}] This is a deterministic stub response from "
            f"'{provider_model_name}'. Configure LLM_GATEWAY_* and disable "
            f"LLM_GATEWAY_STUB_MODE to call a real provider.\n\n"
            "Trip overview: A sample multi-day itinerary would appear here.\n"
            "Day 1 - Arrival: morning, afternoon, evening highlights.\n"
            "Notes: budget tips and practical advice."
        )
        # Approximate usage so downstream metrics have realistic-looking values.
        input_tokens = max(1, len(prompt_text) // 4)
        output_tokens = max(1, len(text) // 4)
        return LLMResult(
            model_key=model_key,
            provider_model_name=provider_model_name,
            gateway_model_identifier=gateway_identifier,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=5,
            status="success",
            prompt_payload=payload,
            raw_response={"stub": True, "model": gateway_identifier},
        )

    def _auth_headers(self) -> dict[str, str]:
        """Build authorization headers for gateway requests.

        Returns:
            A headers mapping including the bearer token when configured.
        """
        headers = {"Content-Type": "application/json"}
        if self._settings.llm_gateway_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_gateway_api_key}"
        return headers
