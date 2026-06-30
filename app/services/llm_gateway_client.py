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
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.ollama_client import OllamaClient, OllamaError

logger = get_logger(__name__)
_MAX_ERROR_BODY_CHARS = 2000

TokenCallback = Callable[[str], Awaitable[None]]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object response body.

    Args:
        text: Raw response text.

    Returns:
        A dict when parsing succeeds, otherwise ``None``.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_provider_error_message(body: dict[str, Any]) -> str | None:
    """Extract a human-readable error message from a provider JSON body.

    Args:
        body: Parsed JSON response object.

    Returns:
        The provider message when found, otherwise ``None``.
    """
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = body.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return None


def _extract_error_detail(response_text: str) -> str | None:
    """Extract a concise error detail string from an HTTP response body.

    Args:
        response_text: Raw HTTP response text.

    Returns:
        A provider message or truncated body text, or ``None`` when empty.
    """
    if not response_text.strip():
        return None
    body = _parse_json_object(response_text)
    if body is not None:
        message = _extract_provider_error_message(body)
        if message:
            return message
    stripped = response_text.strip()
    if len(stripped) > _MAX_ERROR_BODY_CHARS:
        return f"{stripped[:_MAX_ERROR_BODY_CHARS]}…"
    return stripped


def _format_gateway_error(exc: Exception, *, response_text: str | None = None) -> str:
    """Format a gateway exception, preferring the provider response body.

    Args:
        exc: The exception raised by the HTTP client.
        response_text: Optional pre-read body text. Required when ``exc.response``
            came from a streaming request and has not been consumed yet.

    Returns:
        A descriptive error string suitable for logs and ``LLMResult.error_message``.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        body_text = response_text if response_text is not None else response.text
        detail = _extract_error_detail(body_text)
        if detail:
            return f"HTTP {response.status_code}: {detail}"
        return str(exc)
    return str(exc)


async def _read_response_text(response: httpx.Response) -> str:
    """Read an httpx response body as text, including unconsumed stream bodies."""
    try:
        content = await response.aread()
    except Exception:
        return ""
    return content.decode("utf-8", errors="replace")


def _error_raw_response(response_text: str) -> dict[str, Any]:
    """Build a persistable raw-response dict from an error body.

    Args:
        response_text: Raw HTTP response text.

    Returns:
        Parsed JSON when available, otherwise a wrapper with truncated text.
    """
    body = _parse_json_object(response_text)
    if body is not None:
        return body
    if not response_text.strip():
        return {}
    stripped = response_text.strip()
    if len(stripped) > _MAX_ERROR_BODY_CHARS:
        stripped = f"{stripped[:_MAX_ERROR_BODY_CHARS]}…"
    return {"errorBody": stripped}


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

    def __init__(self, settings: Settings, ollama: OllamaClient | None = None) -> None:
        """Initialise the client.

        Args:
            settings: Application settings (gateway URL/key, model catalogue).
            ollama: Optional Ollama client for models with ``provider: ollama``.
        """
        self._settings = settings
        self._ollama = ollama

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

        if self._settings.is_ollama_model(model_key):
            return await self._ollama_completion(
                model_key, provider_model_name, gateway_identifier, prompt_text
            )

        # Resolve where to send the request. When a direct provider credential is
        # configured for this model we call the provider's OpenAI-compatible API
        # directly (using its native model name); otherwise we route through the
        # unified gateway (using its gateway identifier).
        direct_target = self._settings.resolve_provider_target(model_cfg)
        if direct_target is not None:
            base_url, api_key = direct_target
            model_id = provider_model_name
        else:
            base_url = self._settings.llm_gateway_base_url or ""
            api_key = self._settings.llm_gateway_api_key
            model_id = gateway_identifier

        # Build an OpenAI-style chat payload.
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self._settings.is_model_stubbed(model_key):
            return self._stub_completion(
                model_key, provider_model_name, gateway_identifier, prompt_text, payload
            )

        return await self._remote_completion(
            model_key, provider_model_name, gateway_identifier, base_url, api_key, payload
        )

    async def chat_completion_stream(
        self,
        model_key: str,
        prompt_text: str,
        *,
        on_token: TokenCallback | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> LLMResult:
        """Execute a streaming chat completion, invoking ``on_token`` for each delta."""
        model_cfg = self._settings.get_model_config(model_key)
        provider_model_name = model_cfg.provider_model_name if model_cfg else model_key
        gateway_identifier = model_cfg.gateway_model_identifier if model_cfg else model_key

        if self._settings.is_ollama_model(model_key):
            return await self._ollama_completion_stream(
                model_key, provider_model_name, gateway_identifier, prompt_text, on_token=on_token
            )

        direct_target = self._settings.resolve_provider_target(model_cfg)
        if direct_target is not None:
            base_url, api_key = direct_target
            model_id = provider_model_name
        else:
            base_url = self._settings.llm_gateway_base_url or ""
            api_key = self._settings.llm_gateway_api_key
            model_id = gateway_identifier

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if self._settings.is_model_stubbed(model_key):
            return await self._stub_completion_stream(
                model_key, provider_model_name, gateway_identifier, prompt_text, payload, on_token
            )

        return await self._remote_completion_stream(
            model_key,
            provider_model_name,
            gateway_identifier,
            base_url,
            api_key,
            payload,
            on_token,
        )

    async def _ollama_completion_stream(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        prompt_text: str,
        *,
        on_token: TokenCallback | None = None,
    ) -> LLMResult:
        """Execute a streaming completion via the local Ollama client."""
        payload: dict[str, Any] = {
            "model": provider_model_name,
            "prompt": prompt_text,
        }
        if self._ollama is None:
            return LLMResult(
                model_key=model_key,
                provider_model_name=provider_model_name,
                gateway_model_identifier=gateway_identifier,
                status="error",
                error_message="Ollama client is not configured",
                prompt_payload=payload,
            )

        start = time.perf_counter()
        parts: list[str] = []
        try:
            async for chunk in self._ollama.generate_stream(prompt_text):
                parts.append(chunk)
                if on_token is not None:
                    await on_token(chunk)
            text = "".join(parts)
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens = max(1, len(prompt_text) // 4)
            output_tokens = max(1, len(text) // 4)
            return LLMResult(
                model_key=model_key,
                provider_model_name=provider_model_name,
                gateway_model_identifier=gateway_identifier,
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                status="success",
                prompt_payload=payload,
                raw_response={"ollama": True, "model": provider_model_name, "stream": True},
            )
        except OllamaError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("Ollama stream completion failed for model '%s': %s", model_key, exc)
            return LLMResult(
                model_key=model_key,
                provider_model_name=provider_model_name,
                gateway_model_identifier=gateway_identifier,
                status="error",
                error_message=str(exc),
                latency_ms=latency_ms,
                prompt_payload=payload,
            )

    async def _stub_completion_stream(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        prompt_text: str,
        payload: dict[str, Any],
        on_token: TokenCallback | None,
    ) -> LLMResult:
        """Stream a deterministic stub response in fixed-size chunks."""
        result = self._stub_completion(
            model_key, provider_model_name, gateway_identifier, prompt_text, payload
        )
        chunk_size = 32
        for index in range(0, len(result.text), chunk_size):
            chunk = result.text[index : index + chunk_size]
            if on_token is not None:
                await on_token(chunk)
        return result

    async def _remote_completion_stream(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        base_url: str,
        api_key: str | None,
        payload: dict[str, Any],
        on_token: TokenCallback | None,
    ) -> LLMResult:
        """Perform a streaming HTTP call with retry/backoff."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self._auth_headers(api_key)
        attempts = self._settings.llm_gateway_max_retries + 1
        last_error: str | None = None
        last_raw_response: dict[str, Any] = {}
        start = time.perf_counter()

        for attempt in range(1, attempts + 1):
            text_parts: list[str] = []
            try:
                async with httpx.AsyncClient(
                    timeout=self._settings.llm_gateway_timeout_seconds
                ) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[len("data:") :].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                text_parts.append(content)
                                if on_token is not None:
                                    await on_token(content)
                            if choices[0].get("finish_reason"):
                                last_raw_response = chunk
                latency_ms = int((time.perf_counter() - start) * 1000)
                text = "".join(text_parts)
                prompt_content = str(payload["messages"][0]["content"])
                input_tokens = max(1, len(prompt_content) // 4)
                output_tokens = max(1, len(text) // 4)
                return LLMResult(
                    model_key=model_key,
                    provider_model_name=provider_model_name,
                    gateway_model_identifier=gateway_identifier,
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    status="success",
                    prompt_payload=payload,
                    raw_response=last_raw_response or {"stream": True},
                )
            except httpx.HTTPStatusError as exc:
                response_text = await _read_response_text(exc.response)
                last_error = _format_gateway_error(exc, response_text=response_text)
                last_raw_response = _error_raw_response(response_text)
                logger.warning(
                    "Gateway stream failed for model '%s' (attempt %d/%d): %s",
                    model_key,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Gateway stream failed for model '%s' (attempt %d/%d): %s",
                    model_key,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
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
            raw_response=last_raw_response,
        )

    async def _ollama_completion(
        self,
        model_key: str,
        provider_model_name: str,
        gateway_identifier: str,
        prompt_text: str,
    ) -> LLMResult:
        """Execute a completion via the local Ollama client.

        Args:
            model_key: Logical model key.
            provider_model_name: Ollama model name from the catalogue.
            gateway_identifier: Gateway-style identifier for audit logs.
            prompt_text: The full prompt text to send.

        Returns:
            A normalised :class:`LLMResult` (errors captured on the result).
        """
        payload: dict[str, Any] = {
            "model": provider_model_name,
            "prompt": prompt_text,
        }
        if self._ollama is None:
            return LLMResult(
                model_key=model_key,
                provider_model_name=provider_model_name,
                gateway_model_identifier=gateway_identifier,
                status="error",
                error_message="Ollama client is not configured",
                prompt_payload=payload,
            )

        start = time.perf_counter()
        try:
            text = await self._ollama.generate(prompt_text)
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens = max(1, len(prompt_text) // 4)
            output_tokens = max(1, len(text) // 4)
            return LLMResult(
                model_key=model_key,
                provider_model_name=provider_model_name,
                gateway_model_identifier=gateway_identifier,
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                status="success",
                prompt_payload=payload,
                raw_response={"ollama": True, "model": provider_model_name},
            )
        except OllamaError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("Ollama completion failed for model '%s': %s", model_key, exc)
            return LLMResult(
                model_key=model_key,
                provider_model_name=provider_model_name,
                gateway_model_identifier=gateway_identifier,
                status="error",
                error_message=str(exc),
                latency_ms=latency_ms,
                prompt_payload=payload,
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
        base_url: str,
        api_key: str | None,
        payload: dict[str, Any],
    ) -> LLMResult:
        """Perform the real HTTP call with retry/backoff.

        Args:
            model_key: Logical model key.
            provider_model_name: Concrete provider model name.
            gateway_identifier: Gateway-specific identifier.
            base_url: Base URL of the target (direct provider or unified gateway).
            api_key: Bearer token for the target, if any.
            payload: The OpenAI-style request payload.

        Returns:
            A normalised :class:`LLMResult` (errors captured on the result).
        """
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self._auth_headers(api_key)
        attempts = self._settings.llm_gateway_max_retries + 1
        last_error: str | None = None
        last_raw_response: dict[str, Any] = {}
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
            except httpx.HTTPStatusError as exc:
                last_error = _format_gateway_error(exc)
                last_raw_response = _error_raw_response(exc.response.text)
                logger.warning(
                    "Gateway call failed for model '%s' (attempt %d/%d): %s",
                    model_key,
                    attempt,
                    attempts,
                    last_error,
                )
                if attempt < attempts:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
            except Exception as exc:  # Network / parsing failures.
                last_error = str(exc)
                logger.warning(
                    "Gateway call failed for model '%s' (attempt %d/%d): %s",
                    model_key,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
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
            raw_response=last_raw_response,
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

    def _auth_headers(self, api_key: str | None = None) -> dict[str, str]:
        """Build authorization headers for an outbound request.

        Args:
            api_key: Bearer token to use. Falls back to the unified gateway key
                when not provided (e.g. for the ``/tokenize`` endpoint).

        Returns:
            A headers mapping including the bearer token when one is available.
        """
        token = api_key if api_key is not None else self._settings.llm_gateway_api_key
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
