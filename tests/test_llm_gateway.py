"""Unit tests for the LLM gateway client in stub mode."""

from __future__ import annotations

import httpx

from app.core.config import Settings, _default_model_catalog
from app.services.llm_gateway_client import (
    LLMGatewayClient,
    _extract_error_detail,
    _format_gateway_error,
)
from app.services.ollama_client import OllamaClient


def test_extract_error_detail_openai_shape() -> None:
    """OpenAI-style error JSON should surface the nested message."""
    body = '{"error":{"message":"You exceeded your current quota","type":"insufficient_quota"}}'
    assert _extract_error_detail(body) == "You exceeded your current quota"


def test_extract_error_detail_anthropic_shape() -> None:
    """Anthropic-style error JSON should surface the nested message."""
    body = (
        '{"error":{"code":"invalid_request_error",'
        '"message":"Your credit balance is too low to access the Anthropic API.",'
        '"type":"invalid_request_error"}}'
    )
    assert _extract_error_detail(body) == (
        "Your credit balance is too low to access the Anthropic API."
    )


def test_format_gateway_error_includes_status_and_provider_message() -> None:
    """HTTPStatusError formatting should include status code and provider detail."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "Your credit balance is too low to access the Anthropic API.",
                "type": "invalid_request_error",
            }
        },
    )
    exc = httpx.HTTPStatusError("Bad Request", request=request, response=response)
    formatted = _format_gateway_error(exc)
    assert formatted.startswith("HTTP 400:")
    assert "credit balance is too low" in formatted


async def test_stub_completion_is_successful() -> None:
    """Stub mode should return a successful result with non-empty text."""
    settings = Settings(llm_gateway_stub_mode=True)
    client = LLMGatewayClient(settings)
    result = await client.chat_completion("anthropic", "Plan a trip to Tokyo.")
    assert result.status == "success"
    assert result.text
    assert result.provider_model_name == "claude-3-5-sonnet-20240620"
    assert result.input_tokens > 0
    assert result.output_tokens > 0


async def test_stub_uses_model_catalog_identifiers() -> None:
    """Stub responses should reflect the configured model catalogue."""
    settings = Settings(llm_gateway_stub_mode=True)
    client = LLMGatewayClient(settings)
    result = await client.chat_completion("perplexity", "hello")
    assert result.provider_model_name == "sonar-pro"
    assert result.gateway_model_identifier == "perplexity/sonar-pro"


async def test_qwen3_uses_ollama_stub_path() -> None:
    """qwen3 completions should route through Ollama (stub mode in tests)."""
    settings = Settings(
        ollama_preview_stub_mode=True,
        model_catalog=_default_model_catalog(),
    )
    client = LLMGatewayClient(settings, OllamaClient(settings))
    result = await client.chat_completion("qwen3", "Plan a trip to Tokyo.")
    assert result.status == "success"
    assert result.text == "[STUB Ollama response]"
    assert result.provider_model_name == "qwen3:8b"
    assert result.gateway_model_identifier == "ollama/qwen3:8b"
    assert result.input_tokens > 0
    assert result.output_tokens > 0


async def test_stub_token_estimate_unavailable() -> None:
    """Stub mode cannot estimate tokens via the gateway (returns None)."""
    settings = Settings(llm_gateway_stub_mode=True)
    client = LLMGatewayClient(settings)
    assert await client.estimate_tokens("hello", model="gpt-4o-mini") is None
