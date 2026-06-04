"""Unit tests for the LLM gateway client in stub mode."""

from __future__ import annotations

from app.core.config import Settings
from app.services.llm_gateway_client import LLMGatewayClient


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


async def test_stub_token_estimate_unavailable() -> None:
    """Stub mode cannot estimate tokens via the gateway (returns None)."""
    settings = Settings(llm_gateway_stub_mode=True)
    client = LLMGatewayClient(settings)
    assert await client.estimate_tokens("hello", model="gpt-4o-mini") is None
