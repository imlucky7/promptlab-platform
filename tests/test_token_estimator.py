"""Unit tests for the token estimator (local and gateway modes)."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.llm_gateway_client import LLMGatewayClient
from app.services.token_estimator import TokenEstimator


@pytest.fixture
def estimator_local() -> TokenEstimator:
    """Build a token estimator configured for local estimation."""
    settings = Settings(token_estimation_mode="local", llm_gateway_stub_mode=True)
    return TokenEstimator(settings, LLMGatewayClient(settings))


async def test_local_mode_returns_local_estimate(estimator_local: TokenEstimator) -> None:
    """Local mode should return a positive local token count and no gateway."""
    result = await estimator_local.estimate("You are a travel planner.", mode_override="local")
    assert result.effective_mode == "local"
    assert result.from_local is not None
    assert result.from_local.input_tokens > 0
    assert result.from_gateway is None


async def test_gateway_mode_falls_back_to_local_when_stubbed(
    estimator_local: TokenEstimator,
) -> None:
    """Gateway mode should fall back to local when the gateway can't estimate."""
    result = await estimator_local.estimate("Plan a trip to Tokyo.", mode_override="gateway")
    # Stub gateway cannot estimate -> effective mode falls back to local.
    assert result.effective_mode == "local"
    assert result.from_local is not None
    assert result.from_local.input_tokens > 0


async def test_empty_text_is_zero_tokens(estimator_local: TokenEstimator) -> None:
    """Empty input should produce a zero local token count."""
    assert estimator_local.count_local_tokens("") == 0
