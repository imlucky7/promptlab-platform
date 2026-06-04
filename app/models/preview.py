"""Schemas for the dedicated ``/preview`` endpoint.

Preview builds the prompt text, estimates tokens and produces suggestions
*without* persisting anything.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.models.common import CamelModel
from app.models.prompt_suggestions import SuggestionItem

# Per-request token estimation override; ``"default"`` defers to configuration.
TokenEstimationModeRequest = Literal["default", "gateway", "local"]


class PreviewRequest(CamelModel):
    """Request body for ``POST /preview``.

    Attributes:
        use_case_key: Use-case key (e.g. ``"travel"``).
        structured_inputs: Structured inputs to render into the template.
        token_estimation_mode: Estimation override (``default``/``gateway``/``local``).
        models: Optional model keys (reserved for future gateway estimation).
    """

    use_case_key: str = Field(default="travel", examples=["travel"])
    structured_inputs: dict[str, Any] = Field(default_factory=dict)
    token_estimation_mode: TokenEstimationModeRequest = "default"
    models: list[str] = Field(default_factory=list)


class GatewayTokenEstimate(CamelModel):
    """Gateway-derived token estimate.

    Attributes:
        input_tokens: Estimated input tokens from the gateway.
        estimation_available: Whether the gateway provided an estimate.
    """

    input_tokens: int
    estimation_available: bool


class LocalTokenEstimate(CamelModel):
    """Local (tiktoken) token estimate.

    Attributes:
        input_tokens: Estimated input tokens from the local tokenizer.
    """

    input_tokens: int


class TokenEstimates(CamelModel):
    """Combined token estimation result.

    Attributes:
        effective_mode: The mode actually used (``"gateway"`` or ``"local"``).
        from_gateway: Gateway estimate, when computed.
        from_local: Local estimate, when computed.
    """

    effective_mode: Literal["gateway", "local"]
    from_gateway: GatewayTokenEstimate | None = None
    from_local: LocalTokenEstimate | None = None


class PreviewResponse(CamelModel):
    """Response body for ``POST /preview``.

    Attributes:
        use_case_key: Echoes the requested use case.
        structured_inputs: Echoes the requested inputs.
        prompt_text: The assembled prompt text.
        token_estimates: Token estimation result.
        suggestions: Rule-based improvement suggestions.
    """

    use_case_key: str
    structured_inputs: dict[str, Any]
    prompt_text: str
    token_estimates: TokenEstimates
    suggestions: list[SuggestionItem] = Field(default_factory=list)
