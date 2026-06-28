"""Schemas for the ``responses`` collection (per-model run outputs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.models.budget import BudgetBreakdown
from app.models.common import CamelModel, TimestampedModel

ResponseStatus = Literal["success", "error"]


class UsageInfo(CamelModel):
    """Token/latency usage metadata for a single response.

    Attributes:
        input_tokens: Number of prompt (input) tokens.
        output_tokens: Number of completion (output) tokens.
        latency_ms: End-to-end gateway call latency in milliseconds.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class ResponseBase(CamelModel):
    """Fields shared by response read/update operations.

    Attributes:
        run_id: Reference to the owning run.
        model_key: Logical model key (e.g. ``"anthropic"``).
        provider_model_name: Concrete provider model name.
        gateway_model_identifier: Gateway-specific identifier.
        prompt_payload: Payload sent to the gateway (may be truncated).
        raw_response: Full gateway response (may be truncated).
        text: Final assistant text to display.
        usage: Token/latency usage metadata.
        status: ``"success"`` or ``"error"``.
        error_message: Error description when ``status == "error"``.
    """

    run_id: str
    model_key: str
    provider_model_name: str = ""
    gateway_model_identifier: str = ""
    prompt_payload: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    status: ResponseStatus = "success"
    error_message: str | None = None
    budget_breakdown: BudgetBreakdown | None = None


class ResponseUpdate(CamelModel):
    """Payload for partially updating a response (``PATCH``)."""

    text: str | None = None
    status: ResponseStatus | None = None
    error_message: str | None = None
    usage: UsageInfo | None = None
    budget_breakdown: BudgetBreakdown | None = None


class ResponseRead(ResponseBase, TimestampedModel):
    """Response as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
    """

    id: str
