"""Schemas for the ``metrics_logs`` collection and dashboard aggregates."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel


class MetricsLogBase(CamelModel):
    """Fields shared by metrics-log create operations.

    Attributes:
        run_id: Reference to the run.
        response_id: Reference to the response.
        model_key: Logical model key.
        token_cost_estimate: Optional derived cost estimate.
        extra: Additional free-form derived metrics.
    """

    run_id: str
    response_id: str
    model_key: str
    token_cost_estimate: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MetricsLogCreate(MetricsLogBase):
    """Payload for creating a metrics log (``POST``)."""


class MetricsLogRead(MetricsLogBase, TimestampedModel):
    """Metrics log as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
    """

    id: str
