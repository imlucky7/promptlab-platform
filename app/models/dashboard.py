"""Schemas for the dashboard aggregation endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.models.common import CamelModel


class ModelScoreSummary(CamelModel):
    """Aggregated metrics for a single model across evaluated responses.

    Attributes:
        model_key: Logical model key.
        avg_overall: Average overall human score.
        avg_correctness: Average correctness score.
        avg_completeness: Average completeness score.
        avg_style_fit: Average style-fit score.
        avg_faithfulness: Average faithfulness score.
        avg_input_tokens: Average input tokens per response.
        avg_output_tokens: Average output tokens per response.
        avg_latency_ms: Average latency in milliseconds.
        evaluation_count: Number of evaluations contributing to averages.
        response_count: Number of responses contributing to averages.
    """

    model_key: str
    avg_overall: float | None = None
    avg_correctness: float | None = None
    avg_completeness: float | None = None
    avg_style_fit: float | None = None
    avg_faithfulness: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_latency_ms: float | None = None
    evaluation_count: int = 0
    response_count: int = 0


class DashboardSummary(CamelModel):
    """Top-level dashboard summary response.

    Attributes:
        use_case_key: The use case the summary is scoped to.
        models: Per-model aggregate metrics.
    """

    use_case_key: str
    models: list[ModelScoreSummary] = Field(default_factory=list)


class RunModelScore(CamelModel):
    """Per-model overall score for a run in the runs-history view.

    Attributes:
        model_key: Logical model key.
        overall: Overall score if the response was evaluated.
    """

    model_key: str
    overall: int | None = None


class DashboardRunItem(CamelModel):
    """A single recent run entry for the runs-history view.

    Attributes:
        run_id: The run identifier.
        created_at: When the run was created.
        scores: Per-model overall scores (when evaluated).
    """

    run_id: str
    created_at: datetime | None = None
    scores: list[RunModelScore] = Field(default_factory=list)


class DashboardRuns(CamelModel):
    """Runs-history response.

    Attributes:
        use_case_key: The use case scope.
        runs: Recent runs with their per-model scores.
    """

    use_case_key: str
    runs: list[DashboardRunItem] = Field(default_factory=list)
