"""Routes for dashboard aggregations (summary + recent runs)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.core.dependencies import MetricsEngineDep
from app.models.dashboard import DashboardRuns, DashboardSummary

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary, summary="Per-model score summary")
async def dashboard_summary(
    metrics_engine: MetricsEngineDep,
    use_case: Annotated[
        str, Query(alias="useCase", description="Use-case key to summarise")
    ] = "travel",
) -> DashboardSummary:
    """Return per-model aggregate scores and metrics for a use case (FR-20).

    Args:
        metrics_engine: Metrics engine.
        use_case: The use-case key to summarise.

    Returns:
        The per-model dashboard summary.
    """
    return await metrics_engine.summary(use_case)


@router.get("/runs", response_model=DashboardRuns, summary="Recent runs with scores")
async def dashboard_runs(
    metrics_engine: MetricsEngineDep,
    use_case: Annotated[
        str, Query(alias="useCase", description="Use-case key to list runs for")
    ] = "travel",
    limit: Annotated[int, Query(ge=1, le=200, description="Maximum runs to return")] = 20,
) -> DashboardRuns:
    """Return recent runs with per-model overall scores (FR-21).

    Args:
        metrics_engine: Metrics engine.
        use_case: The use-case key.
        limit: Maximum number of runs to return.

    Returns:
        The recent-runs listing.
    """
    return await metrics_engine.recent_runs(use_case, limit=limit)
