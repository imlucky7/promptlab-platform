"""Routes for the ``runs`` resource (list, create+execute, get, patch, delete)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.api.v1.streaming import sse_streaming_response
from app.core.dependencies import BudgetServiceDep, RunServiceDep, RunsRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.responses import ResponseRead
from app.models.runs import RunCreate, RunRead, RunUpdate, RunWithResponses

logger = get_logger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"], dependencies=[Depends(log_request(logger))])


@router.get("", response_model=Page[RunRead], summary="List runs")
async def list_runs(
    repo: RunsRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    use_case_key: Annotated[
        str | None, Query(alias="useCaseKey", description="Filter by use-case key")
    ] = None,
) -> Page[RunRead]:
    """List runs, optionally filtered by use-case key.

    Args:
        repo: Runs repository.
        pagination: Pagination parameters.
        use_case_key: Optional use-case key filter.

    Returns:
        A page of runs.
    """
    filters = {"useCaseKey": use_case_key} if use_case_key else None
    items, total = await repo.list(
        filters=filters, limit=pagination.limit, offset=pagination.offset
    )
    return build_page([RunRead.model_validate(item) for item in items], total, pagination)


@router.post(
    "",
    response_model=RunWithResponses,
    status_code=201,
    summary="Create and execute a run",
)
async def create_run(payload: RunCreate, run_service: RunServiceDep) -> RunWithResponses:
    """Execute a run for one model and persist the result.

    qwen3 uses the previewed ``promptText`` as-is. Other models receive a compact
    templated prompt built from ``inputs`` for token-efficient execution.

    Args:
        payload: The run creation payload.
        run_service: Run orchestration service.

    Returns:
        The persisted run together with its response.
    """
    return await run_service.create_and_execute(payload)


@router.post("/stream", summary="Create and execute a run (streaming)")
async def create_run_stream(payload: RunCreate, run_service: RunServiceDep):
    """Stream run progress, model output tokens, and the final run payload."""

    async def event_generator():
        async for event in run_service.create_and_execute_stream(payload):
            yield event

    return sse_streaming_response(event_generator())


@router.get("/{run_id}", response_model=RunWithResponses, summary="Get a run with responses")
async def get_run(run_id: str, run_service: RunServiceDep) -> RunWithResponses:
    """Fetch a run with its responses and evaluations (FR-10).

    Args:
        run_id: The run id.
        run_service: Run orchestration service.

    Returns:
        The run with its responses and evaluations.

    Raises:
        NotFoundError: If the run does not exist.
    """
    result = await run_service.get_with_children(run_id)
    if result is None:
        raise NotFoundError("Run not found.", details={"id": run_id})
    return result


@router.post(
    "/{run_id}/responses/{response_id}/budget",
    response_model=ResponseRead,
    summary="Generate budget breakdown for a response",
)
async def generate_response_budget(
    run_id: str,
    response_id: str,
    budget_service: BudgetServiceDep,
) -> ResponseRead:
    """Generate an itemized budget breakdown from a successful travel plan response.

    Args:
        run_id: The owning run id.
        response_id: The response id to budget.
        budget_service: Budget generation service.

    Returns:
        The updated response including ``budgetBreakdown``.
    """
    return await budget_service.generate_for_response(run_id, response_id)


@router.patch("/{run_id}", response_model=RunRead, summary="Update run metadata")
async def update_run(run_id: str, payload: RunUpdate, repo: RunsRepoDep) -> RunRead:
    """Partially update run metadata.

    Args:
        run_id: The run id.
        payload: The fields to update.
        repo: Runs repository.

    Returns:
        The updated run.

    Raises:
        NotFoundError: If the run does not exist.
    """
    updated = await repo.patch(run_id, payload.model_dump(by_alias=True, exclude_unset=True))
    if updated is None:
        raise NotFoundError("Run not found.", details={"id": run_id})
    return RunRead.model_validate(updated)


@router.delete("/{run_id}", response_model=MessageResponse, summary="Delete a run")
async def delete_run(run_id: str, repo: RunsRepoDep) -> MessageResponse:
    """Delete a run.

    Args:
        run_id: The run id.
        repo: Runs repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the run does not exist.
    """
    deleted = await repo.delete(run_id)
    if not deleted:
        raise NotFoundError("Run not found.", details={"id": run_id})
    return MessageResponse(message="Run deleted.")
