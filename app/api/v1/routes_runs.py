"""Routes for the ``runs`` resource (list, create+execute, get, patch, delete)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import RunServiceDep, RunsRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
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
    """Execute previewed prompts, persisting the prompt, version and run.

    The caller submits one previewed prompt per model (from ``POST /preview``);
    each variant is sent to its model through the gateway. ``promptId`` and
    ``promptVersionId`` may be supplied to reuse an existing prompt/version, or
    omitted to have the system generate fresh ``ObjectId`` references.

    Args:
        payload: The run creation payload (per-model previewed prompts).
        run_service: Run orchestration service.

    Returns:
        The persisted run together with its per-model responses.
    """
    return await run_service.create_and_execute(payload)


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
