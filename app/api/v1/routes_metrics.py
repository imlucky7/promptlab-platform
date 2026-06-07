"""Routes for the ``metrics_logs`` resource (list, create, get, delete)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import MetricsRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.metrics import MetricsLogCreate, MetricsLogRead

logger = get_logger(__name__)
router = APIRouter(
    prefix="/metrics-logs", tags=["metrics-logs"], dependencies=[Depends(log_request(logger))]
)


@router.get("", response_model=Page[MetricsLogRead], summary="List metrics logs")
async def list_metrics_logs(
    repo: MetricsRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    run_id: Annotated[str | None, Query(alias="runId", description="Filter by run id")] = None,
    response_id: Annotated[
        str | None, Query(alias="responseId", description="Filter by response id")
    ] = None,
) -> Page[MetricsLogRead]:
    """List metrics logs, optionally filtered by run/response.

    Args:
        repo: Metrics repository.
        pagination: Pagination parameters.
        run_id: Optional run id filter.
        response_id: Optional response id filter.

    Returns:
        A page of metrics logs.
    """
    filters: dict[str, str] = {}
    if run_id:
        filters["runId"] = run_id
    if response_id:
        filters["responseId"] = response_id
    items, total = await repo.list(
        filters=filters or None, limit=pagination.limit, offset=pagination.offset
    )
    return build_page([MetricsLogRead.model_validate(item) for item in items], total, pagination)


@router.post(
    "",
    response_model=MetricsLogRead,
    status_code=201,
    summary="Create a metrics log",
)
async def create_metrics_log(payload: MetricsLogCreate, repo: MetricsRepoDep) -> MetricsLogRead:
    """Create a metrics log entry (typically internal usage).

    Args:
        payload: The metrics log to create.
        repo: Metrics repository.

    Returns:
        The created metrics log.
    """
    created = await repo.create(payload.model_dump(by_alias=True))
    return MetricsLogRead.model_validate(created)


@router.get("/{metrics_id}", response_model=MetricsLogRead, summary="Get a metrics log")
async def get_metrics_log(metrics_id: str, repo: MetricsRepoDep) -> MetricsLogRead:
    """Fetch a single metrics log by id.

    Args:
        metrics_id: The metrics log id.
        repo: Metrics repository.

    Returns:
        The requested metrics log.

    Raises:
        NotFoundError: If the metrics log does not exist.
    """
    doc = await repo.get(metrics_id)
    if doc is None:
        raise NotFoundError("Metrics log not found.", details={"id": metrics_id})
    return MetricsLogRead.model_validate(doc)


@router.delete("/{metrics_id}", response_model=MessageResponse, summary="Delete a metrics log")
async def delete_metrics_log(metrics_id: str, repo: MetricsRepoDep) -> MessageResponse:
    """Delete a metrics log.

    Args:
        metrics_id: The metrics log id.
        repo: Metrics repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the metrics log does not exist.
    """
    deleted = await repo.delete(metrics_id)
    if not deleted:
        raise NotFoundError("Metrics log not found.", details={"id": metrics_id})
    return MessageResponse(message="Metrics log deleted.")
