"""Routes for the ``evaluations`` resource (list, upsert, get, patch, delete)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import EvaluationEngineDep, EvaluationsRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.evaluations import EvaluationCreate, EvaluationRead, EvaluationUpdate

logger = get_logger(__name__)
router = APIRouter(
    prefix="/evaluations", tags=["evaluations"], dependencies=[Depends(log_request(logger))]
)


@router.get("", response_model=Page[EvaluationRead], summary="List evaluations")
async def list_evaluations(
    repo: EvaluationsRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    run_id: Annotated[str | None, Query(alias="runId", description="Filter by run id")] = None,
    response_id: Annotated[
        str | None, Query(alias="responseId", description="Filter by response id")
    ] = None,
    model_key: Annotated[
        str | None, Query(alias="modelKey", description="Filter by model key")
    ] = None,
) -> Page[EvaluationRead]:
    """List evaluations, optionally filtered by run/response/model.

    Args:
        repo: Evaluations repository.
        pagination: Pagination parameters.
        run_id: Optional run id filter.
        response_id: Optional response id filter.
        model_key: Optional model key filter.

    Returns:
        A page of evaluations.
    """
    filters: dict[str, str] = {}
    if run_id:
        filters["runId"] = run_id
    if response_id:
        filters["responseId"] = response_id
    if model_key:
        filters["modelKey"] = model_key
    items, total = await repo.list(
        filters=filters or None, limit=pagination.limit, offset=pagination.offset
    )
    return build_page([EvaluationRead.model_validate(item) for item in items], total, pagination)


@router.post(
    "",
    response_model=EvaluationRead,
    status_code=201,
    summary="Create or upsert an evaluation",
)
async def upsert_evaluation(
    payload: EvaluationCreate, engine: EvaluationEngineDep
) -> EvaluationRead:
    """Create or update an evaluation by its natural key (FR-14).

    Args:
        payload: The evaluation to upsert.
        engine: Evaluation engine.

    Returns:
        The upserted evaluation.
    """
    saved = await engine.upsert(payload.model_dump(by_alias=True))
    return EvaluationRead.model_validate(saved)


@router.get("/{evaluation_id}", response_model=EvaluationRead, summary="Get an evaluation")
async def get_evaluation(evaluation_id: str, repo: EvaluationsRepoDep) -> EvaluationRead:
    """Fetch a single evaluation by id.

    Args:
        evaluation_id: The evaluation id.
        repo: Evaluations repository.

    Returns:
        The requested evaluation.

    Raises:
        NotFoundError: If the evaluation does not exist.
    """
    doc = await repo.get(evaluation_id)
    if doc is None:
        raise NotFoundError("Evaluation not found.", details={"id": evaluation_id})
    return EvaluationRead.model_validate(doc)


@router.patch("/{evaluation_id}", response_model=EvaluationRead, summary="Update an evaluation")
async def update_evaluation(
    evaluation_id: str, payload: EvaluationUpdate, repo: EvaluationsRepoDep
) -> EvaluationRead:
    """Partially update an evaluation.

    Args:
        evaluation_id: The evaluation id.
        payload: The fields to update.
        repo: Evaluations repository.

    Returns:
        The updated evaluation.

    Raises:
        NotFoundError: If the evaluation does not exist.
    """
    updated = await repo.patch(
        evaluation_id, payload.model_dump(by_alias=True, exclude_unset=True)
    )
    if updated is None:
        raise NotFoundError("Evaluation not found.", details={"id": evaluation_id})
    return EvaluationRead.model_validate(updated)


@router.delete("/{evaluation_id}", response_model=MessageResponse, summary="Delete an evaluation")
async def delete_evaluation(evaluation_id: str, repo: EvaluationsRepoDep) -> MessageResponse:
    """Delete an evaluation.

    Args:
        evaluation_id: The evaluation id.
        repo: Evaluations repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the evaluation does not exist.
    """
    deleted = await repo.delete(evaluation_id)
    if not deleted:
        raise NotFoundError("Evaluation not found.", details={"id": evaluation_id})
    return MessageResponse(message="Evaluation deleted.")
