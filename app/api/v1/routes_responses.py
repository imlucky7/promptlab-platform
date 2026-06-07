"""Routes for the ``responses`` resource (list, get, patch, delete)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import ResponsesRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.responses import ResponseRead, ResponseUpdate

logger = get_logger(__name__)
router = APIRouter(
    prefix="/responses", tags=["responses"], dependencies=[Depends(log_request(logger))]
)


@router.get("", response_model=Page[ResponseRead], summary="List responses")
async def list_responses(
    repo: ResponsesRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    run_id: Annotated[str | None, Query(alias="runId", description="Filter by run id")] = None,
    model_key: Annotated[
        str | None, Query(alias="modelKey", description="Filter by model key")
    ] = None,
) -> Page[ResponseRead]:
    """List responses, optionally filtered by run id and/or model key.

    Args:
        repo: Responses repository.
        pagination: Pagination parameters.
        run_id: Optional run id filter.
        model_key: Optional model key filter.

    Returns:
        A page of responses.
    """
    filters: dict[str, str] = {}
    if run_id:
        filters["runId"] = run_id
    if model_key:
        filters["modelKey"] = model_key
    items, total = await repo.list(
        filters=filters or None, limit=pagination.limit, offset=pagination.offset
    )
    return build_page([ResponseRead.model_validate(item) for item in items], total, pagination)


@router.get("/{response_id}", response_model=ResponseRead, summary="Get a response")
async def get_response(response_id: str, repo: ResponsesRepoDep) -> ResponseRead:
    """Fetch a single response by id.

    Args:
        response_id: The response id.
        repo: Responses repository.

    Returns:
        The requested response.

    Raises:
        NotFoundError: If the response does not exist.
    """
    doc = await repo.get(response_id)
    if doc is None:
        raise NotFoundError("Response not found.", details={"id": response_id})
    return ResponseRead.model_validate(doc)


@router.patch("/{response_id}", response_model=ResponseRead, summary="Update a response")
async def update_response(
    response_id: str, payload: ResponseUpdate, repo: ResponsesRepoDep
) -> ResponseRead:
    """Partially update response metadata/status.

    Args:
        response_id: The response id.
        payload: The fields to update.
        repo: Responses repository.

    Returns:
        The updated response.

    Raises:
        NotFoundError: If the response does not exist.
    """
    updated = await repo.patch(response_id, payload.model_dump(by_alias=True, exclude_unset=True))
    if updated is None:
        raise NotFoundError("Response not found.", details={"id": response_id})
    return ResponseRead.model_validate(updated)


@router.delete("/{response_id}", response_model=MessageResponse, summary="Delete a response")
async def delete_response(response_id: str, repo: ResponsesRepoDep) -> MessageResponse:
    """Delete a response.

    Args:
        response_id: The response id.
        repo: Responses repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the response does not exist.
    """
    deleted = await repo.delete(response_id)
    if not deleted:
        raise NotFoundError("Response not found.", details={"id": response_id})
    return MessageResponse(message="Response deleted.")
