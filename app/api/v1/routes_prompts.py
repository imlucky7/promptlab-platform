"""Routes for the ``prompts`` resource (full CRUD)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import PromptsRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.prompts import PromptCreate, PromptRead, PromptReplace, PromptUpdate

logger = get_logger(__name__)
router = APIRouter(prefix="/prompts", tags=["prompts"], dependencies=[Depends(log_request(logger))])


@router.get("", response_model=Page[PromptRead], summary="List prompts")
async def list_prompts(
    repo: PromptsRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    use_case_key: Annotated[
        str | None, Query(alias="useCaseKey", description="Filter by use-case key")
    ] = None,
) -> Page[PromptRead]:
    """List prompts, optionally filtered by use-case key.

    Args:
        repo: Prompts repository.
        pagination: Pagination parameters.
        use_case_key: Optional use-case key filter.

    Returns:
        A page of prompts.
    """
    filters = {"useCaseKey": use_case_key} if use_case_key else None
    items, total = await repo.list(
        filters=filters, limit=pagination.limit, offset=pagination.offset
    )
    return build_page([PromptRead.model_validate(item) for item in items], total, pagination)


@router.post("", response_model=PromptRead, status_code=201, summary="Create a prompt")
async def create_prompt(payload: PromptCreate, repo: PromptsRepoDep) -> PromptRead:
    """Create a new prompt.

    Args:
        payload: The prompt to create.
        repo: Prompts repository.

    Returns:
        The created prompt.
    """
    created = await repo.create(payload.model_dump(by_alias=True))
    return PromptRead.model_validate(created)


@router.get("/{prompt_id}", response_model=PromptRead, summary="Get a prompt")
async def get_prompt(prompt_id: str, repo: PromptsRepoDep) -> PromptRead:
    """Fetch a single prompt by id.

    Args:
        prompt_id: The prompt id.
        repo: Prompts repository.

    Returns:
        The requested prompt.

    Raises:
        NotFoundError: If the prompt does not exist.
    """
    doc = await repo.get(prompt_id)
    if doc is None:
        raise NotFoundError("Prompt not found.", details={"id": prompt_id})
    return PromptRead.model_validate(doc)


@router.put("/{prompt_id}", response_model=PromptRead, summary="Replace a prompt")
async def replace_prompt(
    prompt_id: str, payload: PromptReplace, repo: PromptsRepoDep
) -> PromptRead:
    """Fully replace a prompt.

    Args:
        prompt_id: The prompt id.
        payload: The replacement prompt.
        repo: Prompts repository.

    Returns:
        The updated prompt.

    Raises:
        NotFoundError: If the prompt does not exist.
    """
    updated = await repo.replace(prompt_id, payload.model_dump(by_alias=True))
    if updated is None:
        raise NotFoundError("Prompt not found.", details={"id": prompt_id})
    return PromptRead.model_validate(updated)


@router.patch("/{prompt_id}", response_model=PromptRead, summary="Update a prompt")
async def update_prompt(
    prompt_id: str, payload: PromptUpdate, repo: PromptsRepoDep
) -> PromptRead:
    """Partially update a prompt.

    Args:
        prompt_id: The prompt id.
        payload: The fields to update.
        repo: Prompts repository.

    Returns:
        The updated prompt.

    Raises:
        NotFoundError: If the prompt does not exist.
    """
    updated = await repo.patch(prompt_id, payload.model_dump(by_alias=True, exclude_unset=True))
    if updated is None:
        raise NotFoundError("Prompt not found.", details={"id": prompt_id})
    return PromptRead.model_validate(updated)


@router.delete("/{prompt_id}", response_model=MessageResponse, summary="Delete a prompt")
async def delete_prompt(prompt_id: str, repo: PromptsRepoDep) -> MessageResponse:
    """Delete a prompt.

    Args:
        prompt_id: The prompt id.
        repo: Prompts repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the prompt does not exist.
    """
    deleted = await repo.delete(prompt_id)
    if not deleted:
        raise NotFoundError("Prompt not found.", details={"id": prompt_id})
    return MessageResponse(message="Prompt deleted.")
