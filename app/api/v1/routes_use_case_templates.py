"""Routes for the ``use_case_templates`` resource (full CRUD)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import TemplatesRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.use_case_templates import (
    UseCaseTemplateCreate,
    UseCaseTemplateRead,
    UseCaseTemplateReplace,
    UseCaseTemplateUpdate,
)

logger = get_logger(__name__)
router = APIRouter(
    prefix="/use-case-templates",
    tags=["use-case-templates"],
    dependencies=[Depends(log_request(logger))],
)


@router.get("", response_model=Page[UseCaseTemplateRead], summary="List templates")
async def list_templates(
    repo: TemplatesRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    key: Annotated[str | None, Query(description="Filter by use-case key")] = None,
) -> Page[UseCaseTemplateRead]:
    """List use-case templates, optionally filtered by ``key``.

    Args:
        repo: Templates repository.
        pagination: Pagination parameters.
        key: Optional use-case key filter.

    Returns:
        A page of templates.
    """
    filters = {"key": key} if key else None
    items, total = await repo.list(
        filters=filters, limit=pagination.limit, offset=pagination.offset
    )
    return build_page(
        [UseCaseTemplateRead.model_validate(item) for item in items], total, pagination
    )


@router.post(
    "",
    response_model=UseCaseTemplateRead,
    status_code=201,
    summary="Create a template",
)
async def create_template(
    payload: UseCaseTemplateCreate, repo: TemplatesRepoDep
) -> UseCaseTemplateRead:
    """Create a new use-case template.

    Args:
        payload: The template to create.
        repo: Templates repository.

    Returns:
        The created template.
    """
    created = await repo.create(payload.model_dump(by_alias=True))
    return UseCaseTemplateRead.model_validate(created)


@router.get("/{template_id}", response_model=UseCaseTemplateRead, summary="Get a template")
async def get_template(template_id: str, repo: TemplatesRepoDep) -> UseCaseTemplateRead:
    """Fetch a single template by id.

    Args:
        template_id: The template id.
        repo: Templates repository.

    Returns:
        The requested template.

    Raises:
        NotFoundError: If the template does not exist.
    """
    doc = await repo.get(template_id)
    if doc is None:
        raise NotFoundError("Use-case template not found.", details={"id": template_id})
    return UseCaseTemplateRead.model_validate(doc)


@router.put("/{template_id}", response_model=UseCaseTemplateRead, summary="Replace a template")
async def replace_template(
    template_id: str, payload: UseCaseTemplateReplace, repo: TemplatesRepoDep
) -> UseCaseTemplateRead:
    """Fully replace a template.

    Args:
        template_id: The template id.
        payload: The replacement template.
        repo: Templates repository.

    Returns:
        The updated template.

    Raises:
        NotFoundError: If the template does not exist.
    """
    updated = await repo.replace(template_id, payload.model_dump(by_alias=True))
    if updated is None:
        raise NotFoundError("Use-case template not found.", details={"id": template_id})
    return UseCaseTemplateRead.model_validate(updated)


@router.patch("/{template_id}", response_model=UseCaseTemplateRead, summary="Update a template")
async def update_template(
    template_id: str, payload: UseCaseTemplateUpdate, repo: TemplatesRepoDep
) -> UseCaseTemplateRead:
    """Partially update a template.

    Args:
        template_id: The template id.
        payload: The fields to update.
        repo: Templates repository.

    Returns:
        The updated template.

    Raises:
        NotFoundError: If the template does not exist.
    """
    updated = await repo.patch(
        template_id, payload.model_dump(by_alias=True, exclude_unset=True)
    )
    if updated is None:
        raise NotFoundError("Use-case template not found.", details={"id": template_id})
    return UseCaseTemplateRead.model_validate(updated)


@router.delete("/{template_id}", response_model=MessageResponse, summary="Delete a template")
async def delete_template(template_id: str, repo: TemplatesRepoDep) -> MessageResponse:
    """Delete a template.

    Args:
        template_id: The template id.
        repo: Templates repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the template does not exist.
    """
    deleted = await repo.delete(template_id)
    if not deleted:
        raise NotFoundError("Use-case template not found.", details={"id": template_id})
    return MessageResponse(message="Template deleted.")
