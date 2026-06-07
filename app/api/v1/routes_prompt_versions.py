"""Routes for the ``prompt_versions`` resource (CRUD + clone + save-from-run)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.api.v1.request_logging import log_request
from app.core.dependencies import VersioningEngineDep, VersionsRepoDep
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.common import MessageResponse, Page
from app.models.prompt_versions import (
    PromptVersionCreate,
    PromptVersionRead,
    PromptVersionReplace,
    PromptVersionUpdate,
    SaveFromLastRunRequest,
)

logger = get_logger(__name__)
router = APIRouter(
    prefix="/prompt-versions", tags=["prompt-versions"], dependencies=[Depends(log_request(logger))]
)


@router.get("", response_model=Page[PromptVersionRead], summary="List prompt versions")
async def list_versions(
    repo: VersionsRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    prompt_id: Annotated[
        str | None, Query(alias="promptId", description="Filter by prompt id")
    ] = None,
) -> Page[PromptVersionRead]:
    """List prompt versions, optionally filtered by ``promptId``.

    Args:
        repo: Versions repository.
        pagination: Pagination parameters.
        prompt_id: Optional prompt id filter.

    Returns:
        A page of prompt versions.
    """
    filters = {"promptId": prompt_id} if prompt_id else None
    items, total = await repo.list(
        filters=filters,
        limit=pagination.limit,
        offset=pagination.offset,
        sort=[("versionNumber", -1)],
    )
    return build_page(
        [PromptVersionRead.model_validate(item) for item in items], total, pagination
    )


@router.post(
    "",
    response_model=PromptVersionRead,
    status_code=201,
    summary="Create a prompt version",
)
async def create_version(
    payload: PromptVersionCreate, versioning: VersioningEngineDep
) -> PromptVersionRead:
    """Create a new prompt version with a server-assigned version number.

    Args:
        payload: The version to create.
        versioning: Versioning engine (assigns the version number).

    Returns:
        The created version.
    """
    created = await versioning.create_version(payload.model_dump(by_alias=True))
    return PromptVersionRead.model_validate(created)


@router.get("/{version_id}", response_model=PromptVersionRead, summary="Get a prompt version")
async def get_version(version_id: str, repo: VersionsRepoDep) -> PromptVersionRead:
    """Fetch a single prompt version by id.

    Args:
        version_id: The version id.
        repo: Versions repository.

    Returns:
        The requested version.

    Raises:
        NotFoundError: If the version does not exist.
    """
    doc = await repo.get(version_id)
    if doc is None:
        raise NotFoundError("Prompt version not found.", details={"id": version_id})
    return PromptVersionRead.model_validate(doc)


@router.put("/{version_id}", response_model=PromptVersionRead, summary="Replace a prompt version")
async def replace_version(
    version_id: str, payload: PromptVersionReplace, repo: VersionsRepoDep
) -> PromptVersionRead:
    """Fully replace a prompt version.

    Args:
        version_id: The version id.
        payload: The replacement version.
        repo: Versions repository.

    Returns:
        The updated version.

    Raises:
        NotFoundError: If the version does not exist.
    """
    updated = await repo.replace(version_id, payload.model_dump(by_alias=True))
    if updated is None:
        raise NotFoundError("Prompt version not found.", details={"id": version_id})
    return PromptVersionRead.model_validate(updated)


@router.patch("/{version_id}", response_model=PromptVersionRead, summary="Update a prompt version")
async def update_version(
    version_id: str, payload: PromptVersionUpdate, repo: VersionsRepoDep
) -> PromptVersionRead:
    """Partially update a prompt version.

    Args:
        version_id: The version id.
        payload: The fields to update.
        repo: Versions repository.

    Returns:
        The updated version.

    Raises:
        NotFoundError: If the version does not exist.
    """
    updated = await repo.patch(version_id, payload.model_dump(by_alias=True, exclude_unset=True))
    if updated is None:
        raise NotFoundError("Prompt version not found.", details={"id": version_id})
    return PromptVersionRead.model_validate(updated)


@router.delete("/{version_id}", response_model=MessageResponse, summary="Delete a prompt version")
async def delete_version(version_id: str, repo: VersionsRepoDep) -> MessageResponse:
    """Delete a prompt version.

    Args:
        version_id: The version id.
        repo: Versions repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the version does not exist.
    """
    deleted = await repo.delete(version_id)
    if not deleted:
        raise NotFoundError("Prompt version not found.", details={"id": version_id})
    return MessageResponse(message="Prompt version deleted.")


@router.post(
    "/{version_id}/clone",
    response_model=PromptVersionRead,
    status_code=201,
    summary="Clone a prompt version",
)
async def clone_version(
    version_id: str,
    versioning: VersioningEngineDep,
    payload: PromptVersionUpdate | None = None,
) -> PromptVersionRead:
    """Clone an existing version into a new one (FR-17).

    Args:
        version_id: The id of the version to clone.
        versioning: Versioning engine.
        payload: Optional overrides (only ``versionName`` is used).

    Returns:
        The newly created clone.
    """
    version_name = payload.version_name if payload else None
    cloned = await versioning.clone(version_id, version_name)
    return PromptVersionRead.model_validate(cloned)


@router.post(
    "/{version_id}/save-from-last-run",
    response_model=PromptVersionRead,
    status_code=201,
    summary="Save a new version from the last run",
)
async def save_from_last_run(
    version_id: str,
    payload: SaveFromLastRunRequest,
    versioning: VersioningEngineDep,
) -> PromptVersionRead:
    """Create a new version snapshot from a version's last-run state (FR-19).

    Args:
        version_id: The source version id.
        payload: Request with an optional version name.
        versioning: Versioning engine.

    Returns:
        The newly created version.
    """
    created = await versioning.save_from_last_run(version_id, payload.version_name)
    return PromptVersionRead.model_validate(created)
