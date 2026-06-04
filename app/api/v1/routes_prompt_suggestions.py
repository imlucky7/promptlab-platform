"""Routes for the ``prompt_suggestions`` resource (CRUD + apply)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.v1.pagination import PaginationParams, build_page, pagination_params
from app.core.dependencies import (
    SuggestionsRepoDep,
    VersioningEngineDep,
    VersionsRepoDep,
)
from app.core.errors import NotFoundError
from app.models.common import MessageResponse, Page
from app.models.prompt_suggestions import (
    ApplySuggestionRequest,
    PromptSuggestionCreate,
    PromptSuggestionRead,
    PromptSuggestionUpdate,
)
from app.models.prompt_versions import PromptVersionRead

router = APIRouter(prefix="/prompt-suggestions", tags=["prompt-suggestions"])


@router.get("", response_model=Page[PromptSuggestionRead], summary="List suggestions")
async def list_suggestions(
    repo: SuggestionsRepoDep,
    pagination: Annotated[PaginationParams, Depends(pagination_params)],
    prompt_version_id: Annotated[
        str | None, Query(alias="promptVersionId", description="Filter by version id")
    ] = None,
) -> Page[PromptSuggestionRead]:
    """List suggestions, optionally filtered by ``promptVersionId``.

    Args:
        repo: Suggestions repository.
        pagination: Pagination parameters.
        prompt_version_id: Optional version id filter.

    Returns:
        A page of suggestions.
    """
    filters = {"promptVersionId": prompt_version_id} if prompt_version_id else None
    items, total = await repo.list(
        filters=filters, limit=pagination.limit, offset=pagination.offset
    )
    return build_page(
        [PromptSuggestionRead.model_validate(item) for item in items], total, pagination
    )


@router.post(
    "",
    response_model=PromptSuggestionRead,
    status_code=201,
    summary="Create a suggestion",
)
async def create_suggestion(
    payload: PromptSuggestionCreate, repo: SuggestionsRepoDep
) -> PromptSuggestionRead:
    """Create a suggestion entry.

    Args:
        payload: The suggestion to create.
        repo: Suggestions repository.

    Returns:
        The created suggestion.
    """
    created = await repo.create(payload.model_dump(by_alias=True))
    return PromptSuggestionRead.model_validate(created)


@router.get("/{suggestion_id}", response_model=PromptSuggestionRead, summary="Get a suggestion")
async def get_suggestion(suggestion_id: str, repo: SuggestionsRepoDep) -> PromptSuggestionRead:
    """Fetch a single suggestion by id.

    Args:
        suggestion_id: The suggestion id.
        repo: Suggestions repository.

    Returns:
        The requested suggestion.

    Raises:
        NotFoundError: If the suggestion does not exist.
    """
    doc = await repo.get(suggestion_id)
    if doc is None:
        raise NotFoundError("Suggestion not found.", details={"id": suggestion_id})
    return PromptSuggestionRead.model_validate(doc)


@router.patch(
    "/{suggestion_id}", response_model=PromptSuggestionRead, summary="Update a suggestion"
)
async def update_suggestion(
    suggestion_id: str, payload: PromptSuggestionUpdate, repo: SuggestionsRepoDep
) -> PromptSuggestionRead:
    """Partially update a suggestion (e.g. mark as applied).

    Args:
        suggestion_id: The suggestion id.
        payload: The fields to update.
        repo: Suggestions repository.

    Returns:
        The updated suggestion.

    Raises:
        NotFoundError: If the suggestion does not exist.
    """
    updated = await repo.patch(
        suggestion_id, payload.model_dump(by_alias=True, exclude_unset=True)
    )
    if updated is None:
        raise NotFoundError("Suggestion not found.", details={"id": suggestion_id})
    return PromptSuggestionRead.model_validate(updated)


@router.delete(
    "/{suggestion_id}", response_model=MessageResponse, summary="Delete a suggestion"
)
async def delete_suggestion(suggestion_id: str, repo: SuggestionsRepoDep) -> MessageResponse:
    """Delete a suggestion.

    Args:
        suggestion_id: The suggestion id.
        repo: Suggestions repository.

    Returns:
        A confirmation message.

    Raises:
        NotFoundError: If the suggestion does not exist.
    """
    deleted = await repo.delete(suggestion_id)
    if not deleted:
        raise NotFoundError("Suggestion not found.", details={"id": suggestion_id})
    return MessageResponse(message="Suggestion deleted.")


@router.post(
    "/{suggestion_id}/apply",
    response_model=PromptVersionRead,
    status_code=201,
    summary="Apply a suggestion",
)
async def apply_suggestion(
    suggestion_id: str,
    payload: ApplySuggestionRequest,
    repo: SuggestionsRepoDep,
    versions_repo: VersionsRepoDep,
    versioning: VersioningEngineDep,
) -> PromptVersionRead:
    """Apply a suggestion by creating a new prompt version (FR-25).

    The application is non-destructive: a fresh version is created from the
    suggestion's source version with the user-provided prompt text, and the
    suggestion is marked as applied.

    Args:
        suggestion_id: The suggestion id.
        payload: The new prompt text (and optional version name).
        repo: Suggestions repository.
        versions_repo: Versions repository (to load the source version).
        versioning: Versioning engine (to create the new version).

    Returns:
        The newly created prompt version.

    Raises:
        NotFoundError: If the suggestion or its source version is missing.
    """
    suggestion = await repo.get(suggestion_id)
    if suggestion is None:
        raise NotFoundError("Suggestion not found.", details={"id": suggestion_id})

    source_version_id = suggestion["promptVersionId"]
    source_version = await versions_repo.get(source_version_id)
    if source_version is None:
        raise NotFoundError(
            "Source prompt version not found.",
            details={"promptVersionId": source_version_id},
        )

    new_version = await versioning.create_version(
        {
            "promptId": source_version["promptId"],
            "versionName": payload.version_name,
            "structuredInputs": source_version.get("structuredInputs", {}),
            "promptText": payload.prompt_text,
            "createdByUserId": source_version.get("createdByUserId"),
            "parentVersionId": source_version["id"],
            "lastRunId": None,
        }
    )
    await repo.patch(suggestion_id, {"applied": True})
    return PromptVersionRead.model_validate(new_version)
