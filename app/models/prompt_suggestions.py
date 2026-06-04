"""Schemas for the ``prompt_suggestions`` collection (rule-based hints)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel

# Known suggestion categories produced by the suggestion engine.
SuggestionType = Literal[
    "add_examples",
    "clarify_output_format",
    "define_persona",
    "add_constraints",
]


class SuggestionItem(CamelModel):
    """A single, non-persisted suggestion (used in preview responses).

    Attributes:
        suggestion_type: Category of the suggestion.
        description: Human-readable recommendation text.
    """

    suggestion_type: SuggestionType
    description: str


class PromptSuggestionBase(CamelModel):
    """Fields shared by suggestion create operations.

    Attributes:
        prompt_version_id: Version the suggestion relates to.
        run_id: Optional run that triggered the suggestion.
        suggestion_type: Category of the suggestion.
        description: Human-readable recommendation text.
        applied: Whether the suggestion has been applied.
    """

    prompt_version_id: str
    run_id: str | None = None
    suggestion_type: SuggestionType
    description: str
    applied: bool = False


class PromptSuggestionCreate(PromptSuggestionBase):
    """Payload for creating a suggestion (``POST``)."""


class PromptSuggestionUpdate(CamelModel):
    """Payload for partially updating a suggestion (``PATCH``)."""

    description: str | None = None
    applied: bool | None = None


class PromptSuggestionRead(PromptSuggestionBase, TimestampedModel):
    """Suggestion as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
    """

    id: str


class ApplySuggestionRequest(CamelModel):
    """Payload for ``POST /prompt-suggestions/{id}/apply``.

    Attributes:
        prompt_text: The new prompt text to persist as a fresh version.
        version_name: Optional label for the new version.
    """

    prompt_text: str
    version_name: str | None = None
