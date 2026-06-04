"""Schemas for the ``prompts`` collection (logical prompt workspaces)."""

from __future__ import annotations

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel


class PromptBase(CamelModel):
    """Fields shared by create/replace operations.

    Attributes:
        title: User-facing prompt name (e.g. ``"Japan Family Vacation"``).
        use_case_key: Use-case this prompt belongs to (e.g. ``"travel"``).
        created_by_user_id: Reserved for future authentication; may be null.
    """

    title: str = Field(examples=["Japan Family Vacation"])
    use_case_key: str = Field(default="travel", examples=["travel"])
    created_by_user_id: str | None = None


class PromptCreate(PromptBase):
    """Payload for creating a prompt (``POST``)."""


class PromptReplace(PromptBase):
    """Payload for fully replacing a prompt (``PUT``)."""


class PromptUpdate(CamelModel):
    """Payload for partially updating a prompt (``PATCH``)."""

    title: str | None = None
    use_case_key: str | None = None
    created_by_user_id: str | None = None


class PromptRead(PromptBase, TimestampedModel):
    """Prompt as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
    """

    id: str
