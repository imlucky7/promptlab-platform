"""Schemas for the ``prompt_versions`` collection.

A prompt version is an immutable-ish snapshot of the structured inputs and the
final assembled prompt text for a given prompt, identified by an incrementing
``version_number``.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel


class PromptVersionBase(CamelModel):
    """Fields shared by create/replace operations.

    Attributes:
        prompt_id: Reference to the owning prompt.
        version_name: Optional human label (e.g. ``"V1 - baseline"``).
        structured_inputs: Snapshot of the inputs at save time.
        prompt_text: Final assembled prompt text.
        created_by_user_id: Reserved for future authentication.
        parent_version_id: Optional lineage reference for clones.
        last_run_id: Optional reference to the run that produced this version.
    """

    prompt_id: str
    version_name: str | None = Field(default=None, examples=["V1 - baseline"])
    structured_inputs: dict[str, Any] = Field(default_factory=dict)
    prompt_text: str = Field(default="")
    created_by_user_id: str | None = None
    parent_version_id: str | None = None
    last_run_id: str | None = None


class PromptVersionCreate(PromptVersionBase):
    """Payload for creating a version (``POST``).

    ``version_number`` is intentionally omitted: it is assigned server-side by
    the versioning engine to guarantee monotonic numbering per prompt.
    """


class PromptVersionReplace(PromptVersionBase):
    """Payload for fully replacing a version (``PUT``).

    Attributes:
        version_number: The version number to set on replacement.
    """

    version_number: int = Field(ge=1)


class PromptVersionUpdate(CamelModel):
    """Payload for partially updating a version (``PATCH``)."""

    version_name: str | None = None
    structured_inputs: dict[str, Any] | None = None
    prompt_text: str | None = None
    parent_version_id: str | None = None
    last_run_id: str | None = None


class PromptVersionRead(PromptVersionBase, TimestampedModel):
    """Version as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
        version_number: Monotonic version number per prompt.
    """

    id: str
    version_number: int


class SaveFromLastRunRequest(CamelModel):
    """Payload for ``POST /prompt-versions/{id}/save-from-last-run``.

    Attributes:
        version_name: Optional label for the newly created version.
    """

    version_name: str | None = None
