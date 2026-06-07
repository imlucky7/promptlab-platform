"""Schemas for the ``use_case_templates`` collection.

A use-case template defines the input schema (form fields) and the normalized
prompt template (system + user sections with placeholders) for a given use case,
e.g. ``"travel"``. Each template is also scoped to a ``model`` so that a single
use case can carry LLM-specific prompt variants (e.g. a Claude-tuned template
alongside the generic default).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel

#: Model key for the default LLM (ChatGPT). A template is uniquely identified by
#: the ``(key, model)`` pair; this is the fallback model used when a request does
#: not target a specific LLM.
DEFAULT_MODEL: str = "chatgpt"


class UseCaseTemplateBase(CamelModel):
    """Fields shared by create/replace operations.

    Attributes:
        key: Stable use-case key (e.g. ``"travel"``).
        model: Logical model key the template is tuned for (e.g. ``"claude"``).
            Together with ``key`` it uniquely identifies a template variant.
        name: Display name (e.g. ``"Travel Planner"``).
        description: Short human description of the use case.
        input_schema: JSON-schema-like description of the form fields.
        normalized_prompt_template: Template text with ``{{ placeholders }}``.
    """

    key: str = Field(examples=["travel"])
    model: str = Field(default=DEFAULT_MODEL, examples=["chatgpt", "claude"])
    name: str = Field(examples=["Travel Planner"])
    description: str = Field(default="", examples=["Design a multi-day trip itinerary"])
    input_schema: dict[str, Any] = Field(default_factory=dict)
    normalized_prompt_template: str = Field(default="")


class UseCaseTemplateCreate(UseCaseTemplateBase):
    """Payload for creating a template (``POST``)."""


class UseCaseTemplateReplace(UseCaseTemplateBase):
    """Payload for fully replacing a template (``PUT``)."""


class UseCaseTemplateUpdate(CamelModel):
    """Payload for partially updating a template (``PATCH``).

    Every field is optional; only provided fields are modified.
    """

    key: str | None = None
    model: str | None = None
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    normalized_prompt_template: str | None = None


class UseCaseTemplateRead(UseCaseTemplateBase, TimestampedModel):
    """Template as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
    """

    id: str
