"""Schemas for the ``use_case_templates`` collection.

A use-case template defines the input schema (form fields) and the normalized
prompt template (system + user sections with placeholders) for a given use case,
e.g. ``"travel"``.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel


class UseCaseTemplateBase(CamelModel):
    """Fields shared by create/replace operations.

    Attributes:
        key: Stable use-case key (e.g. ``"travel"``).
        name: Display name (e.g. ``"Travel Planner"``).
        description: Short human description of the use case.
        input_schema: JSON-schema-like description of the form fields.
        normalized_prompt_template: Template text with ``{{ placeholders }}``.
    """

    key: str = Field(examples=["travel"])
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
