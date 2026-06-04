"""Shared model primitives used across all entity schemas.

This module centralises common Pydantic building blocks:

* :class:`CamelModel` - a base model that serialises field names in camelCase
  (matching the JSON contract in the architecture doc) while allowing snake_case
  internally and population by either name.
* :class:`TimestampedModel` - adds ``created_at`` / ``updated_at`` helpers.
* Pagination and generic envelope helpers.

Entity ``_id`` values are represented as plain strings in the API layer; the
repository layer is responsible for translating between MongoDB ``ObjectId`` and
these string identifiers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

T = TypeVar("T")


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Returns:
        The current time in UTC.
    """
    return datetime.now(tz=timezone.utc)


class CamelModel(BaseModel):
    """Base model exposing a camelCase JSON contract.

    Internally fields use Python-idiomatic snake_case, but they are aliased to
    camelCase for serialisation/deserialisation so the API matches the contract
    documented in the backend architecture (e.g. ``useCaseKey``, ``promptId``).
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        ser_json_timedelta="iso8601",
    )


class TimestampedModel(CamelModel):
    """Mixin adding standard creation/update timestamps.

    Attributes:
        created_at: When the entity was created (UTC).
        updated_at: When the entity was last updated (UTC), if applicable.
    """

    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class PageMeta(CamelModel):
    """Pagination metadata returned alongside list responses.

    Attributes:
        total: Total number of matching documents.
        limit: Page size requested by the client.
        offset: Number of documents skipped.
    """

    total: int
    limit: int
    offset: int


class Page(CamelModel, Generic[T]):
    """A generic paginated list envelope.

    Attributes:
        items: The page of items.
        meta: Pagination metadata.
    """

    items: list[T]
    meta: PageMeta


class MessageResponse(CamelModel):
    """Simple message envelope for operations without a body (e.g. delete).

    Attributes:
        message: Human-readable status message.
    """

    message: str
