"""Shared pagination helpers for list endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import Query
from pydantic import BaseModel

from app.models.common import Page, PageMeta


class PaginationParams(BaseModel):
    """Standard pagination query parameters.

    Attributes:
        limit: Maximum number of items to return (1-200).
        offset: Number of items to skip.
    """

    limit: int = 50
    offset: int = 0


def pagination_params(
    limit: Annotated[int, Query(ge=1, le=200, description="Maximum items to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of items to skip")] = 0,
) -> PaginationParams:
    """FastAPI dependency producing validated pagination parameters.

    Args:
        limit: Maximum number of items to return.
        offset: Number of items to skip.

    Returns:
        A :class:`PaginationParams` instance.
    """
    return PaginationParams(limit=limit, offset=offset)


def build_page(items: list, total: int, params: PaginationParams) -> Page:
    """Wrap a list of items and total count into a :class:`Page` envelope.

    Args:
        items: The page items (already validated models).
        total: Total number of matching documents.
        params: The pagination parameters used.

    Returns:
        A populated :class:`Page`.
    """
    return Page(
        items=items,
        meta=PageMeta(total=total, limit=params.limit, offset=params.offset),
    )
