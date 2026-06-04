"""Repository for the ``prompts`` collection."""

from __future__ import annotations

from app.db.repositories.base_repo import BaseRepository


class PromptsRepository(BaseRepository):
    """Data access for logical prompts (workspaces)."""

    collection_name = "prompts"
