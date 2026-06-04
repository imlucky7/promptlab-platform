"""Repository for the ``prompt_suggestions`` collection."""

from __future__ import annotations

from app.db.repositories.base_repo import BaseRepository


class PromptSuggestionsRepository(BaseRepository):
    """Data access for prompt suggestions."""

    collection_name = "prompt_suggestions"
