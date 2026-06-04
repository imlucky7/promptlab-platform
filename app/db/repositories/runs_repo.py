"""Repository for the ``runs`` collection."""

from __future__ import annotations

from app.db.repositories.base_repo import BaseRepository


class RunsRepository(BaseRepository):
    """Data access for runs."""

    collection_name = "runs"
