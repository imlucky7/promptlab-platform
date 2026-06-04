"""Repository for the ``responses`` collection."""

from __future__ import annotations

from typing import Any

from app.db.repositories.base_repo import BaseRepository, serialize_doc


class ResponsesRepository(BaseRepository):
    """Data access for per-model responses."""

    collection_name = "responses"

    async def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        """List all responses for a given run.

        Args:
            run_id: The owning run id.

        Returns:
            A list of serialised response documents.
        """
        cursor = self.collection.find({"runId": run_id}).sort([("createdAt", 1)])
        raw = await cursor.to_list(length=1000)
        return [doc for doc in (serialize_doc(item) for item in raw) if doc is not None]

    async def list_by_use_case(self, use_case_key: str, run_ids: list[str]) -> list[dict[str, Any]]:
        """List responses belonging to a set of runs.

        Args:
            use_case_key: Use-case key (used by callers for context only).
            run_ids: The run ids to include.

        Returns:
            A list of serialised response documents.
        """
        if not run_ids:
            return []
        cursor = self.collection.find({"runId": {"$in": run_ids}})
        raw = await cursor.to_list(length=10000)
        return [doc for doc in (serialize_doc(item) for item in raw) if doc is not None]
