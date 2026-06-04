"""Repository for the ``evaluations`` collection."""

from __future__ import annotations

from typing import Any

from app.db.repositories.base_repo import BaseRepository, serialize_doc
from app.models.common import utcnow


class EvaluationsRepository(BaseRepository):
    """Data access for human evaluations."""

    collection_name = "evaluations"

    async def upsert(self, document: dict[str, Any]) -> dict[str, Any]:
        """Create or update an evaluation by its natural key (FR-14).

        The natural key is ``(runId, responseId, modelKey)``; submitting again
        for the same triple updates the existing evaluation rather than creating
        a duplicate.

        Args:
            document: Evaluation fields (camelCase keys).

        Returns:
            The upserted, serialised evaluation.
        """
        now = utcnow()
        key = {
            "runId": document["runId"],
            "responseId": document["responseId"],
            "modelKey": document["modelKey"],
        }
        payload = dict(document)
        payload["updatedAt"] = now
        updated = await self.collection.find_one_and_update(
            key,
            {"$set": payload, "$setOnInsert": {"createdAt": now}},
            upsert=True,
            return_document=True,
        )
        serialized = serialize_doc(updated)
        assert serialized is not None  # upsert always yields a document.
        return serialized

    async def list_by_run(self, run_id: str) -> list[dict[str, Any]]:
        """List evaluations for a run.

        Args:
            run_id: The run id.

        Returns:
            A list of serialised evaluation documents.
        """
        cursor = self.collection.find({"runId": run_id})
        raw = await cursor.to_list(length=1000)
        return [doc for doc in (serialize_doc(item) for item in raw) if doc is not None]

    async def list_by_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """List evaluations for multiple runs.

        Args:
            run_ids: The run ids to include.

        Returns:
            A list of serialised evaluation documents.
        """
        if not run_ids:
            return []
        cursor = self.collection.find({"runId": {"$in": run_ids}})
        raw = await cursor.to_list(length=10000)
        return [doc for doc in (serialize_doc(item) for item in raw) if doc is not None]
