"""Repository for the ``use_case_templates`` collection."""

from __future__ import annotations

from typing import Any

from app.db.repositories.base_repo import BaseRepository, serialize_doc
from app.models.use_case_templates import DEFAULT_MODEL


class UseCaseTemplatesRepository(BaseRepository):
    """Data access for use-case templates.

    A template is uniquely identified by the ``(key, model)`` pair, so a single
    use case can carry multiple LLM-specific prompt variants.
    """

    collection_name = "use_case_templates"

    async def get_by_key(
        self, key: str, model: str = DEFAULT_MODEL
    ) -> dict[str, Any] | None:
        """Fetch a template by its ``key`` and ``model``.

        Args:
            key: The use-case key (e.g. ``"travel"``).
            model: The model key the template is tuned for (e.g. ``"claude"``).

        Returns:
            The serialised template, or ``None`` if no template exists for the
            given ``(key, model)`` pair.
        """
        doc = await self.collection.find_one({"key": key, "model": model})
        return serialize_doc(doc)

    async def list_by_key(self, key: str) -> list[dict[str, Any]]:
        """List every template variant registered for a use-case ``key``.

        Args:
            key: The use-case key (e.g. ``"travel"``).

        Returns:
            The serialised templates for all models, ordered by ``model``.
        """
        cursor = self.collection.find({"key": key}).sort("model", 1)
        raw_items = await cursor.to_list(length=None)
        items = [serialize_doc(item) for item in raw_items]
        return [item for item in items if item is not None]

    async def upsert_by_key(self, key: str, document: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a template addressed by its ``(key, model)`` pair.

        Used by the startup seeder to make seeding idempotent.

        Args:
            key: The use-case key.
            document: Template fields (camelCase keys). May include ``model``;
                defaults to :data:`DEFAULT_MODEL` when omitted.

        Returns:
            The upserted, serialised template.
        """
        from app.models.common import utcnow

        now = utcnow()
        payload = dict(document)
        payload["key"] = key
        model = payload.get("model", DEFAULT_MODEL)
        payload["model"] = model
        payload["updatedAt"] = now
        updated = await self.collection.find_one_and_update(
            {"key": key, "model": model},
            {"$set": payload, "$setOnInsert": {"createdAt": now}},
            upsert=True,
            return_document=True,
        )
        serialized = serialize_doc(updated)
        assert serialized is not None  # upsert always yields a document.
        return serialized
