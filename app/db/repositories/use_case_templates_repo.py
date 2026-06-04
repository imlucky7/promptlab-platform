"""Repository for the ``use_case_templates`` collection."""

from __future__ import annotations

from typing import Any

from app.db.repositories.base_repo import BaseRepository, serialize_doc


class UseCaseTemplatesRepository(BaseRepository):
    """Data access for use-case templates."""

    collection_name = "use_case_templates"

    async def get_by_key(self, key: str) -> dict[str, Any] | None:
        """Fetch a template by its unique ``key``.

        Args:
            key: The use-case key (e.g. ``"travel"``).

        Returns:
            The serialised template, or ``None`` if not found.
        """
        doc = await self.collection.find_one({"key": key})
        return serialize_doc(doc)

    async def upsert_by_key(self, key: str, document: dict[str, Any]) -> dict[str, Any]:
        """Insert or update a template addressed by its ``key``.

        Used by the startup seeder to make seeding idempotent.

        Args:
            key: The use-case key.
            document: Template fields (camelCase keys).

        Returns:
            The upserted, serialised template.
        """
        from app.models.common import utcnow

        now = utcnow()
        payload = dict(document)
        payload["key"] = key
        payload["updatedAt"] = now
        updated = await self.collection.find_one_and_update(
            {"key": key},
            {"$set": payload, "$setOnInsert": {"createdAt": now}},
            upsert=True,
            return_document=True,
        )
        serialized = serialize_doc(updated)
        assert serialized is not None  # upsert always yields a document.
        return serialized
