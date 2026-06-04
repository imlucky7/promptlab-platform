"""Generic MongoDB repository base class.

The base repository encapsulates the boilerplate shared by all collections:

* translating between MongoDB ``ObjectId`` and the string ``id`` exposed by the
  API,
* stamping ``createdAt`` / ``updatedAt`` timestamps,
* paginated listing with filtering and sorting,
* standard CRUD (create / get / list / replace / patch / delete).

Documents are stored with camelCase keys to match the contract and indexes
defined in the architecture document.
"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.models.common import utcnow


def to_object_id(value: str) -> ObjectId | None:
    """Convert a string id to an :class:`ObjectId`.

    Args:
        value: The candidate string identifier.

    Returns:
        The corresponding :class:`ObjectId`, or ``None`` if ``value`` is not a
        valid ObjectId (callers typically treat this as "not found").
    """
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def serialize_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialise a raw Mongo document for the API layer.

    Converts the ``_id`` ObjectId into a string ``id`` field and removes the
    original ``_id`` key. Reference fields are stored as strings already, so no
    further conversion is required.

    Args:
        doc: The raw MongoDB document, or ``None``.

    Returns:
        A serialised document with a string ``id``, or ``None`` if ``doc`` is
        ``None``.
    """
    if doc is None:
        return None
    result = dict(doc)
    object_id = result.pop("_id", None)
    if object_id is not None:
        result["id"] = str(object_id)
    return result


class BaseRepository:
    """Base class providing CRUD operations over a single collection.

    Attributes:
        db: The active Mongo database handle.
        collection_name: Name of the backing collection.
        collection: The Motor collection handle.
    """

    #: Subclasses must set the backing collection name.
    collection_name: str = ""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        """Initialise the repository.

        Args:
            db: The active Mongo database handle.

        Raises:
            ValueError: If ``collection_name`` was not set by the subclass.
        """
        if not self.collection_name:
            raise ValueError("Subclasses must define 'collection_name'.")
        self.db = db
        self.collection: AsyncIOMotorCollection = db[self.collection_name]

    async def create(self, document: dict[str, Any]) -> dict[str, Any]:
        """Insert a new document, stamping creation/update timestamps.

        Args:
            document: The document fields to insert (camelCase keys).

        Returns:
            The created document, serialised for the API (with string ``id``).
        """
        payload = dict(document)
        now = utcnow()
        payload.setdefault("createdAt", now)
        payload["updatedAt"] = now
        result = await self.collection.insert_one(payload)
        payload["_id"] = result.inserted_id
        serialized = serialize_doc(payload)
        assert serialized is not None  # insert always yields a document.
        return serialized

    async def get(self, entity_id: str) -> dict[str, Any] | None:
        """Fetch a single document by id.

        Args:
            entity_id: The string id of the document.

        Returns:
            The serialised document, or ``None`` if not found.
        """
        object_id = to_object_id(entity_id)
        if object_id is None:
            return None
        doc = await self.collection.find_one({"_id": object_id})
        return serialize_doc(doc)

    async def list(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
        sort: list[tuple[str, int]] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List documents matching a filter, with pagination.

        Args:
            filters: Mongo filter document (camelCase keys). Defaults to ``{}``.
            limit: Maximum number of documents to return.
            offset: Number of documents to skip.
            sort: Optional list of ``(field, direction)`` sort pairs. Defaults to
                newest first by ``createdAt``.

        Returns:
            A tuple of ``(items, total_count)`` where ``items`` are serialised.
        """
        query = filters or {}
        sort_spec = sort or [("createdAt", -1)]
        cursor = self.collection.find(query).sort(sort_spec).skip(offset).limit(limit)
        raw_items = await cursor.to_list(length=limit)
        items = [serialize_doc(item) for item in raw_items]
        total = await self.collection.count_documents(query)
        # ``serialize_doc`` never returns None for non-None inputs.
        return [item for item in items if item is not None], total

    async def replace(self, entity_id: str, document: dict[str, Any]) -> dict[str, Any] | None:
        """Fully replace a document's fields (preserving creation time).

        Args:
            entity_id: The string id of the document.
            document: The replacement fields (camelCase keys).

        Returns:
            The updated document, or ``None`` if not found.
        """
        object_id = to_object_id(entity_id)
        if object_id is None:
            return None
        payload = dict(document)
        payload.pop("id", None)
        payload.pop("_id", None)
        payload["updatedAt"] = utcnow()
        updated = await self.collection.find_one_and_update(
            {"_id": object_id},
            {"$set": payload},
            return_document=True,
        )
        return serialize_doc(updated)

    async def patch(self, entity_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Partially update a document.

        ``None`` values are ignored so that PATCH only modifies provided fields.

        Args:
            entity_id: The string id of the document.
            updates: Mapping of fields to update (camelCase keys).

        Returns:
            The updated document, or ``None`` if not found.
        """
        object_id = to_object_id(entity_id)
        if object_id is None:
            return None
        clean_updates = {k: v for k, v in updates.items() if v is not None}
        clean_updates.pop("id", None)
        clean_updates.pop("_id", None)
        clean_updates["updatedAt"] = utcnow()
        updated = await self.collection.find_one_and_update(
            {"_id": object_id},
            {"$set": clean_updates},
            return_document=True,
        )
        return serialize_doc(updated)

    async def delete(self, entity_id: str) -> bool:
        """Delete a document by id.

        Args:
            entity_id: The string id of the document.

        Returns:
            ``True`` if a document was deleted, ``False`` otherwise.
        """
        object_id = to_object_id(entity_id)
        if object_id is None:
            return False
        result = await self.collection.delete_one({"_id": object_id})
        return result.deleted_count > 0
