"""MongoDB client lifecycle and database access.

A single :class:`MongoConnection` manages the async ``motor`` client. The app
opens the connection during startup (lifespan) and closes it on shutdown. The
active database handle is then injected into repositories via FastAPI
dependencies.

Indexes that support the documented query patterns (filtering by ``runId``,
``promptId``, ``useCaseKey`` etc.) are created on startup.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MongoConnection:
    """Holds the async Mongo client and database handle.

    The instance is created once and shared for the lifetime of the process.

    Attributes:
        client: The underlying ``AsyncIOMotorClient`` (``None`` until connected).
        db: The active database handle (``None`` until connected).
    """

    def __init__(self) -> None:
        """Initialise an unconnected connection holder."""
        self.client: AsyncIOMotorClient | None = None
        self.db: AsyncIOMotorDatabase | None = None

    async def connect(self, settings: Settings) -> None:
        """Open the Mongo client and create supporting indexes.

        Args:
            settings: Application settings providing the URI and DB name.
        """
        logger.info("Connecting to MongoDB at %s (db=%s)", settings.mongodb_uri, settings.mongodb_db)
        self.client = AsyncIOMotorClient(settings.mongodb_uri, tz_aware=True)
        self.db = self.client[settings.mongodb_db]
        await self._ensure_indexes()

    async def close(self) -> None:
        """Close the Mongo client if it is open."""
        if self.client is not None:
            logger.info("Closing MongoDB connection")
            self.client.close()
            self.client = None
            self.db = None

    def get_database(self) -> AsyncIOMotorDatabase:
        """Return the active database handle.

        Returns:
            The connected database handle.

        Raises:
            RuntimeError: If called before :meth:`connect`.
        """
        if self.db is None:
            raise RuntimeError("MongoDB is not connected. Did the app start correctly?")
        return self.db

    async def _ensure_indexes(self) -> None:
        """Create indexes that back the documented query/filter patterns."""
        assert self.db is not None  # narrowed by caller; aids type checkers.

        # Common filters across collections.
        await self.db.prompts.create_index("useCaseKey")
        await self.db.prompt_versions.create_index("promptId")
        await self.db.prompt_versions.create_index([("promptId", 1), ("versionNumber", 1)])
        await self.db.runs.create_index("useCaseKey")
        await self.db.runs.create_index("promptId")
        await self.db.responses.create_index("runId")
        await self.db.responses.create_index([("runId", 1), ("modelKey", 1)])
        await self.db.evaluations.create_index("runId")
        # Natural upsert key for evaluations (FR-14).
        await self.db.evaluations.create_index(
            [("runId", 1), ("responseId", 1), ("modelKey", 1)], unique=True
        )
        await self.db.prompt_suggestions.create_index("promptVersionId")
        await self.db.metrics_logs.create_index("runId")
        logger.info("MongoDB indexes ensured")


# Process-wide connection holder. Wired up in ``app.main`` lifespan.
mongo = MongoConnection()
