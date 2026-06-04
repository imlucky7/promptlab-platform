"""Repository for the ``prompt_versions`` collection."""

from __future__ import annotations

from app.db.repositories.base_repo import BaseRepository


class PromptVersionsRepository(BaseRepository):
    """Data access for prompt versions."""

    collection_name = "prompt_versions"

    async def get_max_version_number(self, prompt_id: str) -> int:
        """Return the highest ``versionNumber`` for a prompt.

        Args:
            prompt_id: The owning prompt id.

        Returns:
            The maximum version number, or ``0`` if the prompt has no versions.
        """
        doc = await self.collection.find_one(
            {"promptId": prompt_id},
            sort=[("versionNumber", -1)],
            projection={"versionNumber": 1},
        )
        if not doc:
            return 0
        return int(doc.get("versionNumber", 0))
