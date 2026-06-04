"""Versioning engine.

Owns the rules around prompt version numbering, cloning and the
"save new version from last run" behaviour (PRD FR-17 to FR-19). Centralising
this logic keeps version numbers monotonic per prompt and lineage consistent.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import NotFoundError
from app.db.repositories.prompt_versions_repo import PromptVersionsRepository


class VersioningEngine:
    """Manages prompt version numbering, cloning and lineage."""

    def __init__(self, versions_repo: PromptVersionsRepository) -> None:
        """Initialise the engine.

        Args:
            versions_repo: Repository for ``prompt_versions``.
        """
        self._versions_repo = versions_repo

    async def next_version_number(self, prompt_id: str) -> int:
        """Return the next version number for a prompt.

        Args:
            prompt_id: The owning prompt id.

        Returns:
            ``max(existing) + 1``, i.e. ``1`` for the first version.
        """
        current_max = await self._versions_repo.get_max_version_number(prompt_id)
        return current_max + 1

    async def create_version(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new version, assigning the next version number.

        Args:
            data: Version fields (camelCase keys) including ``promptId``.

        Returns:
            The created, serialised version document.
        """
        payload = dict(data)
        payload["versionNumber"] = await self.next_version_number(payload["promptId"])
        if not payload.get("versionName"):
            payload["versionName"] = f"V{payload['versionNumber']}"
        return await self._versions_repo.create(payload)

    async def clone(self, version_id: str, version_name: str | None = None) -> dict[str, Any]:
        """Clone an existing version into a new one (FR-17).

        Args:
            version_id: The id of the version to clone.
            version_name: Optional label for the clone.

        Returns:
            The newly created clone document.

        Raises:
            NotFoundError: If the source version does not exist.
        """
        source = await self._versions_repo.get(version_id)
        if source is None:
            raise NotFoundError(
                f"Prompt version '{version_id}' not found.",
                details={"id": version_id},
            )
        new_version = {
            "promptId": source["promptId"],
            "versionName": version_name or f"Clone of {source.get('versionName', '')}".strip(),
            "structuredInputs": source.get("structuredInputs", {}),
            "promptText": source.get("promptText", ""),
            "createdByUserId": source.get("createdByUserId"),
            "parentVersionId": source["id"],
            "lastRunId": None,
        }
        return await self.create_version(new_version)

    async def save_from_last_run(
        self, version_id: str, version_name: str | None = None
    ) -> dict[str, Any]:
        """Create a new version snapshot from an existing version's state (FR-19).

        Reuses the source version's structured inputs, prompt text and
        ``lastRunId`` to materialise a fresh, named version.

        Args:
            version_id: The id of the source version.
            version_name: Optional label for the new version.

        Returns:
            The newly created version document.

        Raises:
            NotFoundError: If the source version does not exist.
        """
        source = await self._versions_repo.get(version_id)
        if source is None:
            raise NotFoundError(
                f"Prompt version '{version_id}' not found.",
                details={"id": version_id},
            )
        new_version = {
            "promptId": source["promptId"],
            "versionName": version_name,
            "structuredInputs": source.get("structuredInputs", {}),
            "promptText": source.get("promptText", ""),
            "createdByUserId": source.get("createdByUserId"),
            "parentVersionId": source["id"],
            "lastRunId": source.get("lastRunId"),
        }
        return await self.create_version(new_version)
