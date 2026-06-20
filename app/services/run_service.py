"""Run orchestration service.

Implements the end-to-end "create run" flow: resolve (or generate) the prompt
and prompt-version ``ObjectId`` references, persist the prompt/version when they
do not yet exist, execute each previewed prompt against its target model through
the gateway, persist everything, and keep the ``lastRunId`` reference current.

Keeping this orchestration in a service keeps the route handlers thin.
"""

from __future__ import annotations

import json
from typing import Any

from bson import ObjectId

from app.core.logging import get_logger
from app.db.repositories.evaluations_repo import EvaluationsRepository
from app.db.repositories.prompt_versions_repo import PromptVersionsRepository
from app.db.repositories.prompts_repo import PromptsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.runs import RunCreate, RunWithResponses
from app.services.execution_engine import ExecutionEngine
from app.services.versioning_engine import VersioningEngine

logger = get_logger(__name__)
_LOG_TEXT_LIMIT = 1000


def _truncate_for_log(text: str, limit: int = _LOG_TEXT_LIMIT) -> str:
    """Clip long text for log output.

    Args:
        text: The string to truncate.
        limit: Maximum number of characters to retain.

    Returns:
        The original text, or a clipped preview with a total-length suffix.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text)} chars total)"


def _prompt_log_entry(model: str, prompt_text: str) -> dict[str, object]:
    """Build a log-safe summary for a single model prompt.

    Args:
        model: Logical model key.
        prompt_text: Full prompt text sent to the model.

    Returns:
        A mapping suitable for JSON logging.
    """
    return {
        "model": model,
        "promptTextLength": len(prompt_text),
        "promptTextPreview": _truncate_for_log(prompt_text),
    }


def _response_log_entry(response: dict[str, Any]) -> dict[str, object]:
    """Build a log-safe summary for a persisted response document.

    Args:
        response: Serialised response document from the repository.

    Returns:
        A mapping suitable for JSON logging (large fields omitted/truncated).
    """
    text = response.get("text") or ""
    return {
        "id": response.get("id"),
        "modelKey": response.get("modelKey"),
        "status": response.get("status"),
        "usage": response.get("usage"),
        "errorMessage": response.get("errorMessage"),
        "textPreview": _truncate_for_log(text),
    }


class RunService:
    """Coordinates prompt/version materialisation and run execution."""

    def __init__(
        self,
        prompts_repo: PromptsRepository,
        versions_repo: PromptVersionsRepository,
        runs_repo: RunsRepository,
        responses_repo: ResponsesRepository,
        evaluations_repo: EvaluationsRepository,
        execution_engine: ExecutionEngine,
        versioning_engine: VersioningEngine,
    ) -> None:
        """Initialise the run service.

        Args:
            prompts_repo: Repository for ``prompts``.
            versions_repo: Repository for ``prompt_versions``.
            runs_repo: Repository for ``runs``.
            responses_repo: Repository for ``responses``.
            evaluations_repo: Repository for ``evaluations``.
            execution_engine: Executes prompts across models.
            versioning_engine: Creates versions with monotonic numbering.
        """
        self._prompts_repo = prompts_repo
        self._versions_repo = versions_repo
        self._runs_repo = runs_repo
        self._responses_repo = responses_repo
        self._evaluations_repo = evaluations_repo
        self._execution_engine = execution_engine
        self._versioning_engine = versioning_engine

    async def create_and_execute(self, data: RunCreate) -> RunWithResponses:
        """Create a run from the previewed prompts, then execute it.

        The run's ``promptId``/``promptVersionId`` are taken from the request
        when provided, otherwise generated as fresh MongoDB ``ObjectId`` values.
        Each previewed prompt is executed against its own model through the
        gateway.

        Args:
            data: The run creation payload (per-model previewed prompts).

        Returns:
            A :class:`RunWithResponses` with the persisted run and responses.
        """
        # 1) Resolve prompt/version ids (honour the request, else system-generate).
        prompt_id = data.prompt_id or str(ObjectId())
        prompt_version_id = data.prompt_version_id or str(ObjectId())

        models = [item.model for item in data.prompts]
        prompts_payload = [
            {"model": item.model, "promptText": item.prompt_text} for item in data.prompts
        ]
        primary_text = data.prompts[0].prompt_text

        # 2) Ensure the backing prompt + version documents exist.
        await self._ensure_prompt(prompt_id, data)
        await self._ensure_version(
            prompt_version_id, prompt_id, data, prompts_payload, primary_text
        )

        # 3) Persist the run document.
        run_doc = await self._runs_repo.create(
            {
                "promptVersionId": prompt_version_id,
                "promptId": prompt_id,
                "useCaseKey": data.use_case_key,
                "inputs": data.inputs,
                "models": models,
                "prompts": prompts_payload,
                "promptText": primary_text,
            }
        )
        run_id = run_doc["id"]

        """logger.info(
            "Run execution request:\n%s",
            json.dumps(
                {
                    "runId": run_id,
                    "promptId": prompt_id,
                    "promptVersionId": prompt_version_id,
                    "useCaseKey": data.use_case_key,
                    "models": models,
                    "prompts": [
                        _prompt_log_entry(item.model, item.prompt_text)
                        for item in data.prompts
                    ],
                },
                indent=2,
                default=str,
                sort_keys=True,
            ),
        )
        """

        # 4) Execute every previewed prompt against its model and persist results.
        responses = await self._execution_engine.execute(
            run_id, [(item.model, item.prompt_text) for item in data.prompts]
        )

        """logger.info(
            "Run execution response:\n%s",
            json.dumps(
                {
                    "runId": run_id,
                    "responses": [_response_log_entry(item) for item in responses],
                },
                indent=2,
                default=str,
                sort_keys=True,
            ),
        )
        """
        # 5) Keep the version's lastRunId reference current.
        await self._versions_repo.patch(prompt_version_id, {"lastRunId": run_id})

        return RunWithResponses.model_validate(
            {"run": run_doc, "responses": responses, "evaluations": []}
        )

    async def get_with_children(self, run_id: str) -> RunWithResponses | None:
        """Fetch a run with its responses and evaluations (FR-10).

        Args:
            run_id: The run id.

        Returns:
            A :class:`RunWithResponses`, or ``None`` if the run does not exist.
        """
        run_doc = await self._runs_repo.get(run_id)
        if run_doc is None:
            return None
        responses = await self._responses_repo.list_by_run(run_id)
        evaluations = await self._evaluations_repo.list_by_run(run_id)
        return RunWithResponses.model_validate(
            {"run": run_doc, "responses": responses, "evaluations": evaluations}
        )

    async def _ensure_prompt(self, prompt_id: str, data: RunCreate) -> None:
        """Create the prompt document for ``prompt_id`` when it does not exist.

        Args:
            prompt_id: The resolved prompt id (a valid ``ObjectId`` string).
            data: The run creation payload.
        """
        if await self._prompts_repo.get(prompt_id) is not None:
            return
        await self._prompts_repo.create(
            {
                "_id": ObjectId(prompt_id),
                "title": data.prompt_title or "Untitled prompt",
                "useCaseKey": data.use_case_key,
                "createdByUserId": None,
            }
        )

    async def _ensure_version(
        self,
        version_id: str,
        prompt_id: str,
        data: RunCreate,
        prompts_payload: list[dict[str, Any]],
        primary_text: str,
    ) -> None:
        """Create the prompt-version document for ``version_id`` when missing.

        Args:
            version_id: The resolved version id (a valid ``ObjectId`` string).
            prompt_id: The owning prompt id.
            data: The run creation payload.
            prompts_payload: The per-model prompts to snapshot on the version.
            primary_text: The primary prompt text (first variant).
        """
        if await self._versions_repo.get(version_id) is not None:
            return
        await self._versioning_engine.create_version(
            {
                "_id": ObjectId(version_id),
                "promptId": prompt_id,
                "versionName": data.version_name,
                "structuredInputs": data.inputs,
                "promptText": primary_text,
                "prompts": prompts_payload,
                "createdByUserId": None,
                "parentVersionId": None,
                "lastRunId": None,
            }
        )
