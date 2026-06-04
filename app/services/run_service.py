"""Run orchestration service.

Implements the end-to-end "create run" flow described in the PRD (FR-06/FR-07):
optionally materialise a prompt + version on first run, build (or reuse) the
prompt text, execute the selected models, persist everything, and keep the
``lastRunId`` reference up to date.

Keeping this orchestration in a service keeps the route handlers thin.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import NotFoundError
from app.db.repositories.evaluations_repo import EvaluationsRepository
from app.db.repositories.prompt_versions_repo import PromptVersionsRepository
from app.db.repositories.prompts_repo import PromptsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.runs import RunCreate, RunWithResponses
from app.services.execution_engine import ExecutionEngine
from app.services.prompt_builder import PromptBuilderService
from app.services.versioning_engine import VersioningEngine


class RunService:
    """Coordinates prompt/version materialisation and run execution."""

    def __init__(
        self,
        prompts_repo: PromptsRepository,
        versions_repo: PromptVersionsRepository,
        runs_repo: RunsRepository,
        responses_repo: ResponsesRepository,
        evaluations_repo: EvaluationsRepository,
        prompt_builder: PromptBuilderService,
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
            prompt_builder: Builds prompt text from inputs/templates.
            execution_engine: Executes prompts across models.
            versioning_engine: Creates versions with monotonic numbering.
        """
        self._prompts_repo = prompts_repo
        self._versions_repo = versions_repo
        self._runs_repo = runs_repo
        self._responses_repo = responses_repo
        self._evaluations_repo = evaluations_repo
        self._prompt_builder = prompt_builder
        self._execution_engine = execution_engine
        self._versioning_engine = versioning_engine

    async def create_and_execute(self, data: RunCreate) -> RunWithResponses:
        """Create a run, materialising prompt/version as needed, then execute.

        Args:
            data: The run creation payload.

        Returns:
            A :class:`RunWithResponses` with the persisted run and responses.

        Raises:
            NotFoundError: If a provided ``promptVersionId`` does not exist.
        """
        prompt_id = data.prompt_id
        prompt_version_id = data.prompt_version_id

        # 1) Resolve the prompt text and (optionally) a reusable version.
        prompt_text, prompt_id, prompt_version_id = await self._resolve_prompt(
            data, prompt_id, prompt_version_id
        )

        # 2) Persist the run document.
        run_doc = await self._runs_repo.create(
            {
                "promptVersionId": prompt_version_id,
                "promptId": prompt_id,
                "useCaseKey": data.use_case_key,
                "inputs": data.inputs,
                "models": data.models,
                "promptText": prompt_text,
            }
        )
        run_id = run_doc["id"]

        # 3) Execute across models and persist responses + metrics.
        responses = await self._execution_engine.execute(run_id, prompt_text, data.models)

        # 4) Keep the version's lastRunId reference current.
        if prompt_version_id:
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

    async def _resolve_prompt(
        self,
        data: RunCreate,
        prompt_id: str | None,
        prompt_version_id: str | None,
    ) -> tuple[str, str | None, str | None]:
        """Resolve prompt text plus the prompt/version ids to link to the run.

        Implements the three documented cases:

        * Existing version provided -> reuse it.
        * No prompt/version -> create a prompt and its first version.
        * Prompt only -> create a new version under that prompt.

        Args:
            data: The run creation payload.
            prompt_id: The provided prompt id, if any.
            prompt_version_id: The provided version id, if any.

        Returns:
            A tuple ``(prompt_text, prompt_id, prompt_version_id)``.

        Raises:
            NotFoundError: If a provided ``promptVersionId`` does not exist.
        """
        # Case A: an explicit version is provided -> reuse it.
        if prompt_version_id:
            version = await self._versions_repo.get(prompt_version_id)
            if version is None:
                raise NotFoundError(
                    f"Prompt version '{prompt_version_id}' not found.",
                    details={"promptVersionId": prompt_version_id},
                )
            prompt_text = data.prompt_text or version.get("promptText") or ""
            if not prompt_text:
                prompt_text = await self._prompt_builder.build_prompt(
                    data.use_case_key, version.get("structuredInputs", data.inputs)
                )
            return prompt_text, version.get("promptId", prompt_id), prompt_version_id

        # Build the prompt text from explicit text or from inputs/template.
        prompt_text = data.prompt_text or await self._prompt_builder.build_prompt(
            data.use_case_key, data.inputs
        )

        # Case B: no prompt at all -> create the prompt workspace first.
        if prompt_id is None:
            prompt_doc = await self._prompts_repo.create(
                {
                    "title": data.prompt_title or "Untitled prompt",
                    "useCaseKey": data.use_case_key,
                    "createdByUserId": None,
                }
            )
            prompt_id = prompt_doc["id"]

        # Case B/C: create a new version under the (existing or new) prompt.
        version_doc = await self._versioning_engine.create_version(
            {
                "promptId": prompt_id,
                "versionName": data.version_name,
                "structuredInputs": data.inputs,
                "promptText": prompt_text,
                "createdByUserId": None,
                "parentVersionId": None,
                "lastRunId": None,
            }
        )
        return prompt_text, prompt_id, version_doc["id"]
