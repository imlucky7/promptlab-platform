"""Run orchestration service.

Creates a run document, resolves the executable prompt (full Ollama preview for
qwen3, compact model template for other models), executes via the gateway, and
persists responses.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.db.repositories.evaluations_repo import EvaluationsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.runs import RunCreate, RunWithResponses
from app.services.execution_engine import ExecutionEngine
from app.services.run_prompt_templates import QWEN3_MODEL_KEY, build_run_prompt_for_model

logger = get_logger(__name__)


class RunService:
    """Coordinates run creation and execution."""

    def __init__(
        self,
        runs_repo: RunsRepository,
        responses_repo: ResponsesRepository,
        evaluations_repo: EvaluationsRepository,
        execution_engine: ExecutionEngine,
    ) -> None:
        self._runs_repo = runs_repo
        self._responses_repo = responses_repo
        self._evaluations_repo = evaluations_repo
        self._execution_engine = execution_engine

    async def create_and_execute(self, data: RunCreate) -> RunWithResponses:
        """Create a run from the previewed prompt, then execute it."""
        if data.model == QWEN3_MODEL_KEY:
            executed_prompt = data.prompt_text
        else:
            executed_prompt = build_run_prompt_for_model(data.inputs, data.model)

        prompts_payload = [{"model": data.model, "promptText": executed_prompt}]

        run_doc = await self._runs_repo.create(
            {
                "promptVersionId": data.prompt_version_id,
                "promptId": data.prompt_id,
                "useCaseKey": data.use_case_key,
                "inputs": data.inputs,
                "models": [data.model],
                "prompts": prompts_payload,
                "promptText": executed_prompt,
            }
        )
        run_id = run_doc["id"]

        responses = await self._execution_engine.execute(
            run_id, [(data.model, executed_prompt)]
        )

        return RunWithResponses.model_validate(
            {"run": run_doc, "responses": responses, "evaluations": []}
        )

    async def get_with_children(self, run_id: str) -> RunWithResponses | None:
        """Fetch a run with its responses and evaluations."""
        run_doc = await self._runs_repo.get(run_id)
        if run_doc is None:
            return None
        responses = await self._responses_repo.list_by_run(run_id)
        evaluations = await self._evaluations_repo.list_by_run(run_id)
        return RunWithResponses.model_validate(
            {"run": run_doc, "responses": responses, "evaluations": evaluations}
        )
