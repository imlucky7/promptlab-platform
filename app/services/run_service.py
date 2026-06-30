"""Run orchestration service.

Creates a run document, resolves the executable prompt (full Ollama preview for
qwen3, compact model template for other models), executes via the gateway, and
persists responses.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.core.logging import get_logger
from app.db.repositories.evaluations_repo import EvaluationsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.runs import RunCreate, RunWithResponses
from app.models.stream import complete_event, progress_event, token_event
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

    def _resolve_executed_prompt(self, data: RunCreate) -> str:
        """Return the prompt text that will be sent to the target model."""
        if data.model == QWEN3_MODEL_KEY:
            return data.prompt_text
        return build_run_prompt_for_model(data.inputs, data.model)

    async def create_and_execute(self, data: RunCreate) -> RunWithResponses:
        """Create a run from the previewed prompt, then execute it."""
        executed_prompt = self._resolve_executed_prompt(data)
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

    async def create_and_execute_stream(
        self, data: RunCreate
    ) -> AsyncIterator[dict[str, Any]]:
        """Create and execute a run while streaming progress and token events."""
        yield progress_event("creating_run", "Creating run…")
        executed_prompt = self._resolve_executed_prompt(data)
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

        yield progress_event("executing", f"Running {data.model}…")

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        responses_holder: list[dict[str, Any]] = []

        async def on_token(delta: str) -> None:
            await queue.put(token_event("response", delta))

        async def execute_task() -> None:
            try:
                responses_holder.extend(
                    await self._execution_engine.execute(
                        run_id,
                        [(data.model, executed_prompt)],
                        on_token=on_token,
                    )
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(execute_task())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await task

        yield progress_event("persisting", "Finalizing run…")
        result = RunWithResponses.model_validate(
            {"run": run_doc, "responses": responses_holder, "evaluations": []}
        )
        yield complete_event(result)

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
