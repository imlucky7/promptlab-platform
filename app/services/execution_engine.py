"""Execution engine.

Runs a prompt against one or more models concurrently via the LLM gateway, then
normalises and persists the resulting responses and metrics. Model calls are
executed with :func:`asyncio.gather` so total latency is bounded by the slowest
provider, and a failure in one provider never blocks the others (FR-08, FR-09).
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.logging import get_logger
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.services.llm_gateway_client import LLMGatewayClient, LLMResult
from app.services.response_normalizer import ResponseNormalizer

logger = get_logger(__name__)


class ExecutionEngine:
    """Executes prompts across models and persists the outcomes."""

    def __init__(
        self,
        gateway: LLMGatewayClient,
        normalizer: ResponseNormalizer,
        responses_repo: ResponsesRepository,
        metrics_repo: MetricsRepository,
    ) -> None:
        """Initialise the execution engine.

        Args:
            gateway: Client used to call the LLM gateway.
            normalizer: Converts gateway results into documents.
            responses_repo: Repository for ``responses``.
            metrics_repo: Repository for ``metrics_logs``.
        """
        self._gateway = gateway
        self._normalizer = normalizer
        self._responses_repo = responses_repo
        self._metrics_repo = metrics_repo

    async def execute(
        self,
        run_id: str,
        model_prompts: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Execute each model with its own prompt and persist results.

        Args:
            run_id: The owning run id.
            model_prompts: A list of ``(model_key, prompt_text)`` pairs. Each
                model receives its own previewed prompt (e.g. a ChatGPT-tuned
                prompt and a Claude-tuned prompt), all sent through the gateway.

        Returns:
            The list of persisted, serialised response documents.
        """
        if not model_prompts:
            return []

        logger.info(
            "Executing run %s across models: %s",
            run_id,
            [model_key for model_key, _ in model_prompts],
        )

        # Fire all gateway calls concurrently. Exceptions are returned (not
        # raised) so a single failure does not abort the gather.
        tasks = [
            self._gateway.chat_completion(model_key, prompt_text)
            for model_key, prompt_text in model_prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        persisted: list[dict[str, Any]] = []
        for (model_key, _prompt_text), result in zip(model_prompts, results, strict=True):
            llm_result = self._coerce_result(model_key, result)
            response_doc = self._normalizer.to_response_document(run_id, llm_result)
            saved = await self._responses_repo.create(response_doc)
            persisted.append(saved)

            # Best-effort metrics logging; never fail the run for metrics.
            try:
                metrics_doc = self._normalizer.to_metrics_document(
                    run_id, saved["id"], llm_result
                )
                await self._metrics_repo.create(metrics_doc)
            except Exception as exc:  # pragma: no cover - defensive.
                logger.warning("Failed to persist metrics for run %s: %s", run_id, exc)

        return persisted

    def _coerce_result(self, model_key: str, result: LLMResult | BaseException) -> LLMResult:
        """Convert a gather outcome into a concrete :class:`LLMResult`.

        Args:
            model_key: The model key the result corresponds to.
            result: Either an :class:`LLMResult` or an exception captured by
                :func:`asyncio.gather`.

        Returns:
            A concrete :class:`LLMResult`; exceptions become error results.
        """
        if isinstance(result, LLMResult):
            return result
        logger.warning("Unexpected error executing model '%s': %s", model_key, result)
        return LLMResult(
            model_key=model_key,
            provider_model_name=model_key,
            gateway_model_identifier=model_key,
            status="error",
            error_message=str(result),
        )
