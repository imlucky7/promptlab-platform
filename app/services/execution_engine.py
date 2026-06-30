"""Execution engine.

Runs a prompt against one or more models concurrently via the LLM gateway, then
normalises and persists the resulting responses and metrics. Model calls are
executed with :func:`asyncio.gather` so total latency is bounded by the slowest
provider, and a failure in one provider never blocks the others (FR-08, FR-09).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.core.logging import get_logger
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.services.llm_gateway_client import LLMGatewayClient, LLMResult, TokenCallback
from app.services.response_normalizer import ResponseNormalizer

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


def _llm_request_payload(run_id: str, model_key: str, prompt_text: str) -> dict[str, object]:
    """Build a log-safe summary for an outbound LLM request.

    Args:
        run_id: The owning run id.
        model_key: Logical model key.
        prompt_text: Full prompt text sent to the model.

    Returns:
        A mapping suitable for JSON logging.
    """
    return {
        "runId": run_id,
        "modelKey": model_key,
        "promptTextLength": len(prompt_text),
        "promptTextPreview": _truncate_for_log(prompt_text),
    }


def _llm_result_payload(run_id: str, result: LLMResult) -> dict[str, object]:
    """Build a log-safe summary for an LLM gateway result.

    Args:
        run_id: The owning run id.
        result: Normalised gateway result.

    Returns:
        A mapping suitable for JSON logging.
    """
    return {
        "runId": run_id,
        "modelKey": result.model_key,
        "providerModelName": result.provider_model_name,
        "gatewayModelIdentifier": result.gateway_model_identifier,
        "status": result.status,
        "usage": {
            "inputTokens": result.input_tokens,
            "outputTokens": result.output_tokens,
            "latencyMs": result.latency_ms,
        },
        "textPreview": _truncate_for_log(result.text),
        "errorMessage": result.error_message,
        "promptPayload": "<omitted>",
        "rawResponse": "<omitted>",
    }


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
        *,
        on_token: TokenCallback | None = None,
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
            self._invoke_model(run_id, model_key, prompt_text, on_token=on_token)
            for model_key, prompt_text in model_prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        persisted: list[dict[str, Any]] = []
        for (model_key, _prompt_text), result in zip(model_prompts, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "LLM error:\n%s",
                    json.dumps(
                        {
                            "runId": run_id,
                            "modelKey": model_key,
                            "errorType": type(result).__name__,
                            "errorMessage": str(result),
                        },
                        indent=2,
                        default=str,
                        sort_keys=True,
                    ),
                )
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

    async def _invoke_model(
        self,
        run_id: str,
        model_key: str,
        prompt_text: str,
        *,
        on_token: TokenCallback | None = None,
    ) -> LLMResult:
        """Call the gateway for one model and log the request/response.

        Args:
            run_id: The owning run id.
            model_key: Logical model key.
            prompt_text: Prompt text to send.

        Returns:
            The normalised gateway result.
        """
        logger.info(
            "LLM request:\n%s",
            json.dumps(
                _llm_request_payload(run_id, model_key, prompt_text),
                indent=2,
                default=str,
                sort_keys=True,
            ),
        )
        if on_token is not None:
            result = await self._gateway.chat_completion_stream(
                model_key, prompt_text, on_token=on_token
            )
        else:
            result = await self._gateway.chat_completion(model_key, prompt_text)
        payload = _llm_result_payload(run_id, result)
        if result.status == "error":
            logger.error(
                "LLM error:\n%s",
                json.dumps(payload, indent=2, default=str, sort_keys=True),
            )
        else:
            logger.info(
                "LLM response:\n%s",
                json.dumps(payload, indent=2, default=str, sort_keys=True),
            )
        return result

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
