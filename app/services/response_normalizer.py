"""Response normalizer.

Converts an internal :class:`LLMResult` into the document shape persisted in the
``responses`` collection (and the matching ``metrics_logs`` entry). Raw payloads
are truncated defensively to keep documents within sensible size limits.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.services.llm_gateway_client import LLMResult

# Maximum number of characters to retain when storing raw payloads/responses.
_MAX_RAW_CHARS = 20_000


class ResponseNormalizer:
    """Normalises gateway results into persistable documents."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the normalizer.

        Args:
            settings: Application settings (used for cost estimation pricing).
        """
        self._settings = settings

    def to_response_document(self, run_id: str, result: LLMResult) -> dict[str, Any]:
        """Build a ``responses`` document from a gateway result.

        Args:
            run_id: The owning run id.
            result: The normalised gateway result.

        Returns:
            A document (camelCase keys) ready to insert into ``responses``.
        """
        return {
            "runId": run_id,
            "modelKey": result.model_key,
            "providerModelName": result.provider_model_name,
            "gatewayModelIdentifier": result.gateway_model_identifier,
            "promptPayload": _truncate_json(result.prompt_payload),
            "rawResponse": _truncate_json(result.raw_response),
            "text": result.text,
            "usage": {
                "inputTokens": result.input_tokens,
                "outputTokens": result.output_tokens,
                "latencyMs": result.latency_ms,
            },
            "status": result.status,
            "errorMessage": result.error_message,
        }

    def to_metrics_document(
        self, run_id: str, response_id: str, result: LLMResult
    ) -> dict[str, Any]:
        """Build a ``metrics_logs`` document from a gateway result.

        Args:
            run_id: The owning run id.
            response_id: The persisted response id.
            result: The normalised gateway result.

        Returns:
            A document (camelCase keys) ready to insert into ``metrics_logs``.
        """
        return {
            "runId": run_id,
            "responseId": response_id,
            "modelKey": result.model_key,
            "tokenCostEstimate": self._estimate_cost(result),
            "extra": {
                "inputTokens": result.input_tokens,
                "outputTokens": result.output_tokens,
                "latencyMs": result.latency_ms,
            },
        }

    def _estimate_cost(self, result: LLMResult) -> float | None:
        """Estimate the monetary cost of a result from configured pricing.

        Args:
            result: The normalised gateway result.

        Returns:
            The estimated cost, or ``None`` if no pricing is configured.
        """
        model_cfg = self._settings.get_model_config(result.model_key)
        if model_cfg is None:
            return None
        if model_cfg.input_cost_per_1k is None and model_cfg.output_cost_per_1k is None:
            return None
        input_cost = (model_cfg.input_cost_per_1k or 0.0) * (result.input_tokens / 1000.0)
        output_cost = (model_cfg.output_cost_per_1k or 0.0) * (result.output_tokens / 1000.0)
        return round(input_cost + output_cost, 6)


def _truncate_json(value: dict[str, Any]) -> dict[str, Any]:
    """Truncate a JSON-serialisable mapping that exceeds the size budget.

    Args:
        value: The mapping to potentially truncate.

    Returns:
        The original mapping, or a ``{"_truncated": ...}`` placeholder when the
        serialised form exceeds :data:`_MAX_RAW_CHARS`.
    """
    try:
        encoded = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return {"_unserializable": True}
    if len(encoded) <= _MAX_RAW_CHARS:
        return value
    return {"_truncated": True, "preview": encoded[:_MAX_RAW_CHARS]}
