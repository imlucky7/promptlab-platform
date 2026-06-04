"""Token estimation supporting both gateway-based and local strategies.

The estimator returns a structured result describing the effective mode used and
the estimates obtained from the gateway and/or the local tokenizer. The behaviour
mirrors the architecture document:

* ``gateway`` - try the gateway, fall back to local on failure.
* ``local``   - use the local tokenizer only.
* ``default`` - use the configured default mode (with the same fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.preview import (
    GatewayTokenEstimate,
    LocalTokenEstimate,
    TokenEstimates,
)

if TYPE_CHECKING:
    from app.services.llm_gateway_client import LLMGatewayClient

logger = get_logger(__name__)


class TokenEstimator:
    """Estimates prompt token usage via the gateway or a local tokenizer."""

    def __init__(self, settings: Settings, gateway: LLMGatewayClient) -> None:
        """Initialise the estimator.

        Args:
            settings: Application settings (default mode + tokenizer model).
            gateway: The LLM gateway client used for gateway-based estimation.
        """
        self._settings = settings
        self._gateway = gateway

    async def estimate(self, text: str, mode_override: str = "default") -> TokenEstimates:
        """Estimate token usage for ``text``.

        Args:
            text: The prompt text to estimate.
            mode_override: ``"default"``, ``"gateway"`` or ``"local"``.

        Returns:
            A :class:`TokenEstimates` describing the effective mode and results.
        """
        # Resolve the requested mode against the configured default.
        requested = (
            self._settings.token_estimation_mode if mode_override == "default" else mode_override
        )

        if requested == "local":
            local = self._estimate_local(text)
            return TokenEstimates(effective_mode="local", from_local=local)

        # requested == "gateway": attempt gateway, fall back to local on failure.
        gateway_estimate = await self._estimate_gateway(text)
        local = self._estimate_local(text)
        if gateway_estimate is not None and gateway_estimate.estimation_available:
            return TokenEstimates(
                effective_mode="gateway",
                from_gateway=gateway_estimate,
                from_local=local,
            )

        # Gateway unavailable -> fall back to local while still surfacing the
        # (unavailable) gateway attempt for transparency.
        logger.info("Gateway token estimation unavailable; falling back to local")
        return TokenEstimates(
            effective_mode="local",
            from_gateway=gateway_estimate,
            from_local=local,
        )

    def _estimate_local(self, text: str) -> LocalTokenEstimate:
        """Estimate tokens locally using tiktoken.

        Args:
            text: The text to tokenize.

        Returns:
            A :class:`LocalTokenEstimate` with the token count.
        """
        return LocalTokenEstimate(input_tokens=self.count_local_tokens(text))

    def count_local_tokens(self, text: str) -> int:
        """Return an approximate local token count for ``text``.

        Falls back to a whitespace heuristic if tiktoken or the requested
        encoding is unavailable.

        Args:
            text: The text to tokenize.

        Returns:
            The estimated number of tokens.
        """
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self._settings.local_tokenizer_model)
            except KeyError:
                # Unknown model name: fall back to a widely-used base encoding.
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:  # pragma: no cover - defensive fallback path.
            # Rough heuristic: ~4 characters per token.
            logger.warning("tiktoken unavailable; using heuristic token count")
            return max(1, len(text) // 4) if text else 0

    async def _estimate_gateway(self, text: str) -> GatewayTokenEstimate | None:
        """Attempt to estimate tokens via the LLM gateway.

        Args:
            text: The text to estimate.

        Returns:
            A :class:`GatewayTokenEstimate`, or ``None`` if the gateway call
            could not be made at all.
        """
        try:
            count = await self._gateway.estimate_tokens(
                text, model=self._settings.token_estimation_model
            )
        except Exception as exc:  # pragma: no cover - network/dependency errors.
            logger.warning("Gateway token estimation failed: %s", exc)
            return GatewayTokenEstimate(input_tokens=0, estimation_available=False)

        if count is None:
            return GatewayTokenEstimate(input_tokens=0, estimation_available=False)
        return GatewayTokenEstimate(input_tokens=count, estimation_available=True)
