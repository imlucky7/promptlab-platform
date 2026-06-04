"""Prompt builder service.

Coordinates template rendering, token estimation and suggestion generation. It
provides:

* :meth:`build_prompt` - render the final prompt text from structured inputs.
* :meth:`preview` - build the prompt, estimate tokens and generate suggestions
  without persisting anything (the dedicated preview flow).
"""

from __future__ import annotations

from typing import Any

from app.models.preview import PreviewResponse
from app.services.suggestion_engine import SuggestionEngine
from app.services.template_engine import TemplateEngine
from app.services.token_estimator import TokenEstimator


class PromptBuilderService:
    """Builds prompt text and assembles preview information."""

    def __init__(
        self,
        template_engine: TemplateEngine,
        token_estimator: TokenEstimator,
        suggestion_engine: SuggestionEngine,
    ) -> None:
        """Initialise the prompt builder.

        Args:
            template_engine: Renders templates into prompt text.
            token_estimator: Estimates token usage.
            suggestion_engine: Generates improvement suggestions.
        """
        self._template_engine = template_engine
        self._token_estimator = token_estimator
        self._suggestion_engine = suggestion_engine

    async def build_prompt(self, use_case_key: str, structured_inputs: dict[str, Any]) -> str:
        """Render the final prompt text for a use case and inputs.

        Args:
            use_case_key: The use-case key (e.g. ``"travel"``).
            structured_inputs: Structured inputs to render.

        Returns:
            The assembled prompt text.

        Raises:
            NotFoundError: If the use-case template does not exist.
        """
        return await self._template_engine.render(use_case_key, structured_inputs)

    async def preview(
        self,
        use_case_key: str,
        structured_inputs: dict[str, Any],
        *,
        token_estimation_mode: str = "default",
    ) -> PreviewResponse:
        """Build a non-persisted preview of a prompt.

        Args:
            use_case_key: The use-case key.
            structured_inputs: Structured inputs to render.
            token_estimation_mode: Estimation override (``default``/``gateway``/``local``).

        Returns:
            A :class:`PreviewResponse` with prompt text, token estimates and
            suggestions.

        Raises:
            NotFoundError: If the use-case template does not exist.
        """
        prompt_text = await self.build_prompt(use_case_key, structured_inputs)
        token_estimates = await self._token_estimator.estimate(
            prompt_text, mode_override=token_estimation_mode
        )
        suggestions = self._suggestion_engine.analyze(prompt_text, structured_inputs)
        return PreviewResponse(
            use_case_key=use_case_key,
            structured_inputs=structured_inputs,
            prompt_text=prompt_text,
            token_estimates=token_estimates,
            suggestions=suggestions,
        )
