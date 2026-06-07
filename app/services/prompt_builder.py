"""Prompt builder service.

Coordinates template rendering, token estimation and suggestion generation. It
provides:

* :meth:`build_prompt` - render the final prompt text from structured inputs.
* :meth:`preview` - build the prompt, estimate tokens and generate suggestions
  without persisting anything (the dedicated preview flow).
"""

from __future__ import annotations

from typing import Any

from app.models.preview import PreviewResponse, TemplatePreview
from app.models.use_case_templates import DEFAULT_MODEL
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

    async def build_prompt(
        self,
        use_case_key: str,
        structured_inputs: dict[str, Any],
        *,
        model: str = DEFAULT_MODEL,
    ) -> str:
        """Render the final prompt text for a use case, model and inputs.

        Args:
            use_case_key: The use-case key (e.g. ``"travel"``).
            structured_inputs: Structured inputs to render.
            model: The model key whose template variant should be rendered.

        Returns:
            The assembled prompt text.

        Raises:
            NotFoundError: If the use-case template does not exist.
        """
        return await self._template_engine.render(use_case_key, structured_inputs, model)

    async def preview(
        self,
        use_case_key: str,
        structured_inputs: dict[str, Any],
        *,
        models: list[str] | None = None,
        token_estimation_mode: str = "default",
    ) -> PreviewResponse:
        """Build a non-persisted preview for one or more template variants.

        When ``models`` is provided, each requested model's template is rendered
        (falling back to the default variant when a model-specific one is
        missing). When ``models`` is empty/``None``, every template variant
        registered for the use case is previewed.

        Args:
            use_case_key: The use-case key.
            structured_inputs: Structured inputs to render.
            models: Optional model keys to preview.
            token_estimation_mode: Estimation override (``default``/``gateway``/``local``).

        Returns:
            A :class:`PreviewResponse` carrying one preview per template variant.

        Raises:
            NotFoundError: If the use-case template does not exist.
        """
        template_docs = await self._resolve_templates(use_case_key, models)
        previews = [
            await self._build_template_preview(
                template_doc, structured_inputs, token_estimation_mode
            )
            for template_doc in template_docs
        ]
        return PreviewResponse(
            use_case_key=use_case_key,
            structured_inputs=structured_inputs,
            previews=previews,
        )

    async def _resolve_templates(
        self, use_case_key: str, models: list[str] | None
    ) -> list[dict[str, Any]]:
        """Resolve the template documents to preview for a request.

        Args:
            use_case_key: The use-case key.
            models: Optional model keys to preview.

        Returns:
            The template documents to render, de-duplicated by their resolved
            ``model`` while preserving request order.

        Raises:
            NotFoundError: If no matching template exists.
        """
        if not models:
            return await self._template_engine.list_templates(use_case_key)

        resolved: list[dict[str, Any]] = []
        seen_models: set[str] = set()
        for model in models:
            template_doc = await self._template_engine.get_template(use_case_key, model)
            resolved_model = template_doc.get("model", model)
            if resolved_model in seen_models:
                continue
            seen_models.add(resolved_model)
            resolved.append(template_doc)
        return resolved

    async def _build_template_preview(
        self,
        template_doc: dict[str, Any],
        structured_inputs: dict[str, Any],
        token_estimation_mode: str,
    ) -> TemplatePreview:
        """Render a single template document into a :class:`TemplatePreview`.

        Args:
            template_doc: The serialised template document.
            structured_inputs: Structured inputs to render.
            token_estimation_mode: Estimation override.

        Returns:
            The assembled per-template preview.
        """
        prompt_text = self._template_engine.render_document(
            template_doc, structured_inputs
        )
        token_estimates = await self._token_estimator.estimate(
            prompt_text, mode_override=token_estimation_mode
        )
        suggestions = self._suggestion_engine.analyze(prompt_text, structured_inputs)
        return TemplatePreview(
            model=template_doc.get("model", DEFAULT_MODEL),
            template_name=template_doc.get("name", ""),
            prompt_text=prompt_text,
            token_estimates=token_estimates,
            suggestions=suggestions,
        )
