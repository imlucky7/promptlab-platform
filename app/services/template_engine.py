"""Template engine: load use-case templates and render prompt text.

The engine loads a template from the ``use_case_templates`` collection by key
and applies the stored ``normalizedPromptTemplate`` to the structured inputs
using Jinja2. A sandboxed environment is used so that arbitrary template logic
cannot perform unsafe operations.
"""

from __future__ import annotations

from typing import Any

from jinja2 import ChainableUndefined
from jinja2.sandbox import SandboxedEnvironment

from app.core.errors import NotFoundError
from app.db.repositories.use_case_templates_repo import UseCaseTemplatesRepository
from app.models.use_case_templates import DEFAULT_MODEL


class TemplateEngine:
    """Renders normalized prompt text from a use-case template and inputs."""

    def __init__(self, templates_repo: UseCaseTemplatesRepository) -> None:
        """Initialise the engine.

        Args:
            templates_repo: Repository used to load templates by key.
        """
        self._templates_repo = templates_repo
        # ``ChainableUndefined`` renders missing/optional placeholders as empty
        # strings (and is falsy in ``{% if %}`` checks) instead of raising, so
        # optional fields like ``budget`` can be omitted safely.
        self._env = SandboxedEnvironment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=ChainableUndefined,
        )

    async def get_template(
        self, use_case_key: str, model: str = DEFAULT_MODEL
    ) -> dict[str, Any]:
        """Load a template document by use-case key and model.

        The model-specific variant is preferred. When it does not exist the
        generic :data:`DEFAULT_MODEL` variant is used as a fallback, so any model
        can still be previewed against a use case.

        Args:
            use_case_key: The use-case key (e.g. ``"travel"``).
            model: The model key to honor (e.g. ``"claude"``).

        Returns:
            The serialised template document.

        Raises:
            NotFoundError: If neither a model-specific nor a default template
                exists for ``use_case_key``.
        """
        template = await self._templates_repo.get_by_key(use_case_key, model)
        if template is None and model != DEFAULT_MODEL:
            template = await self._templates_repo.get_by_key(use_case_key, DEFAULT_MODEL)
        if template is None:
            raise NotFoundError(
                f"No use-case template found for key '{use_case_key}'.",
                details={"useCaseKey": use_case_key, "model": model},
            )
        return template

    async def list_templates(self, use_case_key: str) -> list[dict[str, Any]]:
        """List every template variant available for a use-case key.

        Args:
            use_case_key: The use-case key (e.g. ``"travel"``).

        Returns:
            The serialised template documents for every registered model.

        Raises:
            NotFoundError: If no template exists for ``use_case_key``.
        """
        templates = await self._templates_repo.list_by_key(use_case_key)
        if not templates:
            raise NotFoundError(
                f"No use-case template found for key '{use_case_key}'.",
                details={"useCaseKey": use_case_key},
            )
        return templates

    async def render(
        self,
        use_case_key: str,
        structured_inputs: dict[str, Any],
        model: str = DEFAULT_MODEL,
    ) -> str:
        """Render the prompt text for a use case, model and its inputs.

        Args:
            use_case_key: The use-case key.
            structured_inputs: Mapping of input field names to values.
            model: The model key whose template variant should be rendered.

        Returns:
            The rendered prompt text, with trailing whitespace trimmed.

        Raises:
            NotFoundError: If the template does not exist.
        """
        template_doc = await self.get_template(use_case_key, model)
        return self.render_document(template_doc, structured_inputs)

    def render_document(
        self, template_doc: dict[str, Any], structured_inputs: dict[str, Any]
    ) -> str:
        """Render an already-loaded template document against inputs.

        Args:
            template_doc: A serialised template document.
            structured_inputs: Mapping of input field names to values.

        Returns:
            The rendered prompt text, with trailing whitespace trimmed.
        """
        template_text: str = template_doc.get("normalizedPromptTemplate", "")
        template = self._env.from_string(template_text)
        rendered = template.render(**structured_inputs)
        return rendered.strip()
