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

    async def get_template(self, use_case_key: str) -> dict[str, Any]:
        """Load a template document by use-case key.

        Args:
            use_case_key: The use-case key (e.g. ``"travel"``).

        Returns:
            The serialised template document.

        Raises:
            NotFoundError: If no template exists for ``use_case_key``.
        """
        template = await self._templates_repo.get_by_key(use_case_key)
        if template is None:
            raise NotFoundError(
                f"No use-case template found for key '{use_case_key}'.",
                details={"useCaseKey": use_case_key},
            )
        return template

    async def render(self, use_case_key: str, structured_inputs: dict[str, Any]) -> str:
        """Render the prompt text for a use case and its inputs.

        Args:
            use_case_key: The use-case key.
            structured_inputs: Mapping of input field names to values.

        Returns:
            The rendered prompt text, with trailing whitespace trimmed.

        Raises:
            NotFoundError: If the template does not exist.
        """
        template_doc = await self.get_template(use_case_key)
        template_text: str = template_doc.get("normalizedPromptTemplate", "")
        template = self._env.from_string(template_text)
        rendered = template.render(**structured_inputs)
        return rendered.strip()
