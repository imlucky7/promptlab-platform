"""Rule-based prompt suggestion engine.

Analyses the structured inputs and assembled prompt text to detect common prompt
engineering gaps (missing constraints, output format, examples or persona) and
returns human-readable suggestions. Suggestions are non-destructive hints; the
caller decides whether to surface them in a preview or persist them.
"""

from __future__ import annotations

from typing import Any

from app.models.prompt_suggestions import SuggestionItem


class SuggestionEngine:
    """Generates improvement suggestions for a prompt."""

    def analyze(
        self,
        prompt_text: str,
        structured_inputs: dict[str, Any],
    ) -> list[SuggestionItem]:
        """Analyse a prompt and return suggestions.

        Args:
            prompt_text: The assembled prompt text.
            structured_inputs: The structured inputs used to build the prompt.

        Returns:
            A list of :class:`SuggestionItem` recommendations (possibly empty).
        """
        suggestions: list[SuggestionItem] = []
        lowered = prompt_text.lower()

        # 1. Constraints: encourage explicit budget/mobility constraints.
        has_constraints = bool(str(structured_inputs.get("constraints", "")).strip())
        if not has_constraints and "constraint" not in lowered:
            suggestions.append(
                SuggestionItem(
                    suggestion_type="add_constraints",
                    description=(
                        "Add explicit budget and mobility constraints so the model "
                        "tailors recommendations (e.g. accessibility, pace, dietary needs)."
                    ),
                )
            )

        # 2. Output format: encourage a clearly specified output structure.
        format_markers = ("output format", "format:", "headings", "bullet")
        if not any(marker in lowered for marker in format_markers):
            suggestions.append(
                SuggestionItem(
                    suggestion_type="clarify_output_format",
                    description=(
                        "Specify the desired output format, e.g. a heading per day with "
                        "bullet points for morning/afternoon/evening."
                    ),
                )
            )

        # 3. Examples: encourage few-shot examples for consistency.
        if "example" not in lowered:
            suggestions.append(
                SuggestionItem(
                    suggestion_type="add_examples",
                    description=(
                        "Consider adding one example day as a few-shot example to anchor "
                        "the structure and level of detail."
                    ),
                )
            )

        # 4. Persona/role: encourage an explicit role definition.
        persona_markers = ("you are", "act as", "as a", "your role")
        if not any(marker in lowered for marker in persona_markers):
            suggestions.append(
                SuggestionItem(
                    suggestion_type="define_persona",
                    description=(
                        "Define a clear persona/role for the assistant (e.g. 'You are an "
                        "expert travel planner') to improve tone and expertise."
                    ),
                )
            )

        return suggestions
