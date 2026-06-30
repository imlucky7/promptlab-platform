"""Ollama-backed prompt preview: generate, optimize, and suggest improvements."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import date, datetime
from typing import Any, cast

from app.core.logging import get_logger
from app.models.preview import PreviewResponse, TemplatePreview
from app.models.prompt_suggestions import SuggestionItem, SuggestionType
from app.models.stream import complete_event, progress_event, token_event
from app.services.ollama_client import OllamaClient
from app.services.token_estimator import TokenEstimator

logger = get_logger(__name__)

DEFAULT_PREVIEW_MODEL = "chatgpt"

_GENERATE_SYSTEM = (
    "You are an expert travel prompt engineer. Given structured trip details as JSON, "
    "write a single high-quality prompt that a large language model should receive to "
    "produce a detailed travel itinerary. Include role/persona, trip facts, output "
    "format expectations, and any constraints. Return only the prompt text."
)

_OPTIMIZE_SYSTEM = (
    "You are an expert prompt optimizer for travel planning. Review the draft prompt "
    "and trip context. Return JSON with an improved prompt and suggestions for missing "
    "trip details the user should add."
)

_OPTIMIZE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "optimizedPrompt": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "suggestionType": {"type": "string"},
                    "description": {"type": "string"},
                    "field": {"type": "string"},
                },
                "required": ["suggestionType", "description"],
            },
        },
    },
    "required": ["optimizedPrompt", "suggestions"],
}

_PREVIEW_TEMPLATE_NAME = "Qwen 3 preview"

_KNOWN_SUGGESTION_TYPES: set[str] = {
    "add_examples",
    "clarify_output_format",
    "define_persona",
    "add_constraints",
    "fill_missing_detail",
}


class OllamaPreviewService:
    """Builds preview prompts via two Ollama generate calls."""

    def __init__(self, ollama: OllamaClient, token_estimator: TokenEstimator) -> None:
        self._ollama = ollama
        self._token_estimator = token_estimator

    async def preview(
        self,
        use_case_key: str,
        structured_inputs: dict[str, Any],
        *,
        models: list[str] | None = None,
        token_estimation_mode: str = "default",
    ) -> PreviewResponse:
        """Generate an optimized prompt and suggestions via Ollama."""
        started = time.perf_counter()
        trip_context = map_structured_inputs_to_trip_context(structured_inputs)
        context_json = json.dumps(trip_context, indent=2)

        draft_prompt = await self._ollama.generate(
            f"Trip context JSON:\n{context_json}\n\nWrite the travel planning prompt.",
            system=_GENERATE_SYSTEM,
        )

        optimize_raw = await self._ollama.generate(
            (
                f"Trip context JSON:\n{context_json}\n\n"
                f"Draft prompt:\n{draft_prompt}\n\n"
                "Optimize the prompt and list missing details the user should add."
            ),
            system=_OPTIMIZE_SYSTEM,
            response_format=_OPTIMIZE_FORMAT,
        )

        optimized_prompt, suggestions = parse_optimize_response(optimize_raw, draft_prompt)
        token_estimates = await self._token_estimator.estimate(
            optimized_prompt, mode_override=token_estimation_mode
        )

        model_keys = models if models else [DEFAULT_PREVIEW_MODEL]
        previews = [
            TemplatePreview(
                model=model,
                template_name=_PREVIEW_TEMPLATE_NAME,
                prompt_text=optimized_prompt,
                token_estimates=token_estimates,
                suggestions=suggestions,
            )
            for model in model_keys
        ]

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return PreviewResponse(
            use_case_key=use_case_key,
            structured_inputs=structured_inputs,
            previews=previews,
            latency_ms=latency_ms,
        )

    async def preview_stream(
        self,
        use_case_key: str,
        structured_inputs: dict[str, Any],
        *,
        models: list[str] | None = None,
        token_estimation_mode: str = "default",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream preview progress, draft tokens, and the final preview payload."""
        started = time.perf_counter()
        trip_context = map_structured_inputs_to_trip_context(structured_inputs)
        context_json = json.dumps(trip_context, indent=2)

        yield progress_event("generating_draft", "Drafting prompt…")
        draft_parts: list[str] = []
        async for chunk in self._ollama.generate_stream(
            f"Trip context JSON:\n{context_json}\n\nWrite the travel planning prompt.",
            system=_GENERATE_SYSTEM,
        ):
            draft_parts.append(chunk)
            yield token_event("preview", chunk)
        draft_prompt = "".join(draft_parts)

        yield progress_event("optimizing", "Optimizing prompt…")
        optimize_raw = await self._ollama.generate(
            (
                f"Trip context JSON:\n{context_json}\n\n"
                f"Draft prompt:\n{draft_prompt}\n\n"
                "Optimize the prompt and list missing details the user should add."
            ),
            system=_OPTIMIZE_SYSTEM,
            response_format=_OPTIMIZE_FORMAT,
        )

        optimized_prompt, suggestions = parse_optimize_response(optimize_raw, draft_prompt)

        yield progress_event("estimating_tokens", "Estimating tokens…")
        token_estimates = await self._token_estimator.estimate(
            optimized_prompt, mode_override=token_estimation_mode
        )

        model_keys = models if models else [DEFAULT_PREVIEW_MODEL]
        previews = [
            TemplatePreview(
                model=model,
                template_name=_PREVIEW_TEMPLATE_NAME,
                prompt_text=optimized_prompt,
                token_estimates=token_estimates,
                suggestions=suggestions,
            )
            for model in model_keys
        ]

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response = PreviewResponse(
            use_case_key=use_case_key,
            structured_inputs=structured_inputs,
            previews=previews,
            latency_ms=latency_ms,
        )
        yield complete_event(response)


def map_structured_inputs_to_trip_context(
    structured_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Map workspace ``structuredInputs`` to the trip context sent to Ollama."""
    destinations = _resolve_destinations(structured_inputs)
    days = _compute_trip_days(
        structured_inputs.get("startDate"),
        structured_inputs.get("endDate"),
    )
    travel_style = str(structured_inputs.get("budgetLevel") or "").strip() or None
    interests = _parse_interests(structured_inputs.get("preferences"))
    family = _format_family(
        structured_inputs.get("adults"),
        structured_inputs.get("children"),
    )

    context: dict[str, Any] = {
        "destinations": destinations,
        "days": days,
        "travelStyle": travel_style,
        "interests": interests,
        "family": family,
    }

    origin = str(structured_inputs.get("origin") or "").strip()
    if origin:
        context["origin"] = origin

    constraints = str(structured_inputs.get("constraints") or "").strip()
    if constraints:
        context["constraints"] = constraints

    special_notes = str(structured_inputs.get("specialNotes") or "").strip()
    if special_notes:
        context["specialNotes"] = special_notes

    start_date = structured_inputs.get("startDate")
    end_date = structured_inputs.get("endDate")
    if start_date:
        context["startDate"] = start_date
    if end_date:
        context["endDate"] = end_date

    return context


def _resolve_destinations(structured_inputs: dict[str, Any]) -> list[str]:
    destinations = structured_inputs.get("destinations")
    if isinstance(destinations, list):
        return [str(item).strip() for item in destinations if str(item).strip()]
    return []


def _compute_trip_days(start: Any, end: Any) -> int | None:
    if not start or not end:
        return None
    try:
        start_date = _parse_date(start)
        end_date = _parse_date(end)
    except ValueError:
        return None
    delta = (end_date - start_date).days + 1
    return max(delta, 1)


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _parse_interests(preferences: Any) -> list[str]:
    if preferences is None:
        return []
    if isinstance(preferences, list):
        return [str(item).strip() for item in preferences if str(item).strip()]
    return [part.strip() for part in str(preferences).split(",") if part.strip()]


def _format_family(adults: Any, children: Any) -> str | None:
    adult_count = _coerce_count(adults)
    child_count = _coerce_count(children)

    parts: list[str] = []
    if adult_count and adult_count > 0:
        parts.append(f"{adult_count} adult{'s' if adult_count != 1 else ''}")
    if child_count and child_count > 0:
        parts.append(f"{child_count} child{'ren' if child_count != 1 else ''}")
    return " and ".join(parts) if parts else None


def _coerce_count(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optimize_response(
    raw: str, fallback_prompt: str
) -> tuple[str, list[SuggestionItem]]:
    """Parse the optimization JSON from Ollama."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ollama optimization response was not valid JSON: %s", raw[:200])
        raise OllamaError("Ollama returned invalid JSON for prompt optimization.") from exc

    if not isinstance(parsed, dict):
        raise OllamaError("Ollama optimization response must be a JSON object.")

    optimized = parsed.get("optimizedPrompt")
    if not isinstance(optimized, str) or not optimized.strip():
        optimized = fallback_prompt

    raw_suggestions = parsed.get("suggestions", [])
    suggestions: list[SuggestionItem] = []
    if isinstance(raw_suggestions, list):
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            suggestion_type = str(item.get("suggestionType") or "fill_missing_detail")
            if suggestion_type not in _KNOWN_SUGGESTION_TYPES:
                suggestion_type = "fill_missing_detail"
            suggestions.append(
                SuggestionItem(
                    suggestion_type=cast(SuggestionType, suggestion_type),
                    description=description.strip(),
                )
            )

    return optimized.strip(), suggestions
