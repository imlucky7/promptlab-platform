"""Compact, model-specific run prompts built from structured trip inputs.

Non-qwen3 runs use these templates instead of LLM adaptation so execution
stays token-efficient and deterministic. qwen3 runs use the full Ollama
preview ``promptText`` unchanged.
"""

from __future__ import annotations

from typing import Any

from app.services.ollama_preview_service import map_structured_inputs_to_trip_context

QWEN3_MODEL_KEY = "qwen3"

_OUTPUT_FORMAT = (
    "Output: 2-3 sentence overview, then one section per day "
    '("Day N - theme") with morning/afternoon/evening bullets.'
)

# Compact wrappers keyed by logical model id. Unknown models use ``default``.
_MODEL_WRAPPERS: dict[str, str] = {
    "chatgpt": (
        "You are an expert travel planner.\n\n"
        "Trip details:\n{facts}\n\n"
        "{output}"
    ),
    "anthropic": (
        "You are Claude, an expert travel planner.\n\n"
        "<trip_request>\n{facts}\n</trip_request>\n\n"
        "{output}"
    ),
    "claude": (
        "You are Claude, an expert travel planner.\n\n"
        "<trip_request>\n{facts}\n</trip_request>\n\n"
        "{output}"
    ),
    "perplexity": (
        "Create a practical travel itinerary from these facts:\n\n"
        "{facts}\n\n"
        "{output}"
    ),
    "default": (
        "Plan a travel itinerary using:\n\n"
        "{facts}\n\n"
        "{output}"
    ),
}


def build_run_prompt_for_model(structured_inputs: dict[str, Any], model: str) -> str:
    """Build a compact execution prompt for a non-qwen3 target model."""
    context = map_structured_inputs_to_trip_context(structured_inputs)
    facts = _format_fact_block(context)
    wrapper = _MODEL_WRAPPERS.get(model, _MODEL_WRAPPERS["default"])
    return wrapper.format(facts=facts, output=_OUTPUT_FORMAT).strip()


def _format_fact_block(context: dict[str, Any]) -> str:
    """Render trip context as minimal bullet lines."""
    lines: list[str] = []

    origin = context.get("origin")
    if origin:
        lines.append(f"- Origin: {origin}")

    destinations = context.get("destinations")
    if isinstance(destinations, list) and destinations:
        lines.append(f"- Destinations: {', '.join(destinations)}")

    start = context.get("startDate")
    end = context.get("endDate")
    if start and end:
        days = context.get("days")
        day_suffix = f" ({days} days)" if days else ""
        lines.append(f"- Dates: {start} to {end}{day_suffix}")

    family = context.get("family")
    if family:
        lines.append(f"- Travelers: {family}")

    travel_style = context.get("travelStyle")
    if travel_style:
        lines.append(f"- Budget: {travel_style}")

    interests = context.get("interests")
    if isinstance(interests, list) and interests:
        lines.append(f"- Preferences: {', '.join(interests)}")

    constraints = context.get("constraints")
    if constraints:
        lines.append(f"- Constraints: {constraints}")

    special_notes = context.get("specialNotes")
    if special_notes:
        lines.append(f"- Notes: {special_notes}")

    return "\n".join(lines) if lines else "- (no trip details provided)"
