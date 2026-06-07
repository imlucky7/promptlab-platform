"""Idempotent database seeding.

On startup (when ``SEED_ON_STARTUP`` is enabled) the default ``travel``
use-case templates are upserted so the application is immediately usable. Each
template defines the input schema (form fields) and the normalized prompt
template rendered by the template engine. Templates are scoped to a ``model``
so a single use case can ship multiple LLM-specific prompt variants (e.g. the
generic default alongside a Claude-tuned variant).
"""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.logging import get_logger
from app.db.repositories.use_case_templates_repo import UseCaseTemplatesRepository
from app.models.use_case_templates import DEFAULT_MODEL

logger = get_logger(__name__)

# JSON-schema-like definition of the Travel Planner form fields (PRD FR-02).
TRAVEL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["origin", "destination", "startDate", "endDate", "travelers"],
    "properties": {
        "origin": {"type": "string", "title": "Origin"},
        "destination": {"type": "string", "title": "Destination"},
        "startDate": {"type": "string", "format": "date", "title": "Start date"},
        "endDate": {"type": "string", "format": "date", "title": "End date"},
        "travelers": {"type": "integer", "minimum": 1, "title": "Travelers"},
        "budget": {"type": "string", "title": "Budget"},
        "preferences": {"type": "string", "title": "Preferences"},
        "constraints": {"type": "string", "title": "Constraints"},
    },
}

# Normalized prompt template (system + user sections) using Jinja2 placeholders,
# tuned for ChatGPT (the default model). The template engine renders this against
# the structured inputs at build time.
TRAVEL_PROMPT_TEMPLATE: str = """\
[SYSTEM]
You are an expert travel planner. Design a clear, realistic, day-by-day \
itinerary tailored to the traveler's inputs. Use concise headings for each day \
and bullet points for activities. Account for travel time, opening hours, and a \
sensible pace. Call out budget-relevant choices and respect all stated \
constraints.

[USER]
Plan a trip with the following details:
- Origin: {{ origin }}
- Destination: {{ destination }}
- Dates: {{ startDate }} to {{ endDate }}
- Travelers: {{ travelers }}
{% if budget %}- Budget: {{ budget }}
{% endif %}{% if preferences %}- Preferences: {{ preferences }}
{% endif %}{% if constraints %}- Constraints: {{ constraints }}
{% endif %}
Output format:
- A short trip overview (2-3 sentences).
- One section per day titled "Day N - <theme>".
- Bullet points for morning, afternoon, and evening.
- A brief notes section with budget tips and practical advice.
"""

# Claude-tuned variant of the travel template. It leans on Anthropic prompting
# conventions (explicit role framing and XML-style sections) while consuming the
# same structured inputs as the default template.
TRAVEL_PROMPT_TEMPLATE_CLAUDE: str = """\
You are Claude, an expert travel planner. Think step by step about travel time, \
opening hours, and a sensible pace before writing the itinerary. Respect every \
stated constraint and call out budget-relevant choices.

<trip_request>
- Origin: {{ origin }}
- Destination: {{ destination }}
- Dates: {{ startDate }} to {{ endDate }}
- Travelers: {{ travelers }}
{% if budget %}- Budget: {{ budget }}
{% endif %}{% if preferences %}- Preferences: {{ preferences }}
{% endif %}{% if constraints %}- Constraints: {{ constraints }}
{% endif %}</trip_request>

<output_format>
- A short trip overview (2-3 sentences).
- One section per day titled "Day N - <theme>".
- Bullet points for morning, afternoon, and evening.
- A brief notes section with budget tips and practical advice.
</output_format>
"""

# Full default template documents keyed by ``(key, model)``.
DEFAULT_TEMPLATES: list[dict[str, Any]] = [
    {
        "key": "travel",
        "model": DEFAULT_MODEL,
        "name": "Travel Planner (ChatGPT)",
        "description": "Design a multi-day trip itinerary, tuned for ChatGPT",
        "inputSchema": TRAVEL_INPUT_SCHEMA,
        "normalizedPromptTemplate": TRAVEL_PROMPT_TEMPLATE,
    },
    {
        "key": "travel",
        "model": "claude",
        "name": "Travel Planner (Claude)",
        "description": "Design a multi-day trip itinerary, tuned for Claude",
        "inputSchema": TRAVEL_INPUT_SCHEMA,
        "normalizedPromptTemplate": TRAVEL_PROMPT_TEMPLATE_CLAUDE,
    },
]


async def seed_default_templates(db: AsyncIOMotorDatabase) -> None:
    """Upsert the default use-case templates.

    This is safe to run repeatedly; each template is matched by its unique
    ``(key, model)`` pair and updated in place.

    Args:
        db: The active Mongo database handle.
    """
    repo = UseCaseTemplatesRepository(db)
    for template in DEFAULT_TEMPLATES:
        await repo.upsert_by_key(template["key"], template)
        logger.info(
            "Seeded use-case template: %s (model=%s)",
            template["key"],
            template["model"],
        )
