"""Unit tests for Ollama preview helpers and client."""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.services.ollama_client import OllamaClient
from app.services.ollama_preview_service import (
    map_structured_inputs_to_trip_context,
    parse_optimize_response,
)


def test_map_structured_inputs_to_trip_context() -> None:
    context = map_structured_inputs_to_trip_context(
        {
            "origin": "Hyderabad",
            "destinations": ["Switzerland"],
            "destination": "Switzerland",
            "startDate": "2026-06-01",
            "endDate": "2026-06-07",
            "adults": 2,
            "children": 1,
            "budgetLevel": "relaxed",
            "preferences": "mountains, lakes, nature",
            "constraints": "avoid long hikes",
        }
    )
    assert context["destination"] == "Switzerland"
    assert context["days"] == 7
    assert context["travelStyle"] == "relaxed"
    assert context["interests"] == ["mountains", "lakes", "nature"]
    assert context["family"] == "2 adults and 1 child"
    assert context["origin"] == "Hyderabad"
    assert context["constraints"] == "avoid long hikes"


def test_parse_optimize_response_maps_suggestions() -> None:
    raw = json.dumps(
        {
            "optimizedPrompt": "Optimized travel prompt",
            "suggestions": [
                {
                    "suggestionType": "fill_missing_detail",
                    "description": "Add dietary restrictions.",
                    "field": "constraints",
                }
            ],
        }
    )
    prompt, suggestions = parse_optimize_response(raw, "fallback")
    assert prompt == "Optimized travel prompt"
    assert len(suggestions) == 1
    assert suggestions[0].suggestion_type == "fill_missing_detail"
    assert "dietary" in suggestions[0].description


def test_parse_optimize_response_falls_back_to_draft_prompt() -> None:
    raw = json.dumps({"optimizedPrompt": "", "suggestions": []})
    prompt, suggestions = parse_optimize_response(raw, "draft prompt")
    assert prompt == "draft prompt"
    assert suggestions == []


@pytest.mark.asyncio
async def test_ollama_client_stub_generate_and_optimize() -> None:
    settings = Settings(ollama_preview_stub_mode=True)
    client = OllamaClient(settings)

    draft = await client.generate("Trip context JSON:\n{}", system="system")
    assert "STUB generated prompt" in draft

    optimized = await client.generate(
        "optimize",
        system="system",
        response_format={"type": "object"},
    )
    parsed = json.loads(optimized)
    assert parsed["optimizedPrompt"]
    assert parsed["suggestions"]
