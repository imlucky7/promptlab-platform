"""Unit tests for compact run prompt templates."""

from __future__ import annotations

from app.services.run_prompt_templates import build_run_prompt_for_model

TRAVEL_INPUTS = {
    "origin": "Hyderabad",
    "destinations": ["Tokyo"],
    "startDate": "2026-11-10",
    "endDate": "2026-11-20",
    "adults": 2,
    "budgetLevel": "mid-range",
    "preferences": "food, culture",
    "constraints": "avoid overnight buses",
}


def test_build_run_prompt_chatgpt_includes_facts() -> None:
    prompt = build_run_prompt_for_model(TRAVEL_INPUTS, "chatgpt")
    assert "Origin: Hyderabad" in prompt
    assert "Destinations: Tokyo" in prompt
    assert "Dates: 2026-11-10 to 2026-11-20" in prompt
    assert "Travelers: 2 adults" in prompt
    assert "Budget: mid-range" in prompt
    assert "Preferences: food, culture" in prompt
    assert "Constraints: avoid overnight buses" in prompt
    assert "expert travel planner" in prompt


def test_build_run_prompt_anthropic_uses_xml_wrapper() -> None:
    prompt = build_run_prompt_for_model(TRAVEL_INPUTS, "anthropic")
    assert "<trip_request>" in prompt
    assert "</trip_request>" in prompt
    assert "Claude" in prompt


def test_build_run_prompt_is_compact() -> None:
    long_preview = "You are an expert travel planner.\n\n" + ("Detail line.\n" * 80)
    templated = build_run_prompt_for_model(TRAVEL_INPUTS, "chatgpt")
    assert len(templated) < len(long_preview)
