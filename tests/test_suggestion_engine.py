"""Unit tests for the rule-based suggestion engine."""

from __future__ import annotations

from app.services.suggestion_engine import SuggestionEngine


def test_sparse_prompt_triggers_all_suggestions() -> None:
    """A bare prompt with no structure should trigger every suggestion type."""
    engine = SuggestionEngine()
    suggestions = engine.analyze("Plan my trip.", {})
    types = {s.suggestion_type for s in suggestions}
    assert types == {
        "add_constraints",
        "clarify_output_format",
        "add_examples",
        "define_persona",
    }


def test_rich_prompt_triggers_no_suggestions() -> None:
    """A well-structured prompt should not trigger any suggestions."""
    engine = SuggestionEngine()
    prompt = (
        "You are an expert travel planner. Respect all constraints. "
        "Output format: a heading per day with bullet points. "
        "Here is an example day to follow."
    )
    suggestions = engine.analyze(prompt, {"constraints": "no overnight buses"})
    assert suggestions == []


def test_missing_constraints_only() -> None:
    """When only constraints are missing, just that suggestion appears."""
    engine = SuggestionEngine()
    prompt = (
        "You are a travel planner. Output format: bullet points. "
        "Example: Day 1 ..."
    )
    suggestions = engine.analyze(prompt, {})
    types = {s.suggestion_type for s in suggestions}
    assert types == {"add_constraints"}
