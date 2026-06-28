"""Unit tests for budget generation parsing."""

from __future__ import annotations

import json

from app.services.budget_service import parse_budget_response


def test_parse_budget_response_stub_text() -> None:
    """Stub gateway text should yield the canned breakdown."""
    breakdown = parse_budget_response("[STUB:chatgpt] sample")
    assert breakdown.total == 3200
    assert len(breakdown.items) == 4
    assert breakdown.currency == "USD"


def test_parse_budget_response_json() -> None:
    """Valid JSON should map to line items and total."""
    raw = json.dumps(
        {
            "currency": "EUR",
            "total": 1500,
            "items": [
                {
                    "category": "Hotels",
                    "description": "3 nights",
                    "amount": 900,
                    "currency": "EUR",
                },
                {
                    "category": "Food",
                    "description": "Meals",
                    "amount": 600,
                    "currency": "EUR",
                },
            ],
            "notes": "Estimate only",
        }
    )
    breakdown = parse_budget_response(raw)
    assert breakdown.currency == "EUR"
    assert breakdown.total == 1500
    assert len(breakdown.items) == 2
    assert breakdown.notes == "Estimate only"
