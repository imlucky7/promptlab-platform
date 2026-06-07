"""Unit tests for the template engine (rendering + missing template)."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import NotFoundError
from app.db.seed import TRAVEL_PROMPT_TEMPLATE
from app.services.template_engine import TemplateEngine


class _FakeTemplatesRepo:
    """Minimal stand-in for the templates repository used in tests."""

    def __init__(self, template: dict[str, Any] | None) -> None:
        self._template = template

    async def get_by_key(
        self, key: str, model: str = "chatgpt"
    ) -> dict[str, Any] | None:
        """Return the configured template regardless of key/model (or None)."""
        return self._template

    async def list_by_key(self, key: str) -> list[dict[str, Any]]:
        """Return the configured template as a single-item list (or empty)."""
        return [self._template] if self._template is not None else []


@pytest.fixture
def travel_engine() -> TemplateEngine:
    """Build a template engine backed by the default travel template."""
    repo = _FakeTemplatesRepo(
        {"key": "travel", "normalizedPromptTemplate": TRAVEL_PROMPT_TEMPLATE}
    )
    return TemplateEngine(repo)  # type: ignore[arg-type]


async def test_render_includes_required_inputs(travel_engine: TemplateEngine) -> None:
    """Rendered prompt should contain the supplied destination and persona."""
    text = await travel_engine.render(
        "travel",
        {
            "origin": "Hyderabad",
            "destination": "Tokyo",
            "startDate": "2026-11-10",
            "endDate": "2026-11-20",
            "travelers": 2,
        },
    )
    assert "Tokyo" in text
    assert "expert travel planner" in text


async def test_render_omits_absent_optional_fields(travel_engine: TemplateEngine) -> None:
    """Optional fields that are absent should not render their labels."""
    text = await travel_engine.render(
        "travel",
        {"origin": "A", "destination": "B", "startDate": "x", "endDate": "y", "travelers": 1},
    )
    assert "Budget:" not in text


async def test_missing_template_raises_not_found() -> None:
    """A missing template should raise :class:`NotFoundError`."""
    engine = TemplateEngine(_FakeTemplatesRepo(None))  # type: ignore[arg-type]
    with pytest.raises(NotFoundError):
        await engine.render("unknown", {})
