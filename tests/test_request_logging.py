"""Tests for v1 request/response logging helpers and middleware."""

from __future__ import annotations

import logging

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.request_logging import truncate_for_log
from app.core.config import Settings
from app.core.dependencies import get_app_settings, get_db
from app.main import create_app

API = "/api/v1"


def test_truncate_for_log_clips_long_strings() -> None:
    """Long strings should be clipped with a total-length suffix."""
    result = truncate_for_log("x" * 1500, max_string=1000)
    assert isinstance(result, str)
    assert result.endswith("(1500 chars total)")
    assert len(result) < 1500


def test_truncate_for_log_respects_max_depth() -> None:
    """Deeply nested structures should stop at the configured depth."""
    nested: dict[str, object] = {"level": 0}
    current = nested
    for index in range(1, 10):
        nxt: dict[str, object] = {"level": index}
        current["child"] = nxt
        current = nxt

    truncated = truncate_for_log(nested, max_depth=3)
    assert truncated["level"] == 0
    assert truncated["child"]["level"] == 1
    assert truncated["child"]["child"]["level"] == 2
    assert truncated["child"]["child"]["child"] == "<max depth reached>"


@pytest.mark.asyncio
async def test_response_middleware_logs_v1_json(
    settings: Settings, mock_db, caplog: pytest.LogCaptureFixture
) -> None:
    """V1 responses should emit an outgoing response log line."""
    app = create_app(settings)
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_app_settings] = lambda: settings

    caplog.set_level(logging.INFO, logger="app.api.v1.response")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get(f"{API}/prompts")

    assert resp.status_code == 200
    assert any("Outgoing response:" in record.message for record in caplog.records)
    assert any('"statusCode": 200' in record.message for record in caplog.records)

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_response_middleware_skips_non_v1(
    settings: Settings, mock_db, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-v1 routes should not emit outgoing response logs."""
    app = create_app(settings)
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_app_settings] = lambda: settings

    caplog.set_level(logging.INFO, logger="app.api.v1.response")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/health")

    assert resp.status_code == 200
    assert not any("Outgoing response:" in record.message for record in caplog.records)

    app.dependency_overrides.clear()
