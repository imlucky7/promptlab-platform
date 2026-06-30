"""Tests for streaming preview and run endpoints."""

from __future__ import annotations

import json

import pytest


def _parse_sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        data_line = next(
            (line[len("data:") :].strip() for line in block.split("\n") if line.startswith("data:")),
            None,
        )
        if data_line:
            events.append(json.loads(data_line))
    return events


@pytest.mark.asyncio
async def test_preview_stream_returns_progress_and_complete(client) -> None:
    payload = {
        "useCaseKey": "travel",
        "structuredInputs": {
            "origin": "Hyderabad",
            "destinations": ["Singapore"],
            "startDate": "2026-06-01",
            "endDate": "2026-06-07",
            "adults": 2,
            "children": 0,
            "budgetLevel": "Mid-range",
            "preferences": "food",
            "constraints": "",
            "specialNotes": "",
        },
        "models": ["qwen3"],
    }
    response = await client.post("/api/v1/preview/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse_events(response.text)
    assert events[0]["type"] == "progress"
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["type"] == "complete"
    assert events[-1]["data"]["previews"][0]["promptText"]


@pytest.mark.asyncio
async def test_create_run_stream_returns_tokens_and_complete(client) -> None:
    preview = await client.post(
        "/api/v1/preview",
        json={
            "useCaseKey": "travel",
            "structuredInputs": {
                "origin": "Hyderabad",
                "destinations": ["Singapore"],
                "startDate": "2026-06-01",
                "endDate": "2026-06-07",
                "adults": 2,
                "children": 0,
                "budgetLevel": "Mid-range",
                "preferences": "food",
                "constraints": "",
                "specialNotes": "",
            },
            "models": ["qwen3"],
        },
    )
    prompt_text = preview.json()["previews"][0]["promptText"]

    response = await client.post(
        "/api/v1/runs/stream",
        json={
            "useCaseKey": "travel",
            "model": "qwen3",
            "promptText": prompt_text,
            "inputs": {
                "origin": "Hyderabad",
                "destinations": ["Singapore"],
                "startDate": "2026-06-01",
                "endDate": "2026-06-07",
                "adults": 2,
                "children": 0,
                "budgetLevel": "Mid-range",
                "preferences": "food",
                "constraints": "",
                "specialNotes": "",
            },
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse_events(response.text)
    assert events[0]["type"] == "progress"
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["type"] == "complete"
    assert events[-1]["data"]["responses"]
