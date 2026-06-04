"""Integration tests exercising the full API against an in-memory database."""

from __future__ import annotations

from httpx import AsyncClient

API = "/api/v1"

# A representative set of travel inputs reused across tests.
TRAVEL_INPUTS = {
    "origin": "Hyderabad",
    "destination": "Tokyo",
    "startDate": "2026-11-10",
    "endDate": "2026-11-20",
    "travelers": 2,
    "budget": "mid-range",
    "preferences": "food, culture",
    "constraints": "avoid overnight buses",
}


async def test_health(client: AsyncClient) -> None:
    """The health endpoint should respond (DB ping may be degraded for mocks)."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


async def test_preview_returns_prompt_and_estimates(client: AsyncClient) -> None:
    """Preview should assemble prompt text and local token estimates."""
    resp = await client.post(
        f"{API}/preview",
        json={
            "useCaseKey": "travel",
            "structuredInputs": TRAVEL_INPUTS,
            "tokenEstimationMode": "local",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Tokyo" in body["promptText"]
    assert body["tokenEstimates"]["effectiveMode"] == "local"
    assert body["tokenEstimates"]["fromLocal"]["inputTokens"] > 0


async def test_run_creation_persists_prompt_version_and_responses(client: AsyncClient) -> None:
    """Creating a run with no prompt should auto-create prompt + version + responses."""
    resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "inputs": TRAVEL_INPUTS,
            "models": ["anthropic", "perplexity"],
            "promptTitle": "Japan Family Vacation",
            "versionName": "V1 - baseline",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    run = body["run"]
    assert run["promptId"]
    assert run["promptVersionId"]
    assert len(body["responses"]) == 2
    assert all(r["status"] == "success" for r in body["responses"])

    # The run is retrievable with its responses (FR-10).
    run_id = run["id"]
    get_resp = await client.get(f"{API}/runs/{run_id}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["responses"]) == 2


async def test_evaluation_upsert_and_dashboard(client: AsyncClient) -> None:
    """Evaluations should upsert and feed the dashboard summary."""
    run_resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "inputs": TRAVEL_INPUTS,
            "models": ["anthropic"],
        },
    )
    run_body = run_resp.json()
    run_id = run_body["run"]["id"]
    response_id = run_body["responses"][0]["id"]

    scores = {
        "correctness": 4,
        "completeness": 4,
        "styleFit": 5,
        "faithfulness": 4,
        "overall": 4,
    }
    eval_payload = {
        "runId": run_id,
        "responseId": response_id,
        "modelKey": "anthropic",
        "scores": scores,
        "comments": "Good itinerary.",
    }
    first = await client.post(f"{API}/evaluations", json=eval_payload)
    assert first.status_code == 201
    first_id = first.json()["id"]

    # Re-posting the same natural key should upsert (not duplicate).
    eval_payload["scores"]["overall"] = 5
    second = await client.post(f"{API}/evaluations", json=eval_payload)
    assert second.json()["id"] == first_id
    assert second.json()["scores"]["overall"] == 5

    summary = await client.get(f"{API}/dashboard/summary", params={"useCase": "travel"})
    assert summary.status_code == 200
    models = {m["modelKey"]: m for m in summary.json()["models"]}
    assert "anthropic" in models
    assert models["anthropic"]["avgOverall"] == 5.0


async def test_prompt_crud_roundtrip(client: AsyncClient) -> None:
    """Prompts should support create/get/patch/delete."""
    create = await client.post(
        f"{API}/prompts", json={"title": "Trip", "useCaseKey": "travel"}
    )
    assert create.status_code == 201
    prompt_id = create.json()["id"]

    patched = await client.patch(f"{API}/prompts/{prompt_id}", json={"title": "Trip v2"})
    assert patched.json()["title"] == "Trip v2"

    deleted = await client.delete(f"{API}/prompts/{prompt_id}")
    assert deleted.status_code == 200

    missing = await client.get(f"{API}/prompts/{prompt_id}")
    assert missing.status_code == 404


async def test_not_found_returns_structured_error(client: AsyncClient) -> None:
    """Unknown ids should return the structured error envelope."""
    resp = await client.get(f"{API}/runs/64b7f9a2c000000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
