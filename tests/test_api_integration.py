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
    """Preview should return Ollama-generated prompt text and local token estimates."""
    resp = await client.post(
        f"{API}/preview",
        json={
            "useCaseKey": "travel",
            "structuredInputs": TRAVEL_INPUTS,
            "tokenEstimationMode": "local",
            "models": ["chatgpt"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["previews"]) == 1
    preview = body["previews"][0]
    assert preview["model"] == "chatgpt"
    assert preview["templateName"] == "Qwen 3 preview"
    assert preview["promptText"].strip()
    assert preview["tokenEstimates"]["effectiveMode"] == "local"
    assert preview["tokenEstimates"]["fromLocal"]["inputTokens"] > 0
    assert isinstance(preview["suggestions"], list)


async def test_preview_defaults_to_single_model(client: AsyncClient) -> None:
    """With no models requested, preview returns one entry for the default model."""
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
    assert len(body["previews"]) == 1
    assert body["previews"][0]["model"] == "chatgpt"
    assert body["previews"][0]["promptText"].strip()


async def test_preview_accepts_multiple_models(client: AsyncClient) -> None:
    """Preview should return one entry per requested model key."""
    resp = await client.post(
        f"{API}/preview",
        json={
            "useCaseKey": "travel",
            "structuredInputs": TRAVEL_INPUTS,
            "tokenEstimationMode": "local",
            "models": ["chatgpt", "claude"],
        },
    )
    assert resp.status_code == 200
    previews = {p["model"]: p for p in resp.json()["previews"]}
    assert set(previews) == {"chatgpt", "claude"}
    assert previews["chatgpt"]["promptText"] == previews["claude"]["promptText"]
    assert previews["chatgpt"]["templateName"] == "Qwen 3 preview"


async def test_run_creation_persists_prompt_version_and_responses(client: AsyncClient) -> None:
    """A run with no ids should auto-generate prompt + version and per-model responses."""
    resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "inputs": TRAVEL_INPUTS,
            "prompts": [
                {"model": "chatgpt", "promptText": "Plan a 10-day Tokyo trip (ChatGPT)."},
                {"model": "claude", "promptText": "Plan a 10-day Tokyo trip (Claude)."},
            ],
            "promptTitle": "Japan Family Vacation",
            "versionName": "V1 - baseline",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    run = body["run"]
    # Prompt/version ids are system-generated (24-char hex ObjectIds).
    assert len(run["promptId"]) == 24
    assert len(run["promptVersionId"]) == 24
    assert run["models"] == ["chatgpt", "claude"]
    # Each model is executed with its own previewed prompt.
    responses = {r["modelKey"]: r for r in body["responses"]}
    assert set(responses) == {"chatgpt", "claude"}
    assert all(r["status"] == "success" for r in body["responses"])

    # The run is retrievable with its responses (FR-10).
    run_id = run["id"]
    get_resp = await client.get(f"{API}/runs/{run_id}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["responses"]) == 2


async def test_run_honours_supplied_object_ids(client: AsyncClient) -> None:
    """Supplying valid prompt/version ObjectIds should be honoured on the run."""
    prompt_id = "507f1f77bcf86cd799439011"
    version_id = "507f1f77bcf86cd799439012"
    resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "inputs": TRAVEL_INPUTS,
            "promptId": prompt_id,
            "promptVersionId": version_id,
            "prompts": [{"model": "chatgpt", "promptText": "Plan a trip to Tokyo."}],
        },
    )
    assert resp.status_code == 201
    run = resp.json()["run"]
    assert run["promptId"] == prompt_id
    assert run["promptVersionId"] == version_id


async def test_run_rejects_invalid_object_id(client: AsyncClient) -> None:
    """A non-ObjectId promptId should be rejected with a validation error."""
    resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "promptId": "not-an-object-id",
            "prompts": [{"model": "chatgpt", "promptText": "Plan a trip to Tokyo."}],
        },
    )
    assert resp.status_code == 422


async def test_evaluation_upsert_and_dashboard(client: AsyncClient) -> None:
    """Evaluations should upsert and feed the dashboard summary."""
    run_resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "inputs": TRAVEL_INPUTS,
            "prompts": [{"model": "anthropic", "promptText": "Plan a trip to Tokyo."}],
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
