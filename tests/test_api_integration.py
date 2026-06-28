"""Integration tests exercising the full API against an in-memory database."""

from __future__ import annotations

from httpx import AsyncClient

API = "/api/v1"

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


def _run_payload(
    *,
    model: str = "chatgpt",
    prompt_text: str = "Plan a 10-day Tokyo trip.",
    prompt_id: str | None = None,
    prompt_version_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "useCaseKey": "travel",
        "inputs": TRAVEL_INPUTS,
        "model": model,
        "promptText": prompt_text,
    }
    if prompt_id is not None:
        payload["promptId"] = prompt_id
    if prompt_version_id is not None:
        payload["promptVersionId"] = prompt_version_id
    return payload


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
    assert body["latencyMs"] >= 0
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
            "models": ["chatgpt", "qwen3"],
        },
    )
    assert resp.status_code == 200
    previews = {p["model"]: p for p in resp.json()["previews"]}
    assert set(previews) == {"chatgpt", "qwen3"}
    assert previews["chatgpt"]["promptText"] == previews["qwen3"]["promptText"]
    assert previews["chatgpt"]["templateName"] == "Qwen 3 preview"


async def test_run_creation_persists_responses(client: AsyncClient) -> None:
    """A run should execute the requested model and persist one response."""
    resp = await client.post(f"{API}/runs", json=_run_payload(model="qwen3"))
    assert resp.status_code == 201
    body = resp.json()
    assert body.get("promptId") in (None, "")
    assert body.get("promptVersionId") in (None, "")
    assert len(body["responses"]) == 1
    assert body["responses"][0]["modelKey"] == "qwen3"
    assert body["responses"][0]["status"] == "success"
    assert body["responses"][0]["text"] == "[STUB Ollama response]"

    run_id = body["id"]
    get_resp = await client.get(f"{API}/runs/{run_id}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["responses"]) == 1


async def test_run_uses_compact_template_for_non_qwen3(client: AsyncClient) -> None:
    """Non-qwen3 runs should build a compact templated prompt from inputs."""
    resp = await client.post(
        f"{API}/runs",
        json=_run_payload(
            model="chatgpt",
            prompt_text="This long preview text should not be sent verbatim to chatgpt.",
        ),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["responses"]) == 1
    assert body["responses"][0]["modelKey"] == "chatgpt"
    assert body["responses"][0]["status"] == "success"


async def test_generate_budget_persists_on_response(client: AsyncClient) -> None:
    """Budget generation should persist breakdown on the response document."""
    run_resp = await client.post(
        f"{API}/runs",
        json=_run_payload(model="chatgpt"),
    )
    assert run_resp.status_code == 201
    body = run_resp.json()
    run_id = body["id"]
    response_id = body["responses"][0]["id"]

    budget_resp = await client.post(
        f"{API}/runs/{run_id}/responses/{response_id}/budget"
    )
    assert budget_resp.status_code == 200
    budget_body = budget_resp.json()
    assert budget_body["budgetBreakdown"] is not None
    assert budget_body["budgetBreakdown"]["total"] > 0
    assert len(budget_body["budgetBreakdown"]["items"]) >= 1

    get_resp = await client.get(f"{API}/runs/{run_id}")
    assert get_resp.status_code == 200
    stored = get_resp.json()["responses"][0]["budgetBreakdown"]
    assert stored is not None
    assert stored["total"] == budget_body["budgetBreakdown"]["total"]


async def test_run_honours_supplied_object_ids(client: AsyncClient) -> None:
    """Supplying valid prompt/version ObjectIds should be stored on the run."""
    prompt_id = "507f1f77bcf86cd799439011"
    version_id = "507f1f77bcf86cd799439012"
    resp = await client.post(
        f"{API}/runs",
        json=_run_payload(
            prompt_id=prompt_id,
            prompt_version_id=version_id,
        ),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["promptId"] == prompt_id
    assert body["promptVersionId"] == version_id


async def test_run_rejects_invalid_object_id(client: AsyncClient) -> None:
    """A non-ObjectId promptId should be rejected with a validation error."""
    resp = await client.post(
        f"{API}/runs",
        json={
            "useCaseKey": "travel",
            "promptId": "not-an-object-id",
            "model": "chatgpt",
            "promptText": "Plan a trip to Tokyo.",
        },
    )
    assert resp.status_code == 422


async def test_evaluation_upsert_and_dashboard(client: AsyncClient) -> None:
    """Evaluations should upsert and feed the dashboard summary."""
    run_resp = await client.post(
        f"{API}/runs",
        json=_run_payload(model="anthropic"),
    )
    run_body = run_resp.json()
    run_id = run_body["id"]
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
