"""Generate and persist structured budget breakdowns from travel plans."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from app.core.errors import NotFoundError, ValidationAppError
from app.core.logging import get_logger
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.budget import BudgetBreakdown, BudgetLineItem
from app.models.responses import ResponseRead
from app.services.llm_gateway_client import LLMGatewayClient

logger = get_logger(__name__)

_BUDGET_PROMPT = """You are a travel budget analyst. Given the trip context and generated travel plan below,
produce a realistic itemized budget breakdown.

Return ONLY valid JSON with this exact shape:
{{
  "currency": "USD",
  "total": 0,
  "items": [
    {{"category": "Flights", "description": "...", "amount": 0, "currency": "USD"}}
  ],
  "notes": "optional summary"
}}

Trip context:
{trip_context}

Travel plan:
{plan_text}
"""

_STUB_BREAKDOWN = BudgetBreakdown(
    currency="USD",
    total=3200,
    items=[
        BudgetLineItem(
            category="Flights",
            description="Round-trip economy flights",
            amount=1200,
            currency="USD",
        ),
        BudgetLineItem(
            category="Hotels",
            description="Mid-range hotels for the trip duration",
            amount=1400,
            currency="USD",
        ),
        BudgetLineItem(
            category="Food",
            description="Meals and local dining",
            amount=400,
            currency="USD",
        ),
        BudgetLineItem(
            category="Activities",
            description="Tours, attractions, and local transport",
            amount=200,
            currency="USD",
        ),
    ],
    notes="Stub budget breakdown for offline development.",
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_trip_context(inputs: dict[str, Any]) -> str:
    """Render run inputs as readable trip context for the budget prompt."""
    lines: list[str] = []
    for key in (
        "origin",
        "destinations",
        "destination",
        "startDate",
        "endDate",
        "adults",
        "children",
        "travelers",
        "budgetLevel",
        "budget",
        "preferences",
        "constraints",
    ):
        value = inputs.get(key)
        if value not in (None, "", []):
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "- (no trip context provided)"


def parse_budget_response(raw_text: str) -> BudgetBreakdown:
    """Parse model output into a :class:`BudgetBreakdown`.

    Args:
        raw_text: Raw LLM completion text.

    Returns:
        A validated budget breakdown.

    Raises:
        ValidationAppError: When the response cannot be parsed into a valid breakdown.
    """
    if "[STUB:" in raw_text or raw_text.strip().startswith("[STUB Ollama"):
        return _STUB_BREAKDOWN.model_copy(
            update={"generated_at": datetime.now(UTC)},
        )

    parsed = _extract_json_object(raw_text)
    if parsed is None:
        raise ValidationAppError(
            "Budget generation returned invalid JSON.",
            details={"preview": raw_text[:500]},
        )

    items_raw = parsed.get("items", [])
    items: list[BudgetLineItem] = []
    if isinstance(items_raw, list):
        for item in items_raw:
            if not isinstance(item, dict):
                continue
            try:
                items.append(
                    BudgetLineItem(
                        category=str(item.get("category") or "Other"),
                        description=str(item.get("description") or ""),
                        amount=float(item.get("amount") or 0),
                        currency=str(item.get("currency") or parsed.get("currency") or "USD"),
                    )
                )
            except (TypeError, ValueError):
                continue

    if not items:
        raise ValidationAppError(
            "Budget generation returned no line items.",
            details={"preview": raw_text[:500]},
        )

    total = parsed.get("total")
    if total is None:
        total = sum(item.amount for item in items)
    else:
        total = float(total)

    return BudgetBreakdown(
        currency=str(parsed.get("currency") or "USD"),
        total=total,
        items=items,
        notes=str(parsed["notes"]) if parsed.get("notes") else None,
        generated_at=datetime.now(UTC),
    )


class BudgetService:
    """Generates and persists budget breakdowns for run responses."""

    def __init__(
        self,
        runs_repo: RunsRepository,
        responses_repo: ResponsesRepository,
        gateway: LLMGatewayClient,
    ) -> None:
        self._runs_repo = runs_repo
        self._responses_repo = responses_repo
        self._gateway = gateway

    async def generate_for_response(
        self,
        run_id: str,
        response_id: str,
        *,
        model_key: str | None = None,
    ) -> ResponseRead:
        """Generate a budget breakdown for a successful run response.

        Args:
            run_id: Owning run id.
            response_id: Target response id.
            model_key: Optional model override for the budget LLM call.

        Returns:
            The updated response including ``budgetBreakdown``.

        Raises:
            NotFoundError: When the run or response is missing.
            ValidationAppError: When the response is not successful or parsing fails.
        """
        run_doc = await self._runs_repo.get(run_id)
        if run_doc is None:
            raise NotFoundError("Run not found.", details={"id": run_id})

        response_doc = await self._responses_repo.get(response_id)
        if response_doc is None or response_doc.get("runId") != run_id:
            raise NotFoundError("Response not found.", details={"id": response_id})

        if response_doc.get("status") != "success":
            raise ValidationAppError(
                "Budget can only be generated for successful responses.",
                details={"responseId": response_id},
            )

        plan_text = (response_doc.get("text") or "").strip()
        if not plan_text:
            raise ValidationAppError(
                "Response has no travel plan text to budget.",
                details={"responseId": response_id},
            )

        target_model = model_key or response_doc.get("modelKey") or "chatgpt"
        trip_context = _format_trip_context(run_doc.get("inputs") or {})
        prompt = _BUDGET_PROMPT.format(trip_context=trip_context, plan_text=plan_text)

        result = await self._gateway.chat_completion(
            target_model,
            prompt,
            temperature=0.2,
            max_tokens=2000,
        )
        if result.status != "success" or not result.text.strip():
            raise ValidationAppError(
                "Budget generation failed.",
                details={"error": result.error_message or "empty response"},
            )

        breakdown = parse_budget_response(result.text)
        updated = await self._responses_repo.patch(
            response_id,
            {"budgetBreakdown": breakdown.model_dump(by_alias=True, mode="json")},
        )
        if updated is None:
            raise NotFoundError("Response not found.", details={"id": response_id})
        return ResponseRead.model_validate(updated)
