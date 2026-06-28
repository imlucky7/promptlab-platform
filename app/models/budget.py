"""Schemas for LLM-generated travel budget breakdowns."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.models.common import CamelModel


class BudgetLineItem(CamelModel):
    """A single cost line in a travel budget breakdown."""

    category: str = Field(examples=["Flights", "Hotels", "Food"])
    description: str
    amount: float = Field(ge=0)
    currency: str = Field(default="USD", examples=["USD", "EUR"])


class BudgetBreakdown(CamelModel):
    """Structured budget breakdown for a generated travel plan."""

    currency: str = Field(default="USD")
    total: float = Field(ge=0)
    items: list[BudgetLineItem] = Field(default_factory=list)
    notes: str | None = None
    generated_at: datetime | None = None
