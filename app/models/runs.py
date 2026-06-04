"""Schemas for the ``runs`` collection (prompt executions)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel
from app.models.evaluations import EvaluationRead
from app.models.responses import ResponseRead


class RunBase(CamelModel):
    """Fields shared by run create/read operations.

    Attributes:
        prompt_version_id: Optional version this run is tied to.
        prompt_id: Optional denormalised prompt reference.
        use_case_key: Use-case key (e.g. ``"travel"``).
        inputs: Raw user inputs for the run.
        models: Model keys to execute (e.g. ``["anthropic", "perplexity"]``).
    """

    prompt_version_id: str | None = None
    prompt_id: str | None = None
    use_case_key: str = Field(default="travel", examples=["travel"])
    inputs: dict[str, Any] = Field(default_factory=dict)
    models: list[str] = Field(default_factory=list)


class RunCreate(RunBase):
    """Payload for creating + executing a run (``POST /runs``).

    Optional attributes drive the "create prompt/version on first run" behaviour
    described in the PRD:

    Attributes:
        prompt_title: Title for a new prompt when none is provided.
        version_name: Label for the auto-created prompt version.
        prompt_text: Optional pre-built/edited prompt text to execute verbatim.
    """

    prompt_title: str | None = None
    version_name: str | None = None
    prompt_text: str | None = None


class RunUpdate(CamelModel):
    """Payload for partially updating run metadata (``PATCH``)."""

    inputs: dict[str, Any] | None = None
    models: list[str] | None = None
    prompt_version_id: str | None = None
    prompt_id: str | None = None


class RunRead(RunBase, TimestampedModel):
    """Run as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
        prompt_text: The prompt text used for this run, when known.
    """

    id: str
    prompt_text: str | None = None


class RunWithResponses(CamelModel):
    """Composite returned by run creation and ``GET /runs/{id}``.

    Attributes:
        run: The run document.
        responses: Per-model responses for the run.
        evaluations: Evaluations linked to the run's responses.
    """

    run: RunRead
    responses: list[ResponseRead] = Field(default_factory=list)
    evaluations: list[EvaluationRead] = Field(default_factory=list)
