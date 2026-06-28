"""Schemas for the ``runs`` collection (prompt executions)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pydantic import Field, field_validator, model_validator

from app.models.common import CamelModel, TimestampedModel
from app.models.evaluations import EvaluationRead
from app.models.responses import ResponseRead


class RunPrompt(CamelModel):
    """A previewed prompt targeted at a single model.

    Attributes:
        model: Logical model key the prompt was previewed for (e.g.
            ``"chatgpt"`` or ``"claude"``).
        prompt_text: The previewed prompt text (from ``POST /preview``) to send
            to that model.
    """

    model: str = Field(examples=["chatgpt", "claude"])
    prompt_text: str = Field(
        min_length=1,
        description="Previewed prompt text for this model (from POST /preview).",
    )


class RunBase(CamelModel):
    """Fields shared by run create/read operations.

    Attributes:
        use_case_key: Use-case key (e.g. ``"travel"``).
        inputs: Raw user inputs for the run.
    """

    use_case_key: str = Field(default="travel", examples=["travel"])
    inputs: dict[str, Any] = Field(default_factory=dict)


class RunCreate(RunBase):
    """Payload for creating + executing a run (``POST /runs``).

    Attributes:
        model: Logical model key to execute (e.g. ``"qwen3"`` or ``"chatgpt"``).
        prompt_text: Previewed prompt text from ``POST /preview``.
        prompt_id: Optional prompt id reference when linking to a saved prompt.
        prompt_version_id: Optional prompt version id reference.
    """

    model: str = Field(examples=["qwen3", "chatgpt"])
    prompt_text: str = Field(
        min_length=1,
        description="Previewed prompt text to execute.",
    )
    prompt_id: str | None = None
    prompt_version_id: str | None = None

    @field_validator("prompt_id", "prompt_version_id")
    @classmethod
    def _validate_object_id(cls, value: str | None) -> str | None:
        """Reject ids that are not valid MongoDB ObjectIds.

        Args:
            value: The candidate id, or ``None`` to defer to system generation.

        Returns:
            The validated id (unchanged), or ``None``.

        Raises:
            ValueError: If a non-``None`` value is not a valid ObjectId.
        """
        if value is None:
            return value
        try:
            ObjectId(value)
        except (InvalidId, TypeError) as exc:
            raise ValueError("must be a valid 24-character hex ObjectId") from exc
        return value


class RunUpdate(CamelModel):
    """Payload for partially updating run metadata (``PATCH``).

    ``promptId``/``promptVersionId`` are system-managed and intentionally not
    updatable through the API.
    """

    inputs: dict[str, Any] | None = None
    models: list[str] | None = None


class RunRead(RunBase, TimestampedModel):
    """Run as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
        prompt_id: Prompt reference (``ObjectId`` string).
        prompt_version_id: Prompt version reference (``ObjectId`` string).
        models: Model keys executed for this run.
        prompts: The per-model previewed prompts that were executed.
        prompt_text: The primary prompt text used for this run (first variant).
    """

    id: str
    prompt_id: str | None = None
    prompt_version_id: str | None = None
    models: list[str] = Field(default_factory=list)
    prompts: list[RunPrompt] = Field(default_factory=list)
    prompt_text: str | None = None


class RunWithResponses(CamelModel):
    """Composite returned by ``POST /runs`` and ``GET /runs/{id}``.

    Run metadata and child collections are returned at the root (no nested
    ``run`` object). ``inputs``, ``models``, ``prompts``, and ``promptText``
    are omitted from this envelope; see :class:`RunRead` for the full document.

    Attributes:
        id: String form of the MongoDB ``_id``.
        prompt_id: Prompt reference (``ObjectId`` string).
        prompt_version_id: Prompt version reference (``ObjectId`` string).
        use_case_key: Use-case key (e.g. ``"travel"``).
        created_at: When the run was created (UTC).
        updated_at: When the run was last updated (UTC).
        responses: Per-model responses for the run.
        evaluations: Evaluations linked to the run's responses.
    """

    id: str
    prompt_id: str | None = None
    prompt_version_id: str | None = None
    use_case_key: str = Field(default="travel")
    created_at: datetime | None = None
    updated_at: datetime | None = None
    responses: list[ResponseRead] = Field(default_factory=list)
    evaluations: list[EvaluationRead] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_nested_or_flat(cls, value: Any) -> Any:
        """Accept legacy nested ``{"run": ..., "responses": ...}`` assembly.

        The run service still builds a nested dict internally; this validator
        flattens it before field validation so callers need not change.

        Args:
            value: Raw input (flat mapping or nested legacy shape).

        Returns:
            A flat mapping suitable for :class:`RunWithResponses`.
        """
        if not isinstance(value, dict) or "run" not in value:
            return value
        run = value.get("run") or {}
        return {
            "id": run.get("id"),
            "promptId": run.get("promptId"),
            "promptVersionId": run.get("promptVersionId"),
            "useCaseKey": run.get("useCaseKey", "travel"),
            "createdAt": run.get("createdAt"),
            "updatedAt": run.get("updatedAt"),
            "responses": value.get("responses", []),
            "evaluations": value.get("evaluations", []),
        }

    @classmethod
    def from_run_doc(
        cls,
        run_doc: dict[str, Any],
        responses: list[dict[str, Any]],
        evaluations: list[dict[str, Any]],
    ) -> RunWithResponses:
        """Build a flat composite from a serialised run document + children.

        Args:
            run_doc: The serialised run document (camelCase keys).
            responses: Serialised response documents.
            evaluations: Serialised evaluation documents.

        Returns:
            A validated :class:`RunWithResponses` instance.
        """
        return cls.model_validate(
            {
                "id": run_doc["id"],
                "promptId": run_doc.get("promptId"),
                "promptVersionId": run_doc.get("promptVersionId"),
                "useCaseKey": run_doc.get("useCaseKey", "travel"),
                "createdAt": run_doc.get("createdAt"),
                "updatedAt": run_doc.get("updatedAt"),
                "responses": responses,
                "evaluations": evaluations,
            }
        )
