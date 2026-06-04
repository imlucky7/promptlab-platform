"""Schemas for the ``evaluations`` collection (human ratings)."""

from __future__ import annotations

from pydantic import Field

from app.models.common import CamelModel, TimestampedModel


class EvaluationScores(CamelModel):
    """The 1-5 rating dimensions captured for a response.

    Attributes:
        correctness: Factual/logical correctness (1-5).
        completeness: How completely the response covers the task (1-5).
        style_fit: How well the style fits the request (1-5).
        faithfulness: Faithfulness to the prompt/inputs (1-5).
        overall: Overall quality (1-5).
    """

    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    style_fit: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)


class EvaluationBase(CamelModel):
    """Fields shared by evaluation create/replace operations.

    Attributes:
        run_id: Reference to the run.
        response_id: Reference to the evaluated response.
        model_key: Logical model key of the evaluated response.
        scores: The rating dimensions.
        comments: Optional free-text reviewer comments.
    """

    run_id: str
    response_id: str
    model_key: str
    scores: EvaluationScores
    comments: str | None = None


class EvaluationCreate(EvaluationBase):
    """Payload for creating/upserting an evaluation (``POST``).

    The combination of ``run_id`` + ``response_id`` + ``model_key`` is treated as
    the natural key for upsert semantics (FR-14).
    """


class EvaluationUpdate(CamelModel):
    """Payload for partially updating an evaluation (``PATCH``)."""

    scores: EvaluationScores | None = None
    comments: str | None = None


class EvaluationRead(EvaluationBase, TimestampedModel):
    """Evaluation as returned by the API.

    Attributes:
        id: String form of the MongoDB ``_id``.
    """

    id: str
