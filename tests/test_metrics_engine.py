"""Unit tests for aggregation logic in the evaluation/metrics engines."""

from __future__ import annotations

from app.services.evaluation_engine import EvaluationEngine
from app.services.metrics_engine import _opt_float


def test_aggregate_scores_by_model_averages_correctly() -> None:
    """Scores should be averaged per model with an evaluation count."""
    engine = EvaluationEngine(evaluations_repo=None)  # type: ignore[arg-type]
    evaluations = [
        {
            "modelKey": "anthropic",
            "scores": {
                "correctness": 4,
                "completeness": 4,
                "styleFit": 5,
                "faithfulness": 5,
                "overall": 4,
            },
        },
        {
            "modelKey": "anthropic",
            "scores": {
                "correctness": 2,
                "completeness": 2,
                "styleFit": 3,
                "faithfulness": 3,
                "overall": 2,
            },
        },
    ]
    result = engine.aggregate_scores_by_model(evaluations)
    assert result["anthropic"]["evaluationCount"] == 2
    assert result["anthropic"]["overall"] == 3.0
    assert result["anthropic"]["styleFit"] == 4.0


def test_aggregate_handles_empty_list() -> None:
    """No evaluations should produce an empty aggregate mapping."""
    engine = EvaluationEngine(evaluations_repo=None)  # type: ignore[arg-type]
    assert engine.aggregate_scores_by_model([]) == {}


def test_opt_float_handles_non_numeric() -> None:
    """``_opt_float`` returns None for non-numeric inputs."""
    assert _opt_float(None) is None
    assert _opt_float("x") is None
    assert _opt_float(3) == 3.0
