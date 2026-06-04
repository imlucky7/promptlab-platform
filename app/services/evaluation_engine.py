"""Evaluation engine.

Handles upsert semantics for human evaluations (FR-14) and provides score
aggregation helpers used by the dashboard/metrics layer (FR-15).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.db.repositories.evaluations_repo import EvaluationsRepository

# The five 1-5 score dimensions captured per evaluation.
_SCORE_FIELDS = ("correctness", "completeness", "styleFit", "faithfulness", "overall")


class EvaluationEngine:
    """Manages evaluation persistence and aggregation."""

    def __init__(self, evaluations_repo: EvaluationsRepository) -> None:
        """Initialise the engine.

        Args:
            evaluations_repo: Repository for ``evaluations``.
        """
        self._evaluations_repo = evaluations_repo

    async def upsert(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create or update an evaluation by its natural key.

        Args:
            data: Evaluation fields (camelCase keys).

        Returns:
            The upserted, serialised evaluation document.
        """
        return await self._evaluations_repo.upsert(data)

    def aggregate_scores_by_model(
        self, evaluations: list[dict[str, Any]]
    ) -> dict[str, dict[str, float | int]]:
        """Average each score dimension per model key.

        Args:
            evaluations: Serialised evaluation documents.

        Returns:
            A mapping of ``model_key`` to a dict of average scores plus an
            ``evaluationCount``.
        """
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[str, int] = defaultdict(int)

        for evaluation in evaluations:
            model_key = evaluation.get("modelKey")
            if model_key is None:
                continue
            scores = evaluation.get("scores") or {}
            counts[model_key] += 1
            for field in _SCORE_FIELDS:
                value = scores.get(field)
                if isinstance(value, (int, float)):
                    sums[model_key][field] += float(value)

        result: dict[str, dict[str, float | int]] = {}
        for model_key, count in counts.items():
            averages: dict[str, float | int] = {"evaluationCount": count}
            for field in _SCORE_FIELDS:
                averages[field] = round(sums[model_key][field] / count, 3) if count else 0.0
            result[model_key] = averages
        return result
