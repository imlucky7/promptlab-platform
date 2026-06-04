"""Metrics engine.

Computes the per-model and per-run aggregates that back the dashboard endpoints
(FR-20, FR-21): average human scores, average token usage and latency per model,
and a recent-runs listing with per-model overall scores.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.db.repositories.evaluations_repo import EvaluationsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.dashboard import (
    DashboardRunItem,
    DashboardRuns,
    DashboardSummary,
    ModelScoreSummary,
    RunModelScore,
)
from app.services.evaluation_engine import EvaluationEngine


class MetricsEngine:
    """Builds dashboard summaries and run histories from stored data."""

    def __init__(
        self,
        runs_repo: RunsRepository,
        responses_repo: ResponsesRepository,
        evaluations_repo: EvaluationsRepository,
        evaluation_engine: EvaluationEngine,
    ) -> None:
        """Initialise the engine.

        Args:
            runs_repo: Repository for ``runs``.
            responses_repo: Repository for ``responses``.
            evaluations_repo: Repository for ``evaluations``.
            evaluation_engine: Provides score aggregation helpers.
        """
        self._runs_repo = runs_repo
        self._responses_repo = responses_repo
        self._evaluations_repo = evaluations_repo
        self._evaluation_engine = evaluation_engine

    async def summary(self, use_case_key: str) -> DashboardSummary:
        """Compute per-model aggregate scores and metrics for a use case.

        Args:
            use_case_key: The use-case key to scope the summary to.

        Returns:
            A :class:`DashboardSummary` with one entry per observed model.
        """
        runs, _ = await self._runs_repo.list(
            filters={"useCaseKey": use_case_key}, limit=10_000, offset=0
        )
        run_ids = [run["id"] for run in runs]

        responses = await self._responses_repo.list_by_use_case(use_case_key, run_ids)
        evaluations = await self._evaluations_repo.list_by_runs(run_ids)

        score_by_model = self._evaluation_engine.aggregate_scores_by_model(evaluations)
        usage_by_model = self._aggregate_usage_by_model(responses)

        # Union of model keys seen in either responses or evaluations.
        model_keys = sorted(set(score_by_model) | set(usage_by_model))
        models: list[ModelScoreSummary] = []
        for model_key in model_keys:
            scores = score_by_model.get(model_key, {})
            usage = usage_by_model.get(model_key, {})
            models.append(
                ModelScoreSummary(
                    model_key=model_key,
                    avg_overall=_opt_float(scores.get("overall")),
                    avg_correctness=_opt_float(scores.get("correctness")),
                    avg_completeness=_opt_float(scores.get("completeness")),
                    avg_style_fit=_opt_float(scores.get("styleFit")),
                    avg_faithfulness=_opt_float(scores.get("faithfulness")),
                    avg_input_tokens=_opt_float(usage.get("avgInputTokens")),
                    avg_output_tokens=_opt_float(usage.get("avgOutputTokens")),
                    avg_latency_ms=_opt_float(usage.get("avgLatencyMs")),
                    evaluation_count=int(scores.get("evaluationCount", 0)),
                    response_count=int(usage.get("responseCount", 0)),
                )
            )
        return DashboardSummary(use_case_key=use_case_key, models=models)

    async def recent_runs(self, use_case_key: str, limit: int = 20) -> DashboardRuns:
        """List recent runs with their per-model overall scores.

        Args:
            use_case_key: The use-case key to scope the listing to.
            limit: Maximum number of runs to return.

        Returns:
            A :class:`DashboardRuns` with the most recent runs first.
        """
        runs, _ = await self._runs_repo.list(
            filters={"useCaseKey": use_case_key},
            limit=limit,
            offset=0,
            sort=[("createdAt", -1)],
        )
        run_ids = [run["id"] for run in runs]
        evaluations = await self._evaluations_repo.list_by_runs(run_ids)

        # Index overall scores by (run_id, model_key) for quick lookup.
        overall_by_run_model: dict[tuple[str, str], int] = {}
        for evaluation in evaluations:
            run_id = evaluation.get("runId")
            model_key = evaluation.get("modelKey")
            overall = (evaluation.get("scores") or {}).get("overall")
            if run_id and model_key and isinstance(overall, int):
                overall_by_run_model[(run_id, model_key)] = overall

        items: list[DashboardRunItem] = []
        for run in runs:
            run_id = run["id"]
            scores = [
                RunModelScore(
                    model_key=model_key,
                    overall=overall_by_run_model.get((run_id, model_key)),
                )
                for model_key in run.get("models", [])
            ]
            items.append(
                DashboardRunItem(
                    run_id=run_id,
                    created_at=run.get("createdAt"),
                    scores=scores,
                )
            )
        return DashboardRuns(use_case_key=use_case_key, runs=items)

    def _aggregate_usage_by_model(
        self, responses: list[dict[str, Any]]
    ) -> dict[str, dict[str, float | int]]:
        """Average token usage and latency per model key.

        Args:
            responses: Serialised response documents.

        Returns:
            A mapping of ``model_key`` to average usage metrics plus a
            ``responseCount``.
        """
        input_sums: dict[str, float] = defaultdict(float)
        output_sums: dict[str, float] = defaultdict(float)
        latency_sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)

        for response in responses:
            model_key = response.get("modelKey")
            if model_key is None:
                continue
            usage = response.get("usage") or {}
            counts[model_key] += 1
            input_sums[model_key] += float(usage.get("inputTokens", 0))
            output_sums[model_key] += float(usage.get("outputTokens", 0))
            latency_sums[model_key] += float(usage.get("latencyMs", 0))

        result: dict[str, dict[str, float | int]] = {}
        for model_key, count in counts.items():
            result[model_key] = {
                "avgInputTokens": round(input_sums[model_key] / count, 2) if count else 0.0,
                "avgOutputTokens": round(output_sums[model_key] / count, 2) if count else 0.0,
                "avgLatencyMs": round(latency_sums[model_key] / count, 2) if count else 0.0,
                "responseCount": count,
            }
        return result


def _opt_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when not numeric.

    Args:
        value: A candidate numeric value.

    Returns:
        The float value, or ``None``.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return None
