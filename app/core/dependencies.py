"""FastAPI dependency providers.

This module wires the object graph together: it exposes provider functions that
FastAPI injects into route handlers. Each layer depends only on the layer below
it (routes -> services -> repositories -> database), which keeps the code modular
and testable (dependencies can be overridden in tests).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import Settings, get_settings
from app.db.mongo import mongo
from app.db.repositories.evaluations_repo import EvaluationsRepository
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.repositories.prompt_suggestions_repo import PromptSuggestionsRepository
from app.db.repositories.prompt_versions_repo import PromptVersionsRepository
from app.db.repositories.prompts_repo import PromptsRepository
from app.db.repositories.responses_repo import ResponsesRepository
from app.db.repositories.runs_repo import RunsRepository
from app.services.evaluation_engine import EvaluationEngine
from app.services.execution_engine import ExecutionEngine
from app.services.llm_gateway_client import LLMGatewayClient
from app.services.metrics_engine import MetricsEngine
from app.services.ollama_client import OllamaClient
from app.services.ollama_preview_service import OllamaPreviewService
from app.services.response_normalizer import ResponseNormalizer
from app.services.budget_service import BudgetService
from app.services.run_service import RunService
from app.services.suggestion_engine import SuggestionEngine
from app.services.token_estimator import TokenEstimator
from app.services.versioning_engine import VersioningEngine


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
def get_app_settings() -> Settings:
    """Return the process-wide settings object.

    Returns:
        The cached :class:`Settings` instance.
    """
    return get_settings()


def get_db() -> AsyncIOMotorDatabase:
    """Return the active Mongo database handle.

    Returns:
        The connected database handle.
    """
    return mongo.get_database()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DbDep = Annotated[AsyncIOMotorDatabase, Depends(get_db)]


# --------------------------------------------------------------------------- #
# Repositories
# --------------------------------------------------------------------------- #
def get_prompts_repo(db: DbDep) -> PromptsRepository:
    """Provide the prompts repository."""
    return PromptsRepository(db)


def get_versions_repo(db: DbDep) -> PromptVersionsRepository:
    """Provide the prompt versions repository."""
    return PromptVersionsRepository(db)


def get_runs_repo(db: DbDep) -> RunsRepository:
    """Provide the runs repository."""
    return RunsRepository(db)


def get_responses_repo(db: DbDep) -> ResponsesRepository:
    """Provide the responses repository."""
    return ResponsesRepository(db)


def get_evaluations_repo(db: DbDep) -> EvaluationsRepository:
    """Provide the evaluations repository."""
    return EvaluationsRepository(db)


def get_suggestions_repo(db: DbDep) -> PromptSuggestionsRepository:
    """Provide the prompt suggestions repository."""
    return PromptSuggestionsRepository(db)


def get_metrics_repo(db: DbDep) -> MetricsRepository:
    """Provide the metrics logs repository."""
    return MetricsRepository(db)


PromptsRepoDep = Annotated[PromptsRepository, Depends(get_prompts_repo)]
VersionsRepoDep = Annotated[PromptVersionsRepository, Depends(get_versions_repo)]
RunsRepoDep = Annotated[RunsRepository, Depends(get_runs_repo)]
ResponsesRepoDep = Annotated[ResponsesRepository, Depends(get_responses_repo)]
EvaluationsRepoDep = Annotated[EvaluationsRepository, Depends(get_evaluations_repo)]
SuggestionsRepoDep = Annotated[PromptSuggestionsRepository, Depends(get_suggestions_repo)]
MetricsRepoDep = Annotated[MetricsRepository, Depends(get_metrics_repo)]


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
def get_ollama_client(settings: SettingsDep) -> OllamaClient:
    """Provide the Ollama HTTP client."""
    return OllamaClient(settings)


def get_gateway_client(
    settings: SettingsDep,
    ollama: Annotated[OllamaClient, Depends(get_ollama_client)],
) -> LLMGatewayClient:
    """Provide the LLM gateway client."""
    return LLMGatewayClient(settings, ollama)


GatewayDep = Annotated[LLMGatewayClient, Depends(get_gateway_client)]


def get_token_estimator(settings: SettingsDep, gateway: GatewayDep) -> TokenEstimator:
    """Provide the token estimator."""
    return TokenEstimator(settings, gateway)


def get_suggestion_engine() -> SuggestionEngine:
    """Provide the suggestion engine."""
    return SuggestionEngine()


def get_response_normalizer(settings: SettingsDep) -> ResponseNormalizer:
    """Provide the response normalizer."""
    return ResponseNormalizer(settings)


TokenEstimatorDep = Annotated[TokenEstimator, Depends(get_token_estimator)]
SuggestionEngineDep = Annotated[SuggestionEngine, Depends(get_suggestion_engine)]
ResponseNormalizerDep = Annotated[ResponseNormalizer, Depends(get_response_normalizer)]


def get_ollama_preview_service(
    ollama: Annotated[OllamaClient, Depends(get_ollama_client)],
    token_estimator: TokenEstimatorDep,
) -> OllamaPreviewService:
    """Provide the Ollama-backed preview service."""
    return OllamaPreviewService(ollama, token_estimator)


OllamaPreviewDep = Annotated[OllamaPreviewService, Depends(get_ollama_preview_service)]


def get_versioning_engine(versions_repo: VersionsRepoDep) -> VersioningEngine:
    """Provide the versioning engine."""
    return VersioningEngine(versions_repo)


def get_evaluation_engine(evaluations_repo: EvaluationsRepoDep) -> EvaluationEngine:
    """Provide the evaluation engine."""
    return EvaluationEngine(evaluations_repo)


def get_execution_engine(
    gateway: GatewayDep,
    normalizer: ResponseNormalizerDep,
    responses_repo: ResponsesRepoDep,
    metrics_repo: MetricsRepoDep,
) -> ExecutionEngine:
    """Provide the execution engine."""
    return ExecutionEngine(gateway, normalizer, responses_repo, metrics_repo)


VersioningEngineDep = Annotated[VersioningEngine, Depends(get_versioning_engine)]
EvaluationEngineDep = Annotated[EvaluationEngine, Depends(get_evaluation_engine)]
ExecutionEngineDep = Annotated[ExecutionEngine, Depends(get_execution_engine)]


def get_metrics_engine(
    runs_repo: RunsRepoDep,
    responses_repo: ResponsesRepoDep,
    evaluations_repo: EvaluationsRepoDep,
    evaluation_engine: EvaluationEngineDep,
) -> MetricsEngine:
    """Provide the metrics engine."""
    return MetricsEngine(runs_repo, responses_repo, evaluations_repo, evaluation_engine)


def get_run_service(
    runs_repo: RunsRepoDep,
    responses_repo: ResponsesRepoDep,
    evaluations_repo: EvaluationsRepoDep,
    execution_engine: ExecutionEngineDep,
) -> RunService:
    """Provide the run orchestration service."""
    return RunService(
        runs_repo,
        responses_repo,
        evaluations_repo,
        execution_engine,
    )


MetricsEngineDep = Annotated[MetricsEngine, Depends(get_metrics_engine)]
RunServiceDep = Annotated[RunService, Depends(get_run_service)]


def get_budget_service(
    runs_repo: RunsRepoDep,
    responses_repo: ResponsesRepoDep,
    gateway: GatewayDep,
) -> BudgetService:
    """Provide the budget generation service."""
    return BudgetService(runs_repo, responses_repo, gateway)


BudgetServiceDep = Annotated[BudgetService, Depends(get_budget_service)]
