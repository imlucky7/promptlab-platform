"""Aggregates all v1 route modules into a single router.

Mounted under the configurable ``API_V1_PREFIX`` (default ``/api/v1``) by the
application factory.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    routes_dashboard,
    routes_evaluations,
    routes_metrics,
    routes_preview,
    routes_prompt_suggestions,
    routes_prompt_versions,
    routes_prompts,
    routes_responses,
    routes_runs,
    routes_use_case_templates,
)

# Single router that composes every resource router for the v1 API.
api_router = APIRouter()
api_router.include_router(routes_use_case_templates.router)
api_router.include_router(routes_prompts.router)
api_router.include_router(routes_prompt_versions.router)
api_router.include_router(routes_runs.router)
api_router.include_router(routes_responses.router)
api_router.include_router(routes_evaluations.router)
api_router.include_router(routes_prompt_suggestions.router)
api_router.include_router(routes_metrics.router)
api_router.include_router(routes_preview.router)
api_router.include_router(routes_dashboard.router)
