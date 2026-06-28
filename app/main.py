"""Application entry point and FastAPI factory.

Wires together configuration, logging, the MongoDB lifecycle, middleware, the v1
API router, exception handlers, OpenAPI metadata and a health check. The exported
``app`` instance is what Uvicorn serves (``uvicorn app.main:app``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.request_logging import log_response_middleware
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.db.mongo import mongo

logger = get_logger(__name__)

# High-level description rendered in the OpenAPI docs (/docs, /redoc).
API_DESCRIPTION = """\
Prompt Lab - Travel Planner backend API.

Design structured travel-planning prompts, execute them across multiple LLMs via
a unified gateway, compare responses, evaluate quality and track experiments.

* **Preview** prompts (token estimates + suggestions) without persistence.
* **Runs** build/execute prompts across models and persist responses.
* **Evaluations** capture human ratings; **dashboard** endpoints aggregate them.
"""

# Tag metadata gives the generated OpenAPI docs a clear, grouped structure.
OPENAPI_TAGS: list[dict[str, Any]] = [
    {"name": "preview", "description": "Build prompt previews with token estimates and suggestions."},
    {"name": "runs", "description": "Create/execute runs and fetch results."},
    {"name": "responses", "description": "Per-model responses for runs."},
    {"name": "evaluations", "description": "Human evaluations of responses."},
    {"name": "dashboard", "description": "Aggregated metrics and run history."},
    {"name": "prompts", "description": "Logical prompt workspaces."},
    {"name": "prompt-versions", "description": "Versioned prompt snapshots."},
    {"name": "prompt-suggestions", "description": "Rule-based improvement suggestions."},
    {"name": "metrics-logs", "description": "Derived per-response metrics."},
    {"name": "system", "description": "Health and service metadata."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown side effects.

    On startup: connect to MongoDB and ensure indexes. On shutdown: close the
    MongoDB connection.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the running application.
    """
    settings: Settings = app.state.settings
    await mongo.connect(settings)
    logger.info("Application startup complete (env=%s)", settings.environment)
    try:
        yield
    finally:
        await mongo.close()
        logger.info("Application shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override (useful for tests). Defaults to the
            cached process settings.

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        # Serve interactive docs and the machine-readable spec.
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    # Expose settings on app state so the lifespan/handlers can read them.
    app.state.settings = settings

    # CORS so the Next.js frontend (different origin) can call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(log_response_middleware(settings.api_v1_prefix))

    register_exception_handlers(app)

    # Mount the versioned API under the configurable prefix.
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    _register_system_routes(app, settings)
    return app


def _register_system_routes(app: FastAPI, settings: Settings) -> None:
    """Register non-versioned system routes (root + health).

    Args:
        app: The FastAPI application instance.
        settings: Application settings (used to advertise the API prefix).
    """

    @app.get("/", tags=["system"], summary="Service metadata")
    async def root() -> dict[str, Any]:
        """Return basic service metadata and useful links."""
        return {
            "name": settings.app_name,
            "version": __version__,
            "environment": settings.environment,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "apiPrefix": settings.api_v1_prefix,
        }

    @app.get("/health", tags=["system"], summary="Health check")
    async def health() -> dict[str, str]:
        """Liveness/readiness probe.

        Pings MongoDB so orchestrators can detect a broken DB connection.
        """
        status = "ok"
        try:
            await mongo.get_database().command("ping")
        except Exception as exc:  # pragma: no cover - exercised in integration.
            logger.warning("Health check DB ping failed: %s", exc)
            status = "degraded"
        return {"status": status}


# The ASGI application served by Uvicorn (``uvicorn app.main:app``).
app = create_app()
