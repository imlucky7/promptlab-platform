"""Shared pytest fixtures.

Integration tests use ``mongomock-motor`` to provide an in-memory, async
MongoDB-compatible database so the full app (routes -> services -> repositories)
can be exercised without a running MongoDB instance. The LLM gateway runs in stub
mode, so no network access or API keys are required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from mongomock_motor import AsyncMongoMockClient

from app.core.config import Settings, _default_model_catalog
from app.core.dependencies import get_app_settings, get_db
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Provide deterministic test settings (stub gateway, local tokenizer).

    Returns:
        A :class:`Settings` configured for offline tests.
    """
    return Settings(
        environment="local",
        debug=True,
        llm_gateway_stub_mode=True,
        ollama_preview_stub_mode=True,
        token_estimation_mode="local",
        seed_on_startup=False,
        model_catalog=_default_model_catalog(),
    )


@pytest_asyncio.fixture
async def mock_db() -> AsyncIterator:
    """Provide a fresh in-memory Mongo database.

    Yields:
        An async Mongo-compatible database handle.
    """
    client = AsyncMongoMockClient()
    db = client["prompt_lab_test"]
    yield db


@pytest_asyncio.fixture
async def client(settings: Settings, mock_db) -> AsyncIterator[AsyncClient]:
    """Provide an HTTP client bound to the app with overridden dependencies.

    The database and settings dependencies are overridden so the app uses the
    in-memory database and deterministic settings. The ASGI transport does not
    run lifespan events, so no real MongoDB connection is attempted.

    Args:
        settings: Test settings fixture.
        mock_db: In-memory database fixture.

    Yields:
        An ``httpx.AsyncClient`` for issuing requests.
    """
    app = create_app(settings)
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_app_settings] = lambda: settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()
