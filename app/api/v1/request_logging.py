"""Shared request-logging dependency for v1 API routes.

Provides :func:`log_request`, a small dependency factory that each route module
wires into its router so every incoming request is logged as pretty-printed JSON
using that module's own logger.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from fastapi import Request

# Header names whose values must never be written to logs verbatim.
_SENSITIVE_HEADERS = frozenset({"authorization", "cookie", "x-api-key", "proxy-authorization"})
_REDACTED = "***redacted***"


def log_request(logger: logging.Logger) -> Callable[[Request], Awaitable[None]]:
    """Build a FastAPI dependency that logs the full request as pretty JSON.

    Args:
        logger: The calling route module's logger, used to emit the log line so
            the source module is visible in the log output.

    Returns:
        An async dependency callable that logs each incoming request.
    """

    async def _dependency(request: Request) -> None:
        await _log_full_request(request, logger)

    return _dependency


async def _log_full_request(request: Request, logger: logging.Logger) -> None:
    """Emit a single pretty-printed JSON log line describing ``request``.

    The request body is consumed via ``request.body()``; Starlette caches it so
    downstream handlers can still read/parse it. JSON bodies are embedded as
    parsed objects, other payloads as raw text.

    Args:
        request: The incoming HTTP request.
        logger: Logger used to emit the log line.
    """
    payload: dict[str, object] = {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "pathParams": dict(request.path_params),
        "queryParams": dict(request.query_params),
        "headers": _safe_headers(request),
        "client": _client_info(request),
        "body": await _read_body(request),
    }
    logger.info("Incoming request:\n%s", json.dumps(payload, indent=2, default=str, sort_keys=True))


def _safe_headers(request: Request) -> dict[str, str]:
    """Return request headers with sensitive values redacted.

    Args:
        request: The incoming HTTP request.

    Returns:
        A mapping of header name to value (redacted where sensitive).
    """
    return {
        name: (_REDACTED if name.lower() in _SENSITIVE_HEADERS else value)
        for name, value in request.headers.items()
    }


def _client_info(request: Request) -> dict[str, object] | None:
    """Return basic client host/port information when available.

    Args:
        request: The incoming HTTP request.

    Returns:
        A mapping with ``host`` and ``port``, or ``None`` if unknown.
    """
    if request.client is None:
        return None
    return {"host": request.client.host, "port": request.client.port}


async def _read_body(request: Request) -> object:
    """Read and best-effort decode the request body for logging.

    Args:
        request: The incoming HTTP request.

    Returns:
        The parsed JSON body, the decoded text, ``None`` when empty, or a short
        placeholder when the body is not UTF-8 decodable.
    """
    raw = await request.body()
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"<{len(raw)} bytes of non-UTF-8 body>"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
