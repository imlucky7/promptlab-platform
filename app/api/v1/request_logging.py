"""Shared request/response logging for v1 API routes.

Provides :func:`log_request`, a dependency factory that each route module wires
into its router so every incoming request is logged as pretty-printed JSON, and
:func:`log_response_middleware`, an HTTP middleware that logs outgoing v1
responses in the same format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from starlette.responses import Response

# Header names whose values must never be written to logs verbatim.
_SENSITIVE_HEADERS = frozenset({"authorization", "cookie", "x-api-key", "proxy-authorization"})
_REDACTED = "***redacted***"
_DEFAULT_MAX_STRING = 1000
_DEFAULT_MAX_DEPTH = 6

_response_logger = logging.getLogger("app.api.v1.response")


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


def log_response_middleware(
    api_prefix: str,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Build HTTP middleware that logs outgoing v1 API responses as pretty JSON.

    Args:
        api_prefix: URL prefix for the v1 API (e.g. ``/api/v1``). Requests
            outside this prefix are passed through without response logging.

    Returns:
        An async middleware callable suitable for ``app.middleware("http")(...)``.
    """

    async def _middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not request.url.path.startswith(api_prefix):
            return await call_next(request)

        response = await call_next(request)
        body = b"".join([chunk async for chunk in response.body_iterator])

        log_payload: dict[str, object] = {
            "method": request.method,
            "path": request.url.path,
            "statusCode": response.status_code,
            "body": _parse_response_body(body, response.headers.get("content-type")),
        }
        _response_logger.info(
            "Outgoing response:\n%s",
            json.dumps(
                truncate_for_log(log_payload),
                indent=2,
                default=str,
                sort_keys=True,
            ),
        )

        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    return _middleware


def truncate_for_log(
    value: Any,
    *,
    max_string: int = _DEFAULT_MAX_STRING,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    _depth: int = 0,
) -> Any:
    """Recursively truncate large strings and nested structures for safe logging.

    Args:
        value: The value to truncate.
        max_string: Maximum length for string values before clipping.
        max_depth: Maximum nesting depth before replacing with a placeholder.
        _depth: Current recursion depth (internal).

    Returns:
        A log-safe copy of ``value``.
    """
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return f"{value[:max_string]}… ({len(value)} chars total)"

    if isinstance(value, dict):
        if _depth >= max_depth:
            return "<max depth reached>"
        return {
            str(key): truncate_for_log(
                item, max_string=max_string, max_depth=max_depth, _depth=_depth + 1
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        if _depth >= max_depth:
            return "<max depth reached>"
        return [
            truncate_for_log(item, max_string=max_string, max_depth=max_depth, _depth=_depth + 1)
            for item in value
        ]

    if isinstance(value, tuple):
        if _depth >= max_depth:
            return "<max depth reached>"
        return [
            truncate_for_log(item, max_string=max_string, max_depth=max_depth, _depth=_depth + 1)
            for item in value
        ]

    return value


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
    logger.info(
        "Incoming request:\n%s",
        json.dumps(
            truncate_for_log(payload),
            indent=2,
            default=str,
            sort_keys=True,
        ),
    )


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
    return _parse_response_body(raw, request.headers.get("content-type"))


def _parse_response_body(raw: bytes, content_type: str | None) -> object:
    """Decode a response/request body for logging.

    Args:
        raw: Raw body bytes.
        content_type: Optional ``Content-Type`` header value.

    Returns:
        Parsed JSON, decoded text, ``None`` when empty, or a short placeholder.
    """
    if not raw:
        return None
    if content_type and "application/json" not in content_type.lower():
        return f"<{len(raw)} bytes non-JSON body>"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"<{len(raw)} bytes of non-UTF-8 body>"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
