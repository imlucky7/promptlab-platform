"""Server-Sent Events helpers for streaming API responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse


def format_sse_frame(event: str, payload: dict[str, Any]) -> str:
    """Format a single SSE frame.

    Args:
        event: SSE event name (usually matches payload ``type``).
        payload: JSON-serialisable event body.

    Returns:
        A wire-format SSE message ending with a blank line.
    """
    data = json.dumps(payload, default=str)
    return f"event: {event}\ndata: {data}\n\n"


async def iter_sse_events(
    generator: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[str]:
    """Convert a stream of event dicts into SSE wire frames.

    Args:
        generator: Async iterator yielding event payloads with a ``type`` key.

    Yields:
        SSE-formatted strings suitable for :class:`StreamingResponse`.
    """
    async for payload in generator:
        event_name = str(payload.get("type", "message"))
        yield format_sse_frame(event_name, payload)


def sse_streaming_response(
    generator: AsyncIterator[dict[str, Any]],
) -> StreamingResponse:
    """Wrap an async event generator as an SSE :class:`StreamingResponse`.

    Args:
        generator: Async iterator of stream event dicts.

    Returns:
        A streaming HTTP response with ``text/event-stream`` media type.
    """
    return StreamingResponse(
        iter_sse_events(generator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
