"""Shared SSE stream event payloads for preview and run endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.models.common import CamelModel

StreamTarget = Literal["preview", "response"]


class StreamProgressEvent(CamelModel):
    """Progress update during a long-running streamed operation."""

    type: Literal["progress"] = "progress"
    phase: str
    message: str | None = None


class StreamTokenEvent(CamelModel):
    """Incremental text chunk for live UI updates."""

    type: Literal["token"] = "token"
    target: StreamTarget
    delta: str


class StreamCompleteEvent(CamelModel):
    """Final payload when streaming finishes successfully."""

    type: Literal["complete"] = "complete"
    data: dict[str, Any] = Field(default_factory=dict)


class StreamErrorEvent(CamelModel):
    """Terminal error emitted on the stream."""

    type: Literal["error"] = "error"
    message: str


def progress_event(phase: str, message: str | None = None) -> dict[str, Any]:
    """Build a progress event dict."""
    return StreamProgressEvent(phase=phase, message=message).model_dump(by_alias=True)


def token_event(target: StreamTarget, delta: str) -> dict[str, Any]:
    """Build a token delta event dict."""
    return StreamTokenEvent(target=target, delta=delta).model_dump(by_alias=True)


def complete_event(data: Any) -> dict[str, Any]:
    """Build a complete event dict from a Pydantic model or mapping."""
    if hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json", by_alias=True)
    else:
        payload = data
    return StreamCompleteEvent(data=payload).model_dump(by_alias=True)


def error_event(message: str) -> dict[str, Any]:
    """Build an error event dict."""
    return StreamErrorEvent(message=message).model_dump(by_alias=True)
