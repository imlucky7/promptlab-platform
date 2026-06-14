"""Domain error types and FastAPI exception handlers.

Defining a small hierarchy of application errors keeps the service/repository
layers framework-agnostic: they raise plain Python exceptions, and a thin set of
handlers translates them into consistent JSON HTTP responses.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for all application-specific errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code to return to the client.
        code: Stable, machine-readable error code string.
        details: Optional structured details for debugging or clients.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable error description.
            details: Optional structured details to expose to the client.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    """Raised when a requested entity does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationAppError(AppError):
    """Raised for domain-level validation failures (beyond schema checks)."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"


class ConflictError(AppError):
    """Raised when an operation conflicts with current state."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class UpstreamError(AppError):
    """Raised when an upstream dependency (e.g. LLM gateway) fails."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "upstream_error"


def _error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the standard error response body.

    Args:
        code: Machine-readable error code.
        message: Human-readable message.
        details: Optional structured details.

    Returns:
        A JSON-serialisable error envelope.
    """
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Translate an :class:`AppError` into a JSON response."""
        # Log at warning for client errors, error for server faults.
        if exc.status_code >= 500:
            logger.error("AppError: %s (%s)", exc.message, exc.code, exc_info=exc)
        else:
            logger.warning("AppError: %s (%s)", exc.message, exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Normalise FastAPI request validation errors.

        ``exc.errors()`` may embed non-JSON-serialisable values (e.g. the
        originating ``ValueError`` under ``ctx`` for custom validators), so the
        error list is passed through ``jsonable_encoder`` before serialisation.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                "validation_error",
                "Request validation failed.",
                {"errors": jsonable_encoder(exc.errors())},
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler so unexpected errors still return clean JSON."""
        logger.exception("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload("internal_error", "An unexpected error occurred."),
        )
