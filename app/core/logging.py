"""Application logging configuration.

Provides a single :func:`configure_logging` entry point used at startup. Logging
is intentionally simple (stdlib ``logging``) but structured enough to be useful
in containers, where logs are typically shipped from stdout.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole process.

    Calling this more than once is a no-op so that repeated imports or test
    setups do not attach duplicate handlers.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    root = logging.getLogger()
    root.setLevel(log_level)
    # Replace any pre-existing handlers (e.g. from uvicorn) for consistent output.
    root.handlers = [handler]

    # Tame noisy third-party loggers in normal operation.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
