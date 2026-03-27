"""
Structured logging — JSON format, correlation_id via contextvars.

Usage in any module:
    from app.logging.logger import get_logger
    log = get_logger()
    log.info("pipeline_started", empresa="Construtiva", setor="obras")

The correlation_id is bound once at the pipeline entry point:
    from app.logging.logger import bind_correlation_id, clear_correlation_id
    bind_correlation_id(str(uuid.uuid4()))
"""
import io
import sys
from contextvars import ContextVar
from typing import IO, Optional

import structlog

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation_id to the current execution context."""
    _correlation_id.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear the correlation_id from the current execution context."""
    _correlation_id.set(None)


def _add_correlation_id(
    logger: object, method: str, event_dict: dict
) -> dict:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict["correlation_id"] = cid
    return event_dict


def get_logger(stream: IO[str] = sys.stdout) -> structlog.BoundLogger:
    """
    Return a structlog logger configured for JSON output.

    The `stream` parameter is exposed for testing — pass an io.StringIO
    to capture output without touching stdout.
    """
    processors = [
        _add_correlation_id,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]

    return structlog.wrap_logger(
        structlog.PrintLogger(stream),
        processors=processors,
    )
