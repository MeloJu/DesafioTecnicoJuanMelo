"""
Structured logging — JSON format, correlation_id via contextvars.

Usage in any module:
    from app.logging.logger import get_logger
    log = get_logger()
    log.info("pipeline_started", empresa="Construtiva", setor="obras")

To redirect output (e.g., to a file in scripts):
    from app.logging.logger import configure_logging
    configure_logging(stream=open("run.log", "w"))

The correlation_id is bound once at the pipeline entry point:
    from app.logging.logger import bind_correlation_id, clear_correlation_id
    bind_correlation_id(str(uuid.uuid4()))
"""
import sys
from contextvars import ContextVar
from typing import IO, Optional

import structlog

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

_PROCESSORS = [
    lambda logger, method, event_dict: (
        event_dict.update(correlation_id=_correlation_id.get())
        or event_dict
        if _correlation_id.get() is not None
        else event_dict
    ),
    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
    structlog.processors.add_log_level,
    structlog.processors.JSONRenderer(),
]


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


def configure_logging(stream: IO[str] = sys.stdout) -> None:
    """
    Configure the global structlog output stream.

    Call this at the start of a script to redirect all JSON logs:
        configure_logging(stream=open("run.log", "w", encoding="utf-8"))

    All loggers obtained via get_logger() will use this stream for
    subsequent log calls — no need to recreate them.
    """
    processors = [
        _add_correlation_id,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=False,
    )


def get_logger(stream: IO[str] = None) -> structlog.BoundLogger:
    """
    Return a structlog logger.

    Without arguments → uses the global configuration (changed via configure_logging).
    With stream       → creates an independent logger for that stream (tests/REPL).
    """
    if stream is not None:
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
    return structlog.get_logger()


# Configure defaults at import time so modules that call get_logger()
# at module level get a working logger immediately.
configure_logging()
