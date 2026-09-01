from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Wires structlog into Python's stdlib logging, so every module can
    just do `structlog.get_logger(__name__)` and get structured, leveled
    output.

    JSON in anything but local dev, so log lines are ready for a real
    aggregator (Datadog, CloudWatch, whatever) without any extra work —
    that's the whole point of structured logging, and `structlog` sat in
    requirements.txt unused through Day 10, so every log line up to this
    point was just a plain string, not actually structured at all. A
    colored, human-readable console format in dev, since nobody wants to
    read raw JSON in a terminal while iterating locally.

    Call this once, at process startup — see app/main.py.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.environment == "development"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(settings.log_level)
