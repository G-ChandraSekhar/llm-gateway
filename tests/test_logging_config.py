from __future__ import annotations

import json

import structlog

from app.core.config import Settings
from app.core.logging_config import configure_logging


def test_production_mode_emits_valid_json(capsys):
    configure_logging(Settings(environment="production", log_level="INFO"))
    log = structlog.get_logger("test_logger")

    log.info("something_happened", model="gpt-4o-mini", status_code=200)

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    parsed = json.loads(line)  # must be valid JSON, not raise

    assert parsed["event"] == "something_happened"
    assert parsed["model"] == "gpt-4o-mini"
    assert parsed["status_code"] == 200
    assert parsed["level"] == "info"
    assert "timestamp" in parsed


def test_development_mode_emits_human_readable_not_json(capsys):
    configure_logging(Settings(environment="development", log_level="INFO"))
    log = structlog.get_logger("test_logger")

    log.info("something_happened", model="gpt-4o-mini")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]

    # Console-rendered output is not valid JSON — that's the point.
    try:
        json.loads(line)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert is_json is False
    assert "something_happened" in line
    assert "gpt-4o-mini" in line


def test_log_level_below_threshold_is_suppressed(capsys):
    configure_logging(Settings(environment="production", log_level="WARNING"))
    log = structlog.get_logger("test_logger")

    log.info("should_not_appear")
    log.warning("should_appear")

    captured = capsys.readouterr()
    assert "should_not_appear" not in captured.out
    assert "should_appear" in captured.out
