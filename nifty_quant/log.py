"""Structured JSON logging.

A single place to obtain loggers that emit one JSON object per line. Structured
logs make later analysis (grep/jq, loading into a DataFrame, shipping to a log
store) trivial compared to free-form text.

Usage
-----
    from nifty_quant.log import get_logger
    log = get_logger("strategy")
    log.event("signal_generated", symbol="NIFTY", score=84, confidence=0.78)

``event(name, **fields)`` is the primary entry point: it logs at INFO with a
machine-readable ``event`` name plus arbitrary structured fields. Standard
``log.info("...")`` etc. still work and are rendered as JSON too.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False

# Attributes present on every LogRecord that we do NOT want to duplicate into
# the JSON "fields" section.
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "event", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        # Promote a structured event name if one was attached.
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        # Merge any extra structured fields the caller attached.
        extras = getattr(record, "fields", None)
        if isinstance(extras, dict):
            payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class StructuredLogger(logging.LoggerAdapter):
    """Adds an ``event(name, **fields)`` convenience method."""

    def event(self, name: str, level: int = logging.INFO, **fields: Any) -> None:
        self.log(level, name, extra={"event": name, "fields": fields})

    def process(self, msg, kwargs):
        return msg, kwargs


def configure(level: int = logging.INFO, stream=None) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    global _CONFIGURED
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    # Replace existing handlers so we don't double-log if reconfigured.
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> StructuredLogger:
    """Return a structured logger for ``name`` (configuring on first use)."""
    if not _CONFIGURED:
        configure()
    return StructuredLogger(logging.getLogger(name), {})


def add_file_logging(path: str | Path, level: int = logging.INFO) -> None:
    """Attach a JSON file handler to the root logger (in addition to console).

    Idempotent: re-calling with the same path does not add a duplicate handler.
    Used by long-running unattended jobs so failures are auditable after the fact.
    """
    if not _CONFIGURED:
        configure(level)
    target = os.path.abspath(str(path))
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target:
            return  # already attached
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(target, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)
