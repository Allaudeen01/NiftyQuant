"""Tests for structured JSON logging."""

import io
import json
import logging

from nifty_quant.log import configure, get_logger


def _capture():
    stream = io.StringIO()
    configure(level=logging.DEBUG, stream=stream)
    return stream


def test_event_emits_valid_json_with_fields():
    stream = _capture()
    log = get_logger("strategy")
    log.event("signal_generated", symbol="NIFTY", score=84, confidence=0.78)
    line = stream.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "signal_generated"
    assert payload["module"] == "strategy"
    assert payload["symbol"] == "NIFTY"
    assert payload["score"] == 84
    assert payload["confidence"] == 0.78
    assert payload["level"] == "INFO"
    assert "timestamp" in payload


def test_plain_info_is_json():
    stream = _capture()
    log = get_logger("misc")
    log.info("hello world")
    payload = json.loads(stream.getvalue().strip())
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"


def test_one_json_object_per_line():
    stream = _capture()
    log = get_logger("multi")
    log.event("a", x=1)
    log.event("b", y=2)
    lines = [l for l in stream.getvalue().splitlines() if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "a"
    assert json.loads(lines[1])["event"] == "b"
