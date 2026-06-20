"""Tests for session metadata and the research journal."""

from datetime import date, datetime

import json
import pytest

from nifty_quant.data.session import (
    load_event_calendar,
    monthly_expiry_set,
    session_metadata,
)
from nifty_quant.research.journal import ResearchJournal


# --- session metadata ------------------------------------------------------


def test_session_metadata_fields():
    ts = datetime(2026, 6, 23, 10, 15)  # 1 hour after open
    expiry = date(2026, 6, 23)
    meta = session_metadata(ts, expiry, monthly_expiries={date(2026, 6, 30)})
    assert meta["session_id"] == "2026-06-23"
    assert meta["minutes_since_open"] == 60
    assert meta["minutes_until_close"] == 315  # 15:30 - 10:15
    assert meta["days_to_expiry"] == 0
    assert meta["is_expiry_day"] is True
    assert meta["is_monthly_expiry"] is False  # 23rd is weekly, 30th is monthly


def test_session_metadata_events():
    ts = datetime(2025, 2, 1, 11, 0)
    expiry = date(2025, 2, 27)
    events = {date(2025, 2, 1): ["Budget"]}
    meta = session_metadata(ts, expiry, events=events)
    assert meta["is_event_day"] is True
    assert meta["events"] == ["Budget"]
    assert meta["days_to_expiry"] == 26


def test_session_metadata_clamps_outside_session():
    ts = datetime(2026, 6, 23, 16, 30)  # after close
    meta = session_metadata(ts, date(2026, 6, 25))
    assert meta["minutes_until_close"] == 0
    assert meta["minutes_since_open"] > 0


def test_monthly_expiry_set():
    expiries = [date(2026, 6, 23), date(2026, 6, 30), date(2026, 7, 7),
                date(2026, 7, 28)]
    monthly = monthly_expiry_set(expiries)
    assert monthly == {date(2026, 6, 30), date(2026, 7, 28)}


def test_load_event_calendar(tmp_path):
    p = tmp_path / "events.json"
    p.write_text(json.dumps({"2025-02-01": ["Budget"], "bad": ["x"]}),
                 encoding="utf-8")
    cal = load_event_calendar(p)
    assert cal[date(2025, 2, 1)] == ["Budget"]
    assert date(2025, 2, 1) in cal and len(cal) == 1  # "bad" key skipped


def test_load_event_calendar_missing(tmp_path):
    assert load_event_calendar(tmp_path / "nope.json") == {}


# --- research journal ------------------------------------------------------


def test_journal_add_and_list(tmp_path):
    j = ResearchJournal(tmp_path / "rj.jsonl")
    h1 = j.add("EMA works only in low VIX", tags=["ema", "vix"])
    h2 = j.add("PCR>1.2 precedes range expansion", status="testing")
    assert h1.id == 1 and h2.id == 2
    assert len(j.list()) == 2
    assert len(j.list(status="testing")) == 1


def test_journal_update_status(tmp_path):
    j = ResearchJournal(tmp_path / "rj.jsonl")
    h = j.add("EMA + Time beats buy&hold")
    j.update(h.id, status="rejected", confidence=0.91,
             reason="Edge only in 2026 H1; failed year-CV.")
    updated = j.list()[0]
    assert updated.status == "rejected"
    assert updated.confidence == pytest.approx(0.91)
    assert "year-CV" in updated.reason


def test_journal_invalid_status(tmp_path):
    j = ResearchJournal(tmp_path / "rj.jsonl")
    with pytest.raises(ValueError):
        j.add("bad", status="not_a_status")


def test_journal_update_missing_id(tmp_path):
    j = ResearchJournal(tmp_path / "rj.jsonl")
    with pytest.raises(KeyError):
        j.update(999, status="rejected")
