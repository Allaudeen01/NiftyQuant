"""Tests for the data-quality validator."""

from datetime import date, datetime, timedelta

import pytest

from nifty_quant.data.models import Candle, OHLCVSeries
from nifty_quant.data.quality import validate_ohlcv, CRITICAL


def _bar(ts, base=100.0, up=0.5):
    o = base
    c = base + up
    hi = max(o, c) + 0.5
    lo = min(o, c) - 0.5
    return Candle(ts, o, hi, lo, c, 0.0)


def _intraday_day(day: date, bars: int = 75):
    start = datetime(day.year, day.month, day.day, 9, 15)
    return [_bar(start + timedelta(minutes=5 * i), base=100 + i * 0.1)
            for i in range(bars)]


def _weekdays(start: date, n: int):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _good_series(days=5, bars=75, timeframe="5m"):
    candles = []
    for day in _weekdays(date(2025, 1, 6), days):  # 2025-01-06 is a Monday
        candles += _intraday_day(day, bars)
    return OHLCVSeries("NIFTY", timeframe, candles)


def test_good_intraday_passes():
    report = validate_ohlcv(
        _good_series(),
        expected_start=date(2025, 1, 6),
        expected_end=date(2025, 1, 10),
    )
    assert report.passed
    assert report.failures() == []


def test_empty_series_fails_critical():
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", []))
    assert not report.passed
    assert report.checks[0].name == "non_empty"


def test_duplicate_timestamp_fails_critical():
    ts = datetime(2025, 1, 6, 9, 15)
    candles = [_bar(ts), _bar(ts, base=101)]  # same timestamp twice
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", candles))
    check = next(c for c in report.checks if c.name == "timestamps_sorted_unique")
    assert not check.passed and check.severity == CRITICAL
    assert not report.passed


def test_ohlc_integrity_violation():
    # high below open: passes Candle's own (high>=low, positive) check,
    # but the quality validator must catch high < open.
    bad = Candle(datetime(2025, 1, 6, 9, 15), open=105, high=101, low=99, close=100)
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", [bad]))
    check = next(c for c in report.checks if c.name == "ohlc_integrity")
    assert not check.passed
    assert not report.passed


def test_weekend_bar_warns_but_not_critical():
    candles = _intraday_day(date(2025, 1, 6))  # Monday
    # Append a Saturday bar.
    sat = datetime(2025, 1, 11, 9, 15)
    candles.append(_bar(sat))
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", candles))
    weekend = next(c for c in report.checks if c.name == "no_weekend_bars")
    assert not weekend.passed
    assert report.passed  # weekend is a WARNING, not CRITICAL


def test_incomplete_session_warns():
    candles = _intraday_day(date(2025, 1, 6), bars=75)
    candles += _intraday_day(date(2025, 1, 7), bars=10)  # short day
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", candles))
    check = next(c for c in report.checks if c.name == "intraday_completeness")
    assert not check.passed
    assert check.stats["incomplete_sessions"] == 1


def test_outlier_detection():
    candles = _intraday_day(date(2025, 1, 6), bars=20)
    # Inject a doubling bar.
    last_ts = candles[-1].timestamp + timedelta(minutes=5)
    candles.append(_bar(last_ts, base=250.0))
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", candles))
    check = next(c for c in report.checks if c.name == "outliers")
    assert not check.passed
    assert check.stats["count"] >= 1


def test_within_session_violation():
    candles = _intraday_day(date(2025, 1, 6), bars=10)
    # A bar at 16:00 is outside the session.
    candles.append(_bar(datetime(2025, 1, 6, 16, 0)))
    report = validate_ohlcv(OHLCVSeries("NIFTY", "5m", candles))
    check = next(c for c in report.checks if c.name == "within_session")
    assert not check.passed


def test_daily_timeframe_one_bar_per_day():
    days = _weekdays(date(2025, 1, 6), 5)
    candles = [_bar(datetime(d.year, d.month, d.day, 9, 15), base=100 + i)
               for i, d in enumerate(days)]
    report = validate_ohlcv(OHLCVSeries("NIFTY", "1d", candles))
    assert report.passed
    check = next(c for c in report.checks if c.name == "daily_one_bar_per_day")
    assert check.passed


def test_report_text_and_dict():
    report = validate_ohlcv(_good_series())
    text = report.to_text()
    assert "DATA QUALITY REPORT" in text
    assert "Overall: PASS" in text
    d = report.as_dict()
    assert d["passed"] is True
    assert d["symbol"] == "NIFTY"
    assert "checks" in d
