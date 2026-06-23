"""Tests for the live lens: descriptive metrics + sample-gated checks."""

from datetime import date, datetime

import math

from nifty_quant.data.models import OptionChain, OptionQuote, OptionType
from nifty_quant.research import live_lens as lens


def _chain(spot=25000.0, expiry=date(2026, 7, 2), ts=None):
    ts = ts or datetime(2026, 6, 25, 10, 0)
    quotes = []
    for k in (24800, 24900, 25000, 25100, 25200):
        for ot, px in ((OptionType.CALL, max(spot - k, 5) + 30),
                       (OptionType.PUT, max(k - spot, 5) + 30)):
            quotes.append(OptionQuote(
                strike=float(k), option_type=ot, expiry=expiry,
                last_price=px, bid=px - 1, ask=px + 1,
                volume=10, open_interest=1000 + k % 100,
            ))
    return OptionChain("NIFTY", spot, expiry, ts, tuple(quotes))


def test_atm_straddle_picks_nearest_strike():
    st = lens.atm_straddle(_chain(spot=25010))
    assert st is not None
    atm, call_mid, put_mid, straddle = st
    assert atm == 25000.0
    assert straddle == call_mid + put_mid > 0


def test_snapshot_metrics_fields():
    m = lens.snapshot_metrics(_chain(), india_vix=13.5)
    for key in ("pcr_oi", "max_pain", "atm_strike", "straddle",
                "implied_move_pct", "india_vix"):
        assert key in m
    assert m["india_vix"] == 13.5
    assert m["implied_move_pct"] > 0


def test_status_line_contains_core_fields():
    line = lens.status_line(_chain(spot=23900), india_vix=14.1, prev_vix=12.5)
    assert "NIFTY" in line and "VIX 14.1" in line and "(+1.60)" in line
    assert "PCR" in line and "max-pain" in line


def test_percentile_and_sample_size():
    ctx = lens.HistoricalContext([
        {"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0},
    ])
    pct, n = ctx.percentile("x", 3.5)
    assert n == 4
    assert pct == 75.0  # 3 of 4 below 3.5


def test_checks_accumulate_until_threshold():
    # Few rows -> every check must report not-ready (ACCUMULATING), never a result.
    rows = [{"date": date(2026, 6, 22), "is_expiry_day": False,
             "realized_move_pct": 0.3, "vix_implied_daily_pct": 0.8,
             "implied_move_open_pct": 0.5}]
    ctx = lens.HistoricalContext(rows)
    checks = ctx.checks()
    assert len(checks) >= 2
    assert all(not c.ready for c in checks)  # never claims a pattern on n=1
    for c in checks:
        assert c.n < c.need


def test_checks_ready_when_enough_data():
    rows = [{"date": date(2026, 6, d % 28 + 1), "is_expiry_day": False,
             "realized_move_pct": 0.4, "vix_implied_daily_pct": 0.9,
             "implied_move_open_pct": 0.6}
            for d in range(lens.MIN_VRP_DAYS + 2)]
    ctx = lens.HistoricalContext(rows)
    vrp = next(c for c in ctx.checks() if c.name.startswith("VRP"))
    assert vrp.ready is True
    assert vrp.n >= lens.MIN_VRP_DAYS
    assert "implied" in vrp.detail
