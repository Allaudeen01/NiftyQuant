"""Tests for incremental indicators, entry filters, and the filtered EMA."""

from datetime import datetime, timedelta

import pytest

from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.filters import (
    AdxFilter,
    AtrPercentileFilter,
    FilterContext,
    TimeWindowFilter,
)
from nifty_quant.backtest.incremental import (
    IncrementalAdx,
    IncrementalAtr,
    IncrementalEma,
    RollingPercentile,
)
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.filtered_ema import FilteredEmaStrategy
from tests.helpers import ListFeed, make_candle_events

INST = Instrument("NIFTY", InstrumentType.INDEX)


# --- incremental indicators ------------------------------------------------


def test_incremental_ema_tracks_constant():
    ema = IncrementalEma(5)
    for _ in range(20):
        v = ema.update(100.0)
    assert v == pytest.approx(100.0)
    assert ema.ready


def test_incremental_atr_positive():
    atr = IncrementalAtr(14)
    v = None
    for i in range(30):
        v = atr.update(101 + i, 99 + i, 100 + i)
    assert v > 0 and atr.ready


def test_incremental_adx_returns_value_after_warmup():
    adx = IncrementalAdx(14)
    val = None
    for i in range(80):
        # Trending series -> ADX should be defined and in [0, 100].
        val = adx.update(101 + i, 99 + i, 100 + i)
    assert adx.ready
    assert val is not None and 0 <= val <= 100


def test_rolling_percentile():
    rp = RollingPercentile(window=5)
    for x in [1, 2, 3, 4, 5]:
        p = rp.update(x)
    assert p == pytest.approx(1.0)  # last value (5) is the max
    rp.update(0)  # new min
    assert rp.percentile() == pytest.approx(1 / 5)


# --- filters ---------------------------------------------------------------


def _ctx(**indicators):
    return FilterContext(timestamp=datetime(2025, 1, 1, 11, 0), indicators=indicators)


def test_adx_filter():
    f = AdxFilter(25)
    assert f.allows(_ctx(adx=30))
    assert not f.allows(_ctx(adx=20))
    assert not f.allows(_ctx(adx=None))  # unavailable -> blocked


def test_time_window_filter():
    f = TimeWindowFilter("10:00", "14:30")
    inside = FilterContext(datetime(2025, 1, 1, 11, 0))
    before = FilterContext(datetime(2025, 1, 1, 9, 30))
    after = FilterContext(datetime(2025, 1, 1, 15, 0))
    assert f.allows(inside)
    assert not f.allows(before)
    assert not f.allows(after)


def test_atr_percentile_filter():
    f = AtrPercentileFilter(0.5)
    assert f.allows(_ctx(atr_pct=0.7))
    assert not f.allows(_ctx(atr_pct=0.3))


# --- filtered strategy -----------------------------------------------------


def _engine(strategy):
    return BacktestEngine(
        strategy,
        portfolio=Portfolio(starting_cash=1_000_000),
        risk_engine=BasicRiskEngine(default_quantity=10),
        broker=SimulatedBroker(fill_model=MidPriceFill()),
    )


CLOSES = [100] * 10 + [100 + 2 * i for i in range(25)] + [150 - 2 * i for i in range(25)]


def test_filtered_ema_no_filters_trades():
    strat = FilteredEmaStrategy(INST, fast=3, slow=5, quantity=10)
    result = _engine(strat).run(ListFeed(make_candle_events(CLOSES, timeframe="5m")))
    assert result.metrics.num_trades >= 1


def test_impossible_filter_blocks_all_entries():
    # ADX can't exceed 100, so this never allows an entry -> no trades.
    strat = FilteredEmaStrategy(
        INST, fast=3, slow=5, quantity=10, filters=[AdxFilter(999)]
    )
    result = _engine(strat).run(ListFeed(make_candle_events(CLOSES, timeframe="5m")))
    assert result.metrics.num_trades == 0


def test_filter_reduces_or_equals_trade_count():
    base = FilteredEmaStrategy(INST, fast=3, slow=5, quantity=10)
    filtered = FilteredEmaStrategy(
        INST, fast=3, slow=5, quantity=10, filters=[AdxFilter(25)]
    )
    events = make_candle_events(CLOSES, timeframe="5m")
    base_trades = _engine(base).run(ListFeed(events)).metrics.num_trades
    filt_trades = _engine(filtered).run(ListFeed(events)).metrics.num_trades
    assert filt_trades <= base_trades
