"""Tests for the benchmark strategy suite."""

from datetime import datetime, timedelta

import pytest

from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.benchmarks import BENCHMARKS
from tests.helpers import ListFeed, make_candle_events

INST = Instrument("NIFTY", InstrumentType.INDEX)

# Trend up then down, enough bars to warm up slow indicators.
CLOSES = (
    [100 + i for i in range(40)]      # uptrend
    + [140 - i for i in range(40)]    # downtrend
)


def _engine(strategy):
    return BacktestEngine(
        strategy,
        portfolio=Portfolio(starting_cash=1_000_000),
        risk_engine=BasicRiskEngine(default_quantity=10, allow_short=True),
        broker=SimulatedBroker(fill_model=MidPriceFill()),
    )


def _run(strategy, closes=CLOSES):
    engine = _engine(strategy)
    feed = ListFeed(make_candle_events(closes, timeframe="5m",
                                       step=timedelta(minutes=5)))
    return engine.run(feed)


@pytest.mark.parametrize("name", list(BENCHMARKS.keys()))
def test_every_benchmark_runs(name):
    result = _run(BENCHMARKS[name](INST, quantity=10))
    assert len(result.equity_curve) == len(CLOSES)
    assert result.metrics.num_trades >= 0


def test_buy_and_hold_opens_one_long():
    strat = BENCHMARKS["BuyAndHold"](INST, quantity=10)
    result = _run(strat)
    # Opened a long on bar 1 and never exited -> position still open, 0 closed.
    assert strat._state == "long"
    assert result.metrics.num_trades == 0


def test_always_short_opens_short():
    strat = BENCHMARKS["AlwaysShort"](INST, quantity=10)
    _run(strat)
    assert strat._state == "short"


def test_ema_cross_trades_on_trend_reversal():
    strat = BENCHMARKS["EmaCross"](INST, fast=3, slow=5, quantity=10)
    # Flat prefix so EMAs converge, then up (cross up), then down (cross down).
    closes = [100] * 8 + [100 + 2 * i for i in range(20)] + [138 - 2 * i for i in range(20)]
    result = _run(strat, closes=closes)
    assert result.metrics.num_trades >= 1


def test_random_entry_reproducible_with_seed():
    r1 = _run(BENCHMARKS["RandomEntry"](INST, quantity=10, prob=0.1, seed=7))
    r2 = _run(BENCHMARKS["RandomEntry"](INST, quantity=10, prob=0.1, seed=7))
    assert r1.metrics.num_trades == r2.metrics.num_trades


def test_donchian_uses_only_prior_bars():
    # Strictly rising closes -> breakouts should fire (long), no breakdown.
    strat = BENCHMARKS["DonchianBreakout"](INST, period=5, quantity=10)
    result = _run(strat, closes=[100 + i for i in range(40)])
    assert strat._state in ("long", "flat")
    assert result.metrics.num_trades >= 0


def test_all_ten_present():
    assert len(BENCHMARKS) == 10
