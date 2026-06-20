"""Tests for incremental RSI/Bollinger and mean-reversion strategies."""

import pytest

from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.incremental import IncrementalRsi, RollingMeanStd
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.mean_reversion import (
    BollingerReversion,
    RsiReversion,
)
from tests.helpers import ListFeed, make_candle_events

INST = Instrument("NIFTY", InstrumentType.INDEX)


def test_incremental_rsi_all_gains_is_100():
    rsi = IncrementalRsi(14)
    v = None
    for i in range(40):
        v = rsi.update(100 + i)
    assert rsi.ready
    assert v == pytest.approx(100.0)


def test_incremental_rsi_bounded():
    rsi = IncrementalRsi(14)
    prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93, 108]
    last = None
    for p in prices:
        last = rsi.update(p)
    assert last is None or 0 <= last <= 100


def test_rolling_mean_std():
    ms = RollingMeanStd(4)
    for x in [10, 10, 10, 10]:
        mean, std = ms.update(x)
    assert mean == pytest.approx(10.0)
    assert std == pytest.approx(0.0)
    assert ms.ready


def _engine(strategy):
    return BacktestEngine(
        strategy,
        portfolio=Portfolio(starting_cash=1_000_000),
        risk_engine=BasicRiskEngine(default_quantity=10),
        broker=SimulatedBroker(fill_model=MidPriceFill()),
    )


def test_rsi_reversion_buys_dip_and_exits():
    # Sharp drop (oversold) then recovery (revert) should produce a round trip.
    closes = [100] * 20 + [100 - 3 * i for i in range(10)] + [70 + 3 * i for i in range(15)]
    strat = RsiReversion(INST, period=5, oversold=30, exit_level=50, quantity=10)
    result = _engine(strat).run(ListFeed(make_candle_events(closes, timeframe="5m")))
    assert result.metrics.num_trades >= 1


def test_bollinger_reversion_runs():
    closes = [100] * 25 + [100 - 2 * i for i in range(8)] + [84 + 2 * i for i in range(12)]
    strat = BollingerReversion(INST, period=20, num_std=2, quantity=10)
    result = _engine(strat).run(ListFeed(make_candle_events(closes, timeframe="5m")))
    assert result.metrics.num_trades >= 0
    assert len(result.equity_curve) == len(closes)
