"""Tests for portfolio accounting (longs, shorts, reversals, PnL, cash)."""

from datetime import datetime

import pytest

from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.intents import Fill
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.data.providers.base import OrderSide

INST = Instrument("NIFTY", InstrumentType.INDEX)
T0 = datetime(2025, 1, 2, 9, 15)
T1 = datetime(2025, 1, 2, 9, 20)


def _fill(side, qty, price, ts=T0, inst=INST, fees=0.0):
    return Fill(ts, inst, side, qty, price, fees)


def test_long_open_then_close_realizes_pnl():
    pf = Portfolio(starting_cash=1_000_000)
    pf.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    assert pf.cash == pytest.approx(999_000)
    events = pf.apply_fill(_fill(OrderSide.SELL, 10, 110.0, ts=T1))
    assert pf.realized_pnl == pytest.approx(100.0)
    assert pf.cash == pytest.approx(1_000_100)
    assert not pf.positions  # flat
    assert any(name == "position_closed" for name, _ in events)
    assert len(pf.trades) == 1
    assert pf.trades[0].direction == "LONG"
    assert pf.trades[0].pnl == pytest.approx(100.0)


def test_short_open_then_cover():
    pf = Portfolio(starting_cash=1_000_000)
    pf.apply_fill(_fill(OrderSide.SELL, 5, 100.0))
    assert pf.cash == pytest.approx(1_000_500)  # proceeds from short
    pf.apply_fill(_fill(OrderSide.BUY, 5, 90.0, ts=T1))
    assert pf.realized_pnl == pytest.approx(50.0)  # (100-90)*5
    assert not pf.positions


def test_scale_in_averages_price():
    pf = Portfolio()
    pf.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    pf.apply_fill(_fill(OrderSide.BUY, 10, 120.0))
    pos = pf.positions[INST.key]
    assert pos.quantity == 20
    assert pos.avg_price == pytest.approx(110.0)


def test_reversal_closes_and_opens_opposite():
    pf = Portfolio(starting_cash=1_000_000)
    pf.apply_fill(_fill(OrderSide.BUY, 5, 100.0))
    events = pf.apply_fill(_fill(OrderSide.SELL, 8, 110.0, ts=T1))
    # Closed 5 long @ +50, opened 3 short @ 110.
    assert pf.realized_pnl == pytest.approx(50.0)
    pos = pf.positions[INST.key]
    assert pos.quantity == -3
    assert pos.avg_price == pytest.approx(110.0)
    names = [n for n, _ in events]
    assert "position_closed" in names and "position_opened" in names


def test_partial_close_keeps_avg_price():
    pf = Portfolio()
    pf.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    pf.apply_fill(_fill(OrderSide.SELL, 4, 130.0, ts=T1))
    pos = pf.positions[INST.key]
    assert pos.quantity == 6
    assert pos.avg_price == pytest.approx(100.0)  # unchanged on reduction
    assert pf.realized_pnl == pytest.approx((130 - 100) * 4)


def test_equity_marks_open_position():
    pf = Portfolio(starting_cash=1_000_000)
    pf.apply_fill(_fill(OrderSide.BUY, 10, 100.0))
    eq = pf.equity({INST.key: 105.0})
    # cash 999000 + 10*105 = 1,000,050
    assert eq == pytest.approx(1_000_050)


def test_fees_reduce_cash():
    pf = Portfolio(starting_cash=1_000_000)
    pf.apply_fill(_fill(OrderSide.BUY, 1, 100.0, fees=20.0))
    assert pf.cash == pytest.approx(1_000_000 - 100 - 20)


def test_multiplier_scales_pnl():
    inst = Instrument("NIFTY", InstrumentType.INDEX, multiplier=75.0)
    pf = Portfolio()
    pf.apply_fill(Fill(T0, inst, OrderSide.BUY, 1, 100.0))
    pf.apply_fill(Fill(T1, inst, OrderSide.SELL, 1, 110.0))
    assert pf.realized_pnl == pytest.approx(10 * 75.0)
