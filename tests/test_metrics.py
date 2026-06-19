"""Tests for performance metrics."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_quant.backtest.metrics import compute_metrics, drawdown_curve
from nifty_quant.backtest.portfolio import Trade


def _equity(values, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def _trade(pnl, hold_s=3600):
    t0 = datetime(2025, 1, 1, 9, 15)
    t1 = t0 + timedelta(seconds=hold_s)
    return Trade("NIFTY", "LONG", 1, 100.0, 100.0 + pnl, t0, t1, pnl)


def test_total_return():
    eq = _equity([100, 110, 105, 121])
    m = compute_metrics(eq, [])
    assert m.total_return == pytest.approx(0.21)


def test_max_drawdown():
    eq = _equity([100, 120, 90, 150])
    dd = drawdown_curve(eq)
    # Worst point: 90 vs peak 120 -> -0.25
    assert dd.min() == pytest.approx(-0.25)
    m = compute_metrics(eq, [])
    assert m.max_drawdown == pytest.approx(-0.25)


def test_trade_statistics():
    trades = [_trade(100), _trade(-50), _trade(200), _trade(-50)]
    eq = _equity([1000, 1100, 1050, 1250, 1200])
    m = compute_metrics(eq, trades)
    assert m.num_trades == 4
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(300 / 100)  # wins 300, losses 100
    assert m.expectancy == pytest.approx((100 - 50 + 200 - 50) / 4)
    assert m.largest_win == pytest.approx(200)
    assert m.largest_loss == pytest.approx(-50)


def test_sharpe_positive_for_uptrend():
    # Steady gains -> positive Sharpe.
    eq = _equity([100 * (1.01**i) for i in range(30)])
    m = compute_metrics(eq, [])
    assert m.sharpe > 0


def test_metrics_handle_empty_trades():
    eq = _equity([100, 101, 102])
    m = compute_metrics(eq, [])
    assert m.num_trades == 0
    assert np.isnan(m.win_rate)


def test_profit_factor_all_wins_is_inf():
    trades = [_trade(100), _trade(50)]
    eq = _equity([1000, 1100, 1150])
    m = compute_metrics(eq, trades)
    assert m.profit_factor == float("inf")
