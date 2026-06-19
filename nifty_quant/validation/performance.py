"""Rolling performance helpers for monitoring.

Lightweight functions that turn an equity curve / trade list into rolling
health signals. The :class:`~nifty_quant.validation.engine.ValidationEngine`
uses these plus the existing :func:`compute_metrics` for the aggregate view.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nifty_quant.backtest.portfolio import Trade

_TRADING_DAYS = 252


def rolling_sharpe(equity: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualised Sharpe from daily returns of the equity curve."""
    equity = equity.sort_index()
    if isinstance(equity.index, pd.DatetimeIndex):
        equity = equity.resample("1D").last().dropna()
    returns = equity.pct_change().dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std) * math.sqrt(_TRADING_DAYS)


def rolling_expectancy(trades: list[Trade], window: int = 20) -> pd.Series:
    """Rolling mean PnL per trade over the last ``window`` trades."""
    if not trades:
        return pd.Series(dtype=float)
    pnls = pd.Series([t.pnl for t in trades])
    return pnls.rolling(window).mean()


def window_summary(equity: pd.Series, trades: list[Trade]) -> dict:
    """Compact current-window stats used by the validation engine."""
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    summary = {
        "num_trades": len(trades),
        "win_rate": float(wins.size / len(trades)) if trades else float("nan"),
        "expectancy": float(pnls.mean()) if trades else float("nan"),
        "avg_hold_seconds": (
            float(np.mean([t.hold_seconds for t in trades])) if trades
            else float("nan")
        ),
    }
    return summary
