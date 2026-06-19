"""Performance analytics.

Computes the statistics that actually distinguish an edge from noise, from two
inputs:
- an equity curve (pandas Series indexed by timestamp), and
- the list of closed :class:`~nifty_quant.backtest.portfolio.Trade` records.

Annualised ratios (Sharpe/Sortino/Calmar) resample the equity curve to daily
closes so the numbers are comparable regardless of event frequency. With too
little data they return NaN rather than a misleading value.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nifty_quant.backtest.portfolio import Trade

_TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    profit_factor: float
    expectancy: float
    win_rate: float
    num_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_hold_seconds: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    equity: pd.Series,
    trades: list[Trade],
    *,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Build a :class:`PerformanceMetrics` from an equity curve and trades."""
    equity = equity.sort_index()
    start_val = float(equity.iloc[0]) if len(equity) else 0.0
    end_val = float(equity.iloc[-1]) if len(equity) else 0.0

    total_return = (end_val / start_val - 1.0) if start_val > 0 else 0.0

    daily = _to_daily_returns(equity)
    sharpe = _sharpe(daily, risk_free_rate)
    sortino = _sortino(daily, risk_free_rate)
    max_dd = _max_drawdown(equity)
    cagr = _cagr(equity)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0
        else (float("inf") if gross_profit > 0 else float("nan"))
    )
    num_trades = len(trades)
    win_rate = (wins.size / num_trades) if num_trades else float("nan")
    expectancy = float(pnls.mean()) if num_trades else float("nan")
    avg_hold = (
        float(np.mean([t.hold_seconds for t in trades])) if num_trades
        else float("nan")
    )

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        expectancy=expectancy,
        win_rate=win_rate,
        num_trades=num_trades,
        avg_win=float(wins.mean()) if wins.size else 0.0,
        avg_loss=float(losses.mean()) if losses.size else 0.0,
        largest_win=float(wins.max()) if wins.size else 0.0,
        largest_loss=float(losses.min()) if losses.size else 0.0,
        avg_hold_seconds=avg_hold,
    )


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each point."""
    equity = equity.sort_index()
    running_max = equity.cummax()
    return equity / running_max - 1.0


# --- internals -------------------------------------------------------------


def _to_daily_returns(equity: pd.Series) -> pd.Series:
    if len(equity) < 2 or not isinstance(equity.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def _sharpe(daily_returns: pd.Series, rf: float) -> float:
    if daily_returns.size < 2:
        return float("nan")
    excess = daily_returns - rf / _TRADING_DAYS
    std = excess.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(excess.mean() / std * math.sqrt(_TRADING_DAYS))


def _sortino(daily_returns: pd.Series, rf: float) -> float:
    if daily_returns.size < 2:
        return float("nan")
    excess = daily_returns - rf / _TRADING_DAYS
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if excess.mean() > 0 else float("nan")
    dd = math.sqrt(float((downside**2).mean()))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * math.sqrt(_TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    dd = drawdown_curve(equity)
    return float(dd.min())


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or not isinstance(equity.index, pd.DatetimeIndex):
        return float("nan")
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return float("nan")
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    if days <= 0:
        return float("nan")
    years = days / 365.25
    if years <= 0:
        return float("nan")
    return float((end_val / start_val) ** (1.0 / years) - 1.0)
