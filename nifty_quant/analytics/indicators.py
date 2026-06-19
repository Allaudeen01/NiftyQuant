"""Technical indicators implemented in pure pandas/NumPy.

Design rules:
- Every function takes a pandas Series/DataFrame and returns a Series/DataFrame
  aligned to the input index (NaN for the warm-up period). No look-ahead.
- No global state, no I/O, no plotting. Easy to unit-test deterministically.
- These can later be swapped for TA-Lib/vectorbt behind the same signatures if
  we need vectorised speed, but pandas keeps install friction near zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    _check_period(period)
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (standard ``adjust=False`` recursion)."""
    _check_period(period)
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, range 0-100.

    Uses Wilder smoothing (an EMA with alpha = 1/period) of average gains and
    losses, matching the canonical definition used by most charting platforms.
    """
    _check_period(period)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 the asset only rose -> RSI 100. When both are 0 (flat)
    # the value is undefined; leave it as produced (NaN/100) rather than guess.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    out.iloc[:period] = np.nan
    return out


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns a DataFrame with columns ``macd``, ``signal``, ``hist``.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range from an OHLC DataFrame (needs high, low, close columns)."""
    _require_columns(df, ("high", "low", "close"))
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    _check_period(period)
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower.

    Returns columns ``mid``, ``upper``, ``lower``, ``bandwidth``.
    """
    _check_period(period)
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth}
    )


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume-Weighted Average Price over the frame.

    Uses the typical price (H+L+C)/3. For intraday VWAP, slice the frame to a
    single session before calling so the cumulative sum resets per day.
    """
    _require_columns(df, ("high", "low", "close", "volume"))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum()
    cum_pv = (typical * df["volume"]).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """SuperTrend indicator.

    Returns columns ``supertrend`` (the trailing line) and ``direction``
    (1 = uptrend / price above line, -1 = downtrend).
    """
    _require_columns(df, ("high", "low", "close"))
    _check_period(period)

    hl2 = (df["high"] + df["low"]) / 2.0
    atr_series = atr(df, period)
    upper_basic = hl2 + multiplier * atr_series
    lower_basic = hl2 - multiplier * atr_series

    close = df["close"].to_numpy()
    ub = upper_basic.to_numpy()
    lb = lower_basic.to_numpy()
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = ub[i]
            final_lower[i] = lb[i]
            trend[i] = 1.0 if close[i] >= final_lower[i] else -1.0
            continue

        # Carry the band tighter unless price breaks through.
        final_upper[i] = (
            ub[i]
            if (ub[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lb[i]
            if (lb[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        prev_trend = trend[i - 1]
        if prev_trend == 1.0:
            trend[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > final_upper[i] else -1.0

    line = np.where(trend == 1.0, final_lower, final_upper)
    return pd.DataFrame(
        {"supertrend": line, "direction": trend}, index=df.index
    )


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI.

    Returns columns ``adx``, ``plus_di``, ``minus_di``.
    """
    _require_columns(df, ("high", "low", "close"))
    _check_period(period)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_s
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        / atr_s
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame(
        {"adx": adx_s, "plus_di": plus_di, "minus_di": minus_di}
    )


# --- helpers ---------------------------------------------------------------


def _check_period(period: int) -> None:
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
