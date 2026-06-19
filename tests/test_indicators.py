"""Tests for technical indicators.

Where possible we assert against known closed-form values rather than just
shape, so a regression in the math is caught.
"""

import numpy as np
import pandas as pd
import pytest

from nifty_quant.analytics import indicators as ind


def _series(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_sma_basic():
    s = _series([1, 2, 3, 4, 5])
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_warmup_and_value():
    s = _series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    out = ind.ema(s, 3)
    # First valid value at index period-1 equals the SMA seed of adjust=False
    # recursion started from the series; just assert monotonic & finite tail.
    assert out.dropna().is_monotonic_increasing
    assert np.isfinite(out.iloc[-1])


def test_rsi_all_gains_is_100():
    s = _series(list(range(1, 30)))  # strictly increasing
    out = ind.rsi(s, 14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_range_bounds():
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, 200))
    prices = np.abs(prices) + 1  # keep positive
    out = ind.rsi(_series(prices), 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_macd_columns_and_hist():
    s = _series(100 + np.cumsum(np.ones(60)))
    df = ind.macd(s)
    assert list(df.columns) == ["macd", "signal", "hist"]
    # hist must equal macd - signal exactly.
    valid = df.dropna()
    assert np.allclose(valid["hist"], valid["macd"] - valid["signal"])


def _ohlc(n=60, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 5
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    low = np.minimum(low, close)
    high = np.maximum(high, close)
    open_ = (high + low) / 2
    vol = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_true_range_nonnegative():
    df = _ohlc()
    tr = ind.true_range(df).dropna()
    assert (tr >= 0).all()


def test_atr_positive():
    df = _ohlc()
    a = ind.atr(df, 14).dropna()
    assert (a > 0).all()


def test_bollinger_band_ordering():
    df = _ohlc()
    bb = ind.bollinger_bands(df["close"], 20, 2.0).dropna()
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()


def test_vwap_within_price_range():
    df = _ohlc()
    v = ind.vwap(df).dropna()
    assert (v >= df["low"].min() - 1).all()
    assert (v <= df["high"].max() + 1).all()


def test_supertrend_direction_values():
    df = _ohlc(n=80)
    st = ind.supertrend(df, 10, 3.0).dropna()
    assert set(st["direction"].unique()).issubset({1.0, -1.0})


def test_adx_bounded():
    df = _ohlc(n=120)
    a = ind.adx(df, 14).dropna()
    assert (a["adx"] >= 0).all() and (a["adx"] <= 100).all()


def test_invalid_period_raises():
    s = _series([1, 2, 3])
    with pytest.raises(ValueError):
        ind.sma(s, 0)
    with pytest.raises(ValueError):
        ind.ema(s, -5)


def test_missing_columns_raises():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError):
        ind.atr(df, 2)
