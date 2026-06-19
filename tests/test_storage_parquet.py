"""Tests for the Parquet storage backend."""

from datetime import date, datetime

import pytest

from nifty_quant.data.models import (
    Candle,
    OHLCVSeries,
    OptionChain,
    OptionQuote,
    OptionType,
)
from nifty_quant.data.storage.parquet import ParquetStorage


def _series(symbol="NIFTY", timeframe="5m", n=6, day=date(2025, 1, 2)):
    candles = []
    for i in range(n):
        ts = datetime(day.year, day.month, day.day, 9, 15 + i * 5)
        candles.append(
            Candle(ts, open=100 + i, high=101 + i, low=99 + i, close=100.5 + i, volume=1000 + i)
        )
    return OHLCVSeries(symbol=symbol, timeframe=timeframe, candles=candles)


def _chain(ts, spot=25050.0):
    expiry = date(2025, 1, 30)
    quotes = []
    for strike in (25000, 25100):
        quotes.append(
            OptionQuote(
                strike=strike,
                option_type=OptionType.CALL,
                expiry=expiry,
                last_price=120.0,
                open_interest=1000,
                volume=500,
                implied_volatility=0.13,
            )
        )
        quotes.append(
            OptionQuote(
                strike=strike,
                option_type=OptionType.PUT,
                expiry=expiry,
                last_price=110.0,
                open_interest=1100,
                volume=600,
                implied_volatility=None,  # exercise NaN round-trip
            )
        )
    return OptionChain("NIFTY", spot, expiry, ts, tuple(quotes))


def test_candles_roundtrip(tmp_path):
    store = ParquetStorage(tmp_path)
    series = _series()
    assert store.write_candles(series) == 6
    out = store.read_candles(
        "NIFTY", "5m", datetime(2025, 1, 2, 0, 0), datetime(2025, 1, 2, 23, 59)
    )
    assert len(out) == 6
    df = out.to_frame()
    assert df["close"].iloc[0] == pytest.approx(100.5)
    assert df["close"].iloc[-1] == pytest.approx(105.5)


def test_candles_idempotent_write(tmp_path):
    store = ParquetStorage(tmp_path)
    series = _series()
    store.write_candles(series)
    store.write_candles(series)  # write twice
    out = store.read_candles(
        "NIFTY", "5m", datetime(2025, 1, 2, 0, 0), datetime(2025, 1, 2, 23, 59)
    )
    assert len(out) == 6  # no duplicates


def test_candles_range_filter(tmp_path):
    store = ParquetStorage(tmp_path)
    store.write_candles(_series(n=6))
    out = store.read_candles(
        "NIFTY", "5m",
        datetime(2025, 1, 2, 9, 20),
        datetime(2025, 1, 2, 9, 30),
    )
    # 9:20, 9:25, 9:30 -> 3 candles
    assert len(out) == 3


def test_candles_partitioned_by_day(tmp_path):
    store = ParquetStorage(tmp_path)
    store.write_candles(_series(n=3, day=date(2025, 1, 2)))
    store.write_candles(_series(n=3, day=date(2025, 1, 3)))
    p1 = tmp_path / "candles" / "5m" / "2025" / "NIFTY_2025-01-02.parquet"
    p2 = tmp_path / "candles" / "5m" / "2025" / "NIFTY_2025-01-03.parquet"
    assert p1.exists() and p2.exists()
    out = store.read_candles(
        "NIFTY", "5m", datetime(2025, 1, 2), datetime(2025, 1, 3, 23, 59)
    )
    assert len(out) == 6


def test_empty_read_returns_empty_series(tmp_path):
    store = ParquetStorage(tmp_path)
    out = store.read_candles(
        "NIFTY", "5m", datetime(2025, 1, 2), datetime(2025, 1, 2, 23, 59)
    )
    assert len(out) == 0


def test_option_chain_roundtrip(tmp_path):
    store = ParquetStorage(tmp_path)
    ts = datetime(2025, 1, 2, 10, 0)
    assert store.write_option_chain(_chain(ts)) == 4
    chains = store.read_option_chains(
        "NIFTY", datetime(2025, 1, 2), datetime(2025, 1, 2, 23, 59)
    )
    assert len(chains) == 1
    chain = chains[0]
    assert chain.spot == pytest.approx(25050.0)
    assert len(chain.quotes) == 4
    # IV None survives the round-trip for puts.
    put = next(q for q in chain.puts() if q.strike == 25000)
    assert put.implied_volatility is None
    call = next(q for q in chain.calls() if q.strike == 25000)
    assert call.implied_volatility == pytest.approx(0.13)


def test_option_chain_multiple_snapshots_ordered(tmp_path):
    store = ParquetStorage(tmp_path)
    store.write_option_chain(_chain(datetime(2025, 1, 2, 10, 5)))
    store.write_option_chain(_chain(datetime(2025, 1, 2, 10, 0)))
    chains = store.read_option_chains(
        "NIFTY", datetime(2025, 1, 2), datetime(2025, 1, 2, 23, 59)
    )
    assert len(chains) == 2
    assert chains[0].timestamp < chains[1].timestamp  # sorted ascending


def test_option_chain_idempotent(tmp_path):
    store = ParquetStorage(tmp_path)
    ts = datetime(2025, 1, 2, 10, 0)
    store.write_option_chain(_chain(ts))
    store.write_option_chain(_chain(ts))  # same snapshot again
    chains = store.read_option_chains(
        "NIFTY", datetime(2025, 1, 2), datetime(2025, 1, 2, 23, 59)
    )
    assert len(chains) == 1
    assert len(chains[0].quotes) == 4  # not 8
