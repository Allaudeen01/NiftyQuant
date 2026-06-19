"""Tests for the FeatureEngine and FeatureStore."""

from datetime import date, datetime

import math
import pytest

from nifty_quant.data.models import OptionChain, OptionQuote, OptionType
from nifty_quant.features.engine import FEATURE_VERSION, FeatureConfig, FeatureEngine
from nifty_quant.features.store import InMemoryFeatureStore, ParquetFeatureStore
from nifty_quant.features.vector import FeatureVector
from nifty_quant.feed.events import OptionChainEvent
from tests.helpers import make_candle_events


def test_feature_vector_is_ready_and_get():
    fv = FeatureVector(
        datetime(2025, 1, 1), "NIFTY", "v1",
        {"ema_20": 100.0, "rsi_14": float("nan")},
    )
    assert fv.is_ready("ema_20")
    assert not fv.is_ready("rsi_14")     # NaN -> not ready
    assert not fv.is_ready("missing")
    assert math.isnan(fv.get("missing"))


def test_engine_emits_expected_keys():
    engine = FeatureEngine(FeatureConfig(ema_periods=(5, 10), bb_period=5))
    events = make_candle_events([100 + i for i in range(30)])
    fv = None
    for e in events:
        fv = engine.on_candle(e)
    assert fv.version == FEATURE_VERSION
    for key in ("close", "ema_5", "ema_10", "rsi_14", "macd_hist",
                "atr_14", "adx_14", "bb_bandwidth"):
        assert key in fv.values


def test_engine_warmup_then_finite():
    engine = FeatureEngine(FeatureConfig(ema_periods=(5,), bb_period=5))
    events = make_candle_events([100 + i for i in range(40)])
    vectors = [engine.on_candle(e) for e in events]
    # Early vector: ema_5 may be NaN; late vector: finite.
    assert not math.isnan(vectors[-1]["ema_5"])
    assert not math.isnan(vectors[-1]["close"])


def _chain(ts):
    expiry = date(2025, 1, 30)
    spot = 25050.0
    quotes = []
    for strike in (25000, 25100):
        quotes.append(OptionQuote(strike, OptionType.CALL, expiry, 120.0,
                                  open_interest=1000, volume=500,
                                  implied_volatility=0.13))
        quotes.append(OptionQuote(strike, OptionType.PUT, expiry, 110.0,
                                  open_interest=1500, volume=600,
                                  implied_volatility=0.14))
    return OptionChain("NIFTY", spot, expiry, ts, tuple(quotes))


def test_option_features_merge_into_candle_vector():
    engine = FeatureEngine(FeatureConfig(ema_periods=(5,)))
    # Feed an option chain first so option features are available.
    engine.on_option_chain(OptionChainEvent(datetime(2025, 1, 1, 9, 0), _chain(datetime(2025, 1, 1, 9, 0))))
    events = make_candle_events([100 + i for i in range(10)])
    fv = None
    for e in events:
        fv = engine.on_candle(e)
    assert "pcr_oi" in fv.values
    assert "atm_iv" in fv.values
    assert fv["pcr_oi"] == pytest.approx(3000 / 2000)  # puts 1500*2 / calls 1000*2


def test_inmemory_store_roundtrip():
    store = InMemoryFeatureStore()
    fv1 = FeatureVector(datetime(2025, 1, 1, 9, 15), "NIFTY", "v1", {"ema_20": 100.0})
    fv2 = FeatureVector(datetime(2025, 1, 1, 9, 20), "NIFTY", "v1", {"ema_20": 101.0})
    store.put(fv1)
    store.put(fv2)
    got = store.get("NIFTY", "v1", datetime(2025, 1, 1), datetime(2025, 1, 2))
    assert len(got) == 2
    assert got[0].timestamp < got[1].timestamp
    df = store.to_frame("NIFTY", "v1")
    assert list(df["ema_20"]) == [100.0, 101.0]


def test_inmemory_store_version_isolation():
    store = InMemoryFeatureStore()
    store.put(FeatureVector(datetime(2025, 1, 1), "NIFTY", "v1", {"x": 1.0}))
    store.put(FeatureVector(datetime(2025, 1, 1), "NIFTY", "v2", {"x": 2.0}))
    v1 = store.get("NIFTY", "v1", datetime(2025, 1, 1), datetime(2025, 1, 2))
    assert len(v1) == 1 and v1[0]["x"] == 1.0


def test_parquet_store_roundtrip(tmp_path):
    store = ParquetFeatureStore(tmp_path)
    fv = FeatureVector(datetime(2025, 1, 1, 9, 15), "NIFTY", "v1",
                       {"ema_20": 100.0, "rsi_14": 55.0})
    store.put(fv)
    got = store.get("NIFTY", "v1", datetime(2025, 1, 1), datetime(2025, 1, 2))
    assert len(got) == 1
    assert got[0]["ema_20"] == pytest.approx(100.0)
    assert got[0]["rsi_14"] == pytest.approx(55.0)


def test_parquet_store_idempotent(tmp_path):
    store = ParquetFeatureStore(tmp_path)
    fv = FeatureVector(datetime(2025, 1, 1, 9, 15), "NIFTY", "v1", {"ema_20": 100.0})
    store.put(fv)
    store.put(fv)
    got = store.get("NIFTY", "v1", datetime(2025, 1, 1), datetime(2025, 1, 2))
    assert len(got) == 1
