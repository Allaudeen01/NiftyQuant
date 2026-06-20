"""Tests for the Angel One provider mapping and chunking (offline, no SDK)."""

from datetime import date, datetime

import pytest

from nifty_quant.data.providers.angelone import (
    AngelOneProvider,
    LiveTradingDisabledError,
    _chunk_ranges,
    _clean_totp_secret,
    _map_candle,
)
from nifty_quant.data.providers.base import OrderRequest, OrderSide


class FakeSmartConnect:
    """Stand-in for SmartConnect returning Angel-shaped candle payloads."""

    def __init__(self):
        self.calls = []
        self.placed = []

    def getCandleData(self, params):
        self.calls.append(params)
        # Return two candles per call, dated within the requested window.
        d = params["fromdate"].split(" ")[0]
        return {
            "status": True,
            "data": [
                [f"{d}T09:15:00+05:30", 100.0, 101.0, 99.5, 100.5, 1000],
                [f"{d}T09:16:00+05:30", 100.5, 102.0, 100.0, 101.5, 1200],
            ],
        }

    def placeOrder(self, params):
        self.placed.append(params)
        return {"status": True, "data": {"orderid": "AO123"}}

    def cancelOrder(self, order_id, variety):
        return {"status": True, "data": {"orderid": order_id}}

    def orderBook(self):
        return {"status": True, "data": [{"orderid": "AO123", "status": "open"}]}


@pytest.fixture
def provider():
    return AngelOneProvider(FakeSmartConnect(), request_pause=0.0)


# --- pure helpers ----------------------------------------------------------


def test_chunk_ranges_splits_by_max_days():
    chunks = _chunk_ranges(date(2025, 1, 1), date(2025, 3, 31), max_days=30)
    assert len(chunks) == 3
    assert chunks[0] == (date(2025, 1, 1), date(2025, 1, 30))
    assert chunks[-1][1] == date(2025, 3, 31)
    # Chunks are contiguous and non-overlapping.
    assert chunks[1][0] == chunks[0][1].fromordinal(chunks[0][1].toordinal() + 1)


def test_chunk_ranges_single_when_short():
    chunks = _chunk_ranges(date(2025, 1, 1), date(2025, 1, 5), max_days=30)
    assert chunks == [(date(2025, 1, 1), date(2025, 1, 5))]


def test_chunk_ranges_empty_when_reversed():
    assert _chunk_ranges(date(2025, 2, 1), date(2025, 1, 1), 30) == []


def test_map_candle_strips_timezone():
    c = _map_candle(["2025-01-02T09:15:00+05:30", 100, 101, 99, 100.5, 500])
    assert c.timestamp == datetime(2025, 1, 2, 9, 15, 0)
    assert c.timestamp.tzinfo is None
    assert c.close == pytest.approx(100.5)


def test_clean_totp_secret_strips_spaces():
    assert _clean_totp_secret("  ABCD EFGH 2345  ") == "ABCDEFGH2345"
    assert _clean_totp_secret("") == ""


# --- provider --------------------------------------------------------------


def test_resolve_token_default_nifty(provider):
    assert provider.resolve_token("NIFTY") == "99926000"


def test_resolve_token_unknown_raises(provider):
    with pytest.raises(KeyError):
        provider.resolve_token("UNKNOWNSYM")


def test_get_ohlcv_maps_and_dedups(provider):
    # 90-day range with ONE_MINUTE (30-day cap) -> 3 chunks, 2 candles each,
    # but identical timestamps per chunk-day so dedup keeps unique ones.
    series = provider.get_ohlcv("NIFTY", "1m", date(2025, 1, 1), date(2025, 3, 31))
    client = provider._client
    assert len(client.calls) == 3            # chunked into 3 requests
    df = series.to_frame()
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # Candles are sorted ascending and de-duplicated by timestamp.
    assert list(df.index) == sorted(df.index)
    assert len(df.index) == len(set(df.index))


def test_get_ohlcv_rejects_bad_timeframe(provider):
    with pytest.raises(ValueError):
        provider.get_ohlcv("NIFTY", "2m", date(2025, 1, 1), date(2025, 1, 2))


def test_get_ohlcv_explicit_token(provider):
    provider.get_ohlcv("ANYTHING", "1d", date(2025, 1, 1), date(2025, 1, 2),
                       token="12345")
    assert provider._client.calls[0]["symboltoken"] == "12345"


def test_orders_blocked_when_live_disabled(provider):
    order = OrderRequest("NIFTY", OrderSide.BUY, 75)
    with pytest.raises(LiveTradingDisabledError):
        provider.place_order(order)
    with pytest.raises(LiveTradingDisabledError):
        provider.cancel_order("AO123")


def test_orders_work_when_enabled():
    client = FakeSmartConnect()
    prov = AngelOneProvider(client, live_trading_enabled=True, request_pause=0.0)
    result = prov.place_order(OrderRequest("NIFTY", OrderSide.BUY, 75))
    assert result.broker_order_id == "AO123"
    assert client.placed[0]["transactiontype"] == "BUY"


def test_orderbook_passthrough(provider):
    book = provider.get_orderbook()
    assert isinstance(book, list)
    assert book[0]["orderid"] == "AO123"
