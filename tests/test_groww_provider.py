"""Tests for the Groww provider mapping logic.

We inject a fake SDK client that returns payloads shaped like the documented
Groww responses, so the entire mapping layer is exercised offline -- no network
and no ``growwapi`` install required.
"""

from datetime import date

import pytest

from nifty_quant.data.models import OptionType
from nifty_quant.data.providers.base import OrderRequest, OrderSide, OrderKind
from nifty_quant.data.providers.groww import (
    GrowwProvider,
    LiveTradingDisabledError,
)


class FakeGrowwClient:
    """Minimal stand-in for growwapi.GrowwAPI with documented payloads."""

    # Constants the SDK exposes as attributes.
    VALIDITY_DAY = "DAY"
    EXCHANGE_NSE = "NSE"
    SEGMENT_FNO = "FNO"
    SEGMENT_CASH = "CASH"
    PRODUCT_MIS = "MIS"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_MARKET = "MARKET"
    TRANSACTION_TYPE_BUY = "BUY"
    TRANSACTION_TYPE_SELL = "SELL"

    def __init__(self):
        self.placed = []
        self.cancelled = []

    def get_ltp(self, segment, exchange_trading_symbols):
        return {"NSE_NIFTY": 25050.9}

    def get_historical_candle_data(self, **kwargs):
        return {
            "candles": [
                [1633072800, 150.0, 155.0, 145.0, 152.0, 10000],
                [1633072860, 152.0, 158.0, 151.0, 157.0, 12000],
            ],
            "start_time": "2025-01-01 09:15:00",
            "end_time": "2025-01-01 15:30:00",
            "interval_in_minutes": 1,
        }

    def get_option_chain(self, exchange, underlying, expiry_date):
        return {
            "underlying_ltp": 25050.9,
            "strikes": {
                "25000": {
                    "CE": {
                        "greeks": {"delta": 0.55, "gamma": 0.001, "iv": 13.4},
                        "trading_symbol": "NIFTY25O0725000CE",
                        "ltp": 180.0,
                        "open_interest": 1200,
                        "volume": 3400,
                    },
                    "PE": {
                        "greeks": {"delta": -0.45, "gamma": 0.001, "iv": 14.1},
                        "trading_symbol": "NIFTY25O0725000PE",
                        "ltp": 150.0,
                        "open_interest": 2200,
                        "volume": 4100,
                    },
                },
                "25100": {
                    "CE": {
                        "greeks": {"delta": 0.40, "gamma": 0.001, "iv": 13.0},
                        "trading_symbol": "NIFTY25O0725100CE",
                        "ltp": 120.0,
                        "open_interest": 1500,
                        "volume": 2900,
                    },
                    # PE intentionally absent to test sparse legs.
                },
            },
        }

    def place_order(self, **kwargs):
        self.placed.append(kwargs)
        return {"groww_order_id": "GW123", "order_status": "OPEN"}

    def cancel_order(self, groww_order_id):
        self.cancelled.append(groww_order_id)
        return {"order_status": "CANCELLED"}

    def get_order_list(self):
        return {"order_list": [{"groww_order_id": "GW123", "status": "OPEN"}]}


@pytest.fixture
def provider():
    return GrowwProvider(FakeGrowwClient())


def test_get_spot(provider):
    assert provider.get_spot("NIFTY") == pytest.approx(25050.9)


def test_get_ohlcv_maps_candles(provider):
    series = provider.get_ohlcv("NIFTY", "1m", date(2025, 1, 1), date(2025, 1, 1))
    assert len(series) == 2
    df = series.to_frame()
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == pytest.approx(152.0)
    assert df["volume"].iloc[1] == pytest.approx(12000)


def test_get_ohlcv_rejects_bad_timeframe(provider):
    with pytest.raises(ValueError):
        provider.get_ohlcv("NIFTY", "2m", date(2025, 1, 1), date(2025, 1, 1))


def test_option_chain_mapping_and_iv_conversion(provider):
    chain = provider.get_option_chain("NIFTY", date(2025, 10, 7))
    assert chain.spot == pytest.approx(25050.9)
    # Spot 25050.9 is 49.1 from 25100 vs 50.9 from 25000 -> ATM is 25100.
    assert chain.atm_strike() == 25100

    # IV must be converted from percent to fraction.
    ce_25000 = next(
        q for q in chain.calls() if q.strike == 25000
    )
    assert ce_25000.implied_volatility == pytest.approx(0.134)
    assert ce_25000.option_type is OptionType.CALL

    # Sparse leg: 25100 has only a CE, no PE.
    puts_25100 = [q for q in chain.puts() if q.strike == 25100]
    assert puts_25100 == []
    assert len(chain.quotes) == 3


def test_place_order_blocked_when_live_disabled(provider):
    order = OrderRequest(
        trading_symbol="NIFTY25O0725000CE",
        side=OrderSide.BUY,
        quantity=75,
        kind=OrderKind.MARKET,
    )
    with pytest.raises(LiveTradingDisabledError):
        provider.place_order(order)


def test_cancel_order_blocked_when_live_disabled(provider):
    with pytest.raises(LiveTradingDisabledError):
        provider.cancel_order("GW123")


def test_place_order_works_when_live_enabled():
    client = FakeGrowwClient()
    prov = GrowwProvider(client, live_trading_enabled=True)
    order = OrderRequest(
        trading_symbol="NIFTY25O0725000CE",
        side=OrderSide.BUY,
        quantity=75,
        kind=OrderKind.LIMIT,
        price=180.0,
        reference_id="ref-00012345",
    )
    result = prov.place_order(order)
    assert result.broker_order_id == "GW123"
    assert result.status == "OPEN"
    assert client.placed[0]["transaction_type"] == "BUY"
    assert client.placed[0]["order_type"] == "LIMIT"
    assert client.placed[0]["price"] == 180.0


def test_cancel_order_when_enabled():
    client = FakeGrowwClient()
    prov = GrowwProvider(client, live_trading_enabled=True)
    assert prov.cancel_order("GW123") is True
    assert client.cancelled == ["GW123"]


def test_orderbook_passthrough(provider):
    book = provider.get_orderbook()
    assert isinstance(book, list)
    assert book[0]["groww_order_id"] == "GW123"


def test_limit_order_requires_price():
    with pytest.raises(ValueError):
        OrderRequest(
            trading_symbol="X",
            side=OrderSide.BUY,
            quantity=75,
            kind=OrderKind.LIMIT,
        )
