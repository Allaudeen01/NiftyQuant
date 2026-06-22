"""Tests for the Angel instrument master and option-chain assembly (offline)."""

from datetime import date

import pytest

from nifty_quant.data.models import OptionType
from nifty_quant.data.providers.angel_instruments import (
    InstrumentMaster,
    _option_type_from_symbol,
    _parse_expiry,
)
from nifty_quant.data.providers.angelone import AngelOneProvider


# --- instrument master -----------------------------------------------------


def _records():
    return [
        {"token": "99926000", "symbol": "Nifty 50", "name": "NIFTY",
         "expiry": "", "strike": "0.000000", "instrumenttype": "AMXIDX",
         "exch_seg": "NSE"},
        {"token": "111", "symbol": "NIFTY30JAN2525000CE", "name": "NIFTY",
         "expiry": "30JAN2025", "strike": "2500000.000000",
         "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
        {"token": "112", "symbol": "NIFTY30JAN2525000PE", "name": "NIFTY",
         "expiry": "30JAN2025", "strike": "2500000.000000",
         "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
        {"token": "113", "symbol": "NIFTY30JAN2525100CE", "name": "NIFTY",
         "expiry": "30JAN2025", "strike": "2510000.000000",
         "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
        # Different expiry -> must be excluded.
        {"token": "114", "symbol": "NIFTY27FEB2525000CE", "name": "NIFTY",
         "expiry": "27FEB2025", "strike": "2500000.000000",
         "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    ]


def test_parse_expiry_formats():
    assert _parse_expiry("30JAN2025") == date(2025, 1, 30)
    assert _parse_expiry("2025-01-30") == date(2025, 1, 30)
    assert _parse_expiry("") is None


def test_option_type_from_symbol():
    assert _option_type_from_symbol("NIFTY30JAN2525000CE") is OptionType.CALL
    assert _option_type_from_symbol("NIFTY30JAN2525000PE") is OptionType.PUT
    assert _option_type_from_symbol("NIFTY50") is None


def test_instrument_master_filters_by_expiry_and_strike():
    master = InstrumentMaster.from_records(_records())
    ins = master.option_instruments("NIFTY", date(2025, 1, 30))
    # 3 contracts at the Jan expiry (two 25000 CE/PE, one 25100 CE).
    assert len(ins) == 3
    strikes = sorted({o.strike for o in ins})
    assert strikes == [25000.0, 25100.0]  # paise converted to rupees
    tokens = {o.token for o in ins}
    assert "114" not in tokens  # Feb expiry excluded


def test_available_expiries():
    master = InstrumentMaster.from_records(_records())
    assert master.available_expiries("NIFTY") == [date(2025, 1, 30), date(2025, 2, 27)]


# --- option-chain assembly -------------------------------------------------


class FakeClientForChain:
    def getMarketData(self, mode, exchange_tokens):
        assert mode in ("FULL", "LTP")
        if mode == "LTP":
            tok = exchange_tokens["NSE"][0]
            # Token-aware LTP: NIFTY index vs India VIX (spot now reads live LTP).
            ltp = 25050.0 if tok == "99926000" else 13.42
            return {"status": True, "data": {
                "fetched": [{"symbolToken": tok, "ltp": ltp}], "unfetched": []}}
        tokens = exchange_tokens["NFO"]
        fetched = []
        for t in tokens:
            fetched.append({
                "symbolToken": t,
                "ltp": 120.0,
                "opnInterest": 1000 + int(t),
                "tradeVolume": 500,
                "depth": {"buy": [{"price": 119.5, "quantity": 50}],
                          "sell": [{"price": 120.5, "quantity": 50}]},
            })
        return {"status": True, "data": {"fetched": fetched, "unfetched": []}}

    def getCandleData(self, params):
        # Used by get_spot (most recent daily candle).
        return {"status": True, "data": [
            ["2025-01-29T09:15:00+05:30", 25040, 25060, 25030, 25050.0, 0],
        ]}


def test_get_option_chain_builds_from_master_and_marketdata():
    master = InstrumentMaster.from_records(_records())
    provider = AngelOneProvider(
        FakeClientForChain(), request_pause=0.0, instrument_master=master
    )
    chain = provider.get_option_chain("NIFTY", date(2025, 1, 30))

    assert chain.underlying == "NIFTY"
    assert chain.spot == pytest.approx(25050.0)
    assert len(chain.quotes) == 3
    # Bid/ask come from depth; mid is between them.
    ce = next(q for q in chain.calls() if q.strike == 25000)
    assert ce.bid == pytest.approx(119.5)
    assert ce.ask == pytest.approx(120.5)
    assert ce.mid == pytest.approx(120.0)
    assert ce.open_interest > 0
    # ATM strike resolves against the spot.
    assert chain.atm_strike() in (25000.0, 25100.0)


def test_get_option_chain_raises_when_no_contracts():
    master = InstrumentMaster.from_records(_records())
    provider = AngelOneProvider(
        FakeClientForChain(), request_pause=0.0, instrument_master=master
    )
    with pytest.raises(RuntimeError):
        provider.get_option_chain("NIFTY", date(2030, 1, 1))


def test_index_token_resolves_india_vix():
    records = _records() + [
        {"token": "26017", "symbol": "India VIX", "name": "INDIA VIX",
         "expiry": "", "strike": "0.000000", "instrumenttype": "AMXIDX",
         "exch_seg": "NSE"},
    ]
    master = InstrumentMaster.from_records(records)
    assert master.index_token("INDIA VIX") == "26017"
    assert master.index_token("NIFTY") == "99926000"
    assert master.index_token("NONEXISTENT") is None


def test_get_india_vix():
    records = _records() + [
        {"token": "26017", "symbol": "India VIX", "name": "INDIA VIX",
         "expiry": "", "strike": "0.000000", "instrumenttype": "AMXIDX",
         "exch_seg": "NSE"},
    ]
    master = InstrumentMaster.from_records(records)
    provider = AngelOneProvider(
        FakeClientForChain(), request_pause=0.0, instrument_master=master
    )
    assert provider.get_india_vix() == pytest.approx(13.42)
