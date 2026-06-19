"""Tests for the replay feed and the historical collector.

Together these verify the 'replay == live' contract: a handler receives a
strictly time-ordered event stream through the same interface a live feed will
use.
"""

from datetime import date, datetime

from nifty_quant.data.collectors.historical import HistoricalCollector
from nifty_quant.data.models import (
    Candle,
    OHLCVSeries,
    OptionChain,
    OptionQuote,
    OptionType,
)
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.feed.base import MarketEventHandler
from nifty_quant.feed.events import CandleEvent, OptionChainEvent
from nifty_quant.feed.replay import ReplayFeed


class RecordingHandler(MarketEventHandler):
    """Captures events in arrival order for assertions."""

    def __init__(self):
        self.candles = []
        self.chains = []
        self.order = []

    def on_candle(self, event: CandleEvent):
        self.candles.append(event)
        self.order.append(("candle", event.timestamp))

    def on_option_chain(self, event: OptionChainEvent):
        self.chains.append(event)
        self.order.append(("chain", event.timestamp))


class FakeProvider:
    """Minimal MarketDataProvider stand-in for the collector test."""

    def get_ohlcv(self, symbol, timeframe, start, end):
        candles = [
            Candle(datetime(2025, 1, 2, 9, 15), 100, 101, 99, 100.5, 1000),
            Candle(datetime(2025, 1, 2, 9, 20), 100.5, 102, 100, 101.5, 1200),
        ]
        return OHLCVSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    def get_option_chain(self, underlying, expiry):
        quotes = (
            OptionQuote(25000, OptionType.CALL, expiry, 120.0, open_interest=1000),
            OptionQuote(25000, OptionType.PUT, expiry, 110.0, open_interest=1100),
        )
        return OptionChain(
            underlying, 25050.0, expiry, datetime(2025, 1, 2, 9, 15), quotes
        )

    def get_spot(self, symbol):
        return 25050.0


def _seed_storage(tmp_path):
    store = ParquetStorage(tmp_path)
    candles = [
        Candle(datetime(2025, 1, 2, 9, 15), 100, 101, 99, 100.5, 1000),
        Candle(datetime(2025, 1, 2, 9, 20), 100.5, 102, 100, 101.5, 1200),
        Candle(datetime(2025, 1, 2, 9, 25), 101.5, 103, 101, 102.5, 1300),
    ]
    store.write_candles(OHLCVSeries("NIFTY", "5m", candles))

    expiry = date(2025, 1, 30)
    # Snapshot at 9:20, same timestamp as the second candle (tie-break test).
    chain = OptionChain(
        "NIFTY",
        25050.0,
        expiry,
        datetime(2025, 1, 2, 9, 20),
        (OptionQuote(25000, OptionType.CALL, expiry, 120.0, open_interest=1000),),
    )
    store.write_option_chain(chain)
    return store


def test_replay_emits_time_ordered_events(tmp_path):
    store = _seed_storage(tmp_path)
    feed = ReplayFeed(
        store,
        start=datetime(2025, 1, 2, 0, 0),
        end=datetime(2025, 1, 2, 23, 59),
        candle_specs=[("NIFTY", "5m")],
        chain_underlyings=["NIFTY"],
    )
    handler = RecordingHandler()
    feed.subscribe(handler)
    count = feed.run()

    assert count == 4  # 3 candles + 1 chain
    assert len(handler.candles) == 3
    assert len(handler.chains) == 1

    # Timestamps must be non-decreasing in arrival order.
    times = [ts for _, ts in handler.order]
    assert times == sorted(times)

    # Tie-break at 09:20: candle before chain.
    tie = [kind for kind, ts in handler.order if ts == datetime(2025, 1, 2, 9, 20)]
    assert tie == ["candle", "chain"]


def test_replay_respects_window(tmp_path):
    store = _seed_storage(tmp_path)
    feed = ReplayFeed(
        store,
        start=datetime(2025, 1, 2, 9, 18),
        end=datetime(2025, 1, 2, 9, 22),
        candle_specs=[("NIFTY", "5m")],
        chain_underlyings=["NIFTY"],
    )
    handler = RecordingHandler()
    feed.subscribe(handler)
    feed.run()
    # Only the 9:20 candle and the 9:20 chain fall in the window.
    assert len(handler.candles) == 1
    assert len(handler.chains) == 1


def test_collector_persists_then_replays(tmp_path):
    """End-to-end: collect from a provider, then replay from storage."""
    store = ParquetStorage(tmp_path)
    collector = HistoricalCollector(FakeProvider(), store)

    assert collector.collect_candles("NIFTY", "5m", date(2025, 1, 2), date(2025, 1, 2)) == 2
    assert collector.collect_option_chain("NIFTY", date(2025, 1, 30)) == 2

    feed = ReplayFeed(
        store,
        start=datetime(2025, 1, 2, 0, 0),
        end=datetime(2025, 1, 2, 23, 59),
        candle_specs=[("NIFTY", "5m")],
        chain_underlyings=["NIFTY"],
    )
    handler = RecordingHandler()
    feed.subscribe(handler)
    feed.run()
    assert len(handler.candles) == 2
    assert len(handler.chains) == 1
