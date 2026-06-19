"""Tests for the PaperFeed and clock (deterministic, no real time/network)."""

from datetime import date, datetime, timedelta

import pytest

from nifty_quant.data.models import (
    Candle,
    OHLCVSeries,
    OptionChain,
    OptionQuote,
    OptionType,
)
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.feed.base import MarketEventHandler
from nifty_quant.feed.clock import ManualClock
from nifty_quant.feed.events import CandleEvent, OptionChainEvent
from nifty_quant.feed.paper import PaperFeed


BASE = datetime(2025, 1, 2, 9, 15)


def _candles(n):
    return [
        Candle(BASE + timedelta(minutes=5 * i), 100 + i, 101 + i, 99 + i, 100 + i, 1000)
        for i in range(n)
    ]


class FakePaperProvider:
    """Reveals one more candle on each poll; serves a fresh chain each call."""

    def __init__(self, total_candles=5, initial_visible=3):
        self._all = _candles(total_candles)
        self._visible = initial_visible
        self._chain_calls = 0

    def get_ohlcv(self, symbol, timeframe, start, end):
        series = OHLCVSeries(symbol, timeframe, list(self._all[: self._visible]))
        self._visible = min(self._visible + 1, len(self._all))
        return series

    def get_option_chain(self, underlying, expiry):
        ts = BASE + timedelta(minutes=self._chain_calls)
        self._chain_calls += 1
        quotes = (
            OptionQuote(25000, OptionType.CALL, expiry, 120.0, open_interest=1000),
            OptionQuote(25000, OptionType.PUT, expiry, 110.0, open_interest=1100),
        )
        return OptionChain(underlying, 25050.0, expiry, ts, quotes)

    def get_spot(self, symbol):
        return 25050.0


class Recorder(MarketEventHandler):
    def __init__(self):
        self.candles = []
        self.chains = []

    def on_candle(self, event: CandleEvent):
        self.candles.append(event)

    def on_option_chain(self, event: OptionChainEvent):
        self.chains.append(event)


class RaisingProvider:
    def get_ohlcv(self, *a, **k):
        raise RuntimeError("provider down")

    def get_option_chain(self, *a, **k):
        raise RuntimeError("provider down")

    def get_spot(self, symbol):
        raise RuntimeError("provider down")


def _feed(provider, **kw):
    defaults = dict(
        candle_specs=[("NIFTY", "5m")],
        chain_specs=[("NIFTY", date(2025, 1, 30))],
        poll_interval_seconds=60.0,
        clock=ManualClock(BASE),
    )
    defaults.update(kw)
    return PaperFeed(provider, **defaults)


def test_manual_clock_sleep_advances():
    clk = ManualClock(BASE)
    clk.sleep(30)
    assert clk.now() == BASE + timedelta(seconds=30)


def test_paper_feed_emits_and_dedups():
    feed = _feed(FakePaperProvider(total_candles=5, initial_visible=3))
    rec = Recorder()
    feed.subscribe(rec)
    feed.run(max_polls=2)

    # Poll 1 emits 3 history candles; poll 2 reveals 1 more -> 4 unique, no dupes.
    timestamps = [c.timestamp for c in rec.candles]
    assert len(timestamps) == 4
    assert len(set(timestamps)) == 4
    # A chain is emitted each poll.
    assert len(rec.chains) == 2


def test_paper_feed_no_initial_history():
    feed = _feed(
        FakePaperProvider(total_candles=5, initial_visible=3),
        emit_initial_history=False,
    )
    rec = Recorder()
    feed.subscribe(rec)
    feed.run(max_polls=2)
    # Poll 1 suppresses history; poll 2 emits only the newly revealed candle.
    assert len(rec.candles) == 1


def test_paper_feed_persists_to_storage(tmp_path):
    storage = ParquetStorage(tmp_path)
    feed = _feed(FakePaperProvider(total_candles=4, initial_visible=2), storage=storage)
    feed.subscribe(Recorder())
    feed.run(max_polls=2)

    candles = storage.read_candles(
        "NIFTY", "5m", BASE - timedelta(days=1), BASE + timedelta(days=1)
    )
    assert len(candles) >= 3  # accumulated and persisted
    chains = storage.read_option_chains(
        "NIFTY", BASE - timedelta(days=1), BASE + timedelta(days=1)
    )
    assert len(chains) == 2  # one snapshot per poll (distinct timestamps)


def test_paper_feed_survives_then_aborts_on_errors():
    feed = _feed(RaisingProvider(), max_consecutive_errors=3)
    rec = Recorder()
    feed.subscribe(rec)
    total = feed.run(max_polls=10)
    # No events, and the loop aborts after hitting the error cap (not 10 polls).
    assert total == 0
    assert rec.candles == [] and rec.chains == []


def test_paper_feed_stop_requested():
    feed = _feed(FakePaperProvider())
    feed.subscribe(Recorder())
    feed.stop()
    total = feed.run(max_polls=5)
    assert total == 0  # stopped before any poll


def test_paper_feed_drives_backtest_engine_unchanged():
    """The exact backtest engine runs forward on the live feed, no changes."""
    from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
    from nifty_quant.backtest.engine import BacktestEngine
    from nifty_quant.backtest.instrument import Instrument, InstrumentType
    from nifty_quant.backtest.portfolio import Portfolio
    from nifty_quant.backtest.risk import BasicRiskEngine
    from nifty_quant.backtest.strategies.ema_cross import EmaCrossStrategy

    inst = Instrument("NIFTY", InstrumentType.INDEX)
    engine = BacktestEngine(
        EmaCrossStrategy(inst, fast=2, slow=3, quantity=10),
        portfolio=Portfolio(starting_cash=1_000_000),
        risk_engine=BasicRiskEngine(default_quantity=10),
        broker=SimulatedBroker(fill_model=MidPriceFill()),
    )
    feed = _feed(FakePaperProvider(total_candles=8, initial_visible=8))
    feed.subscribe(engine)
    feed.run(max_polls=2)

    result = engine.build_result()
    # Every emitted event produced an equity point.
    assert len(result.equity_curve) > 0

