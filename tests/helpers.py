"""Shared test helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

from nifty_quant.data.models import Candle
from nifty_quant.feed.base import MarketFeed
from nifty_quant.feed.events import CandleEvent


class ListFeed(MarketFeed):
    """A feed that emits a fixed list of events (test helper)."""

    def __init__(self, events):
        super().__init__()
        self._events = list(events)

    def run(self) -> int:
        for e in self._events:
            self._dispatch(e)
        return len(self._events)


def make_candle_events(
    closes,
    *,
    symbol: str = "NIFTY",
    timeframe: str = "1d",
    start: datetime = datetime(2025, 1, 1, 9, 15),
    step: timedelta = timedelta(days=1),
):
    """Build a list of CandleEvents from a sequence of closes."""
    events = []
    ts = start
    for c in closes:
        events.append(
            CandleEvent(
                timestamp=ts,
                symbol=symbol,
                timeframe=timeframe,
                candle=Candle(ts, c, c + 1, c - 1, c, 1000),
            )
        )
        ts += step
    return events
