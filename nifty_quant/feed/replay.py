"""Historical replay feed.

Reads candles and option-chain snapshots from :class:`Storage` and emits them
through the same :class:`~nifty_quant.feed.base.MarketFeed` interface a live
WebSocket feed will use. A strategy subscribed to a ReplayFeed behaves exactly
as it will in paper/live trading -- only the feed implementation differs.

Events from multiple sources are merged into one strictly time-ordered stream.
When timestamps tie, candles are emitted before option-chain snapshots (price
prints, then the derived chain), which is a deterministic, defensible ordering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from nifty_quant.data.storage.base import Storage
from nifty_quant.feed.base import MarketFeed
from nifty_quant.feed.events import (
    CandleEvent,
    MarketEvent,
    OptionChainEvent,
)
from nifty_quant.log import get_logger

_log = get_logger("feed.replay")

# Tie-breaker priority: lower sorts first when timestamps are equal.
_CANDLE_PRIORITY = 0
_CHAIN_PRIORITY = 1


class ReplayFeed(MarketFeed):
    """Replays stored history as a time-ordered event stream."""

    def __init__(
        self,
        storage: Storage,
        start: datetime,
        end: datetime,
        *,
        candle_specs: Iterable[tuple[str, str]] = (),
        chain_underlyings: Iterable[str] = (),
    ) -> None:
        """
        Parameters
        ----------
        storage:
            Backend to read from.
        start, end:
            Inclusive time window to replay.
        candle_specs:
            Iterable of (symbol, timeframe) pairs to stream candles for.
        chain_underlyings:
            Iterable of underlyings to stream option-chain snapshots for.
        """
        super().__init__()
        self.storage = storage
        self.start = start
        self.end = end
        self.candle_specs = list(candle_specs)
        self.chain_underlyings = list(chain_underlyings)

    def _collect_events(self) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        for symbol, timeframe in self.candle_specs:
            series = self.storage.read_candles(
                symbol, timeframe, self.start, self.end
            )
            for candle in series.candles:
                events.append(
                    CandleEvent(
                        timestamp=candle.timestamp,
                        symbol=symbol,
                        timeframe=timeframe,
                        candle=candle,
                    )
                )
        for underlying in self.chain_underlyings:
            for chain in self.storage.read_option_chains(
                underlying, self.start, self.end
            ):
                events.append(
                    OptionChainEvent(timestamp=chain.timestamp, chain=chain)
                )
        events.sort(key=_sort_key)
        return events

    def run(self) -> int:
        """Emit all events in the window to subscribers. Returns event count."""
        events = self._collect_events()
        _log.event(
            "replay_started",
            start=self.start.isoformat(),
            end=self.end.isoformat(),
            events=len(events),
        )
        for event in events:
            self._dispatch(event)
        _log.event("replay_finished", events=len(events))
        return len(events)


def _sort_key(event: MarketEvent) -> tuple[datetime, int]:
    priority = _CANDLE_PRIORITY if isinstance(event, CandleEvent) else _CHAIN_PRIORITY
    return (event.timestamp, priority)
