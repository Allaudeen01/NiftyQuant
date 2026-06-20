"""Storage interface.

Deliberately minimal and access-pattern driven: the backtester and replay
engine read candles and option-chain snapshots over a time range; the collector
writes them. The schema/indexing details are left to concrete backends so we
can optimise storage *after* the access patterns are proven (per the agreed
roadmap), not before.
"""

from __future__ import annotations

import abc
from datetime import datetime

from nifty_quant.data.models import OHLCVSeries, OptionChain


class Storage(abc.ABC):
    """Persistence for time-series market data."""

    @abc.abstractmethod
    def write_candles(self, series: OHLCVSeries) -> int:
        """Persist a candle series. Returns the number of candles written.

        Implementations must be idempotent: re-writing overlapping candles
        must not create duplicates (dedupe on timestamp).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def read_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> OHLCVSeries:
        """Read candles for [start, end] inclusive, ascending by timestamp."""
        raise NotImplementedError

    @abc.abstractmethod
    def write_option_chain(self, chain: OptionChain) -> int:
        """Persist one option-chain snapshot. Returns rows (quotes) written.

        Idempotent on (snapshot timestamp, strike, option_type).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def read_option_chains(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
    ) -> list[OptionChain]:
        """Read all snapshots in [start, end], ascending by snapshot time."""
        raise NotImplementedError
