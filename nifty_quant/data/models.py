"""Immutable data models for market data.

These are plain dataclasses so they are easy to construct in tests, serialise,
and reason about. They are intentionally provider-agnostic: every concrete
data source (NSE public API, Zerodha Kite, Dhan, Upstox, a CSV backtest feed)
maps its raw payload into these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Sequence

import pandas as pd


class OptionType(str, Enum):
    """Call or Put."""

    CALL = "CE"
    PUT = "PE"


@dataclass(frozen=True)
class Candle:
    """A single OHLCV candle for one timeframe bucket."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        for name in ("open", "high", "low", "close"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


@dataclass(frozen=True)
class OHLCVSeries:
    """An ordered series of candles for a single symbol/timeframe.

    Provides a cached conversion to a pandas DataFrame indexed by timestamp,
    which is the form the analytics layer consumes.
    """

    symbol: str
    timeframe: str  # e.g. "1d", "1w", "5m"
    candles: Sequence[Candle]

    def __post_init__(self) -> None:
        ts = [c.timestamp for c in self.candles]
        if ts != sorted(ts):
            raise ValueError("candles must be sorted ascending by timestamp")

    def to_frame(self) -> pd.DataFrame:
        """Return an OHLCV DataFrame indexed by timestamp (ascending)."""
        if not self.candles:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).astype(float)
        df = pd.DataFrame(
            {
                "open": [c.open for c in self.candles],
                "high": [c.high for c in self.candles],
                "low": [c.low for c in self.candles],
                "close": [c.close for c in self.candles],
                "volume": [c.volume for c in self.candles],
            },
            index=pd.DatetimeIndex(
                [c.timestamp for c in self.candles], name="timestamp"
            ),
        )
        return df

    def __len__(self) -> int:
        return len(self.candles)


@dataclass(frozen=True)
class OptionQuote:
    """A single option contract snapshot from the chain."""

    strike: float
    option_type: OptionType
    expiry: date
    last_price: float
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0
    oi_change: float = 0.0
    implied_volatility: float | None = None  # as a fraction, e.g. 0.13 = 13%

    @property
    def mid(self) -> float:
        """Mid price if a two-sided quote exists, else last traded price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last_price


@dataclass(frozen=True)
class OptionChain:
    """A full option-chain snapshot for one underlying and one expiry."""

    underlying: str
    spot: float
    expiry: date
    timestamp: datetime
    quotes: Sequence[OptionQuote] = field(default_factory=tuple)

    def calls(self) -> list[OptionQuote]:
        return sorted(
            (q for q in self.quotes if q.option_type is OptionType.CALL),
            key=lambda q: q.strike,
        )

    def puts(self) -> list[OptionQuote]:
        return sorted(
            (q for q in self.quotes if q.option_type is OptionType.PUT),
            key=lambda q: q.strike,
        )

    def strikes(self) -> list[float]:
        return sorted({q.strike for q in self.quotes})

    def atm_strike(self) -> float:
        """Strike closest to spot."""
        strikes = self.strikes()
        if not strikes:
            raise ValueError("chain has no strikes")
        return min(strikes, key=lambda k: abs(k - self.spot))
