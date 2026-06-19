"""Market events.

Every event carries a ``timestamp`` so a feed can emit a strictly time-ordered
stream. Concrete event types wrap the existing data models, so the same
``Candle`` / ``OptionChain`` objects flow through replay and (later) live feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from nifty_quant.data.models import Candle, OptionChain


@dataclass(frozen=True)
class MarketEvent:
    """Base class for all events. ``timestamp`` drives ordering."""

    timestamp: datetime


@dataclass(frozen=True)
class CandleEvent(MarketEvent):
    """A completed OHLCV candle for one symbol/timeframe."""

    symbol: str
    timeframe: str
    candle: Candle


@dataclass(frozen=True)
class OptionChainEvent(MarketEvent):
    """A full option-chain snapshot."""

    chain: OptionChain
