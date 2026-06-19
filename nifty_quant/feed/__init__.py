"""Event feed layer.

Defines the single event interface that *every* data source pushes through:
historical replay today, live WebSocket later. Strategy code subscribes once
and runs unchanged across backtest, paper, and live modes.
"""

from nifty_quant.feed.events import (
    MarketEvent,
    CandleEvent,
    OptionChainEvent,
)
from nifty_quant.feed.base import MarketEventHandler, MarketFeed
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.feed.clock import Clock, RealClock, ManualClock
from nifty_quant.feed.paper import PaperFeed

__all__ = [
    "MarketEvent",
    "CandleEvent",
    "OptionChainEvent",
    "MarketEventHandler",
    "MarketFeed",
    "ReplayFeed",
    "Clock",
    "RealClock",
    "ManualClock",
    "PaperFeed",
]
