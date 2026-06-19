"""The unified feed and handler interfaces.

This is the contract that makes "replay == paper == live": any feed (historical
:class:`~nifty_quant.feed.replay.ReplayFeed` now, a live WebSocket feed later)
pushes :class:`~nifty_quant.feed.events.MarketEvent` instances to subscribed
:class:`MarketEventHandler` objects. Strategies implement the handler once and
never learn whether the data is historical or live.
"""

from __future__ import annotations

import abc

from nifty_quant.feed.events import (
    CandleEvent,
    MarketEvent,
    OptionChainEvent,
)


class MarketEventHandler(abc.ABC):
    """Receives market events. Strategies subclass this.

    Default implementations dispatch by event type to ``on_candle`` /
    ``on_option_chain`` so subclasses override only what they care about.
    """

    def handle(self, event: MarketEvent) -> None:
        """Dispatch an event to the appropriate typed handler."""
        if isinstance(event, CandleEvent):
            self.on_candle(event)
        elif isinstance(event, OptionChainEvent):
            self.on_option_chain(event)
        else:  # pragma: no cover - defensive
            self.on_event(event)

    def on_candle(self, event: CandleEvent) -> None:
        """Handle a candle event. Override as needed."""

    def on_option_chain(self, event: OptionChainEvent) -> None:
        """Handle an option-chain snapshot. Override as needed."""

    def on_event(self, event: MarketEvent) -> None:
        """Fallback for unrecognised event types. Override as needed."""


class MarketFeed(abc.ABC):
    """A source of time-ordered market events.

    Implementations: ReplayFeed (historical), and later a LiveFeed (WebSocket).
    Both share this interface, so the runner/strategy code is identical.
    """

    def __init__(self) -> None:
        self._handlers: list[MarketEventHandler] = []

    def subscribe(self, handler: MarketEventHandler) -> None:
        """Register a handler to receive every emitted event."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def _dispatch(self, event: MarketEvent) -> None:
        for handler in self._handlers:
            handler.handle(event)

    @abc.abstractmethod
    def run(self) -> None:
        """Emit events to subscribers until the source is exhausted/stopped."""
        raise NotImplementedError
