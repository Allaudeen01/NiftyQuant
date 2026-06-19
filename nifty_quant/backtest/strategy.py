"""Strategy base class and metadata.

A Strategy reacts to events and/or feature vectors and *emits signals*. It never
touches a broker or portfolio, which keeps every strategy testable in isolation
and identical across backtest, paper, and live feeds.

Two consumption styles, both supported by the engine:
- raw-event strategies override ``on_candle`` / ``on_option_chain``;
- feature strategies override ``on_features`` and consume :class:`FeatureVector`
  objects produced by the FeatureEngine (the preferred style for research).

Every strategy describes itself via :class:`StrategyMetadata` so experiment
tracking can record exactly what produced a result.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from nifty_quant.backtest.signals import Signal
from nifty_quant.feed.base import MarketEventHandler
from nifty_quant.features.vector import FeatureVector


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    version: str = "0.0.0"
    author: str = "research"
    parameters: dict = field(default_factory=dict)
    feature_version: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "parameters": dict(self.parameters),
            "feature_version": self.feature_version,
        }


class Strategy(MarketEventHandler, abc.ABC):
    """Abstract base for all strategies (rule-based, ML, or AI-driven)."""

    def __init__(self) -> None:
        self._pending: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        """Queue a signal for the engine to process after this event."""
        self._pending.append(signal)

    def drain(self) -> list[Signal]:
        """Return and clear buffered signals."""
        out = self._pending
        self._pending = []
        return out

    def on_features(self, features: FeatureVector) -> None:
        """Handle a computed feature vector. Feature strategies override this."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (used in logs/journal)."""
        raise NotImplementedError

    @property
    def metadata(self) -> StrategyMetadata:
        """Self-description. Override to record version/params/author."""
        return StrategyMetadata(name=self.name)
