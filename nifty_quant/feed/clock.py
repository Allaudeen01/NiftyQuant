"""Clock abstraction for time-driven feeds.

Injecting the clock keeps the :class:`~nifty_quant.feed.paper.PaperFeed` fully
testable: tests use :class:`ManualClock` to advance time deterministically with
no real sleeping, while production uses :class:`RealClock`.
"""

from __future__ import annotations

import abc
import time
from datetime import datetime, timedelta


class Clock(abc.ABC):
    @abc.abstractmethod
    def now(self) -> datetime:
        raise NotImplementedError

    @abc.abstractmethod
    def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class RealClock(Clock):
    """Wall-clock time with real sleeping."""

    def now(self) -> datetime:
        return datetime.now()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ManualClock(Clock):
    """Test clock: ``sleep`` advances a virtual clock instead of blocking."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
