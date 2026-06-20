"""Composable entry filters.

A filter answers one question: "given current conditions, should this entry be
ALLOWED?" Filters gate *entries only* — exits are never blocked (you must always
be able to get out). Layer filters onto a base entry without changing the entry
logic, which is the core of the "when should it NOT trade?" research approach.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime, time


@dataclass(frozen=True)
class FilterContext:
    """Snapshot of conditions at the moment an entry signal is considered."""

    timestamp: datetime
    indicators: dict = field(default_factory=dict)

    def value(self, name: str) -> float | None:
        v = self.indicators.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return v


class EntryFilter(abc.ABC):
    @abc.abstractmethod
    def allows(self, ctx: FilterContext) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def label(self) -> str:
        raise NotImplementedError


class AdxFilter(EntryFilter):
    """Allow entries only in a trending market (ADX >= threshold)."""

    def __init__(self, min_adx: float) -> None:
        self.min_adx = min_adx

    def allows(self, ctx: FilterContext) -> bool:
        v = ctx.value("adx")
        return v is not None and v >= self.min_adx

    @property
    def label(self) -> str:
        return f"ADX>={self.min_adx:g}"


class TimeWindowFilter(EntryFilter):
    """Allow entries only within an intraday time window (inclusive)."""

    def __init__(self, start: str, end: str) -> None:
        self.start = _parse_time(start)
        self.end = _parse_time(end)
        self._s, self._e = start, end

    def allows(self, ctx: FilterContext) -> bool:
        t = ctx.timestamp.time()
        return self.start <= t <= self.end

    @property
    def label(self) -> str:
        return f"Time[{self._s}-{self._e}]"


class AtrPercentileFilter(EntryFilter):
    """Allow entries only when volatility (ATR percentile) is elevated."""

    def __init__(self, min_pct: float) -> None:
        self.min_pct = min_pct

    def allows(self, ctx: FilterContext) -> bool:
        v = ctx.value("atr_pct")
        return v is not None and v >= self.min_pct

    @property
    def label(self) -> str:
        return f"ATRpct>={self.min_pct:g}"


def _parse_time(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))
