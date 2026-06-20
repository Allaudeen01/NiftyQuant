"""Incremental (streaming) indicators for fast, filter-based strategies.

Each updates in O(1) (or O(window)) per bar, so strategies that need ADX/ATR/
percentile gating run quickly over large intraday datasets without recomputing
over a rolling buffer. These are intentionally lightweight, self-consistent
approximations of the Wilder-smoothed indicators in ``analytics.indicators``;
they're used for entry *filters*, where threshold robustness matters more than
matching a charting platform to the decimal.
"""

from __future__ import annotations

import math
from collections import deque


class IncrementalEma:
    def __init__(self, period: int) -> None:
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value: float | None = None
        self.count = 0

    def update(self, x: float) -> float:
        self.count += 1
        if self.value is None:
            self.value = x
        else:
            self.value += self.alpha * (x - self.value)
        return self.value

    @property
    def ready(self) -> bool:
        return self.count >= self.period


class IncrementalAtr:
    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.value: float | None = None
        self._prev_close: float | None = None
        self.count = 0

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close),
                     abs(low - self._prev_close))
        self._prev_close = close
        self.count += 1
        if self.value is None:
            self.value = tr
        else:
            self.value += (tr - self.value) / self.period
        return self.value

    @property
    def ready(self) -> bool:
        return self.count >= self.period


class IncrementalAdx:
    """Wilder-style ADX with +DI/-DI, updated incrementally."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._ph: float | None = None
        self._pl: float | None = None
        self._pc: float | None = None
        self._atr: float | None = None
        self._pdm: float | None = None
        self._ndm: float | None = None
        self.adx: float | None = None
        self.count = 0

    def update(self, high: float, low: float, close: float) -> float | None:
        if self._ph is None:
            self._ph, self._pl, self._pc = high, low, close
            self.count = 1
            return None

        up_move = high - self._ph
        down_move = self._pl - low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr = max(high - low, abs(high - self._pc), abs(low - self._pc))

        self._atr = tr if self._atr is None else self._atr + (tr - self._atr) / self.period
        self._pdm = plus_dm if self._pdm is None else self._pdm + (plus_dm - self._pdm) / self.period
        self._ndm = minus_dm if self._ndm is None else self._ndm + (minus_dm - self._ndm) / self.period

        self._ph, self._pl, self._pc = high, low, close
        self.count += 1

        if not self._atr:
            return self.adx
        pdi = 100.0 * self._pdm / self._atr
        ndi = 100.0 * self._ndm / self._atr
        denom = pdi + ndi
        dx = 100.0 * abs(pdi - ndi) / denom if denom > 0 else 0.0
        self.adx = dx if self.adx is None else self.adx + (dx - self.adx) / self.period
        return self.adx

    @property
    def ready(self) -> bool:
        # ADX needs roughly two smoothing windows to stabilise.
        return self.count >= 2 * self.period


class RollingPercentile:
    """Percentile rank of the most recent value within a rolling window."""

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, x: float) -> float:
        self._buf.append(x)
        return self.percentile()

    def percentile(self) -> float:
        if not self._buf:
            return math.nan
        last = self._buf[-1]
        below = sum(1 for v in self._buf if v <= last)
        return below / len(self._buf)

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window


class IncrementalRsi:
    """Wilder's RSI, updated incrementally."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev: float | None = None
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self.value: float | None = None
        self.count = 0

    def update(self, close: float) -> float | None:
        if self._prev is None:
            self._prev = close
            self.count = 1
            return None
        change = close - self._prev
        self._prev = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if self._avg_gain is None:
            self._avg_gain, self._avg_loss = gain, loss
        else:
            self._avg_gain += (gain - self._avg_gain) / self.period
            self._avg_loss += (loss - self._avg_loss) / self.period
        self.count += 1
        if self._avg_loss == 0:
            self.value = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = 100.0 - 100.0 / (1.0 + rs)
        return self.value

    @property
    def ready(self) -> bool:
        return self.count >= self.period + 1


class RollingMeanStd:
    """Rolling mean and population std over a window (for Bollinger bands)."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, x: float) -> tuple[float, float]:
        self._buf.append(x)
        return self.stats()

    def stats(self) -> tuple[float, float]:
        n = len(self._buf)
        if n == 0:
            return (math.nan, math.nan)
        mean = sum(self._buf) / n
        var = sum((v - mean) ** 2 for v in self._buf) / n
        return (mean, math.sqrt(var))

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window
