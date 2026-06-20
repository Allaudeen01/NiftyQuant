"""Benchmark strategies (the baseline any real strategy must beat after costs).

Ten simple, transparent strategies computed with incremental state (O(1) or
O(window) per bar) so they run fast over large intraday datasets without the
FeatureEngine. All are long/flat except AlwaysShort. They are baselines, NOT
recommendations.

Trading model: each emits BUY to open a long, EXIT to flatten, SELL to open a
short. The engine's risk engine sizes positions.

Note on VWAP: index candles carry zero volume, so a true volume weighting is
impossible here; VwapTrend falls back to a session-anchored mean of the typical
price. This is flagged in its metadata.
"""

from __future__ import annotations

import random
from collections import deque

from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.strategy import Strategy, StrategyMetadata
from nifty_quant.feed.events import CandleEvent


class _Benchmark(Strategy):
    """Shared base: single-instrument, tracks flat/long/short state."""

    def __init__(self, instrument: Instrument, *, quantity: int = 1) -> None:
        super().__init__()
        self.instrument = instrument
        self.quantity = quantity
        self._state = "flat"  # flat | long | short

    def _go_long(self, ts, reason: str) -> None:
        if self._state != "long":
            self._state = "long"
            self.emit(Signal(ts, self.instrument, SignalAction.BUY,
                             reason=reason, quantity=self.quantity))

    def _go_short(self, ts, reason: str) -> None:
        if self._state != "short":
            self._state = "short"
            self.emit(Signal(ts, self.instrument, SignalAction.SELL,
                             reason=reason, quantity=self.quantity))

    def _flatten(self, ts, reason: str) -> None:
        if self._state != "flat":
            self._state = "flat"
            self.emit(Signal(ts, self.instrument, SignalAction.EXIT, reason=reason))

    def _accepts(self, event: CandleEvent) -> bool:
        return event.symbol == self.instrument.symbol

    def _meta(self, name: str, params: dict) -> StrategyMetadata:
        return StrategyMetadata(name=name, version="1.0.0",
                                author="benchmark", parameters=params)


# --- 1. Buy & Hold ---------------------------------------------------------


class BuyAndHold(_Benchmark):
    @property
    def name(self) -> str:
        return "BuyAndHold"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("BuyAndHold", {"quantity": self.quantity})

    def on_candle(self, event: CandleEvent) -> None:
        if self._accepts(event):
            self._go_long(event.timestamp, "buy and hold")


# --- 2. Always Long --------------------------------------------------------


class AlwaysLong(_Benchmark):
    @property
    def name(self) -> str:
        return "AlwaysLong"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("AlwaysLong", {"quantity": self.quantity})

    def on_candle(self, event: CandleEvent) -> None:
        if self._accepts(event):
            self._go_long(event.timestamp, "always long")


# --- 3. Always Short -------------------------------------------------------


class AlwaysShort(_Benchmark):
    @property
    def name(self) -> str:
        return "AlwaysShort"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("AlwaysShort", {"quantity": self.quantity})

    def on_candle(self, event: CandleEvent) -> None:
        if self._accepts(event):
            self._go_short(event.timestamp, "always short")


# --- 4. Random Entry -------------------------------------------------------


class RandomEntry(_Benchmark):
    def __init__(self, instrument, *, quantity=1, prob=0.02, seed=42):
        super().__init__(instrument, quantity=quantity)
        self.prob = prob
        self._rng = random.Random(seed)
        self._seed = seed

    @property
    def name(self) -> str:
        return "RandomEntry"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("RandomEntry",
                          {"quantity": self.quantity, "prob": self.prob,
                           "seed": self._seed})

    def on_candle(self, event: CandleEvent) -> None:
        if not self._accepts(event):
            return
        if self._rng.random() < self.prob:
            if self._state == "long":
                self._flatten(event.timestamp, "random exit")
            else:
                self._go_long(event.timestamp, "random entry")


# --- 5 & 6. EMA / SMA cross ------------------------------------------------


class _CrossBase(_Benchmark):
    def __init__(self, instrument, *, fast, slow, quantity=1):
        super().__init__(instrument, quantity=quantity)
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self._prev_diff = None

    def _update_fast_slow(self, price: float) -> tuple[float, float] | None:
        raise NotImplementedError

    def on_candle(self, event: CandleEvent) -> None:
        if not self._accepts(event):
            return
        vals = self._update_fast_slow(event.candle.close)
        if vals is None:
            return
        fast, slow = vals
        diff = fast - slow
        prev = self._prev_diff
        self._prev_diff = diff
        if prev is None:
            return
        if prev <= 0 < diff:
            self._go_long(event.timestamp, f"{self.name} bullish cross")
        elif prev >= 0 > diff:
            self._flatten(event.timestamp, f"{self.name} bearish cross")


class EmaCross(_CrossBase):
    def __init__(self, instrument, *, fast=20, slow=50, quantity=1):
        super().__init__(instrument, fast=fast, slow=slow, quantity=quantity)
        self._af = 2.0 / (fast + 1)
        self._as = 2.0 / (slow + 1)
        self._ef = None
        self._es = None
        self._count = 0

    @property
    def name(self) -> str:
        return f"EmaCross({self.fast}/{self.slow})"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("EmaCross", {"fast": self.fast, "slow": self.slow})

    def _update_fast_slow(self, price):
        self._count += 1
        if self._ef is None:
            self._ef = self._es = price
        else:
            self._ef += self._af * (price - self._ef)
            self._es += self._as * (price - self._es)
        return (self._ef, self._es) if self._count >= self.slow else None


class SmaCross(_CrossBase):
    def __init__(self, instrument, *, fast=20, slow=50, quantity=1):
        super().__init__(instrument, fast=fast, slow=slow, quantity=quantity)
        self._fbuf = deque(maxlen=fast)
        self._sbuf = deque(maxlen=slow)

    @property
    def name(self) -> str:
        return f"SmaCross({self.fast}/{self.slow})"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("SmaCross", {"fast": self.fast, "slow": self.slow})

    def _update_fast_slow(self, price):
        self._fbuf.append(price)
        self._sbuf.append(price)
        if len(self._sbuf) < self.slow:
            return None
        return (sum(self._fbuf) / len(self._fbuf),
                sum(self._sbuf) / len(self._sbuf))


# --- 7. VWAP trend (session-anchored; volume-less fallback) ----------------


class VwapTrend(_Benchmark):
    def __init__(self, instrument, *, quantity=1):
        super().__init__(instrument, quantity=quantity)
        self._day = None
        self._cum_pv = 0.0
        self._cum_vol = 0.0
        self._cum_tp = 0.0
        self._n = 0

    @property
    def name(self) -> str:
        return "VwapTrend"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("VwapTrend",
                          {"note": "session-anchored; falls back to mean "
                                   "typical price when volume is zero"})

    def on_candle(self, event: CandleEvent) -> None:
        if not self._accepts(event):
            return
        c = event.candle
        day = event.timestamp.date()
        if day != self._day:
            self._day = day
            self._cum_pv = self._cum_vol = self._cum_tp = 0.0
            self._n = 0
        typical = (c.high + c.low + c.close) / 3.0
        self._cum_pv += typical * c.volume
        self._cum_vol += c.volume
        self._cum_tp += typical
        self._n += 1
        vwap = (self._cum_pv / self._cum_vol if self._cum_vol > 0
                else self._cum_tp / self._n)
        if c.close > vwap:
            self._go_long(event.timestamp, "close above VWAP")
        else:
            self._flatten(event.timestamp, "close below VWAP")


# --- 8. SuperTrend ---------------------------------------------------------


class SuperTrend(_Benchmark):
    def __init__(self, instrument, *, period=10, multiplier=3.0, quantity=1):
        super().__init__(instrument, quantity=quantity)
        self.period = period
        self.multiplier = multiplier
        self._prev_close = None
        self._atr = None
        self._final_upper = None
        self._final_lower = None
        self._trend = None
        self._count = 0

    @property
    def name(self) -> str:
        return f"SuperTrend({self.period},{self.multiplier:g})"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("SuperTrend",
                          {"period": self.period, "multiplier": self.multiplier})

    def on_candle(self, event: CandleEvent) -> None:
        if not self._accepts(event):
            return
        c = event.candle
        self._count += 1
        if self._prev_close is None:
            tr = c.high - c.low
        else:
            tr = max(c.high - c.low, abs(c.high - self._prev_close),
                     abs(c.low - self._prev_close))
        self._atr = tr if self._atr is None else self._atr + (tr - self._atr) / self.period
        self._prev_close = c.close
        if self._count < self.period:
            return

        hl2 = (c.high + c.low) / 2.0
        ub = hl2 + self.multiplier * self._atr
        lb = hl2 - self.multiplier * self._atr
        if self._final_upper is None:
            self._final_upper, self._final_lower = ub, lb
            self._trend = 1 if c.close >= lb else -1
        else:
            self._final_upper = ub if (ub < self._final_upper or c.close > self._final_upper) else self._final_upper
            self._final_lower = lb if (lb > self._final_lower or c.close < self._final_lower) else self._final_lower
            if self._trend == 1:
                self._trend = -1 if c.close < self._final_lower else 1
            else:
                self._trend = 1 if c.close > self._final_upper else -1

        if self._trend == 1:
            self._go_long(event.timestamp, "supertrend up")
        else:
            self._flatten(event.timestamp, "supertrend down")


# --- 9. MACD cross ---------------------------------------------------------


class MacdCross(_Benchmark):
    def __init__(self, instrument, *, fast=12, slow=26, signal=9, quantity=1):
        super().__init__(instrument, quantity=quantity)
        self.fast, self.slow, self.signal = fast, slow, signal
        self._af = 2.0 / (fast + 1)
        self._as = 2.0 / (slow + 1)
        self._asig = 2.0 / (signal + 1)
        self._ef = self._es = self._sig = None
        self._count = 0
        self._prev_hist = None

    @property
    def name(self) -> str:
        return f"MacdCross({self.fast}/{self.slow}/{self.signal})"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("MacdCross",
                          {"fast": self.fast, "slow": self.slow,
                           "signal": self.signal})

    def on_candle(self, event: CandleEvent) -> None:
        if not self._accepts(event):
            return
        price = event.candle.close
        self._count += 1
        if self._ef is None:
            self._ef = self._es = price
        else:
            self._ef += self._af * (price - self._ef)
            self._es += self._as * (price - self._es)
        macd = self._ef - self._es
        if self._count < self.slow:
            return
        self._sig = macd if self._sig is None else self._sig + self._asig * (macd - self._sig)
        hist = macd - self._sig
        prev = self._prev_hist
        self._prev_hist = hist
        if prev is None:
            return
        if prev <= 0 < hist:
            self._go_long(event.timestamp, "macd bullish")
        elif prev >= 0 > hist:
            self._flatten(event.timestamp, "macd bearish")


# --- 10. Donchian breakout -------------------------------------------------


class DonchianBreakout(_Benchmark):
    def __init__(self, instrument, *, period=20, quantity=1):
        super().__init__(instrument, quantity=quantity)
        self.period = period
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)

    @property
    def name(self) -> str:
        return f"DonchianBreakout({self.period})"

    @property
    def metadata(self) -> StrategyMetadata:
        return self._meta("DonchianBreakout", {"period": self.period})

    def on_candle(self, event: CandleEvent) -> None:
        if not self._accepts(event):
            return
        c = event.candle
        if len(self._highs) == self.period:
            upper = max(self._highs)
            lower = min(self._lows)
            if c.close > upper:
                self._go_long(event.timestamp, "donchian breakout up")
            elif c.close < lower:
                self._flatten(event.timestamp, "donchian breakdown")
        # Append AFTER testing so the channel uses only prior bars.
        self._highs.append(c.high)
        self._lows.append(c.low)


# Registry of benchmark factories: name -> callable(instrument, quantity).
BENCHMARKS: dict[str, type] = {
    "BuyAndHold": BuyAndHold,
    "AlwaysLong": AlwaysLong,
    "AlwaysShort": AlwaysShort,
    "RandomEntry": RandomEntry,
    "EmaCross": EmaCross,
    "SmaCross": SmaCross,
    "VwapTrend": VwapTrend,
    "SuperTrend": SuperTrend,
    "MacdCross": MacdCross,
    "DonchianBreakout": DonchianBreakout,
}
