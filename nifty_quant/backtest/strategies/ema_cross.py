"""Reference EMA-crossover strategy.

PURPOSE: exercise the backtest framework end-to-end. This is a textbook example
to validate plumbing -- NOT a recommended or profitable trading strategy. Do
not trade it.

Logic: maintain a fast and slow EMA of candle closes for one symbol. When the
fast EMA crosses above the slow EMA, emit a BUY; when it crosses below, emit an
EXIT (flatten). EMAs are updated incrementally so the strategy works identically
on a streaming live feed.
"""

from __future__ import annotations

from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.strategy import Strategy, StrategyMetadata
from nifty_quant.feed.events import CandleEvent


class EmaCrossStrategy(Strategy):
    def __init__(
        self,
        instrument: Instrument,
        *,
        fast: int = 10,
        slow: int = 20,
        quantity: int = 1,
    ) -> None:
        super().__init__()
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.instrument = instrument
        self.fast = fast
        self.slow = slow
        self.quantity = quantity
        self._alpha_fast = 2.0 / (fast + 1)
        self._alpha_slow = 2.0 / (slow + 1)
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._count = 0
        self._prev_diff: float | None = None
        self._long = False

    @property
    def name(self) -> str:
        return f"EmaCross({self.fast}/{self.slow})"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="EmaCross",
            version="1.0.0",
            author="research",
            parameters={
                "fast": self.fast,
                "slow": self.slow,
                "quantity": self.quantity,
            },
        )

    def on_candle(self, event: CandleEvent) -> None:
        if event.symbol != self.instrument.symbol:
            return
        price = event.candle.close
        self._count += 1

        if self._ema_fast is None:
            self._ema_fast = price
            self._ema_slow = price
        else:
            self._ema_fast += self._alpha_fast * (price - self._ema_fast)
            self._ema_slow += self._alpha_slow * (price - self._ema_slow)

        # Wait for enough data before acting.
        if self._count < self.slow:
            self._prev_diff = self._ema_fast - self._ema_slow
            return

        diff = self._ema_fast - self._ema_slow
        prev = self._prev_diff
        self._prev_diff = diff
        if prev is None:
            return

        crossed_up = prev <= 0 < diff
        crossed_down = prev >= 0 > diff

        if crossed_up and not self._long:
            self._long = True
            self.emit(
                Signal(
                    timestamp=event.timestamp,
                    instrument=self.instrument,
                    action=SignalAction.BUY,
                    confidence=1.0,
                    reason="EMA fast crossed above slow",
                    quantity=self.quantity,
                )
            )
        elif crossed_down and self._long:
            self._long = False
            self.emit(
                Signal(
                    timestamp=event.timestamp,
                    instrument=self.instrument,
                    action=SignalAction.EXIT,
                    confidence=1.0,
                    reason="EMA fast crossed below slow",
                )
            )
