"""EMA-cross entry with composable entry filters.

The entry logic is UNCHANGED from the benchmark EMA cross (long on bullish
cross, flat on bearish cross). The only addition is a list of
:class:`EntryFilter` objects that can *block* a long entry when conditions are
poor. Exits are never blocked.

This is the vehicle for the "when should it NOT trade?" research: keep the
entry fixed, vary the filters, and measure whether excluding bad-condition
trades beats the unfiltered baseline after costs.
"""

from __future__ import annotations

from nifty_quant.backtest.filters import EntryFilter, FilterContext
from nifty_quant.backtest.incremental import (
    IncrementalAdx,
    IncrementalAtr,
    IncrementalEma,
    RollingPercentile,
)
from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.strategy import Strategy, StrategyMetadata
from nifty_quant.feed.events import CandleEvent


class FilteredEmaStrategy(Strategy):
    def __init__(
        self,
        instrument: Instrument,
        *,
        fast: int = 20,
        slow: int = 50,
        quantity: int = 1,
        filters: list[EntryFilter] | None = None,
        adx_period: int = 14,
        atr_period: int = 14,
        atr_pct_window: int = 100,
    ) -> None:
        super().__init__()
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.instrument = instrument
        self.fast = fast
        self.slow = slow
        self.quantity = quantity
        self.filters = list(filters or [])

        self._ema_fast = IncrementalEma(fast)
        self._ema_slow = IncrementalEma(slow)
        self._adx = IncrementalAdx(adx_period)
        self._atr = IncrementalAtr(atr_period)
        self._atr_pct = RollingPercentile(atr_pct_window)
        self._prev_diff: float | None = None
        self._long = False

    @property
    def name(self) -> str:
        if not self.filters:
            return f"EMA({self.fast}/{self.slow})"
        tags = "+".join(f.label for f in self.filters)
        return f"EMA({self.fast}/{self.slow})[{tags}]"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FilteredEma",
            version="1.0.0",
            author="research",
            parameters={
                "fast": self.fast,
                "slow": self.slow,
                "filters": [f.label for f in self.filters],
            },
        )

    def on_candle(self, event: CandleEvent) -> None:
        if event.symbol != self.instrument.symbol:
            return
        c = event.candle

        fast = self._ema_fast.update(c.close)
        slow = self._ema_slow.update(c.close)
        adx = self._adx.update(c.high, c.low, c.close)
        atr = self._atr.update(c.high, c.low, c.close)
        atr_pct = self._atr_pct.update(atr)

        if not (self._ema_fast.ready and self._ema_slow.ready):
            self._prev_diff = fast - slow
            return

        diff = fast - slow
        prev = self._prev_diff
        self._prev_diff = diff
        if prev is None:
            return

        ctx = FilterContext(
            timestamp=event.timestamp,
            indicators={
                "adx": adx if self._adx.ready else None,
                "atr": atr,
                "atr_pct": atr_pct if self._atr_pct.ready else None,
            },
        )

        if prev <= 0 < diff and not self._long:
            if all(f.allows(ctx) for f in self.filters):
                self._long = True
                self.emit(Signal(
                    event.timestamp, self.instrument, SignalAction.BUY,
                    reason=f"bullish cross | {self.name}", quantity=self.quantity,
                ))
        elif prev >= 0 > diff and self._long:
            self._long = False
            self.emit(Signal(
                event.timestamp, self.instrument, SignalAction.EXIT,
                reason="bearish cross",
            ))
