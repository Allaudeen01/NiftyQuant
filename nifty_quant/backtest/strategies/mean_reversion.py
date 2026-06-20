"""Mean-reversion strategies.

Evidence-driven hypothesis: trend-following lost across the board on choppy,
flat NIFTY 5m data, so the natural counter-hypothesis is mean reversion -- buy
dips, exit on reversion to the mean. Long/flat, OHLCV-only, incremental.

NOT recommendations; candidates to be tested with year-by-year CV.
"""

from __future__ import annotations

from nifty_quant.backtest.incremental import IncrementalRsi, RollingMeanStd
from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.strategy import Strategy, StrategyMetadata
from nifty_quant.feed.events import CandleEvent


class RsiReversion(Strategy):
    """Long when RSI is oversold; exit when it reverts above an exit level."""

    def __init__(self, instrument: Instrument, *, period: int = 14,
                 oversold: float = 30.0, exit_level: float = 50.0,
                 quantity: int = 1) -> None:
        super().__init__()
        self.instrument = instrument
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.quantity = quantity
        self._rsi = IncrementalRsi(period)
        self._long = False

    @property
    def name(self) -> str:
        return f"RsiReversion({self.period},{self.oversold:g}/{self.exit_level:g})"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="RsiReversion", version="1.0.0", author="research",
            parameters={"period": self.period, "oversold": self.oversold,
                        "exit_level": self.exit_level},
        )

    def on_candle(self, event: CandleEvent) -> None:
        if event.symbol != self.instrument.symbol:
            return
        rsi = self._rsi.update(event.candle.close)
        if not self._rsi.ready or rsi is None:
            return
        if rsi < self.oversold and not self._long:
            self._long = True
            self.emit(Signal(event.timestamp, self.instrument, SignalAction.BUY,
                             reason=f"RSI {rsi:.1f} < {self.oversold}",
                             quantity=self.quantity))
        elif rsi > self.exit_level and self._long:
            self._long = False
            self.emit(Signal(event.timestamp, self.instrument, SignalAction.EXIT,
                             reason=f"RSI {rsi:.1f} > {self.exit_level}"))


class BollingerReversion(Strategy):
    """Long when price closes below the lower band; exit on return to the mean."""

    def __init__(self, instrument: Instrument, *, period: int = 20,
                 num_std: float = 2.0, quantity: int = 1) -> None:
        super().__init__()
        self.instrument = instrument
        self.period = period
        self.num_std = num_std
        self.quantity = quantity
        self._ms = RollingMeanStd(period)
        self._long = False

    @property
    def name(self) -> str:
        return f"BollingerReversion({self.period},{self.num_std:g})"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="BollingerReversion", version="1.0.0", author="research",
            parameters={"period": self.period, "num_std": self.num_std},
        )

    def on_candle(self, event: CandleEvent) -> None:
        if event.symbol != self.instrument.symbol:
            return
        close = event.candle.close
        mean, std = self._ms.update(close)
        if not self._ms.ready:
            return
        lower = mean - self.num_std * std
        if close < lower and not self._long:
            self._long = True
            self.emit(Signal(event.timestamp, self.instrument, SignalAction.BUY,
                             reason="close below lower band", quantity=self.quantity))
        elif close >= mean and self._long:
            self._long = False
            self.emit(Signal(event.timestamp, self.instrument, SignalAction.EXIT,
                             reason="reverted to mean"))
