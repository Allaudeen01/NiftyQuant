"""Reference feature-consuming strategy.

PURPOSE: demonstrate the FeatureVector consumption path end-to-end. NOT a
trading recommendation. Do not trade it.

Logic: go long when the fast EMA is above the slow EMA and RSI is not yet
overbought; flatten when the fast EMA drops below the slow EMA. It reads only
from the :class:`FeatureVector`, never from raw candles -- so it depends on the
FeatureEngine/Store and benefits from identical, versioned inputs.
"""

from __future__ import annotations

from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.strategy import Strategy, StrategyMetadata
from nifty_quant.features.vector import FeatureVector


class FeatureMomentumStrategy(Strategy):
    def __init__(
        self,
        instrument: Instrument,
        *,
        fast: int = 20,
        slow: int = 50,
        rsi_period: int = 14,
        rsi_ceiling: float = 70.0,
        quantity: int = 1,
    ) -> None:
        super().__init__()
        self.instrument = instrument
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_ceiling = rsi_ceiling
        self.quantity = quantity
        self._long = False
        self._fast_key = f"ema_{fast}"
        self._slow_key = f"ema_{slow}"
        self._rsi_key = f"rsi_{rsi_period}"

    @property
    def name(self) -> str:
        return f"FeatureMomentum({self.fast}/{self.slow})"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FeatureMomentum",
            version="1.0.0",
            parameters={
                "fast": self.fast,
                "slow": self.slow,
                "rsi_period": self.rsi_period,
                "rsi_ceiling": self.rsi_ceiling,
                "quantity": self.quantity,
            },
        )

    def on_features(self, features: FeatureVector) -> None:
        if features.symbol != self.instrument.symbol:
            return
        if not features.is_ready(self._fast_key, self._slow_key, self._rsi_key):
            return

        fast = features[self._fast_key]
        slow = features[self._slow_key]
        rsi = features[self._rsi_key]

        bullish = fast > slow and rsi < self.rsi_ceiling
        if bullish and not self._long:
            self._long = True
            self.emit(
                Signal(
                    timestamp=features.timestamp,
                    instrument=self.instrument,
                    action=SignalAction.BUY,
                    confidence=1.0,
                    reason=f"ema{self.fast}>ema{self.slow}, rsi {rsi:.1f}<{self.rsi_ceiling}",
                    quantity=self.quantity,
                )
            )
        elif fast < slow and self._long:
            self._long = False
            self.emit(
                Signal(
                    timestamp=features.timestamp,
                    instrument=self.instrument,
                    action=SignalAction.EXIT,
                    confidence=1.0,
                    reason=f"ema{self.fast}<ema{self.slow}",
                )
            )
