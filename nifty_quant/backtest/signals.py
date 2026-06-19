"""Strategy signals.

A Signal is the strategy's *expressed intent to act* -- "I want to buy this" --
not an order. It is intentionally decoupled from execution: the engine turns a
Signal into a :class:`~nifty_quant.backtest.intents.TradeIntent`, the risk
engine decides whether/how much, and only then does the broker act.

The ``reason`` and ``confidence`` fields are first-class so that later an
LLM/ML layer can populate them and the journal records *why* every trade fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from nifty_quant.backtest.instrument import Instrument


class SignalAction(str, Enum):
    BUY = "BUY"        # open/increase a long
    SELL = "SELL"      # open/increase a short
    EXIT = "EXIT"      # flatten any existing position in this instrument
    HOLD = "HOLD"      # explicit no-op (useful for logging/AI rationale)


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    instrument: Instrument
    action: SignalAction
    confidence: float = 1.0
    reason: str = ""
    quantity: int | None = None      # optional hint; risk engine may override
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity hint must be positive when provided")
