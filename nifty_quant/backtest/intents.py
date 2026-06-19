"""Intent, approved order, and fill models.

The deliberate three-step separation:

    TradeIntent     -- what we'd like to do (from a Signal), unsized or hinted
    ApprovedOrder   -- what the risk engine sanctioned (concrete quantity)
    Fill            -- what the broker actually executed (price, fees, time)

This makes the path from idea to execution fully auditable: every stage is a
distinct, logged object.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from nifty_quant.backtest.instrument import Instrument
from nifty_quant.data.providers.base import OrderKind, OrderSide


@dataclass(frozen=True)
class TradeIntent:
    """A desired trade derived from a signal (quantity may be unset)."""

    timestamp: datetime
    instrument: Instrument
    side: OrderSide
    quantity: int | None = None
    kind: OrderKind = OrderKind.MARKET
    limit_price: float | None = None
    reason: str = ""
    confidence: float = 1.0
    flatten: bool = False   # True => close whatever position exists


@dataclass(frozen=True)
class ApprovedOrder:
    """A risk-approved, fully-sized order ready for execution."""

    timestamp: datetime
    instrument: Instrument
    side: OrderSide
    quantity: int
    kind: OrderKind = OrderKind.MARKET
    limit_price: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("approved order quantity must be positive")


@dataclass(frozen=True)
class Fill:
    """The realised execution of an order."""

    timestamp: datetime
    instrument: Instrument
    side: OrderSide
    quantity: int
    price: float
    fees: float = 0.0

    @property
    def signed_quantity(self) -> int:
        """+qty for a buy, -qty for a sell."""
        return self.quantity if self.side is OrderSide.BUY else -self.quantity
