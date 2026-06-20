"""Provider-agnostic market-data and broker interfaces.

Every concrete source (Groww, Zerodha Kite, Dhan, Upstox, or a CSV/parquet
backtest feed) implements one of these. The rest of the system depends only on
these interfaces, so swapping brokers never touches analytics, strategy, or
backtest code.

Two layers, deliberately separated:

- :class:`MarketDataProvider` -- read-only market data. A backtest CSV feed
  only needs this; it can never place an order.
- :class:`BrokerProvider`     -- adds live execution (orders, positions). Only
  real brokers implement this, and execution is expected to be gated behind a
  risk engine and an explicit live-trading switch.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from enum import Enum

from nifty_quant.data.models import OHLCVSeries, OptionChain


class MarketDataProvider(abc.ABC):
    """Abstract base class for all read-only market-data sources."""

    @abc.abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> OHLCVSeries:
        """Return historical OHLCV candles for ``symbol`` over [start, end]."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_option_chain(
        self,
        underlying: str,
        expiry: date,
    ) -> OptionChain:
        """Return the current option-chain snapshot for one expiry."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_spot(self, symbol: str) -> float:
        """Return the latest spot/last price for ``symbol``."""
        raise NotImplementedError


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderKind(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class OrderRequest:
    """A broker-agnostic order instruction.

    Concrete brokers translate this into their own enums/fields. Keeping it
    minimal and explicit makes the risk engine easy to reason about.
    """

    trading_symbol: str
    side: OrderSide
    quantity: int
    kind: OrderKind = OrderKind.MARKET
    price: float | None = None          # required for LIMIT
    trigger_price: float | None = None
    reference_id: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.kind is OrderKind.LIMIT and self.price is None:
            raise ValueError("LIMIT orders require a price")


@dataclass(frozen=True)
class OrderResult:
    """Normalised result of an order placement."""

    broker_order_id: str
    status: str
    raw: dict


class BrokerProvider(MarketDataProvider):
    """Market data + live execution. Implemented only by real brokers."""

    @abc.abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order. Implementations MUST honour a live-trading gate."""
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a working order by its broker id."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_orderbook(self) -> list[dict]:
        """Return current working/historical orders for the session."""
        raise NotImplementedError
