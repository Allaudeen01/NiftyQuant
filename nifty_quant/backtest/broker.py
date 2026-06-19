"""Simulated broker with pluggable fill models.

The broker tracks the latest market quote per instrument (updated from the
event stream) and converts an :class:`ApprovedOrder` into a :class:`Fill` using
a configurable :class:`FillModel`. It deliberately does NOT assume fills at the
last traded price -- that is the single biggest cause of over-optimistic
backtests.

Implements the same conceptual surface a live broker adapter will, so the
engine code is unchanged when execution is later pointed at a real broker.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.intents import ApprovedOrder, Fill
from nifty_quant.data.providers.base import OrderKind, OrderSide
from nifty_quant.feed.events import CandleEvent, MarketEvent, OptionChainEvent
from nifty_quant.log import get_logger

_log = get_logger("backtest.broker")


@dataclass(frozen=True)
class Quote:
    """Latest market view for one instrument."""

    last: float
    bid: float = 0.0
    ask: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


class FillModel(abc.ABC):
    """Decides the execution price for an order given the current quote."""

    @abc.abstractmethod
    def fill_price(self, side: OrderSide, quote: Quote) -> float:
        raise NotImplementedError


class MidPriceFill(FillModel):
    """Fill at the mid price (optimistic but symmetric)."""

    def fill_price(self, side: OrderSide, quote: Quote) -> float:
        return quote.mid


class BidAskFill(FillModel):
    """Buy at the ask, sell at the bid (pays the spread). Falls back to last."""

    def fill_price(self, side: OrderSide, quote: Quote) -> float:
        if side is OrderSide.BUY:
            return quote.ask if quote.ask > 0 else quote.last
        return quote.bid if quote.bid > 0 else quote.last


class FixedSlippage(FillModel):
    """Last price plus a fixed adverse amount per unit."""

    def __init__(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("slippage amount must be >= 0")
        self.amount = amount

    def fill_price(self, side: OrderSide, quote: Quote) -> float:
        sign = 1.0 if side is OrderSide.BUY else -1.0
        return quote.last + sign * self.amount


class PercentSlippage(FillModel):
    """Last price plus an adverse percentage (e.g. 0.001 = 10 bps)."""

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("slippage pct must be >= 0")
        self.pct = pct

    def fill_price(self, side: OrderSide, quote: Quote) -> float:
        sign = 1.0 if side is OrderSide.BUY else -1.0
        return quote.last * (1.0 + sign * self.pct)


class SimulatedBroker:
    """Tracks quotes from the event stream and fills approved orders."""

    def __init__(
        self,
        fill_model: FillModel | None = None,
        *,
        fee_per_order: float = 0.0,
    ) -> None:
        self.fill_model = fill_model or BidAskFill()
        self.fee_per_order = fee_per_order
        self._quotes: dict[str, Quote] = {}

    # --- market state -------------------------------------------------------

    def on_event(self, event: MarketEvent) -> None:
        """Update the latest quote(s) from a market event."""
        if isinstance(event, CandleEvent):
            # Underlying key is just the symbol (see Instrument.key).
            close = event.candle.close
            self._quotes[event.symbol] = Quote(last=close, bid=close, ask=close)
        elif isinstance(event, OptionChainEvent):
            chain = event.chain
            for q in chain.quotes:
                key = (
                    f"{chain.underlying}|OPT|{q.expiry.isoformat()}"
                    f"|{q.strike:g}|{q.option_type.value}"
                )
                self._quotes[key] = Quote(
                    last=q.last_price, bid=q.bid, ask=q.ask
                )

    def quote(self, instrument: Instrument) -> Quote | None:
        return self._quotes.get(instrument.key)

    def last_prices(self) -> dict[str, float]:
        """Latest last-price per instrument key (for portfolio marking)."""
        return {k: q.last for k, q in self._quotes.items()}

    # --- execution ----------------------------------------------------------

    def execute(self, order: ApprovedOrder) -> Fill | None:
        """Attempt to fill an order at the current quote. None if unfillable."""
        quote = self._quotes.get(order.instrument.key)
        if quote is None:
            _log.event(
                "fill_skipped_no_quote",
                level=30,  # WARNING
                instrument=order.instrument.key,
            )
            return None

        price = self.fill_model.fill_price(order.side, quote)

        # Respect a limit price if one was supplied.
        if order.kind is OrderKind.LIMIT and order.limit_price is not None:
            if order.side is OrderSide.BUY and price > order.limit_price:
                return None
            if order.side is OrderSide.SELL and price < order.limit_price:
                return None
            price = order.limit_price

        if price <= 0:
            return None

        return Fill(
            timestamp=order.timestamp,
            instrument=order.instrument,
            side=order.side,
            quantity=order.quantity,
            price=price,
            fees=self.fee_per_order,
        )
