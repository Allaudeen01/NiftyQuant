"""Portfolio: cash, positions, realised/unrealised PnL, exposure.

All position accounting lives here (not in the broker) using a signed
average-cost method that handles longs, shorts, scaling in/out, and reversals.
A closed (or reduced) position produces a :class:`Trade` record with entry/exit
prices, PnL, and hold time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from nifty_quant.backtest.instrument import Instrument
from nifty_quant.backtest.intents import Fill


@dataclass
class Position:
    """A signed position: quantity > 0 long, < 0 short."""

    instrument: Instrument
    quantity: int = 0
    avg_price: float = 0.0
    opened_at: datetime | None = None

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def market_value(self, price: float) -> float:
        return self.quantity * price * self.instrument.multiplier

    def unrealized(self, price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return (price - self.avg_price) * self.quantity * self.instrument.multiplier


@dataclass
class Trade:
    """A realised (closed or partially-closed) round-trip slice."""

    instrument_key: str
    direction: str           # "LONG" or "SHORT"
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float = 0.0

    @property
    def hold_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


@dataclass
class Portfolio:
    """Tracks cash and positions; applies fills; reports equity."""

    starting_cash: float = 1_000_000.0
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.starting_cash

    # --- mutation -----------------------------------------------------------

    def apply_fill(self, fill: Fill) -> list[tuple[str, dict]]:
        """Apply a fill. Returns a list of (event_name, payload) for journaling.

        Event names: 'position_opened', 'position_closed'.
        """
        events: list[tuple[str, dict]] = []
        inst = fill.instrument
        key = inst.key
        mult = inst.multiplier
        pos = self.positions.get(key) or Position(instrument=inst)

        signed = fill.signed_quantity
        price = fill.price

        # Cash: buying reduces cash, selling increases it. Fees always reduce.
        self.cash -= signed * price * mult
        self.cash -= fill.fees

        if pos.quantity == 0 or _same_sign(pos.quantity, signed):
            # Opening or adding to a position -> update weighted average.
            was_flat = pos.quantity == 0
            total_qty = pos.quantity + signed
            pos.avg_price = (
                (pos.avg_price * abs(pos.quantity) + price * abs(signed))
                / abs(total_qty)
            )
            pos.quantity = total_qty
            if was_flat:
                pos.opened_at = fill.timestamp
                events.append(
                    ("position_opened", _open_payload(pos, price, fill.timestamp))
                )
        else:
            # Reducing, closing, or reversing.
            closing = min(abs(signed), abs(pos.quantity))
            direction = "LONG" if pos.quantity > 0 else "SHORT"
            pnl = (price - pos.avg_price) * _sign(pos.quantity) * closing * mult
            self.realized_pnl += pnl
            self.trades.append(
                Trade(
                    instrument_key=key,
                    direction=direction,
                    quantity=closing,
                    entry_price=pos.avg_price,
                    exit_price=price,
                    entry_time=pos.opened_at or fill.timestamp,
                    exit_time=fill.timestamp,
                    pnl=pnl,
                    fees=fill.fees,
                )
            )
            remaining = pos.quantity + signed  # signed opposes pos here
            pos.quantity = remaining
            events.append(
                ("position_closed", {
                    "instrument": key,
                    "direction": direction,
                    "quantity": closing,
                    "pnl": pnl,
                })
            )
            if remaining == 0:
                pos.opened_at = None
                pos.avg_price = 0.0
            elif _same_sign(remaining, signed):
                # Reversal: the order more than closed the position, the
                # leftover opens a fresh position (opposite side) at fill price.
                pos.avg_price = price
                pos.opened_at = fill.timestamp
                events.append(
                    ("position_opened", _open_payload(pos, price, fill.timestamp))
                )
            # else: simple partial reduction -> avg_price/opened_at unchanged.

        if pos.quantity == 0:
            self.positions.pop(key, None)
        else:
            self.positions[key] = pos
        return events

    # --- valuation ----------------------------------------------------------

    def equity(self, prices: dict[str, float]) -> float:
        """Total equity = cash + marked value of all open positions."""
        total = self.cash
        for key, pos in self.positions.items():
            price = prices.get(key, pos.avg_price)
            total += pos.market_value(price)
        return total

    def unrealized(self, prices: dict[str, float]) -> float:
        return sum(
            pos.unrealized(prices.get(key, pos.avg_price))
            for key, pos in self.positions.items()
        )

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(
            abs(pos.market_value(prices.get(key, pos.avg_price)))
            for key, pos in self.positions.items()
        )


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _same_sign(a: float, b: float) -> bool:
    return _sign(a) == _sign(b) and a != 0 and b != 0


def _open_payload(pos: Position, price: float, ts: datetime) -> dict:
    return {
        "instrument": pos.instrument.key,
        "quantity": pos.quantity,
        "avg_price": pos.avg_price,
        "opened_at": ts.isoformat(),
    }
