"""Risk engine: the gate between intent and execution.

A strategy only ever *intends*. The risk engine decides whether an intent
becomes an :class:`ApprovedOrder`, and at what size. Keeping this separate and
explicit is what makes the system auditable and is the same component that will
later guard live execution.

:class:`BasicRiskEngine` implements a small but meaningful rule set; richer
engines (per-strategy limits, portfolio VaR, Greeks budgets) plug in via the
same interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from nifty_quant.backtest.intents import ApprovedOrder, TradeIntent
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.data.providers.base import OrderSide


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    order: ApprovedOrder | None
    reason: str


class RiskEngine(abc.ABC):
    """Evaluates a trade intent against current portfolio state."""

    @abc.abstractmethod
    def evaluate(
        self,
        intent: TradeIntent,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> RiskDecision:
        raise NotImplementedError


class BasicRiskEngine(RiskEngine):
    """Position-sizing + simple guardrails.

    Rules:
    - Flatten intents are always approved (closing risk is allowed).
    - Default quantity is used when the intent gives no hint.
    - Per-instrument net quantity is capped at ``max_units_per_instrument``.
    - Shorting can be disabled.
    - Gross exposure is capped at ``max_gross_exposure`` (skipped if None).
    """

    def __init__(
        self,
        *,
        default_quantity: int = 1,
        max_units_per_instrument: int | None = None,
        allow_short: bool = True,
        max_gross_exposure: float | None = None,
    ) -> None:
        if default_quantity <= 0:
            raise ValueError("default_quantity must be positive")
        self.default_quantity = default_quantity
        self.max_units_per_instrument = max_units_per_instrument
        self.allow_short = allow_short
        self.max_gross_exposure = max_gross_exposure

    def evaluate(
        self,
        intent: TradeIntent,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> RiskDecision:
        inst = intent.instrument
        key = inst.key
        pos = portfolio.positions.get(key)

        # --- flatten: approve closing the exact existing quantity ----------
        if intent.flatten:
            if pos is None or pos.quantity == 0:
                return RiskDecision(False, None, "nothing to flatten")
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            order = ApprovedOrder(
                timestamp=intent.timestamp,
                instrument=inst,
                side=side,
                quantity=abs(pos.quantity),
                kind=intent.kind,
                limit_price=intent.limit_price,
                reason=intent.reason or "flatten",
            )
            return RiskDecision(True, order, "flatten approved")

        qty = intent.quantity or self.default_quantity

        if intent.side is OrderSide.SELL and not self.allow_short:
            current = pos.quantity if pos else 0
            # Selling is only allowed up to the long quantity held.
            if current <= 0:
                return RiskDecision(False, None, "short selling disabled")
            qty = min(qty, current)

        # --- per-instrument cap --------------------------------------------
        if self.max_units_per_instrument is not None:
            current = pos.quantity if pos else 0
            signed = qty if intent.side is OrderSide.BUY else -qty
            projected = abs(current + signed)
            if projected > self.max_units_per_instrument:
                room = self.max_units_per_instrument - abs(current)
                if room <= 0:
                    return RiskDecision(
                        False, None, "per-instrument position cap reached"
                    )
                qty = room

        # --- gross exposure cap --------------------------------------------
        if self.max_gross_exposure is not None:
            price = prices.get(key)
            if price is not None:
                add_exposure = qty * price * inst.multiplier
                if (
                    portfolio.gross_exposure(prices) + add_exposure
                    > self.max_gross_exposure
                ):
                    return RiskDecision(
                        False, None, "gross exposure cap exceeded"
                    )

        if qty <= 0:
            return RiskDecision(False, None, "sized quantity is zero")

        order = ApprovedOrder(
            timestamp=intent.timestamp,
            instrument=inst,
            side=intent.side,
            quantity=qty,
            kind=intent.kind,
            limit_price=intent.limit_price,
            reason=intent.reason,
        )
        return RiskDecision(True, order, "approved")
