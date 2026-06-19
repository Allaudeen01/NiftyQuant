"""Tradable instrument descriptor.

A single type identifies what is being traded and how to price/key it. The
``key`` is used both as the portfolio position id and as the broker's price
lookup key, so marking-to-market and fills line up automatically.

Keying convention:
- Options:   ``SYMBOL|OPT|YYYY-MM-DD|STRIKE|CE/PE``
- Otherwise: ``SYMBOL``   (matches the symbol on a CandleEvent for easy pricing)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from nifty_quant.data.models import OptionType


class InstrumentType(str, Enum):
    INDEX = "INDEX"
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


@dataclass(frozen=True)
class Instrument:
    """Identifies a tradable contract.

    ``multiplier`` scales PnL (e.g. an option lot's contract multiplier). For
    an index/equity traded in single units it stays 1.
    """

    symbol: str
    instrument_type: InstrumentType
    expiry: date | None = None
    strike: float | None = None
    option_type: OptionType | None = None
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.instrument_type is InstrumentType.OPTION:
            if self.strike is None or self.option_type is None or self.expiry is None:
                raise ValueError(
                    "OPTION instruments require strike, option_type and expiry"
                )
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")

    @property
    def key(self) -> str:
        if self.instrument_type is InstrumentType.OPTION:
            return (
                f"{self.symbol}|OPT|{self.expiry.isoformat()}"
                f"|{self.strike:g}|{self.option_type.value}"
            )
        return self.symbol

    @property
    def is_option(self) -> bool:
        return self.instrument_type is InstrumentType.OPTION

    def describe(self) -> str:
        if self.is_option:
            return (
                f"{self.symbol} {self.strike:g}{self.option_type.value} "
                f"exp {self.expiry.isoformat()}"
            )
        return f"{self.symbol} ({self.instrument_type.value})"
