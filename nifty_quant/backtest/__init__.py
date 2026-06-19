"""Modular backtesting framework.

Pipeline (every box is replaceable and communicates through the event system):

    ReplayFeed -> Strategy -> Signal -> TradeIntent -> RiskEngine
               -> ApprovedOrder -> SimulatedBroker -> Fill -> Portfolio -> Metrics

The same :class:`~nifty_quant.backtest.strategy.Strategy` interface attaches to
a ReplayFeed today and a PaperFeed/LiveFeed later without modification, because
all of them speak the :mod:`nifty_quant.feed` event protocol.
"""

from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.intents import TradeIntent, ApprovedOrder, Fill
from nifty_quant.backtest.strategy import Strategy, StrategyMetadata
from nifty_quant.backtest.risk import RiskEngine, BasicRiskEngine, RiskDecision
from nifty_quant.backtest.broker import (
    SimulatedBroker,
    Quote,
    FillModel,
    MidPriceFill,
    BidAskFill,
    FixedSlippage,
    PercentSlippage,
)
from nifty_quant.backtest.portfolio import Portfolio, Position, Trade
from nifty_quant.backtest.journal import Journal, JournalRecord, EventType
from nifty_quant.backtest.engine import BacktestEngine, BacktestResult

__all__ = [
    "Instrument",
    "InstrumentType",
    "Signal",
    "SignalAction",
    "TradeIntent",
    "ApprovedOrder",
    "Fill",
    "Strategy",
    "StrategyMetadata",
    "RiskEngine",
    "BasicRiskEngine",
    "RiskDecision",
    "SimulatedBroker",
    "Quote",
    "FillModel",
    "MidPriceFill",
    "BidAskFill",
    "FixedSlippage",
    "PercentSlippage",
    "Portfolio",
    "Position",
    "Trade",
    "Journal",
    "JournalRecord",
    "EventType",
    "BacktestEngine",
    "BacktestResult",
]
