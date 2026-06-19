"""Backtest engine: orchestrates the full pipeline over an event feed.

The engine is itself a :class:`MarketEventHandler`. It subscribes to any
:class:`~nifty_quant.feed.base.MarketFeed` (ReplayFeed now; PaperFeed/LiveFeed
later) and, for each event:

    1. updates the broker's market view and marks the portfolio,
    2. feeds the event to the strategy,
    3. turns each emitted signal into an intent, runs it through the risk
       engine, executes approved orders on the broker, and applies fills to the
       portfolio,
    4. journals every step (event sourcing) and records an equity point.

Because the strategy and these stages all speak the event protocol, the exact
same setup runs unchanged against a live feed later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from nifty_quant.backtest.broker import SimulatedBroker
from nifty_quant.backtest.intents import TradeIntent
from nifty_quant.backtest.journal import EventType, Journal
from nifty_quant.backtest.metrics import (
    PerformanceMetrics,
    compute_metrics,
    drawdown_curve,
)
from nifty_quant.backtest.portfolio import Portfolio, Trade
from nifty_quant.backtest.risk import BasicRiskEngine, RiskEngine
from nifty_quant.backtest.signals import Signal, SignalAction
from nifty_quant.backtest.strategy import Strategy
from nifty_quant.data.providers.base import OrderSide
from nifty_quant.feed.base import MarketEventHandler, MarketFeed
from nifty_quant.feed.events import CandleEvent, MarketEvent, OptionChainEvent
from nifty_quant.features.engine import FeatureEngine
from nifty_quant.features.store import FeatureStore
from nifty_quant.log import get_logger

_log = get_logger("backtest.engine")


class BacktestEngine(MarketEventHandler):
    """Drives strategy -> risk -> broker -> portfolio over a feed."""

    def __init__(
        self,
        strategy: Strategy,
        *,
        portfolio: Portfolio | None = None,
        risk_engine: RiskEngine | None = None,
        broker: SimulatedBroker | None = None,
        journal: Journal | None = None,
        feature_engine: FeatureEngine | None = None,
        feature_store: FeatureStore | None = None,
        record_market_events: bool = False,
    ) -> None:
        self.strategy = strategy
        self.portfolio = portfolio or Portfolio()
        self.risk_engine = risk_engine or BasicRiskEngine()
        self.broker = broker or SimulatedBroker()
        self.journal = journal or Journal()
        self.feature_engine = feature_engine
        self.feature_store = feature_store
        self.record_market_events = record_market_events
        self._equity_times: list[datetime] = []
        self._equity_values: list[float] = []

    # --- event handling -----------------------------------------------------

    def handle(self, event: MarketEvent) -> None:
        self.broker.on_event(event)
        if self.record_market_events:
            self.journal.record(
                event.timestamp,
                EventType.MARKET_EVENT,
                kind=type(event).__name__,
            )

        # Feature computation: option chains update state; candles emit a vector.
        if self.feature_engine is not None:
            if isinstance(event, OptionChainEvent):
                self.feature_engine.on_option_chain(event)
            elif isinstance(event, CandleEvent):
                fv = self.feature_engine.on_candle(event)
                if self.feature_store is not None:
                    self.feature_store.put(fv)
                self.strategy.on_features(fv)

        # Raw-event path (no-op for pure feature strategies).
        self.strategy.handle(event)

        for signal in self.strategy.drain():
            self._process_signal(signal)

        # Record one equity point per event (post-decision).
        prices = self.broker.last_prices()
        self._equity_times.append(event.timestamp)
        self._equity_values.append(self.portfolio.equity(prices))

    def _process_signal(self, signal: Signal) -> None:
        self.journal.record(
            signal.timestamp,
            EventType.SIGNAL_GENERATED,
            strategy=self.strategy.name,
            action=signal.action.value,
            instrument=signal.instrument.key,
            confidence=signal.confidence,
            reason=signal.reason,
        )
        if signal.action is SignalAction.HOLD:
            return

        intent = self._to_intent(signal)
        self.journal.record(
            signal.timestamp,
            EventType.INTENT_CREATED,
            instrument=intent.instrument.key,
            side=intent.side.value,
            flatten=intent.flatten,
            quantity=intent.quantity,
        )

        prices = self.broker.last_prices()
        decision = self.risk_engine.evaluate(intent, self.portfolio, prices)
        if not decision.approved or decision.order is None:
            self.journal.record(
                signal.timestamp,
                EventType.RISK_REJECTED,
                instrument=intent.instrument.key,
                reason=decision.reason,
            )
            return

        order = decision.order
        self.journal.record(
            order.timestamp,
            EventType.RISK_APPROVED,
            instrument=order.instrument.key,
            side=order.side.value,
            quantity=order.quantity,
            reason=decision.reason,
        )
        self.journal.record(
            order.timestamp,
            EventType.ORDER_SUBMITTED,
            instrument=order.instrument.key,
            side=order.side.value,
            quantity=order.quantity,
        )

        fill = self.broker.execute(order)
        if fill is None:
            self.journal.record(
                order.timestamp,
                EventType.ORDER_UNFILLED,
                instrument=order.instrument.key,
            )
            return

        self.journal.record(
            fill.timestamp,
            EventType.ORDER_FILLED,
            instrument=fill.instrument.key,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.price,
            fees=fill.fees,
        )
        for name, payload in self.portfolio.apply_fill(fill):
            evt = (
                EventType.POSITION_OPENED
                if name == "position_opened"
                else EventType.POSITION_CLOSED
            )
            self.journal.record(fill.timestamp, evt, **payload)

    def _to_intent(self, signal: Signal) -> TradeIntent:
        if signal.action is SignalAction.EXIT:
            return TradeIntent(
                timestamp=signal.timestamp,
                instrument=signal.instrument,
                side=OrderSide.SELL,  # placeholder; flatten resolves direction
                quantity=None,
                reason=signal.reason,
                confidence=signal.confidence,
                flatten=True,
            )
        side = OrderSide.BUY if signal.action is SignalAction.BUY else OrderSide.SELL
        return TradeIntent(
            timestamp=signal.timestamp,
            instrument=signal.instrument,
            side=side,
            quantity=signal.quantity,
            reason=signal.reason,
            confidence=signal.confidence,
        )

    # --- run ----------------------------------------------------------------

    def run(self, feed: MarketFeed) -> "BacktestResult":
        """Subscribe to ``feed``, replay it, and return results."""
        feed.subscribe(self)
        _log.event("backtest_started", strategy=self.strategy.name)
        feed.run()
        return self.build_result()

    def build_result(self) -> "BacktestResult":
        """Assemble a result from accumulated equity/trades/journal.

        Usable mid/after a paper session as well as after a backtest run.
        """
        equity = pd.Series(
            self._equity_values,
            index=pd.DatetimeIndex(self._equity_times, name="timestamp"),
            dtype=float,
        )
        # Collapse duplicate timestamps to their last value for clean analytics.
        if not equity.empty:
            equity = equity[~equity.index.duplicated(keep="last")]

        metrics = compute_metrics(equity, self.portfolio.trades)
        _log.event(
            "backtest_finished",
            strategy=self.strategy.name,
            trades=metrics.num_trades,
            total_return=metrics.total_return,
            sharpe=metrics.sharpe,
            max_drawdown=metrics.max_drawdown,
        )
        return BacktestResult(
            strategy_name=self.strategy.name,
            equity_curve=equity,
            trades=list(self.portfolio.trades),
            metrics=metrics,
            journal=self.journal,
        )


@dataclass
class BacktestResult:
    """Outputs of a backtest: equity curve, trades, metrics, and journal."""

    strategy_name: str
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: PerformanceMetrics
    journal: Journal

    def trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "instrument": t.instrument_key,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "pnl": t.pnl,
                    "fees": t.fees,
                    "hold_seconds": t.hold_seconds,
                }
                for t in self.trades
            ]
        )

    def drawdown_curve(self) -> pd.Series:
        return drawdown_curve(self.equity_curve)

    def save(self, directory: str | Path, *, plots: bool = True) -> None:
        """Persist equity curve, trades, metrics, journal, and (optionally) plots."""
        import json

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.equity_curve.to_csv(out / "equity_curve.csv", header=["equity"])
        self.trades_dataframe().to_csv(out / "trades.csv", index=False)
        (out / "metrics.json").write_text(
            json.dumps(self.metrics.as_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        self.journal.save(out / "journal.jsonl")
        if plots:
            self._render_plots(out)

    def _render_plots(self, out: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - optional dependency
            _log.event("plots_skipped_no_matplotlib", level=30)
            return
        if self.equity_curve.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        self.equity_curve.plot(ax=ax, color="#1f77b4")
        ax.set_title(f"Equity Curve — {self.strategy_name}")
        ax.set_ylabel("Equity")
        fig.tight_layout()
        fig.savefig(out / "equity_curve.png", dpi=120)
        plt.close(fig)

        dd = self.drawdown_curve()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(dd.index, dd.values, 0.0, color="#d62728", alpha=0.5)
        ax.set_title(f"Drawdown — {self.strategy_name}")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()
        fig.savefig(out / "drawdown.png", dpi=120)
        plt.close(fig)
