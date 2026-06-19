"""End-to-end backtest: EMA reference strategy through the full pipeline."""

from datetime import datetime, timedelta

from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.journal import EventType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
from nifty_quant.backtest.strategies.ema_cross import EmaCrossStrategy
from nifty_quant.data.models import Candle
from nifty_quant.feed.base import MarketFeed
from nifty_quant.feed.events import CandleEvent


class ListFeed(MarketFeed):
    """A trivial feed that emits a fixed list of events (test helper)."""

    def __init__(self, events):
        super().__init__()
        self._events = events

    def run(self) -> int:
        for e in self._events:
            self._dispatch(e)
        return len(self._events)


def _candle_events(closes):
    base = datetime(2025, 1, 2, 9, 15)
    events = []
    for i, c in enumerate(closes):
        ts = base + timedelta(minutes=5 * i)
        events.append(
            CandleEvent(
                timestamp=ts,
                symbol="NIFTY",
                timeframe="5m",
                candle=Candle(ts, c, c + 1, c - 1, c, 1000),
            )
        )
    return events


# A path that rises (up-cross) then falls (down-cross).
CLOSES = (
    [100] * 5
    + [101, 103, 106, 110, 115, 120]   # rally -> fast crosses above slow
    + [118, 113, 107, 100, 94, 90]     # selloff -> fast crosses below slow
)


def _engine():
    inst = Instrument("NIFTY", InstrumentType.INDEX)
    strat = EmaCrossStrategy(inst, fast=3, slow=5, quantity=10)
    return BacktestEngine(
        strat,
        portfolio=Portfolio(starting_cash=1_000_000),
        risk_engine=BasicRiskEngine(default_quantity=10),
        broker=SimulatedBroker(fill_model=MidPriceFill()),
    )


def test_backtest_produces_trades_and_curve():
    engine = _engine()
    feed = ListFeed(_candle_events(CLOSES))
    result = engine.run(feed)

    assert len(result.equity_curve) == len(CLOSES)
    assert result.metrics.num_trades >= 1
    # A buy followed by an exit should have occurred.
    fills = result.journal.filter(EventType.ORDER_FILLED)
    assert len(fills) >= 2


def test_journal_event_sequence_present():
    engine = _engine()
    result = engine.run(ListFeed(_candle_events(CLOSES)))
    types = {r.event_type for r in result.journal.records}
    assert EventType.SIGNAL_GENERATED in types
    assert EventType.INTENT_CREATED in types
    assert EventType.RISK_APPROVED in types
    assert EventType.ORDER_FILLED in types
    assert EventType.POSITION_OPENED in types
    assert EventType.POSITION_CLOSED in types


def test_backtest_save_outputs(tmp_path):
    engine = _engine()
    result = engine.run(ListFeed(_candle_events(CLOSES)))
    result.save(tmp_path, plots=True)
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "journal.jsonl").exists()
    # matplotlib is installed in this env, so plots should render.
    assert (tmp_path / "equity_curve.png").exists()
    assert (tmp_path / "drawdown.png").exists()


def test_flat_market_no_trades():
    engine = _engine()
    result = engine.run(ListFeed(_candle_events([100] * 30)))
    assert result.metrics.num_trades == 0
