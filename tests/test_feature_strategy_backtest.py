"""End-to-end: a feature-consuming strategy through the engine with a store."""

from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.feature_momentum import FeatureMomentumStrategy
from nifty_quant.features.engine import FeatureConfig, FeatureEngine
from nifty_quant.features.store import InMemoryFeatureStore
from tests.helpers import ListFeed, make_candle_events


CLOSES = (
    [100] * 12
    + [100 + 2 * i for i in range(20)]   # strong uptrend -> long
    + [140 - 2 * i for i in range(20)]   # downtrend -> exit
)


def test_feature_strategy_trades_and_store_populated():
    inst = Instrument("NIFTY", InstrumentType.INDEX)
    strat = FeatureMomentumStrategy(inst, fast=5, slow=10, quantity=10)
    store = InMemoryFeatureStore()
    engine = BacktestEngine(
        strat,
        portfolio=Portfolio(starting_cash=1_000_000),
        risk_engine=BasicRiskEngine(default_quantity=10),
        broker=SimulatedBroker(fill_model=MidPriceFill()),
        feature_engine=FeatureEngine(FeatureConfig(ema_periods=(5, 10), bb_period=5)),
        feature_store=store,
    )
    result = engine.run(ListFeed(make_candle_events(CLOSES)))

    # The strategy should have opened and later exited a long.
    assert result.metrics.num_trades >= 1

    # Every candle should have produced and stored a feature vector.
    from datetime import datetime

    stored = store.get("NIFTY", "v1", datetime(2024, 1, 1), datetime(2026, 1, 1))
    assert len(stored) == len(CLOSES)
    assert "ema_5" in stored[-1].values


def test_strategy_metadata_exposed():
    inst = Instrument("NIFTY", InstrumentType.INDEX)
    strat = FeatureMomentumStrategy(inst, fast=20, slow=50)
    meta = strat.metadata
    assert meta.name == "FeatureMomentum"
    assert meta.parameters["fast"] == 20
    assert meta.as_dict()["version"] == "1.0.0"
