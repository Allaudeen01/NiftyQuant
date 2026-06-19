"""Parameter sweeps.

Run a strategy across a grid of parameters and compare results. Each run gets a
fresh portfolio/broker/feed so runs are independent and reproducible.

``feed_factory`` must return a *fresh* feed per call (feeds are consumed once).
Component factories let callers control starting cash, slippage model, feature
engine, etc., while keeping each run isolated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from nifty_quant.backtest.broker import SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.metrics import PerformanceMetrics
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine, RiskEngine
from nifty_quant.backtest.strategy import Strategy
from nifty_quant.features.engine import FeatureEngine
from nifty_quant.features.store import FeatureStore
from nifty_quant.feed.base import MarketFeed
from nifty_quant.log import get_logger

_log = get_logger("research.sweep")

StrategyFactory = Callable[..., Strategy]
FeedFactory = Callable[[], MarketFeed]


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a parameter grid into a list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in combos]


@dataclass
class SweepResult:
    params: dict
    metrics: PerformanceMetrics
    strategy_name: str

    def row(self) -> dict:
        return {
            "strategy": self.strategy_name,
            **self.params,
            **self.metrics.as_dict(),
        }


@dataclass
class SweepReport:
    results: list[SweepResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.row() for r in self.results])

    def best(self, metric: str = "sharpe", *, maximize: bool = True) -> SweepResult:
        """Best result by a PerformanceMetrics attribute, ignoring NaNs."""
        import math

        candidates = [
            r for r in self.results
            if not math.isnan(getattr(r.metrics, metric, float("nan")))
        ]
        if not candidates:
            raise ValueError(f"no runs with a finite '{metric}'")
        key = lambda r: getattr(r.metrics, metric)
        return max(candidates, key=key) if maximize else min(candidates, key=key)


def run_sweep(
    strategy_factory: StrategyFactory,
    param_grid: dict[str, list],
    feed_factory: FeedFactory,
    *,
    portfolio_factory: Callable[[], Portfolio] = Portfolio,
    risk_factory: Callable[[], RiskEngine] = BasicRiskEngine,
    broker_factory: Callable[[], SimulatedBroker] = SimulatedBroker,
    feature_engine_factory: Callable[[], FeatureEngine] | None = None,
    feature_store_factory: Callable[[], FeatureStore] | None = None,
) -> SweepReport:
    """Run ``strategy_factory(**params)`` across the grid; return a report."""
    report = SweepReport()
    for params in expand_grid(param_grid):
        strategy = strategy_factory(**params)
        engine = BacktestEngine(
            strategy,
            portfolio=portfolio_factory(),
            risk_engine=risk_factory(),
            broker=broker_factory(),
            feature_engine=(
                feature_engine_factory() if feature_engine_factory else None
            ),
            feature_store=(
                feature_store_factory() if feature_store_factory else None
            ),
        )
        result = engine.run(feed_factory())
        report.results.append(
            SweepResult(
                params=params,
                metrics=result.metrics,
                strategy_name=strategy.name,
            )
        )
        _log.event(
            "sweep_run_complete",
            params=params,
            sharpe=result.metrics.sharpe,
            total_return=result.metrics.total_return,
            trades=result.metrics.num_trades,
        )
    return report
