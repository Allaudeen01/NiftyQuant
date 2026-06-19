"""Walk-forward analysis.

Guards against overfitting by selecting parameters on a training window and
evaluating them on the *next, unseen* window. Repeating this across rolling
windows yields an out-of-sample performance profile rather than a single
in-sample-optimised number.

    [ train ][ test ]
            [ train ][ test ]
                    [ train ][ test ]   ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from nifty_quant.backtest.broker import SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.metrics import PerformanceMetrics
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine, RiskEngine
from nifty_quant.features.engine import FeatureEngine
from nifty_quant.features.store import FeatureStore
from nifty_quant.feed.base import MarketFeed
from nifty_quant.research.sweep import StrategyFactory, run_sweep
from nifty_quant.log import get_logger

_log = get_logger("research.walkforward")

RangeFeedFactory = Callable[[datetime, datetime], MarketFeed]


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class WalkForwardResult:
    window: WalkForwardWindow
    best_params: dict
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics


def generate_windows(
    start: datetime,
    end: datetime,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[WalkForwardWindow]:
    """Generate rolling train/test windows over [start, end]."""
    step = timedelta(days=step_days if step_days is not None else test_days)
    train = timedelta(days=train_days)
    test = timedelta(days=test_days)

    windows: list[WalkForwardWindow] = []
    cursor = start
    while cursor + train + test <= end + timedelta(days=1):
        train_start = cursor
        train_end = cursor + train
        test_start = train_end
        test_end = train_end + test
        windows.append(
            WalkForwardWindow(train_start, train_end, test_start, test_end)
        )
        cursor = cursor + step
    return windows


def run_walk_forward(
    strategy_factory: StrategyFactory,
    param_grid: dict[str, list],
    feed_factory_for_range: RangeFeedFactory,
    windows: list[WalkForwardWindow],
    *,
    select_metric: str = "sharpe",
    maximize: bool = True,
    portfolio_factory: Callable[[], Portfolio] = Portfolio,
    risk_factory: Callable[[], RiskEngine] = BasicRiskEngine,
    broker_factory: Callable[[], SimulatedBroker] = SimulatedBroker,
    feature_engine_factory: Callable[[], FeatureEngine] | None = None,
    feature_store_factory: Callable[[], FeatureStore] | None = None,
) -> list[WalkForwardResult]:
    """Optimise on each training window; evaluate on the following test window."""
    results: list[WalkForwardResult] = []
    component_kwargs = dict(
        portfolio_factory=portfolio_factory,
        risk_factory=risk_factory,
        broker_factory=broker_factory,
        feature_engine_factory=feature_engine_factory,
        feature_store_factory=feature_store_factory,
    )

    for window in windows:
        train_report = run_sweep(
            strategy_factory,
            param_grid,
            lambda: feed_factory_for_range(window.train_start, window.train_end),
            **component_kwargs,
        )
        best = train_report.best(select_metric, maximize=maximize)

        # Evaluate the chosen params out-of-sample on the test window.
        test_engine = BacktestEngine(
            strategy_factory(**best.params),
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
        test_result = test_engine.run(
            feed_factory_for_range(window.test_start, window.test_end)
        )
        results.append(
            WalkForwardResult(
                window=window,
                best_params=best.params,
                train_metrics=best.metrics,
                test_metrics=test_result.metrics,
            )
        )
        _log.event(
            "walk_forward_window_complete",
            train_start=window.train_start.isoformat(),
            test_end=window.test_end.isoformat(),
            best_params=best.params,
            test_sharpe=test_result.metrics.sharpe,
        )
    return results
