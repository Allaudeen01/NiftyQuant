"""Tests for regime detection, sweeps, walk-forward, and experiment tracking."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.broker import MidPriceFill, SimulatedBroker
from nifty_quant.backtest.strategies.ema_cross import EmaCrossStrategy
from nifty_quant.research.experiment import ExperimentTracker, git_commit
from nifty_quant.research.regime import classify_regime
from nifty_quant.research.sweep import expand_grid, run_sweep
from nifty_quant.research.walkforward import generate_windows, run_walk_forward
from tests.helpers import ListFeed, make_candle_events


# --- regime ----------------------------------------------------------------


def _prices(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_regime_uptrend():
    r = classify_regime(_prices([100 * 1.01**i for i in range(60)]))
    assert r.trend == "UP"


def test_regime_downtrend():
    r = classify_regime(_prices([100 * 0.99**i for i in range(60)]))
    assert r.trend == "DOWN"


def test_regime_sideways():
    rng = np.random.default_rng(0)
    noise = 100 + rng.normal(0, 0.2, 60)
    r = classify_regime(_prices(noise))
    assert r.trend == "SIDEWAYS"


def test_regime_high_volatility_tag():
    rng = np.random.default_rng(1)
    vals = [100.0]
    for _ in range(60):
        vals.append(vals[-1] * (1 + rng.normal(0, 0.05)))  # ~5% daily moves
    r = classify_regime(_prices(vals))
    assert r.volatility == "HIGH"
    assert "GAP" in r.tags


# --- sweep -----------------------------------------------------------------


def test_expand_grid():
    grid = {"fast": [3, 5], "slow": [10, 20]}
    combos = expand_grid(grid)
    assert len(combos) == 4
    assert {"fast": 3, "slow": 10} in combos


def test_expand_grid_empty():
    assert expand_grid({}) == [{}]


CLOSES = (
    [100] * 5
    + [101, 103, 106, 110, 115, 120]
    + [118, 113, 107, 100, 94, 90]
)


def _ema_factory(**params):
    inst = Instrument("NIFTY", InstrumentType.INDEX)
    return EmaCrossStrategy(inst, quantity=5, **params)


def _feed_factory():
    return ListFeed(make_candle_events(CLOSES))


def test_run_sweep_and_best():
    grid = {"fast": [3, 5], "slow": [8, 10]}
    report = run_sweep(
        _ema_factory,
        grid,
        _feed_factory,
        broker_factory=lambda: SimulatedBroker(fill_model=MidPriceFill()),
    )
    assert len(report.results) == 4
    df = report.to_frame()
    assert "fast" in df.columns and "sharpe" in df.columns
    # best() should not raise when at least one run has a finite metric.
    best = report.best("total_return", maximize=True)
    assert "fast" in best.params


# --- walk-forward ----------------------------------------------------------


def test_generate_windows():
    start = datetime(2021, 1, 1)
    end = datetime(2021, 12, 31)
    windows = generate_windows(start, end, train_days=90, test_days=30, step_days=30)
    assert len(windows) >= 1
    w = windows[0]
    assert w.train_start == start
    assert (w.train_end - w.train_start).days == 90
    assert (w.test_end - w.test_start).days == 30
    assert w.test_start == w.train_end  # no gap


def test_run_walk_forward_oos():
    # Build a long candle stream and slice by date for each window.
    closes = ([100] * 10 + [100 + i for i in range(40)] + [140 - i for i in range(40)])
    all_events = make_candle_events(closes, start=datetime(2021, 1, 1))

    def feed_for_range(start, end):
        sel = [e for e in all_events if start <= e.timestamp <= end]
        return ListFeed(sel)

    windows = generate_windows(
        datetime(2021, 1, 1), datetime(2021, 4, 30),
        train_days=40, test_days=20, step_days=20,
    )
    results = run_walk_forward(
        _ema_factory,
        {"fast": [3, 5], "slow": [8, 12]},
        feed_for_range,
        windows,
        select_metric="total_return",
        broker_factory=lambda: SimulatedBroker(fill_model=MidPriceFill()),
    )
    assert len(results) == len(windows)
    for r in results:
        assert "fast" in r.best_params
        assert r.test_metrics is not None


# --- experiment tracking ---------------------------------------------------


def test_experiment_tracker_roundtrip(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.log(
        strategy_name="EmaCross",
        strategy_version="1.0.0",
        parameters={"fast": 10, "slow": 20},
        metrics={"sharpe": 1.2, "total_return": 0.15},
        feature_version="v1",
        data_version="nifty-2025",
        regime={"trend": "UP", "volatility": "LOW"},
        tags=["reference"],
    )
    assert exp.id

    listed = tracker.list_experiments()
    assert len(listed) == 1
    loaded = tracker.load(exp.id)
    assert loaded.strategy_name == "EmaCross"
    assert loaded.parameters["fast"] == 10

    df = tracker.to_frame()
    assert "param.fast" in df.columns
    assert "metric.sharpe" in df.columns
    assert "regime.trend" in df.columns


def test_experiment_appends_multiple(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    for i in range(3):
        tracker.log(
            strategy_name="S",
            strategy_version="1.0.0",
            parameters={"p": i},
            metrics={"sharpe": float(i)},
        )
    assert len(tracker.list_experiments()) == 3


def test_git_commit_returns_str_or_none():
    # In this repo it should be a string; tolerate None if git is unavailable.
    val = git_commit()
    assert val is None or isinstance(val, str)
