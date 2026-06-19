"""Tests for drift detection and the validation engine."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nifty_quant.backtest.portfolio import Trade
from nifty_quant.validation.alerts import AlertLevel
from nifty_quant.validation.drift import (
    detect_feature_drift,
    ks_statistic,
    population_stability_index,
)
from nifty_quant.validation.engine import (
    Baseline,
    ValidationEngine,
    ValidationThresholds,
)


# --- drift -----------------------------------------------------------------


def test_psi_identical_is_near_zero():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)
    assert population_stability_index(base, cur) < 0.1


def test_psi_shifted_is_large():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)  # mean shifted by 3 sigma
    assert population_stability_index(base, cur) > 0.25


def test_ks_identical_small_disjoint_one():
    a = np.linspace(0, 1, 100)
    assert ks_statistic(a, a) == pytest.approx(0.0, abs=1e-9)
    b = np.linspace(10, 11, 100)
    assert ks_statistic(a, b) == pytest.approx(1.0)


def test_detect_feature_drift_flags_shift():
    rng = np.random.default_rng(1)
    base = {"rsi": rng.normal(50, 5, 1000), "atr": rng.normal(20, 2, 1000)}
    cur = {"rsi": rng.normal(80, 5, 1000), "atr": rng.normal(20, 2, 1000)}
    results = detect_feature_drift(base, cur)
    by_feature = {r.feature: r for r in results}
    assert by_feature["rsi"].drifted is True
    assert by_feature["atr"].drifted is False


# --- validation engine -----------------------------------------------------


def _equity(values, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def _trades(pnls):
    t0 = datetime(2025, 1, 1, 9, 15)
    return [
        Trade("NIFTY", "LONG", 1, 100.0, 100.0 + p, t0, t0 + timedelta(hours=1), p)
        for p in pnls
    ]


def _baseline(**kw):
    defaults = dict(
        sharpe=2.0, win_rate=0.6, max_drawdown=-0.10, expectancy=100.0
    )
    defaults.update(kw)
    return Baseline(**defaults)


def test_healthy_strategy_passes():
    eng = ValidationEngine(_baseline())
    eq = _equity([1000 * 1.01**i for i in range(40)])
    trades = _trades([100, 120, -40, 90, 110, -30, 80, 130, 95, -20, 105, 100])
    report = eng.validate(eq, trades)
    assert report.passed
    assert all(a.level < AlertLevel.WARNING for a in report.alerts)


def test_degraded_strategy_alerts():
    eng = ValidationEngine(_baseline())
    eq = _equity([1000 * 0.99**i for i in range(40)])
    trades = _trades([-50, -40, 20, -60, -30, -45, 10, -55, -35, -25, -40, -50])
    report = eng.validate(eq, trades)
    assert not report.passed
    codes = {a.code for a in report.alerts}
    assert "expectancy_negative" in codes
    assert any(a.level == AlertLevel.CRITICAL for a in report.alerts)


def test_insufficient_data_defers_checks():
    eng = ValidationEngine(_baseline(), ValidationThresholds(min_trades=10))
    eq = _equity([1000, 990, 1010])
    report = eng.validate(eq, _trades([-50, -40, 20]))
    codes = {a.code for a in report.alerts}
    assert "insufficient_data" in codes
    # INFO only -> still "passed".
    assert report.passed


def test_regime_mismatch_alert():
    eng = ValidationEngine(_baseline(regime_trend="UP"))
    eq = _equity([1000 * 1.01**i for i in range(40)])
    trades = _trades([100, 120, -40, 90, 110, -30, 80, 130, 95, -20, 105, 100])
    report = eng.validate(eq, trades, current_regime_trend="DOWN")
    assert any(a.code == "regime_mismatch" for a in report.alerts)
    assert not report.passed


def test_feature_drift_alert():
    rng = np.random.default_rng(2)
    base_dist = {"rsi": rng.normal(50, 5, 1000)}
    eng = ValidationEngine(_baseline(feature_distributions=base_dist))
    eq = _equity([1000 * 1.01**i for i in range(40)])
    trades = _trades([100, 120, -40, 90, 110, -30, 80, 130, 95, -20, 105, 100])
    current = {"rsi": rng.normal(85, 5, 500)}
    report = eng.validate(eq, trades, current_features=current)
    assert any(a.code == "feature_drift" for a in report.alerts)
