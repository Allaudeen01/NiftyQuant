"""Continuous validation.

A health monitor for strategies. Compares live/paper behaviour against the
backtest baseline and flags degradation:

    paper trades / features
            |
            v
      ValidationEngine  --(baseline distribution)-->  drift & metric checks
            |
            v
      structured Alerts

Includes feature-distribution drift detection (PSI / KS), rolling performance
metrics, and regime-conformance checks. Purely analytical -- it raises alerts,
it does not make trading decisions.
"""

from nifty_quant.validation.alerts import Alert, AlertLevel
from nifty_quant.validation.drift import (
    DriftResult,
    population_stability_index,
    ks_statistic,
    detect_feature_drift,
)
from nifty_quant.validation.performance import (
    rolling_sharpe,
    rolling_expectancy,
    window_summary,
)
from nifty_quant.validation.engine import (
    Baseline,
    ValidationEngine,
    ValidationReport,
    ValidationThresholds,
)

__all__ = [
    "Alert",
    "AlertLevel",
    "DriftResult",
    "population_stability_index",
    "ks_statistic",
    "detect_feature_drift",
    "rolling_sharpe",
    "rolling_expectancy",
    "window_summary",
    "Baseline",
    "ValidationEngine",
    "ValidationReport",
    "ValidationThresholds",
]
