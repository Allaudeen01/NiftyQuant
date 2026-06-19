"""ValidationEngine: compare live/paper behaviour against the backtest baseline.

Given a :class:`Baseline` captured from a validated backtest (its metrics,
feature distributions, and regime), the engine evaluates current performance
and features and emits structured :class:`Alert` objects when behaviour drifts
outside expectations. It is analytical only -- it never changes positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nifty_quant.backtest.metrics import compute_metrics
from nifty_quant.backtest.portfolio import Trade
from nifty_quant.validation.alerts import Alert, AlertLevel
from nifty_quant.validation.drift import DriftResult, detect_feature_drift
from nifty_quant.validation.performance import window_summary
from nifty_quant.log import get_logger

_log = get_logger("validation.engine")


@dataclass
class Baseline:
    """The validated reference a strategy is monitored against."""

    sharpe: float
    win_rate: float
    max_drawdown: float
    expectancy: float
    feature_distributions: dict[str, np.ndarray] = field(default_factory=dict)
    regime_trend: str | None = None

    @classmethod
    def from_backtest(
        cls,
        metrics,
        *,
        feature_distributions: dict[str, np.ndarray] | None = None,
        regime_trend: str | None = None,
    ) -> "Baseline":
        return cls(
            sharpe=metrics.sharpe,
            win_rate=metrics.win_rate,
            max_drawdown=metrics.max_drawdown,
            expectancy=metrics.expectancy,
            feature_distributions=feature_distributions or {},
            regime_trend=regime_trend,
        )


@dataclass
class ValidationThresholds:
    sharpe_drop: float = 0.5          # current below baseline by this => alert
    win_rate_drop: float = 0.10       # absolute drop in win rate
    drawdown_excess: float = 0.05     # current DD worse than baseline by this
    min_trades: int = 10              # below this, metric checks are unreliable
    psi_threshold: float = 0.25
    ks_threshold: float = 0.3


@dataclass
class ValidationReport:
    alerts: list[Alert] = field(default_factory=list)
    current: dict = field(default_factory=dict)
    drift: list[DriftResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if nothing at WARNING level or above fired."""
        return not any(a.level >= AlertLevel.WARNING for a in self.alerts)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "alerts": [a.as_dict() for a in self.alerts],
            "current": dict(self.current),
            "drift": [d.as_dict() for d in self.drift],
        }


class ValidationEngine:
    def __init__(
        self,
        baseline: Baseline,
        thresholds: ValidationThresholds | None = None,
    ) -> None:
        self.baseline = baseline
        self.t = thresholds or ValidationThresholds()

    def validate(
        self,
        current_equity: pd.Series,
        current_trades: list[Trade],
        *,
        current_features: dict[str, np.ndarray] | None = None,
        current_regime_trend: str | None = None,
    ) -> ValidationReport:
        report = ValidationReport()
        metrics = compute_metrics(current_equity, current_trades)
        summary = window_summary(current_equity, current_trades)
        report.current = {**metrics.as_dict(), **summary}

        n = metrics.num_trades
        if n < self.t.min_trades:
            report.alerts.append(
                Alert(
                    AlertLevel.INFO,
                    "insufficient_data",
                    f"Only {n} trades (< {self.t.min_trades}); "
                    "performance checks deferred.",
                    metric="num_trades",
                    observed=float(n),
                    expected=float(self.t.min_trades),
                )
            )
        else:
            self._check_sharpe(metrics, report)
            self._check_win_rate(metrics, report)
            self._check_drawdown(metrics, report)
            self._check_expectancy(metrics, report)

        self._check_regime(current_regime_trend, report)
        self._check_drift(current_features, report)

        _log.event(
            "validation_complete",
            passed=report.passed,
            alerts=len(report.alerts),
            num_trades=n,
        )
        return report

    # --- individual checks --------------------------------------------------

    def _check_sharpe(self, metrics, report: ValidationReport) -> None:
        b, c = self.baseline.sharpe, metrics.sharpe
        if _finite(b) and _finite(c) and c < b - self.t.sharpe_drop:
            level = AlertLevel.CRITICAL if c < 0 <= b else AlertLevel.WARNING
            report.alerts.append(
                Alert(
                    level, "sharpe_drift",
                    f"Sharpe {c:.2f} fell below baseline {b:.2f} "
                    f"by more than {self.t.sharpe_drop}.",
                    metric="sharpe", observed=c, expected=b,
                )
            )

    def _check_win_rate(self, metrics, report: ValidationReport) -> None:
        b, c = self.baseline.win_rate, metrics.win_rate
        if _finite(b) and _finite(c) and (b - c) > self.t.win_rate_drop:
            report.alerts.append(
                Alert(
                    AlertLevel.WARNING, "win_rate_drift",
                    f"Win rate {c:.1%} dropped from baseline {b:.1%}.",
                    metric="win_rate", observed=c, expected=b,
                )
            )

    def _check_drawdown(self, metrics, report: ValidationReport) -> None:
        b, c = self.baseline.max_drawdown, metrics.max_drawdown
        if _finite(b) and _finite(c) and c < b - self.t.drawdown_excess:
            report.alerts.append(
                Alert(
                    AlertLevel.WARNING, "drawdown_excess",
                    f"Max drawdown {c:.1%} exceeds baseline {b:.1%}.",
                    metric="max_drawdown", observed=c, expected=b,
                )
            )

    def _check_expectancy(self, metrics, report: ValidationReport) -> None:
        b, c = self.baseline.expectancy, metrics.expectancy
        if _finite(b) and _finite(c) and b > 0 and c <= 0:
            report.alerts.append(
                Alert(
                    AlertLevel.CRITICAL, "expectancy_negative",
                    f"Expectancy turned non-positive ({c:.2f}) "
                    f"vs baseline {b:.2f}.",
                    metric="expectancy", observed=c, expected=b,
                )
            )

    def _check_regime(
        self, current_trend: str | None, report: ValidationReport
    ) -> None:
        b = self.baseline.regime_trend
        if b is not None and current_trend is not None and current_trend != b:
            report.alerts.append(
                Alert(
                    AlertLevel.WARNING, "regime_mismatch",
                    f"Operating in '{current_trend}' regime; "
                    f"strategy was validated in '{b}'.",
                    metric="regime", context={"baseline": b, "current": current_trend},
                )
            )

    def _check_drift(
        self,
        current_features: dict[str, np.ndarray] | None,
        report: ValidationReport,
    ) -> None:
        if not current_features or not self.baseline.feature_distributions:
            return
        results = detect_feature_drift(
            self.baseline.feature_distributions,
            current_features,
            psi_threshold=self.t.psi_threshold,
            ks_threshold=self.t.ks_threshold,
        )
        report.drift = results
        for d in results:
            if d.drifted:
                report.alerts.append(
                    Alert(
                        AlertLevel.WARNING, "feature_drift",
                        f"Feature '{d.feature}' drifted "
                        f"(PSI={d.psi:.2f}, KS={d.ks:.2f}).",
                        metric=d.feature, observed=d.current_mean,
                        expected=d.baseline_mean,
                        context={"psi": d.psi, "ks": d.ks},
                    )
                )


def _finite(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not math.isinf(x)
