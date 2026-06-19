"""Feature-distribution drift detection.

Compares a feature's *baseline* distribution (training/backtest period) against
its *current* distribution (recent live/paper period). Two complementary,
dependency-free measures:

- PSI (Population Stability Index): bins the baseline by quantiles and measures
  how much current mass shifted between bins. Common bands: <0.10 stable,
  0.10-0.25 moderate, >0.25 significant.
- KS statistic: the maximum gap between the two empirical CDFs (0..1).

These are descriptive drift signals, not hypothesis tests with p-values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


def population_stability_index(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """PSI of ``current`` vs ``baseline`` using baseline quantile bins."""
    baseline = _clean(baseline)
    current = _clean(current)
    if baseline.size == 0 or current.size == 0:
        return float("nan")

    # Quantile edges from the baseline; dedupe for near-constant data.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if edges.size < 2:
        return 0.0  # baseline is effectively constant -> no meaningful PSI
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    base_pct = base_counts / base_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    base_pct = np.clip(base_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)

    return float(np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct)))


def ks_statistic(baseline: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), in [0, 1]."""
    baseline = _clean(baseline)
    current = _clean(current)
    if baseline.size == 0 or current.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([baseline, current]))
    cdf_b = np.searchsorted(np.sort(baseline), grid, side="right") / baseline.size
    cdf_c = np.searchsorted(np.sort(current), grid, side="right") / current.size
    return float(np.max(np.abs(cdf_b - cdf_c)))


@dataclass(frozen=True)
class DriftResult:
    feature: str
    psi: float
    ks: float
    baseline_mean: float
    current_mean: float
    drifted: bool

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "ks": self.ks,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "drifted": self.drifted,
        }


def detect_feature_drift(
    baseline: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    *,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.3,
) -> list[DriftResult]:
    """Compute drift per shared feature. ``drifted`` if PSI or KS exceeds limit."""
    results: list[DriftResult] = []
    for feature in sorted(set(baseline) & set(current)):
        b = _clean(np.asarray(baseline[feature], dtype=float))
        c = _clean(np.asarray(current[feature], dtype=float))
        if b.size == 0 or c.size == 0:
            continue
        psi = population_stability_index(b, c)
        ks = ks_statistic(b, c)
        drifted = (
            (not np.isnan(psi) and psi >= psi_threshold)
            or (not np.isnan(ks) and ks >= ks_threshold)
        )
        results.append(
            DriftResult(
                feature=feature,
                psi=psi,
                ks=ks,
                baseline_mean=float(b.mean()),
                current_mean=float(c.mean()),
                drifted=drifted,
            )
        )
    return results


def _clean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).ravel()
    return arr[~np.isnan(arr)]
