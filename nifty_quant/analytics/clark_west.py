"""Clark-West MSPE-adjusted test for NESTED forecast comparison.

Diebold-Mariano is invalid for nested models under the null: the larger model
estimates parameters that are zero under H0, which inflates its MSPE and biases
the DM statistic toward the small model. Clark & West (2007) correct this with
an adjustment term.

For a small (restricted) forecast `f_s` nested inside a large forecast `f_l`:

    cw_t = (y_t - f_s,t)^2 - (y_t - f_l,t)^2 + (f_s,t - f_l,t)^2

    H0: E[cw_t] <= 0   (the large model does NOT beat the small one)
    H1: E[cw_t] >  0

EXP034-owned (`stat_tests.py` carries no DM/CW). Pure functions, no I/O.
"""

from __future__ import annotations

import numpy as np


def cw_series(y: np.ndarray, f_small: np.ndarray, f_large: np.ndarray) -> np.ndarray:
    """Per-observation Clark-West MSPE-adjusted terms."""
    y = np.asarray(y, dtype=float)
    fs = np.asarray(f_small, dtype=float)
    fl = np.asarray(f_large, dtype=float)
    return (y - fs) ** 2 - (y - fl) ** 2 + (fs - fl) ** 2


def day_block_bootstrap_test(day_means: np.ndarray, block_len: float,
                              n_boot: int, seed: int,
                              bootstrap_indices) -> dict:
    """One-sided bootstrap test of H0: mean <= 0, resampling WHOLE trading days.

    `day_means` is one value per out-of-sample trading day (the day's mean CW
    term). Resampling at day level -- rather than row level -- is what respects
    intraday dependence; row-level resampling would badly understate the
    standard error.

    `bootstrap_indices` is injected (the project's stationary-bootstrap
    generator) so this module stays dependency-free and testable.
    """
    x = np.asarray(day_means, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return {"mean": float(np.mean(x)) if n else float("nan"),
                "se": float("nan"), "p_one_sided": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_days": n}

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(n, block_len, n_boot, rng)
    boot_means = x[idx].mean(axis=1)

    observed = float(x.mean())
    centered = boot_means - observed          # impose H0: true mean = 0
    p = float(np.mean(centered >= observed))
    return {
        "mean": observed,
        "se": float(np.std(boot_means, ddof=1)),
        "p_one_sided": p,
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "n_days": n,
    }


def hac_one_sided(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Newey-West HAC t-stat and one-sided p-value for H0: mean <= 0.

    Reported as a cross-check on the day-block bootstrap, which is primary.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # NB: `np.allclose(std, 0.0)` would be WRONG here -- its default absolute
    # tolerance of 1e-8 declares a genuine CW series (std ~1e-10) constant and
    # silently returns NaN. Degeneracy must be tested exactly.
    if len(x) < 3 or np.std(x) == 0.0:
        return float("nan"), float("nan")
    import statsmodels.api as sm
    lag = max(1, min(int(lag), len(x) - 2))
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    t = float(res.tvalues[0])
    p_two = float(res.pvalues[0])
    p_one = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    return t, p_one


def oos_r2(y: np.ndarray, pred: np.ndarray, naive: np.ndarray) -> float:
    """Campbell-Thompson style out-of-sample R^2 against a naive benchmark."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(naive)
    if ok.sum() < 2:
        return float("nan")
    sse_m = float(np.sum((y[ok] - np.asarray(pred)[ok]) ** 2))
    sse_n = float(np.sum((y[ok] - np.asarray(naive)[ok]) ** 2))
    return 1.0 - sse_m / sse_n if sse_n > 0 else float("nan")
