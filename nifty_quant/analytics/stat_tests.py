"""Statistical primitives for mean-reversion / regime / sizing research.

Pure functions (no I/O, no trading decisions, no global state) implementing the
standard, public-domain quantitative-finance tools:

  * hurst_exponent      -- trending (>0.5) vs mean-reverting (<0.5) vs RW (~0.5)
  * adf_test            -- Augmented Dickey-Fuller stationarity test (statsmodels)
  * half_life           -- Ornstein-Uhlenbeck half-life of mean reversion
  * variance_ratio      -- Lo-MacKinlay variance-ratio (heteroskedasticity-robust z)
  * kelly_fraction      -- Kelly criterion from win-rate / payoff (and Gaussian form)

These are RESEARCH PRIMITIVES, not strategies. They quantify structure and give
principled (non-curve-fit) parameters such as a holding-period estimate. The
formulas are standard textbook results (Dickey-Fuller 1979; Lo-MacKinlay 1988;
Kelly 1956); implemented here from first principles, validated against known
synthetic cases in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --- mean reversion / persistence ------------------------------------------

def hurst_exponent(series, max_lag: int = 50) -> float:
    """Hurst exponent via the rescaled-range / lagged-variance method.

    H < 0.5 -> mean-reverting, H ~ 0.5 -> random walk, H > 0.5 -> trending.
    Estimated from the slope of log(var of lagged differences) vs log(lag).
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    max_lag = min(max_lag, n // 2)
    lags = range(2, max_lag)
    tau = []
    valid_lags = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        std = diff.std()
        if std > 0:
            tau.append(std)
            valid_lags.append(lag)
    if len(valid_lags) < 3:
        return float("nan")
    # std of lagged diffs ~ lag**H  ->  slope of log-log is H.
    slope = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
    return float(slope)


def half_life(series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion (in samples).

    Regress dy_t = a + lambda * y_{t-1}; half-life = -ln(2) / lambda.
    Returns +inf if not mean-reverting (lambda >= 0), NaN if undefined.
    A principled holding-period estimate (no parameter tuning).
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return float("nan")
    y_lag = y[:-1]
    dy = np.diff(y)
    # OLS of dy on [1, y_lag]
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    lam = beta[1]
    if lam >= 0:
        return float("inf")  # not mean-reverting
    return float(-math.log(2) / lam)


def adf_test(series, max_lag: int | None = None) -> dict:
    """Augmented Dickey-Fuller stationarity test (via statsmodels).

    Returns {stat, pvalue, used_lag, nobs, crit_1/5/10, stationary_5pct}.
    H0: a unit root exists (non-stationary). Low p-value => reject => stationary.
    """
    from statsmodels.tsa.stattools import adfuller

    x = pd.Series(series, dtype=float).dropna()
    if len(x) < 10:
        return {"stat": float("nan"), "pvalue": float("nan"),
                "stationary_5pct": False, "nobs": len(x)}
    kwargs = {"autolag": "AIC"} if max_lag is None else {"maxlag": max_lag, "autolag": None}
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(x.to_numpy(), **kwargs)
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "crit_1": float(crit["1%"]),
        "crit_5": float(crit["5%"]),
        "crit_10": float(crit["10%"]),
        "stationary_5pct": bool(pvalue < 0.05),
    }


@dataclass
class VarianceRatioResult:
    vr: float
    z: float          # heteroskedasticity-robust z-statistic
    pvalue: float
    q: int


def variance_ratio(series, q: int = 2) -> VarianceRatioResult:
    """Lo-MacKinlay variance-ratio test on a price/log-price series.

    VR(q) < 1 => mean reversion, > 1 => momentum/trending, ~1 => random walk.
    Uses the heteroskedasticity-consistent z-statistic (robust to vol clustering).
    Input should be a level series (e.g. log price); it differences internally.
    """
    p = np.asarray(series, dtype=float)
    p = p[~np.isnan(p)]
    n = len(p) - 1
    if n < q * 2 or q < 2:
        return VarianceRatioResult(float("nan"), float("nan"), float("nan"), q)
    r = np.diff(p)                      # one-period returns
    mu = r.mean()
    # one-period variance (unbiased)
    var1 = np.sum((r - mu) ** 2) / n
    # q-period overlapping variance
    rq = p[q:] - p[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    if var1 == 0:
        return VarianceRatioResult(float("nan"), float("nan"), float("nan"), q)
    vr = varq / var1
    # heteroskedasticity-robust variance of VR (Lo-MacKinlay 1988)
    phi = 0.0
    for j in range(1, q):
        dj = (r[j:] - mu)
        dlag = (r[:n - j] - mu)
        delta = (np.sum(dj ** 2 * dlag ** 2)
                 / (np.sum((r - mu) ** 2) ** 2))
        delta *= n
        weight = (2 * (q - j) / q) ** 2
        phi += weight * delta
    z = (vr - 1) / math.sqrt(phi) if phi > 0 else float("nan")
    from scipy import stats as ss
    pvalue = 2 * (1 - ss.norm.cdf(abs(z))) if not math.isnan(z) else float("nan")
    return VarianceRatioResult(float(vr), float(z), float(pvalue), q)


# --- tail risk / extreme value theory --------------------------------------

def hill_estimator(data, k: int | None = None, tail: str = "right") -> dict:
    """Hill estimator of the tail index for a heavy-tailed sample.

    Fits a power-law tail using the ``k`` largest order statistics (right tail)
    or ``k`` most negative (left tail). Returns the tail index ``alpha`` (larger
    => thinner tail; ``alpha`` also bounds the number of finite moments), the
    shape ``xi = 1/alpha`` (the EVT/GPD shape), and an asymptotic std error
    ``alpha/sqrt(k)``.

    Only defined for a strictly positive tail: values are taken as-is for the
    right tail and negated for the left tail, then the positive exceedances are
    used. Default ``k`` is ~10% of the sample (a common rule of thumb; the
    estimate is sensitive to ``k`` -- scan a range in practice).
    """
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    if tail == "left":
        x = -x
    x = x[x > 0]                       # power-law tail is defined on positives
    x = np.sort(x)[::-1]               # descending
    n = len(x)
    if n < 20:
        return {"alpha": float("nan"), "xi": float("nan"), "se": float("nan"),
                "k": 0, "n": n}
    if k is None:
        k = max(10, int(0.10 * n))
    k = min(k, n - 1)
    top = x[:k + 1]
    # Hill: xi = (1/k) sum_{i=1..k} ln(x_i) - ln(x_{k+1})
    xi = float(np.mean(np.log(top[:k])) - math.log(top[k]))
    if xi <= 0:
        return {"alpha": float("inf"), "xi": 0.0, "se": float("nan"),
                "k": k, "n": n}
    alpha = 1.0 / xi
    return {"alpha": alpha, "xi": xi, "se": alpha / math.sqrt(k), "k": k, "n": n}


def pot_gpd_fit(data, threshold_pct: float = 95.0, tail: str = "left") -> dict:
    """Peaks-over-threshold: fit a Generalized Pareto to threshold exceedances.

    For ``tail='left'`` the sample is negated so left-tail losses become the
    upper tail (the usual risk-management convention). Fits a GPD to exceedances
    above the ``threshold_pct`` percentile via ``scipy.stats.genpareto``.

    Returns the GPD shape ``xi`` (>0 heavy/power-law tail, =0 exponential,
    <0 bounded), scale ``beta``, the threshold ``u``, the number of exceedances
    ``n_exceed``, and ``tail_df = 1/xi`` (equivalent Student-t degrees of freedom
    for xi>0). A positive ``xi`` is the EVT signature of fat tails.
    """
    from scipy import stats as _ss

    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    if tail == "left":
        x = -x
    if len(x) < 50:
        return {"xi": float("nan"), "beta": float("nan"), "u": float("nan"),
                "n_exceed": 0, "tail_df": float("nan")}
    u = float(np.percentile(x, threshold_pct))
    exceed = x[x > u] - u
    if len(exceed) < 10:
        return {"xi": float("nan"), "beta": float("nan"), "u": u,
                "n_exceed": len(exceed), "tail_df": float("nan")}
    # genpareto.fit with floc=0 (exceedances are >= 0 by construction)
    xi, _loc, beta = _ss.genpareto.fit(exceed, floc=0.0)
    tail_df = (1.0 / xi) if xi > 1e-6 else float("inf")
    return {"xi": float(xi), "beta": float(beta), "u": u,
            "n_exceed": int(len(exceed)), "tail_df": float(tail_df)}


# --- position sizing -------------------------------------------------------

def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly fraction for a discrete bet.

    f* = p - (1 - p) / b   where b = avg_win/avg_loss (payoff ratio).
    Returns the fraction of capital to risk. Negative => no edge (don't bet).
    """
    if win_loss_ratio <= 0:
        return 0.0
    f = win_prob - (1.0 - win_prob) / win_loss_ratio
    return float(f)


def kelly_gaussian(mean_return: float, variance: float) -> float:
    """Continuous Kelly leverage for a Gaussian return: f* = mean / variance.

    This is the optimal-growth leverage for a near-Gaussian edge. In practice
    use a FRACTION of this (e.g. half-Kelly): full Kelly assumes the true
    distribution is known and is far too aggressive under estimation error and
    fat tails (critical for option-selling / VRP, which has negative skew).
    """
    if variance <= 0:
        return 0.0
    return float(mean_return / variance)


def fractional_kelly(full_kelly: float, fraction: float = 0.5,
                     cap: float | None = 1.0) -> float:
    """Scale a Kelly fraction down (default half-Kelly) and optionally cap it.

    Half-Kelly keeps ~3/4 of the growth at ~1/2 the volatility and is far more
    robust to mis-estimated edge -- the sane default for real deployment.
    """
    f = max(0.0, full_kelly) * fraction
    if cap is not None:
        f = min(f, cap)
    return float(f)
