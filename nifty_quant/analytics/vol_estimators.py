"""OHLC-based volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell,
Yang-Zhang).

These estimate daily return VARIANCE from daily open/high/low/close (and the
prior close for the overnight gap). Range-based estimators are far more
efficient than the close-to-close squared return (they use the intraday path),
and Yang-Zhang additionally accounts for the overnight jump and is
drift-independent.

All functions return VARIANCE (vol squared). Standard textbook forms:
  * Parkinson (1980)      -- high/low range only.
  * Garman-Klass (1980)   -- adds open/close; more efficient than Parkinson.
  * Rogers-Satchell (1991)-- drift-independent (handles trending days).
  * Yang-Zhang (2000)     -- overnight + open + Rogers-Satchell; minimum variance,
                             drift-independent, handles gaps. A WINDOW estimator.

Pure functions; no I/O. Inputs are array-likes of equal length.
"""

from __future__ import annotations

import math

import numpy as np

_INV_4LN2 = 1.0 / (4.0 * math.log(2.0))
_2LN2_MINUS_1 = 2.0 * math.log(2.0) - 1.0


def parkinson(high, low) -> np.ndarray:
    """Per-day Parkinson variance: (1/(4 ln2)) * (ln(H/L))^2."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    return _INV_4LN2 * np.log(h / l) ** 2


def garman_klass(open_, high, low, close) -> np.ndarray:
    """Per-day Garman-Klass variance: 0.5*(ln(H/L))^2 - (2ln2-1)*(ln(C/O))^2."""
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    return 0.5 * np.log(h / l) ** 2 - _2LN2_MINUS_1 * np.log(c / o) ** 2


def rogers_satchell(open_, high, low, close) -> np.ndarray:
    """Per-day Rogers-Satchell variance (drift-independent):
    ln(H/C)ln(H/O) + ln(L/C)ln(L/O)."""
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def yang_zhang(open_, high, low, close, prev_close) -> float:
    """Yang-Zhang variance over a WINDOW of days (returns a scalar variance).

    Combines overnight variance, open-to-close variance, and Rogers-Satchell:
        YZ = V_overnight + k * V_open + (1-k) * V_rs,
        k = 0.34 / (1.34 + (N+1)/(N-1)).
    ``prev_close`` is the close of the day BEFORE each day (same length).
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    pc = np.asarray(prev_close, dtype=float)
    N = len(o)
    if N < 2:
        return float("nan")
    overnight = np.log(o / pc)                     # close(t-1) -> open(t)
    open_close = np.log(c / o)                     # open(t) -> close(t)
    v_on = np.var(overnight, ddof=1)
    v_oc = np.var(open_close, ddof=1)
    v_rs = np.mean(rogers_satchell(o, h, l, c))
    k = 0.34 / (1.34 + (N + 1) / (N - 1))
    return float(v_on + k * v_oc + (1 - k) * v_rs)


def yang_zhang_daily(open_, high, low, close, prev_close) -> np.ndarray:
    """A per-day, single-day 'full-day' variance proxy in the Yang-Zhang spirit:
    overnight squared return + Rogers-Satchell intraday (drift-free).

    Unlike the window ``yang_zhang``, this is a single-day estimator suitable as
    a per-day HAR input; it includes the overnight gap (which close-to-close 5m
    realized variance omits) and is drift-independent intraday.
    """
    o = np.asarray(open_, dtype=float)
    pc = np.asarray(prev_close, dtype=float)
    overnight_sq = np.log(o / pc) ** 2
    return overnight_sq + rogers_satchell(o, high, low, close)
