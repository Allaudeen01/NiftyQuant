"""Black-Scholes pricing, Greeks, and implied-volatility solving.

European options on an index (Nifty options are European, cash-settled), which
makes Black-Scholes a reasonable analytic model. All functions are pure and
vectorise over NumPy arrays as well as accepting scalars.

Conventions:
- ``s``  : spot price of the underlying
- ``k``  : strike
- ``t``  : time to expiry in YEARS (e.g. 7/365 for a weekly)
- ``r``  : risk-free rate (annual, continuous), as a fraction
- ``sigma``: volatility (annual), as a fraction
- ``q``  : continuous dividend yield (default 0 for an index proxy)
- option type is the string "CE"/"call" or "PE"/"put"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    """Standard normal CDF (uses erf; works on scalars and arrays)."""
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def _erf(x):
    # numpy has no erf; use math.erf elementwise via vectorize for arrays.
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return math.erf(float(arr))
    return np.vectorize(math.erf)(arr)


def _is_call(option_type: str) -> bool:
    ot = option_type.strip().upper()
    if ot in ("CE", "C", "CALL"):
        return True
    if ot in ("PE", "P", "PUT"):
        return False
    raise ValueError(f"unknown option_type: {option_type!r}")


def _d1_d2(s, k, t, r, sigma, q):
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def price(s, k, t, r, sigma, option_type: str, q: float = 0.0):
    """Black-Scholes option price."""
    if np.any(np.asarray(t) <= 0) or np.any(np.asarray(sigma) <= 0):
        return _intrinsic(s, k, option_type)
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    disc_r = np.exp(-r * np.asarray(t, dtype=float))
    disc_q = np.exp(-q * np.asarray(t, dtype=float))
    if _is_call(option_type):
        return s * disc_q * _norm_cdf(d1) - k * disc_r * _norm_cdf(d2)
    return k * disc_r * _norm_cdf(-d2) - s * disc_q * _norm_cdf(-d1)


def _intrinsic(s, k, option_type: str):
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    if _is_call(option_type):
        return np.maximum(s - k, 0.0)
    return np.maximum(k - s, 0.0)


@dataclass(frozen=True)
class Greeks:
    """Option sensitivities.

    delta : d(price)/d(spot)
    gamma : d(delta)/d(spot)
    vega  : d(price)/d(vol), PER 1.00 (100%) change in vol
    theta : d(price)/d(time), PER YEAR (negative for long options typically)
    rho   : d(price)/d(rate), per 1.00 change in r

    Convenience scaled views are provided as properties.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    @property
    def vega_per_pct(self) -> float:
        """Vega per 1% (0.01) change in implied volatility."""
        return self.vega / 100.0

    @property
    def theta_per_day(self) -> float:
        """Theta per calendar day."""
        return self.theta / 365.0


def greeks(s, k, t, r, sigma, option_type: str, q: float = 0.0) -> Greeks:
    """Full set of Black-Scholes Greeks for a single contract (scalars)."""
    s = float(s)
    k = float(k)
    t = float(t)
    sigma = float(sigma)
    if t <= 0 or sigma <= 0:
        # Degenerate: at/after expiry. Delta is a step; others vanish.
        intrinsic_delta = (
            1.0 if (_is_call(option_type) and s > k)
            else -1.0 if (not _is_call(option_type) and s < k)
            else 0.0
        )
        return Greeks(intrinsic_delta, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    d1 = float(d1)
    d2 = float(d2)
    pdf_d1 = float(_norm_pdf(d1))
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    call = _is_call(option_type)

    gamma = disc_q * pdf_d1 / (s * sigma * math.sqrt(t))
    vega = s * disc_q * pdf_d1 * math.sqrt(t)

    if call:
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -(s * disc_q * pdf_d1 * sigma) / (2.0 * math.sqrt(t))
            - r * k * disc_r * _norm_cdf(d2)
            + q * s * disc_q * _norm_cdf(d1)
        )
        rho = k * t * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -(s * disc_q * pdf_d1 * sigma) / (2.0 * math.sqrt(t))
            + r * k * disc_r * _norm_cdf(-d2)
            - q * s * disc_q * _norm_cdf(-d1)
        )
        rho = -k * t * disc_r * _norm_cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_volatility(
    market_price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Solve for implied volatility via bisection on a bracketed root.

    Bisection is slower than Newton but robust: it cannot diverge and handles
    near-zero-vega wings gracefully. Returns None if the price is below
    intrinsic value (no real solution) or the search fails to converge.
    """
    if t <= 0:
        return None
    intrinsic = float(_intrinsic(s, k, option_type))
    if market_price < intrinsic - tol:
        return None  # arbitrage / bad quote

    lo, hi = 1e-4, 5.0  # 0.01% .. 500% annual vol
    price_lo = float(price(s, k, t, r, lo, option_type, q))
    price_hi = float(price(s, k, t, r, hi, option_type, q))
    if not (price_lo <= market_price <= price_hi):
        # Target outside the achievable range for this bracket.
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = float(price(s, k, t, r, mid, option_type, q))
        if abs(val - market_price) < tol:
            return mid
        if val < market_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
