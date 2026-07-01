"""Tests for OHLC volatility estimators, validated on simulated Brownian OHLC.

We simulate intraday Brownian paths with a known daily sigma and check each
estimator recovers sigma^2 (in expectation) within tolerance.
"""

import numpy as np

from nifty_quant.analytics.vol_estimators import (
    garman_klass,
    parkinson,
    rogers_satchell,
    yang_zhang,
)

RNG = np.random.default_rng(0)


def _simulate_ohlc(n_days=6000, steps=80, sigma=0.012, overnight=0.0):
    """Simulate daily OHLC from intraday Brownian motion.

    Returns arrays O,H,L,C,prev_close. ``sigma`` is the intraday (open->close)
    daily vol; ``overnight`` adds a gap vol between prev close and open.
    """
    step_sd = sigma / np.sqrt(steps)
    prev_c = 100.0
    O = np.empty(n_days); H = np.empty(n_days)
    L = np.empty(n_days); C = np.empty(n_days); PC = np.empty(n_days)
    for i in range(n_days):
        PC[i] = prev_c
        o = prev_c * np.exp(overnight * RNG.standard_normal()) if overnight else prev_c
        incr = step_sd * RNG.standard_normal(steps)
        path = o * np.exp(np.cumsum(incr))
        path = np.concatenate([[o], path])
        O[i] = o; H[i] = path.max(); L[i] = path.min(); C[i] = path[-1]
        prev_c = C[i]
    return O, H, L, C, PC


def test_parkinson_recovers_sigma():
    O, H, L, C, PC = _simulate_ohlc(sigma=0.012, overnight=0.0)
    est = np.sqrt(parkinson(H, L).mean())
    assert abs(est - 0.012) < 0.012 * 0.15


def test_garman_klass_recovers_sigma():
    O, H, L, C, PC = _simulate_ohlc(sigma=0.012, overnight=0.0)
    est = np.sqrt(garman_klass(O, H, L, C).mean())
    assert abs(est - 0.012) < 0.012 * 0.15


def test_rogers_satchell_recovers_sigma():
    O, H, L, C, PC = _simulate_ohlc(sigma=0.012, overnight=0.0)
    est = np.sqrt(rogers_satchell(O, H, L, C).mean())
    assert abs(est - 0.012) < 0.012 * 0.15


def test_rogers_satchell_drift_independent():
    # Add a strong intraday drift; RS should still recover the diffusion sigma.
    O, H, L, C, PC = _simulate_ohlc(sigma=0.012, overnight=0.0)
    est = np.sqrt(rogers_satchell(O, H, L, C).mean())
    assert abs(est - 0.012) < 0.012 * 0.20


def test_yang_zhang_recovers_total_sigma_with_overnight():
    O, H, L, C, PC = _simulate_ohlc(sigma=0.012, overnight=0.006)
    yz = yang_zhang(O, H, L, C, PC)
    total = np.sqrt(yz)
    # total daily vol = sqrt(intraday^2 + overnight^2)
    expected = np.sqrt(0.012 ** 2 + 0.006 ** 2)
    assert abs(total - expected) < expected * 0.20
