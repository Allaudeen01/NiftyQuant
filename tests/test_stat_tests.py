"""Tests for analytics.stat_tests, validated against known synthetic processes."""

import math

import numpy as np
import pytest

from nifty_quant.analytics.stat_tests import (
    adf_test,
    fractional_kelly,
    half_life,
    hurst_exponent,
    kelly_fraction,
    kelly_gaussian,
    variance_ratio,
)

RNG = np.random.default_rng(0)


def _random_walk(n=3000):
    return np.cumsum(RNG.standard_normal(n))


def _ar1(n=3000, phi=0.7):
    """Stationary mean-reverting AR(1): x_t = phi*x_{t-1} + e."""
    x = np.zeros(n)
    e = RNG.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _trending(n=3000, drift=0.5):
    return np.cumsum(drift + RNG.standard_normal(n))


# --- Hurst ----------------------------------------------------------------

def test_hurst_random_walk_near_half():
    h = hurst_exponent(_random_walk())
    assert 0.40 < h < 0.60


def test_hurst_mean_reverting_below_half():
    h = hurst_exponent(_ar1(phi=0.5))
    assert h < 0.5


def test_hurst_trending_above_half():
    h = hurst_exponent(_trending(drift=0.8))
    assert h > 0.5


def test_hurst_too_short_is_nan():
    assert math.isnan(hurst_exponent([1, 2, 3]))


# --- half-life ------------------------------------------------------------

def test_half_life_mean_reverting_finite_positive():
    hl = half_life(_ar1(phi=0.7))
    # AR(1) phi=0.7 -> lambda=phi-1=-0.3 -> half-life = -ln2/ln? approx -ln2/(phi-1)
    assert 0 < hl < 20
    # theoretical OU half-life ~ -ln(2)/(phi-1)
    assert hl == pytest.approx(-math.log(2) / (0.7 - 1), rel=0.4)


def test_half_life_random_walk_is_large_or_inf():
    hl = half_life(_random_walk())
    assert hl > 50 or math.isinf(hl)


# --- ADF ------------------------------------------------------------------

def test_adf_stationary_for_ar1():
    res = adf_test(_ar1(phi=0.5))
    assert res["stationary_5pct"] is True
    assert res["pvalue"] < 0.05


def test_adf_nonstationary_for_random_walk():
    res = adf_test(_random_walk())
    assert res["stationary_5pct"] is False
    assert res["pvalue"] > 0.05


# --- variance ratio -------------------------------------------------------

def test_variance_ratio_random_walk_near_one():
    vr = variance_ratio(_random_walk(), q=2)
    assert 0.85 < vr.vr < 1.15
    assert vr.pvalue > 0.05  # cannot reject random walk


def test_variance_ratio_mean_reverting_below_one():
    vr = variance_ratio(_ar1(phi=0.3), q=2)
    assert vr.vr < 1.0


def test_variance_ratio_trending_above_one():
    # strong positive autocorrelation in increments -> VR > 1
    base = _ar1(phi=0.6)
    momentum_prices = np.cumsum(base)  # integrate -> trending increments
    vr = variance_ratio(momentum_prices, q=2)
    assert vr.vr > 1.0


# --- Kelly ----------------------------------------------------------------

def test_kelly_fraction_even_money_edge():
    # 60% win, 1:1 payoff -> f* = 0.6 - 0.4/1 = 0.2
    assert kelly_fraction(0.60, 1.0) == pytest.approx(0.20)


def test_kelly_fraction_no_edge_nonpositive():
    # 50% win, 1:1 payoff -> f* = 0
    assert kelly_fraction(0.50, 1.0) == pytest.approx(0.0)
    # losing edge -> negative (don't bet)
    assert kelly_fraction(0.40, 1.0) < 0


def test_kelly_fraction_payoff_skew():
    # 50% win, 2:1 payoff -> f* = 0.5 - 0.5/2 = 0.25
    assert kelly_fraction(0.50, 2.0) == pytest.approx(0.25)


def test_kelly_gaussian_and_fractional():
    f = kelly_gaussian(mean_return=0.001, variance=0.01)  # = 0.1
    assert f == pytest.approx(0.1)
    # half-kelly, capped
    assert fractional_kelly(f, fraction=0.5) == pytest.approx(0.05)
    assert fractional_kelly(2.0, fraction=0.5, cap=1.0) == pytest.approx(1.0)
    # negative full-kelly -> 0 (no bet)
    assert fractional_kelly(-0.3) == 0.0


def test_kelly_gaussian_zero_variance():
    assert kelly_gaussian(0.01, 0.0) == 0.0
