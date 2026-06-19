"""Tests for Black-Scholes pricing, Greeks, and implied vol."""

import math

import pytest

from nifty_quant.analytics import black_scholes as bs


# Reference values for ATM-ish option, standard textbook inputs.
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_put_call_parity():
    call = bs.price(S, K, T, R, SIGMA, "CE")
    put = bs.price(S, K, T, R, SIGMA, "PE")
    # C - P = S - K * e^{-rT}
    lhs = call - put
    rhs = S - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_call_price_known_value():
    # Standard reference: S=100,K=100,T=1,r=5%,sigma=20% -> call ~ 10.4506
    call = bs.price(S, K, T, R, SIGMA, "call")
    assert call == pytest.approx(10.4506, abs=1e-3)


def test_call_delta_bounds():
    g = bs.greeks(S, K, T, R, SIGMA, "CE")
    assert 0.0 < g.delta < 1.0
    # ATM call delta is a bit above 0.5 with positive drift.
    assert g.delta == pytest.approx(0.6368, abs=1e-3)


def test_put_delta_negative():
    g = bs.greeks(S, K, T, R, SIGMA, "PE")
    assert -1.0 < g.delta < 0.0


def test_gamma_equal_for_call_and_put():
    gc = bs.greeks(S, K, T, R, SIGMA, "CE")
    gp = bs.greeks(S, K, T, R, SIGMA, "PE")
    assert gc.gamma == pytest.approx(gp.gamma, abs=1e-9)
    assert gc.vega == pytest.approx(gp.vega, abs=1e-9)


def test_theta_negative_for_long_call():
    g = bs.greeks(S, K, T, R, SIGMA, "CE")
    assert g.theta < 0  # long option bleeds time value
    assert g.theta_per_day == pytest.approx(g.theta / 365.0)


def test_implied_vol_roundtrip():
    target = bs.price(S, K, T, R, SIGMA, "CE")
    iv = bs.implied_volatility(target, S, K, T, R, "CE")
    assert iv == pytest.approx(SIGMA, abs=1e-4)


def test_implied_vol_below_intrinsic_returns_none():
    # Price below intrinsic value has no valid IV.
    deep_itm_intrinsic = max(S - K, 0.0)
    iv = bs.implied_volatility(deep_itm_intrinsic - 1.0, 150.0, 100.0, T, R, "CE")
    assert iv is None


def test_price_at_expiry_is_intrinsic():
    assert bs.price(110, 100, 0.0, R, SIGMA, "CE") == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, R, SIGMA, "PE") == pytest.approx(10.0)


def test_unknown_option_type_raises():
    with pytest.raises(ValueError):
        bs.price(S, K, T, R, SIGMA, "XX")
