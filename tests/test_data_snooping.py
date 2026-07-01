"""Tests for White Reality Check / Hansen SPA, on synthetic null & alternative."""

import numpy as np

from nifty_quant.analytics.data_snooping import (
    reality_check_spa,
    stationary_bootstrap_indices,
)

RNG = np.random.default_rng(7)


def test_stationary_bootstrap_shape_and_range():
    idx = stationary_bootstrap_indices(100, 10.0, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_null_all_noise_does_not_reject():
    # K models, all pure zero-mean noise -> best is luck -> large p-values.
    T, K = 500, 10
    perf = RNG.standard_normal((T, K)) * 0.01
    res = reality_check_spa(perf, block_len=10.0, n_boot=2000, seed=1)
    assert res.reality_check_p > 0.10
    assert res.spa_p > 0.10


def test_alternative_one_genuine_winner_rejects():
    # One model has a real positive mean; the rest are noise -> reject H0.
    T, K = 500, 10
    perf = RNG.standard_normal((T, K)) * 0.01
    perf[:, 3] += 0.01  # genuine edge ~1 "unit" of the noise std
    res = reality_check_spa(perf, block_len=10.0, n_boot=2000, seed=2)
    assert res.best_model == 3
    assert res.spa_p < 0.05
    assert res.reality_check_p < 0.10


def test_spa_at_least_as_powerful_as_rc():
    # With many hopeless models diluting RC, SPA p should be <= RC p.
    T, K = 400, 20
    perf = RNG.standard_normal((T, K)) * 0.01
    perf[:, 0] += 0.006
    perf[:, 1:] -= 0.02  # many clearly-bad models
    res = reality_check_spa(perf, block_len=8.0, n_boot=2000, seed=3)
    assert res.spa_p <= res.reality_check_p + 1e-9


def test_too_short_returns_nan():
    res = reality_check_spa(np.zeros((5, 3)), n_boot=100)
    assert np.isnan(res.spa_p)
