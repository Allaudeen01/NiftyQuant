"""Data-snooping controls: White's Reality Check and Hansen's SPA test.

When the BEST of K models/strategies is selected after searching over all of
them, its apparent outperformance is biased upward -- some of it is just luck
from the multiplicity of the search. These bootstrap tests give a family-wise,
dependence-aware p-value for the null "the best model does not beat the
benchmark once the search over K models is accounted for."

References (standard, public-domain):
  * Politis & Romano (1994) -- the stationary bootstrap.
  * White (2000) -- the "Reality Check" for data snooping.
  * Hansen (2005) -- the "Superior Predictive Ability" (SPA) test, which
    studentizes and removes hopeless models so the test is not diluted.

Convention: the performance matrix ``perf`` is shape (T, K); ``perf[t, k]`` is a
per-period performance of model k RELATIVE TO A BENCHMARK, oriented so that
HIGHER IS BETTER (e.g. benchmark_loss - model_loss). Positive column mean =>
model beat the benchmark on average.

Pure functions; no I/O, no global state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def stationary_bootstrap_indices(n: int, block_len: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index matrix, shape (n_boot, n).

    Geometric block lengths with mean ``block_len`` (restart probability
    p = 1/block_len). Wraps around the sample (circular).
    """
    if block_len < 1:
        block_len = 1.0
    p = 1.0 / block_len
    idx = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        cur = rng.integers(0, n)
        for t in range(n):
            if t == 0 or rng.random() < p:
                cur = rng.integers(0, n)
            else:
                cur = (cur + 1) % n
            idx[b, t] = cur
    return idx


@dataclass
class SnoopingResult:
    reality_check_p: float     # White (2000) p-value
    spa_p: float               # Hansen (2005) consistent SPA p-value
    best_model: int            # index of the max-mean model
    best_mean: float           # its mean performance (benchmark-relative)
    t_stats: np.ndarray        # studentized sqrt(T)*mean/omega per model
    n_boot: int
    block_len: float


def reality_check_spa(perf, block_len: float = 10.0, n_boot: int = 5000,
                      seed: int = 12345) -> SnoopingResult:
    """White Reality Check + Hansen SPA (consistent) from one bootstrap draw.

    ``perf``: (T, K) array, benchmark-relative, higher = better.
    Returns both p-values plus diagnostics. Small p => the best model's
    outperformance is unlikely to be pure data-snooping luck.
    """
    f = np.asarray(perf, dtype=float)
    if f.ndim != 2:
        raise ValueError("perf must be 2-D (T, K)")
    T, K = f.shape
    if T < 20 or K < 1:
        return SnoopingResult(float("nan"), float("nan"), -1, float("nan"),
                              np.full(K, np.nan), n_boot, block_len)
    rng = np.random.default_rng(seed)
    fbar = f.mean(axis=0)                       # (K,)
    sqrtT = math.sqrt(T)

    idx = stationary_bootstrap_indices(T, block_len, n_boot, rng)
    # bootstrap means: (n_boot, K)
    boot_means = np.empty((n_boot, K), dtype=float)
    for b in range(n_boot):
        boot_means[b] = f[idx[b]].mean(axis=0)

    # omega_k: bootstrap std of sqrt(T)*(fbar* - fbar)
    centered = sqrtT * (boot_means - fbar[None, :])
    omega = centered.std(axis=0, ddof=1)
    omega = np.where(omega < 1e-12, 1e-12, omega)

    # --- White's Reality Check ---------------------------------------------
    V = np.max(sqrtT * fbar)
    V_boot = np.max(centered, axis=1)           # g_k = fbar for all k
    rc_p = float(np.mean(V_boot >= V))

    # --- Hansen's SPA (consistent) -----------------------------------------
    t_stats = sqrtT * fbar / omega
    T_spa = max(0.0, float(np.max(t_stats)))
    # consistent recentering: drop models with mean far below zero
    thresh = -np.sqrt(2.0 * math.log(math.log(T))) if T >= 3 else 0.0
    keep = (t_stats >= thresh)                  # bool (K,)
    g = np.where(keep, fbar, 0.0)               # recenter kept models by their mean
    # studentized bootstrap stat; models not kept contribute Z*=sqrt(T)*fbar*/omega
    Z = sqrtT * (boot_means - g[None, :]) / omega[None, :]
    spa_boot = np.maximum(0.0, np.max(Z, axis=1))
    spa_p = float(np.mean(spa_boot >= T_spa))

    best = int(np.argmax(fbar))
    return SnoopingResult(rc_p, spa_p, best, float(fbar[best]), t_stats,
                          n_boot, block_len)
