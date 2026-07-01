"""Experiment 023: Data-snooping control (White Reality Check + Hansen SPA).

Pre-registered in docs/preregistrations/exp023_reality_check_spa.md.

Applies the family-wise, dependence-aware bootstrap tests to two real model
sets built earlier in this campaign:

  App 1  RV forecast horse-race (Exp 013): does the best RV forecaster beat the
         naive RW benchmark after accounting for trying 4 models?
  App 2  The 15-feature HAR screen (Exp 012 provenance): does the best feature
         (semivar_ratio) survive a data-snooping-corrected test, or was it a
         lucky draw from 15 candidates?

    python scripts/exp023_reality_check.py

Inference/validation only. No trades, no strategy.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling scripts

from nifty_quant.analytics.data_snooping import reality_check_spa

BLOCK_LEN = 10.0
N_BOOT = 5000


# --- App 1: Exp 013 forecast horse-race ------------------------------------

def app1_forecast_horserace() -> None:
    from exp013_rv_forecast_comparison import daily_rv, walk_forward, qlike, MODELS

    print("=" * 92)
    print("APP 1  RV FORECAST HORSE-RACE (Exp 013 set)  benchmark = RW (naive)")
    rv = daily_rv()
    fc = walk_forward(rv)
    fc = fc.dropna(subset=MODELS + ["actual"]).reset_index(drop=True)
    actual = fc["actual"].to_numpy()
    T = len(fc)
    print(f"  common OOS days: {T}")

    # QLIKE loss per model; performance = loss_RW - loss_model (higher=better)
    ql = {m: qlike(actual, fc[m].to_numpy()) for m in MODELS}
    challengers = [m for m in MODELS if m != "RW"]
    perf = np.column_stack([ql["RW"] - ql[m] for m in challengers])  # (T, K)

    res = reality_check_spa(perf, block_len=BLOCK_LEN, n_boot=N_BOOT, seed=101)
    _report(challengers, perf, res, benchmark="RW")


# --- App 2: the 15-feature HAR screen --------------------------------------

def _pooled_walk_forward_se(d: pd.DataFrame, candidates: list[str]):
    """Expanding one-step-ahead OOS squared errors for baseline HAR and each
    HAR+candidate model, on a COMMON set of OOS dates.

    Returns (se_base (T,), se_full dict[cand]->(T,)). Refit every step.
    """
    base_cols = ["har_d", "har_w", "har_m"]
    need = base_cols + ["y"] + candidates
    sub = d.dropna(subset=need).reset_index(drop=True)
    n = len(sub)
    start = n // 2
    y = sub["y"].to_numpy()
    Xb_all = sub[base_cols].to_numpy()
    cand_arr = {c: sub[c].to_numpy() for c in candidates}

    se_base = []
    se_full = {c: [] for c in candidates}

    def _fit_predict(Xtr_raw, ytr, Xte_raw):
        # explicit intercept column (avoid add_constant's constant-detection quirk)
        Xtr = np.column_stack([np.ones(len(Xtr_raw)), Xtr_raw])
        Xte = np.column_stack([np.ones(len(Xte_raw)), Xte_raw])
        beta = sm.OLS(ytr, Xtr).fit().params
        return float((Xte @ beta)[0])

    for t in range(start, n):
        ytr = y[:t]
        pb = _fit_predict(Xb_all[:t], ytr, Xb_all[t:t+1])
        se_base.append((y[t] - pb) ** 2)
        for c in candidates:
            Xf_tr = np.column_stack([Xb_all[:t], cand_arr[c][:t]])
            Xf_te = np.column_stack([Xb_all[t:t+1], cand_arr[c][t:t+1]])
            pf = _fit_predict(Xf_tr, ytr, Xf_te)
            se_full[c].append((y[t] - pf) ** 2)
    return np.array(se_base), {c: np.array(v) for c, v in se_full.items()}


def app2_feature_screen() -> None:
    from volatility_feature_search import (
        build_daily, load_bars, har_features, CANDIDATES)

    print("\n" + "=" * 92)
    print("APP 2  15-FEATURE HAR SCREEN (Exp 012 provenance)  benchmark = HAR-only")
    daily = build_daily(load_bars())
    d = har_features(daily)
    se_base, se_full = _pooled_walk_forward_se(d, CANDIDATES)
    T = len(se_base)
    print(f"  pooled one-step OOS days: {T}  candidates: {len(CANDIDATES)}")

    # performance = baseline SE - candidate SE (higher=better forecast)
    cols = [c for c in CANDIDATES if c in se_full]
    perf = np.column_stack([se_base - se_full[c] for c in cols])  # (T, K)
    res = reality_check_spa(perf, block_len=BLOCK_LEN, n_boot=N_BOOT, seed=202)
    _report(cols, perf, res, benchmark="HAR-only")


# --- shared reporting -------------------------------------------------------

def _report(names, perf, res, benchmark: str) -> None:
    fbar = perf.mean(axis=0)
    order = np.argsort(fbar)[::-1]
    print(f"  per-model mean performance vs {benchmark} (higher=better), top 6:")
    for j in order[:6]:
        print(f"    {names[j]:<22} mean={fbar[j]:+.3e}  t={res.t_stats[j]:+.2f}")
    print(f"  BEST: {names[res.best_model]}  (mean {res.best_mean:+.3e})")
    print(f"  White Reality Check p = {res.reality_check_p:.4f}")
    print(f"  Hansen SPA (consistent) p = {res.spa_p:.4f}")
    if res.spa_p < 0.05 and res.reality_check_p < 0.10:
        verd = "PASSES snooping control (edge not explained by the search)"
    elif res.spa_p >= 0.10:
        verd = "FAILS / snooping-explained (edge consistent with luck)"
    else:
        verd = "BORDERLINE (weak; 0.05 <= SPA p < 0.10)"
    print(f"  --> VERDICT: {verd}")


def main() -> int:
    app1_forecast_horserace()
    app2_feature_screen()
    print("\n" + "=" * 92)
    print("Scope: data-snooping inference control. No trading rule implied.")
    print("Pre-committed: if App 2 fails, Exp 012 confidence is downgraded in its note.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
