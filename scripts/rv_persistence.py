"""Experiment 003: NIFTY realized-volatility persistence (read-only research).

ONE question: is NIFTY daily realized volatility serially predictable
(volatility clustering), or is it serially unpredictable?

Object of study: next-day INTRADAY realized volatility, RV_{t+1}, built from
summed squared 5-minute open-to-close log returns (overnight gap excluded -- it
was characterised in Exp 001/002). Predictor: RV_t and its HAR components.

No parameters are optimised. HAR components are the definitional 1/5/22-day
windows. Out-of-sample is judged against two naive baselines.

    python scripts/rv_persistence.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss
from statsmodels.stats.diagnostic import acorr_ljungbox

ANNUALISE = math.sqrt(252)


def daily_realized_vol() -> pd.DataFrame:
    """One row per trading day: intraday realized vol from 5m open-to-close."""
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    if not files:
        raise SystemExit("No 5m NIFTY parquet found")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()

    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        if len(g) < 12:                      # skip very short/partial sessions
            continue
        logret = np.log(g["close"].to_numpy()[1:] / g["close"].to_numpy()[:-1])
        rv = math.sqrt(float(np.sum(logret ** 2)))   # daily realized vol
        rows.append({"date": d, "year": d.year, "rv": rv, "bars": len(g)})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    out["rv_ann"] = out["rv"] * ANNUALISE
    return out


def hac_ols(y, X, lags=10):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def oos_r2(actual, pred, bench):
    actual, pred, bench = map(np.asarray, (actual, pred, bench))
    sse = np.sum((actual - pred) ** 2)
    sst = np.sum((actual - bench) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def main() -> int:
    rv = daily_realized_vol()
    n = len(rv)
    print("=" * 92)
    print("EXP 003  NIFTY REALIZED-VOLATILITY PERSISTENCE  (intraday RV from 5m)")
    print(f"Trading days: {n}  | {rv['date'].min().date()} -> {rv['date'].max().date()}")
    print(f"Mean annualised RV: {rv['rv_ann'].mean()*100:.1f}%  "
          f"min {rv['rv_ann'].min()*100:.1f}%  max {rv['rv_ann'].max()*100:.1f}%")

    lr = rv["log_rv"].to_numpy()

    # --- 1) Autocorrelation of log-RV + Ljung-Box ----------------------------
    print("\n--- Autocorrelation of log-RV ---")
    def acf(x, k):
        x = x - x.mean()
        return float(np.sum(x[k:] * x[:-k]) / np.sum(x * x))
    for k in (1, 2, 3, 5, 10, 22):
        print(f"  lag {k:>2}: rho = {acf(lr, k): .3f}")
    lb = acorr_ljungbox(lr, lags=[1, 5, 10, 22], return_df=True)
    print("  Ljung-Box (H0=no autocorrelation):")
    for lag, row in lb.iterrows():
        print(f"    lag {lag:>2}: stat={row['lb_stat']:.1f}  p={row['lb_pvalue']:.2e}")

    # --- 2) AR(1) on log-RV with HAC SE --------------------------------------
    print("\n--- AR(1):  log_RV[t+1] ~ const + log_RV[t]   (HAC SE) ---")
    y = lr[1:]
    x = lr[:-1]
    m = hac_ols(y, x)
    beta = m.params[1]; tval = m.tvalues[1]
    print(f"  beta = {beta:.3f}  HAC t = {tval:.2f}  R2 = {m.rsquared:.3f}")
    half_life = math.log(0.5) / math.log(abs(beta)) if 0 < beta < 1 else float("nan")
    print(f"  shock half-life = {half_life:.1f} days")

    # --- 3) HAR-RV (daily/weekly/monthly), HAC SE ----------------------------
    print("\n--- HAR-RV:  RV[t+1] ~ RV_d + RV_w + RV_m   (log space, HAC SE) ---")
    s = pd.Series(lr)
    rv_d = s.shift(0)
    rv_w = s.rolling(5).mean()
    rv_m = s.rolling(22).mean()
    har = pd.DataFrame({"y": s.shift(-1), "d": rv_d, "w": rv_w, "m": rv_m}).dropna()
    hm = hac_ols(har["y"].to_numpy(), har[["d", "w", "m"]].to_numpy())
    for nm, b, t in zip(["const", "RV_d", "RV_w", "RV_m"], hm.params, hm.tvalues):
        print(f"    {nm:<6} coef={b: .3f}  HAC t={t: .2f}")
    print(f"  In-sample R2 = {hm.rsquared:.3f}")

    # --- 4) Walk-forward OOS vs naive baselines ------------------------------
    print("\n--- 4-fold expanding walk-forward (OOS R2 vs unconditional mean) ---")
    har_full = pd.DataFrame({"y": s.shift(-1), "d": rv_d, "w": rv_w, "m": rv_m,
                             "rw_naive": rv_d}).dropna().reset_index(drop=True)
    N = len(har_full)
    start = N // 2  # first half always in-sample
    fold_bounds = np.linspace(start, N, 5, dtype=int)
    for i in range(4):
        a, b = fold_bounds[i], fold_bounds[i + 1]
        train = har_full.iloc[:a]
        test = har_full.iloc[a:b]
        if len(test) < 20:
            print(f"  fold {i+1}: Insufficient Evidence (n_test={len(test)})")
            continue
        Xtr = sm.add_constant(train[["d", "w", "m"]].to_numpy())
        fit = sm.OLS(train["y"].to_numpy(), Xtr).fit()
        Xte = sm.add_constant(test[["d", "w", "m"]].to_numpy(), has_constant="add")
        pred = fit.predict(Xte)
        bench_mean = np.full(len(test), train["y"].mean())   # unconditional mean
        pred_rw = test["rw_naive"].to_numpy()                # "tomorrow=today"
        r2_mean = oos_r2(test["y"], pred, bench_mean)
        r2_rw = oos_r2(test["y"], pred, pred_rw)
        print(f"  fold {i+1}: n_test={len(test):3d}  "
              f"OOS_R2 vs mean = {r2_mean: .3f}   vs random-walk = {r2_rw: .3f}")

    # --- 5) Robustness: per-year AR(1) ---------------------------------------
    print("\n--- Per-year AR(1) beta (log-RV) ---")
    for yv in sorted(rv["year"].unique()):
        sub = rv[rv["year"] == yv]["log_rv"].to_numpy()
        if len(sub) < 30:
            print(f"  {yv}: Insufficient Evidence (n={len(sub)})")
            continue
        b = hac_ols(sub[1:], sub[:-1])
        print(f"  {yv}: n={len(sub):3d}  beta={b.params[1]: .3f}  t={b.tvalues[1]: .2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
