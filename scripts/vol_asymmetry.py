"""Experiment 006A: Volatility Asymmetry (leverage effect). Read-only research.

ONE new variable vs Exp 003: today's signed daily return. Question: after
controlling for today's realized volatility, does today's return SIGN explain
tomorrow's realized volatility (negative returns -> higher future RV)?

Object: RV_{t+1} (intraday realized vol from 5m open-to-close, as Exp 003).
Control: RV_t (log). New predictor: today's signed daily return (and a
downside-only term). No parameters optimised.

    python scripts/vol_asymmetry.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss

RNG = np.random.default_rng(23)


def daily() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    rows = []
    for d, g in df.groupby("date"):
        if len(g) < 12:
            continue
        g = g.sort_values("timestamp")
        c = g["close"].to_numpy()
        o = float(g["open"].iloc[0])
        lr = np.log(c[1:] / c[:-1])
        rv = math.sqrt(float(np.sum(lr ** 2)))
        # today's intraday return (open->close); sign is the new variable
        ret = c[-1] / o - 1.0
        rows.append({"date": d, "year": d.year, "rv": rv, "ret": ret})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    out["rv_next"] = out["rv"].shift(-1)
    out["log_rv_next"] = out["log_rv"].shift(-1)
    out["ret_neg"] = np.minimum(out["ret"], 0.0)   # downside-only term
    return out.dropna(subset=["rv_next"]).reset_index(drop=True)


def hac(y, X, lags=10):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC",
                                             cov_kwds={"maxlags": lags})


def partial_corr(x, y, z):
    """corr(x,y | z) via residuals of x~z and y~z."""
    def resid(a, b):
        b1 = sm.add_constant(b)
        return a - sm.OLS(a, b1).fit().predict(b1)
    rx = resid(x, z); ry = resid(y, z)
    r, p = ss.pearsonr(rx, ry)
    return r, p


def main() -> int:
    d = daily()
    n = len(d)
    print("=" * 92)
    print("EXP 006A  VOLATILITY ASYMMETRY (LEVERAGE EFFECT)  NIFTY 5m")
    print(f"Obs: {n}  {d['date'].min().date()} -> {d['date'].max().date()}")
    print(f"mean ret {d['ret'].mean()*100:.3f}%  "
          f"neg-return days {(d['ret']<0).mean()*100:.1f}%")

    # 1) Raw correlations with next-day RV
    print("\n--- 1) Correlations with RV_next ---")
    for nm, col in [("today RV", "rv"), ("today ret", "ret"),
                    ("today |ret|", None), ("today ret_neg", "ret_neg")]:
        x = d["ret"].abs() if col is None else d[col]
        r, p = ss.pearsonr(x, d["rv_next"])
        print(f"  corr(RV_next, {nm:<12}) = {r:+.3f}  p={p:.2e}")

    # 2) Partial correlation: ret vs RV_next controlling for today's RV
    pr, pp = partial_corr(d["ret"].to_numpy(), d["rv_next"].to_numpy(),
                          d["log_rv"].to_numpy())
    prn, ppn = partial_corr(d["ret_neg"].to_numpy(), d["rv_next"].to_numpy(),
                            d["log_rv"].to_numpy())
    print("\n--- 2) Partial correlation (control = today's log RV) ---")
    print(f"  partial corr(RV_next, today ret    | RV) = {pr:+.3f}  p={pp:.2e}")
    print(f"  partial corr(RV_next, ret_neg      | RV) = {prn:+.3f}  p={ppn:.2e}")

    # 3) Regression: log RV_next ~ log RV_t + ret_t  (HAC)
    print("\n--- 3) HAC regression  log_RV_next ~ log_RV + ret ---")
    base = hac(d["log_rv_next"], d[["log_rv"]].to_numpy())
    m1 = hac(d["log_rv_next"], d[["log_rv", "ret"]].to_numpy())
    for nm, b, t in zip(["const", "log_RV", "ret"], m1.params, m1.tvalues):
        print(f"    {nm:<7} coef={b: .3f}  HAC t={t: .2f}")
    print(f"    R2 base(RV only)={base.rsquared:.3f}  +ret={m1.rsquared:.3f}  "
          f"partial_R2={(m1.rsquared-base.rsquared):.4f}")

    # 3b) asymmetric spec: add downside-only term
    print("\n--- 3b) HAC regression  log_RV_next ~ log_RV + ret + ret_neg ---")
    m2 = hac(d["log_rv_next"], d[["log_rv", "ret", "ret_neg"]].to_numpy())
    for nm, b, t in zip(["const", "log_RV", "ret", "ret_neg"],
                        m2.params, m2.tvalues):
        print(f"    {nm:<8} coef={b: .3f}  HAC t={t: .2f}")
    print(f"    R2={m2.rsquared:.3f}  partial_R2 over RV={(m2.rsquared-base.rsquared):.4f}")

    # 4) Quantile buckets of today's return -> next-day RV
    print("\n--- 4) Next-day RV (annualised %) by today's return quintile ---")
    d2 = d.copy()
    d2["bucket"] = pd.qcut(d2["ret"], 5,
                           labels=["LargeDown", "SmallDown", "Neutral",
                                   "SmallUp", "LargeUp"])
    ann = math.sqrt(252) * 100
    for b in ["LargeDown", "SmallDown", "Neutral", "SmallUp", "LargeUp"]:
        sub = d2[d2["bucket"] == b]["rv_next"].to_numpy()
        bs = sub[RNG.integers(0, len(sub), size=(4000, len(sub)))].mean(axis=1)
        print(f"  {b:<10} n={len(sub):>3}  meanRV_next={sub.mean()*ann:>5.2f}%  "
              f"median={np.median(sub)*ann:>5.2f}%  "
              f"CI95=[{np.percentile(bs,2.5)*ann:.2f},{np.percentile(bs,97.5)*ann:.2f}]%")
    ld = d2[d2["bucket"] == "LargeDown"]["rv_next"].to_numpy()
    lu = d2[d2["bucket"] == "LargeUp"]["rv_next"].to_numpy()
    tt = ss.ttest_ind(ld, lu, equal_var=False)
    mw = ss.mannwhitneyu(ld, lu, alternative="greater")
    print(f"  LargeDown vs LargeUp: Welch t p={tt.pvalue:.3f}  "
          f"MWU(greater) p={mw.pvalue:.3f}")

    # 5) Walk-forward OOS: does ret add OOS predictive power over RV-only?
    print("\n--- 5) Walk-forward OOS R2 (RV+ret vs RV-only baseline) ---")
    feat = d[["log_rv", "ret", "ret_neg"]].to_numpy()
    y = d["log_rv_next"].to_numpy()
    N = len(d); start = N // 2
    edges = np.linspace(start, N, 5, dtype=int)
    for i in range(4):
        a, b = edges[i], edges[i + 1]
        Xtr, ytr = feat[:a], y[:a]
        Xte, yte = feat[a:b], y[a:b]
        if len(yte) < 20:
            print(f"  fold {i+1}: Insufficient (n={len(yte)})"); continue
        # baseline: RV only
        fb = sm.OLS(ytr, sm.add_constant(Xtr[:, :1])).fit()
        pb = fb.predict(sm.add_constant(Xte[:, :1], has_constant="add"))
        # full: RV + ret + ret_neg
        ff = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
        pf = ff.predict(sm.add_constant(Xte, has_constant="add"))
        sse_f = np.sum((yte - pf) ** 2); sse_b = np.sum((yte - pb) ** 2)
        r2_incr = 1 - sse_f / sse_b
        print(f"  fold {i+1}: n={len(yte):3d}  OOS_R2(full vs RV-only)={r2_incr:+.4f}")

    # 6) Per-year regression sign of ret / ret_neg
    print("\n--- 6) Per-year HAC: log_RV_next ~ log_RV + ret + ret_neg ---")
    for yv in sorted(d["year"].unique()):
        s = d[d["year"] == yv]
        if len(s) < 40:
            print(f"  {yv}: Insufficient (n={len(s)})"); continue
        m = hac(s["log_rv_next"], s[["log_rv", "ret", "ret_neg"]].to_numpy())
        pr = np.asarray(m.params); tv = np.asarray(m.tvalues)
        print(f"  {yv}: n={len(s):3d}  ret coef={pr[2]:+.2f} t={tv[2]:+.2f}"
              f"  ret_neg coef={pr[3]:+.2f} t={tv[3]:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
