"""Experiment 009: Intraday return autocorrelation after volatility normalization.

ONE new variable vs Exp 008: the SIGNED 5-minute return. We deseasonalize each
return by its time-of-day volatility (the Exp 008 clock), then test whether the
normalized returns show significant autocorrelation -- a structural property,
not a strategy.

normalized_return = 5m_return / sd_of_returns_at_that_time_of_day

Tests: ACF/PACF at lags 1,2,3,5,10,20; Ljung-Box; variance-ratio; runs test;
bootstrap CIs. Repeated by session, by volatility regime (Exp 003/004 HAR
forecast terciles), walk-forward, and per-year. BH-FDR over the lag-1 tests.

No parameters optimised.  python scripts/intraday_autocorr.py
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
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(37)
LAGS = [1, 2, 3, 5, 10, 20]


def load_bars() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    df["year"] = df["timestamp"].dt.year
    df = df[(df["tod"] >= "09:15") & (df["tod"] <= "15:30")]   # quality gate
    return df


def build_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar signed log returns within each day (no overnight)."""
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        if len(g) < 12:
            continue
        c = g["close"].to_numpy()
        tod = g["tod"].to_numpy()
        yr = int(g["year"].iloc[0])
        r = np.log(c[1:] / c[:-1])
        for i in range(len(r)):
            rows.append({"date": d, "year": yr, "tod": tod[i + 1],
                         "ret": r[i], "pos": i})
    return pd.DataFrame(rows)


def regime_map() -> dict:
    """Daily HAR one-step-ahead RV forecast -> LOW/NORMAL/HIGH (Exp 003/004)."""
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
        c = g.sort_values("timestamp")["close"].to_numpy()
        rows.append({"date": d, "log_rv": math.log(max(
            math.sqrt(float(np.sum(np.log(c[1:] / c[:-1]) ** 2))), 1e-9))})
    rv = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    s = rv["log_rv"]
    feat = pd.DataFrame({"d": s.shift(1), "w": s.shift(1).rolling(5).mean(),
                         "m": s.shift(1).rolling(22).mean(), "y": s,
                         "date": rv["date"]}).dropna().reset_index(drop=True)
    preds = [np.nan] * len(feat)
    for i in range(60, len(feat)):
        tr = feat.iloc[:i]
        fit = sm.OLS(tr["y"].to_numpy(),
                     sm.add_constant(tr[["d", "w", "m"]].to_numpy())).fit()
        xr = np.r_[1.0, feat.loc[i, ["d", "w", "m"]].to_numpy(dtype=float)]
        preds[i] = float(fit.predict(xr.reshape(1, -1))[0])
    feat["f"] = preds
    feat = feat.dropna(subset=["f"])
    lo, hi = feat["f"].quantile([0.30, 0.70])
    lab = np.where(feat["f"] < lo, "LOW", np.where(feat["f"] >= hi, "HIGH", "NORMAL"))
    return dict(zip(feat["date"], lab))


def session_of(tod: str) -> str:
    h, m = map(int, tod.split(":")); x = h * 60 + m
    if x <= 9 * 60 + 20:
        return "Opening"
    if x < 11 * 60 + 30:
        return "Morning"
    if x < 13 * 60:
        return "Lunch"
    if x < 15 * 60 + 5:
        return "Afternoon"
    return "Closing"


def acf_within_day(dfn: pd.DataFrame, lag: int) -> float:
    """Pooled lag-k autocorr of normalized returns, computed WITHIN each day
    (never crossing the overnight boundary)."""
    xs, ys = [], []
    for _, g in dfn.groupby("date"):
        v = g.sort_values("pos")["z"].to_numpy()
        if len(v) > lag:
            xs.append(v[:-lag]); ys.append(v[lag:])
    if not xs:
        return float("nan")
    x = np.concatenate(xs); y = np.concatenate(ys)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def boot_acf_ci(dfn, lag, n=1000):
    """Bootstrap CI by resampling whole days (preserves within-day structure)."""
    days = dfn["date"].unique()
    series_by_day = {d: g.sort_values("pos")["z"].to_numpy()
                     for d, g in dfn.groupby("date")}
    acfs = []
    for _ in range(n):
        samp = RNG.choice(days, size=len(days), replace=True)
        xs, ys = [], []
        for d in samp:
            v = series_by_day[d]
            if len(v) > lag:
                xs.append(v[:-lag]); ys.append(v[lag:])
        if not xs:
            continue
        x = np.concatenate(xs); y = np.concatenate(ys)
        if x.std() > 0 and y.std() > 0:
            acfs.append(np.corrcoef(x, y)[0, 1])
    if len(acfs) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(acfs, 2.5)), float(np.percentile(acfs, 97.5)))


def main() -> int:
    df = load_bars()
    rets = build_returns(df)

    # --- Volatility clock: sd of returns by time-of-day -> normalize -------
    tod_sd = rets.groupby("tod")["ret"].std()
    rets["z"] = rets["ret"] / rets["tod"].map(tod_sd)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
    print("=" * 92)
    print("EXP 009  INTRADAY RETURN AUTOCORRELATION (vol-normalized)  NIFTY 5m")
    print(f"Bars: {len(rets)}  days: {rets['date'].nunique()}")
    print(f"normalized z: mean={rets['z'].mean():.4f} std={rets['z'].std():.3f}")

    # --- Overall ACF + bootstrap CI + Ljung-Box ---------------------------
    print("\n--- Pooled within-day ACF of normalized returns ---")
    raw_p = {}
    for lag in LAGS:
        r = acf_within_day(rets, lag)
        lo, hi = boot_acf_ci(rets, lag, n=600)
        sig = "" if (lo <= 0 <= hi) else "  *CI excl 0*"
        # approx p via normal SE ~ 1/sqrt(N)
        N = len(rets)
        z = r * math.sqrt(N)
        p = 2 * (1 - ss.norm.cdf(abs(z)))
        raw_p[lag] = p
        print(f"  lag {lag:>2}: acf={r:+.4f}  boot95=[{lo:+.4f},{hi:+.4f}]  "
              f"approx p={p:.2e}{sig}")

    # Ljung-Box on a representative day-stacked series (within-day, lag up to 20)
    # Build a single concatenated series with NaN breaks handled by per-day LB avg
    lb_lag1 = []
    for _, g in rets.groupby("date"):
        v = g.sort_values("pos")["z"].to_numpy()
        if len(v) > 21:
            lb = acorr_ljungbox(v, lags=[1], return_df=True)
            lb_lag1.append(lb["lb_pvalue"].iloc[0])
    print(f"\n  Ljung-Box lag1 per-day: median p={np.median(lb_lag1):.3f}  "
          f"frac p<0.05 = {np.mean(np.array(lb_lag1) < 0.05):.2f}")

    # --- Variance Ratio test (Lo-MacKinlay style) on lag-1 within days -----
    def variance_ratio(dfn, q=2):
        vrs = []
        for _, g in dfn.groupby("date"):
            v = g.sort_values("pos")["z"].to_numpy()
            if len(v) <= q + 1:
                continue
            var1 = np.var(v, ddof=1)
            agg = np.add.reduceat(v, np.arange(0, len(v) - len(v) % q, q))
            varq = np.var(agg, ddof=1) / q
            if var1 > 0:
                vrs.append(varq / var1)
        return np.mean(vrs)
    vr2 = variance_ratio(rets, 2)
    print(f"  Variance Ratio VR(2) (mean over days) = {vr2:.3f}  "
          f"(<1 => mean reversion, >1 => momentum)")

    # --- Runs test (sign randomness) on pooled z --------------------------
    signs = np.sign(rets["z"].to_numpy()); signs = signs[signs != 0]
    n_pos = (signs > 0).sum(); n_neg = (signs < 0).sum()
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    mu = 2 * n_pos * n_neg / (n_pos + n_neg) + 1
    var = (mu - 1) * (mu - 2) / (n_pos + n_neg - 1)
    z_runs = (runs - mu) / math.sqrt(var)
    print(f"  Runs test: runs={runs} expected={mu:.0f} z={z_runs:.2f} "
          f"p={2*(1-ss.norm.cdf(abs(z_runs))):.2e}")

    # --- By session -------------------------------------------------------
    print("\n--- Lag-1 ACF by session ---")
    rets["session"] = rets["tod"].map(session_of)
    sess_p = {}
    for s in ["Opening", "Morning", "Lunch", "Afternoon", "Closing"]:
        sub = rets[rets["session"] == s]
        r = acf_within_day(sub, 1)
        lo, hi = boot_acf_ci(sub, 1, n=500)
        sess_p[s] = 2 * (1 - ss.norm.cdf(abs(r * math.sqrt(len(sub)))))
        print(f"  {s:<10} n={len(sub):>6} acf1={r:+.4f} boot95=[{lo:+.4f},{hi:+.4f}]")

    # --- By volatility regime --------------------------------------------
    print("\n--- Lag-1 ACF by volatility regime (HAR forecast tercile) ---")
    reg = regime_map()
    rets["regime"] = rets["date"].map(reg)
    reg_p = {}
    for g in ["LOW", "NORMAL", "HIGH"]:
        sub = rets[rets["regime"] == g]
        if len(sub) < 100:
            print(f"  {g}: insufficient"); continue
        r = acf_within_day(sub, 1)
        lo, hi = boot_acf_ci(sub, 1, n=500)
        reg_p[g] = 2 * (1 - ss.norm.cdf(abs(r * math.sqrt(len(sub)))))
        print(f"  {g:<7} n={len(sub):>6} acf1={r:+.4f} boot95=[{lo:+.4f},{hi:+.4f}]")

    # --- Per-year & walk-forward (lag-1) ----------------------------------
    print("\n--- Lag-1 ACF per year ---")
    for yv in sorted(rets["year"].unique()):
        sub = rets[rets["year"] == yv]
        r = acf_within_day(sub, 1)
        print(f"  {yv}: n={len(sub):>6} acf1={r:+.4f}")
    print("\n--- Lag-1 ACF walk-forward (4 folds) ---")
    days = np.array(sorted(rets["date"].unique()))
    for i, fold in enumerate(np.array_split(days, 4)):
        sub = rets[rets["date"].isin(pd.DatetimeIndex(fold))]
        r = acf_within_day(sub, 1)
        print(f"  fold {i+1}: n={len(sub):>6} acf1={r:+.4f}")

    # --- BH-FDR over the 6 lag tests --------------------------------------
    pv = [raw_p[l] for l in LAGS]
    _, adj, _, _ = multipletests(pv, method="fdr_bh")
    print("\n--- Benjamini-Hochberg FDR over lags ---")
    for l, p, a in zip(LAGS, pv, adj):
        print(f"  lag {l:>2}: raw p={p:.2e}  adj p={a:.2e}  "
              f"{'sig' if a < 0.05 else 'ns'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
