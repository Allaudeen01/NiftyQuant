"""Experiment 011: Leverage effect x intraday session interaction (read-only).

ONE new variable vs Exp 006A: intraday SESSION (morning/lunch/afternoon).
Question: is the leverage effect (negative recent return -> higher future RV,
controlling for current RV) uniform across the day, or concentrated in a session?

Per evaluation bar t (intraday, same-day windows only, no overnight crossing):
  Current RV  = sqrt(sum of squared 5m log returns over the past 12 bars)
  Future RV   = sqrt(sum of squared 5m log returns over the next 12 bars)
  neg         = 1 if trailing-60min return (c[t]/c[t-12]) < 0 else 0
  session     = Morning(09:15-12:00) / Lunch(12:00-13:00) / Afternoon(13:00-15:30)

All regressions in log-RV space (as Exp 003/006A) with HAC (Newey-West) errors
to handle the overlapping-window autocorrelation. No parameters optimised.

    python scripts/leverage_session.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(43)
HAC_LAGS = 12
WIN = 12  # 60 min = 12 bars


def session_of(tod: str) -> str:
    h, m = map(int, tod.split(":"))
    x = h * 60 + m
    if x < 12 * 60:
        return "Morning"
    if x < 13 * 60:
        return "Lunch"
    return "Afternoon"


def build() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df = df[(df["timestamp"].dt.strftime("%H:%M") >= "09:15")
            & (df["timestamp"].dt.strftime("%H:%M") <= "15:30")]
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        c = g["close"].to_numpy()
        tod = g["timestamp"].dt.strftime("%H:%M").to_numpy()
        n = len(c)
        if n < 2 * WIN + 2:
            continue
        r = np.log(c[1:] / c[:-1])  # r[k] is return into bar k+1 (index 0..n-2)
        for t in range(WIN, n - WIN - 1):
            cur = math.sqrt(float(np.sum(r[t - WIN:t] ** 2)))      # past 12
            fut = math.sqrt(float(np.sum(r[t:t + WIN] ** 2)))      # next 12
            if cur <= 0 or fut <= 0:
                continue
            trail = math.log(c[t] / c[t - WIN])
            rows.append({
                "date": d, "year": d.year, "tod": tod[t],
                "session": session_of(tod[t]),
                "log_cur": math.log(cur), "log_fut": math.log(fut),
                "neg": 1.0 if trail < 0 else 0.0,
            })
    return pd.DataFrame(rows)


def hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC",
                                             cov_kwds={"maxlags": HAC_LAGS})


def session_beta(sub: pd.DataFrame):
    """Leverage coefficient (neg dummy) within a subset, HAC."""
    if len(sub) < 50:
        return None
    m = hac(sub["log_fut"].to_numpy(),
            sub[["log_cur", "neg"]].to_numpy())
    p = np.asarray(m.params); t = np.asarray(m.tvalues); pv = np.asarray(m.pvalues)
    return {"n": len(sub), "beta": p[2], "t": t[2], "p": pv[2]}


def boot_beta_diff(a: pd.DataFrame, b: pd.DataFrame, n=400):
    """Block(day)-bootstrap CI for beta(a) - beta(b)."""
    def fit_days(df, days):
        sub = df[df["date"].isin(days)]
        if len(sub) < 50 or sub["neg"].nunique() < 2:
            return np.nan
        X = sm.add_constant(sub[["log_cur", "neg"]].to_numpy())
        return sm.OLS(sub["log_fut"].to_numpy(), X).fit().params[2]
    da = a["date"].unique(); db = b["date"].unique()
    diffs = []
    for _ in range(n):
        ba = fit_days(a, RNG.choice(da, len(da), replace=True))
        bb = fit_days(b, RNG.choice(db, len(db), replace=True))
        if not (np.isnan(ba) or np.isnan(bb)):
            diffs.append(ba - bb)
    if len(diffs) < 30:
        return (float("nan"), float("nan"))
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main() -> int:
    d = build()
    print("=" * 90)
    print("EXP 011  LEVERAGE EFFECT x INTRADAY SESSION  NIFTY 5m")
    print(f"Observations: {len(d)}  days: {d['date'].nunique()}  "
          f"neg-return bars: {d['neg'].mean()*100:.1f}%")
    counts = d["session"].value_counts().to_dict()
    print(f"By session: {counts}")

    sessions = ["Morning", "Lunch", "Afternoon"]

    # --- 1) Session-specific leverage coefficient ------------------------
    print("\n--- Session-specific leverage coef (log_fut ~ log_cur + neg), HAC ---")
    betas = {}
    raw_p = {}
    for s in sessions:
        res = session_beta(d[d["session"] == s])
        betas[s] = res
        raw_p[s] = res["p"]
        print(f"  {s:<10} n={res['n']:>6}  beta(neg)={res['beta']:+.4f}  "
              f"t={res['t']:+.2f}  p={res['p']:.2e}")

    # --- 2) Full interaction model (lunch = baseline) --------------------
    print("\n--- Interaction model (Lunch baseline): neg x Morning, neg x Afternoon ---")
    dd = d.copy()
    dd["s_morn"] = (dd["session"] == "Morning").astype(float)
    dd["s_aft"] = (dd["session"] == "Afternoon").astype(float)
    dd["neg_morn"] = dd["neg"] * dd["s_morn"]
    dd["neg_aft"] = dd["neg"] * dd["s_aft"]
    cols = ["log_cur", "neg", "s_morn", "s_aft", "neg_morn", "neg_aft"]
    full = hac(dd["log_fut"].to_numpy(), dd[cols].to_numpy())
    names = ["const"] + cols
    pr = np.asarray(full.params); tv = np.asarray(full.tvalues); pv = np.asarray(full.pvalues)
    for nm, b, t, p in zip(names, pr, tv, pv):
        mark = " <-- interaction" if nm in ("neg_morn", "neg_aft") else ""
        print(f"    {nm:<10} coef={b:+.4f}  t={t:+.2f}  p={p:.2e}{mark}")
    # joint F-test on the two interactions
    R = np.zeros((2, len(names)))
    R[0, names.index("neg_morn")] = 1
    R[1, names.index("neg_aft")] = 1
    ftest = full.f_test(R)
    print(f"  Joint F-test (both interactions = 0): F={float(ftest.fvalue):.2f} "
          f"p={float(ftest.pvalue):.2e}")

    # --- 3) Pairwise bootstrap differences -------------------------------
    print("\n--- Pairwise leverage-coef differences (day-block bootstrap 95% CI) ---")
    for a, b in [("Morning", "Lunch"), ("Afternoon", "Lunch"), ("Morning", "Afternoon")]:
        lo, hi = boot_beta_diff(d[d["session"] == a], d[d["session"] == b])
        diff = betas[a]["beta"] - betas[b]["beta"]
        excl = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  {a:<10} - {b:<10}: Δbeta={diff:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  {excl}")

    # --- 4) Regime-specific (HAR forecast tercile) -----------------------
    print("\n--- Session leverage coef by volatility regime ---")
    reg = regime_map()
    d2 = d.copy(); d2["regime"] = d2["date"].map(reg)
    for rg in ["LOW", "NORMAL", "HIGH"]:
        line = f"  [{rg:<6}] "
        for s in sessions:
            res = session_beta(d2[(d2["regime"] == rg) & (d2["session"] == s)])
            line += f"{s[:3]}:{res['beta']:+.3f}(t{res['t']:+.1f}) " if res else f"{s[:3]}:n/a "
        print(line)

    # --- 5) Per-year ------------------------------------------------------
    print("\n--- Session leverage coef per year ---")
    for y in sorted(d["year"].unique()):
        line = f"  {y}: "
        for s in sessions:
            res = session_beta(d[(d["year"] == y) & (d["session"] == s)])
            line += f"{s[:3]}:{res['beta']:+.3f}(t{res['t']:+.1f}) " if res else f"{s[:3]}:n/a "
        print(line)

    # --- 6) Walk-forward (4 folds): dominant session sign ----------------
    print("\n--- Walk-forward (4 folds): session leverage coef ---")
    days = np.array(sorted(d["date"].unique()))
    for i, fold in enumerate(np.array_split(days, 4)):
        sub = d[d["date"].isin(pd.DatetimeIndex(fold))]
        line = f"  fold {i+1}: "
        best = None; bestb = -9
        for s in sessions:
            res = session_beta(sub[sub["session"] == s])
            if res:
                line += f"{s[:3]}:{res['beta']:+.3f} "
                if res["beta"] > bestb:
                    bestb = res["beta"]; best = s
        line += f"| strongest: {best}"
        print(line)

    # --- BH-FDR over session p-values ------------------------------------
    pv_list = [raw_p[s] for s in sessions]
    _, adj, _, _ = multipletests(pv_list, method="fdr_bh")
    print("\n--- BH-FDR over session leverage p-values ---")
    for s, p, a in zip(sessions, pv_list, adj):
        print(f"  {s:<10} raw p={p:.2e}  adj p={a:.2e}  {'sig' if a < 0.05 else 'ns'}")
    return 0


def regime_map() -> dict:
    """Daily HAR one-step-ahead RV forecast -> LOW/NORMAL/HIGH (as Exp 003/004)."""
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    rows = []
    for dd, g in df.groupby("date"):
        c = g.sort_values("timestamp")["close"].to_numpy()
        if len(c) < 12:
            continue
        rows.append({"date": dd, "log_rv": math.log(max(
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


if __name__ == "__main__":
    raise SystemExit(main())
