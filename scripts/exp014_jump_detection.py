"""Experiment 014: Jump detection — continuous vs jump decomposition of NIFTY RV.

Pre-registered in docs/preregistrations/exp014_jump_detection.md.

Questions:
  Q1  How much of NIFTY intraday volatility is jumps vs continuous diffusion?
  Q2  Are jump days clustered, or serially independent?
  Q3  Does a continuous+jump (HAR-CJ) split improve next-day RV forecasting
      beyond plain HAR?

Standard estimators (Barndorff-Nielsen-Shephard; Huang-Tauchen; Andersen-
Bollerslev-Diebold):
  RV = sum r_i^2                          (realized variance, total)
  BV = mu1^-2 * (M/(M-1)) * sum|r_i||r_{i-1}|   (bipower, continuous part)
  JV = max(RV-BV, 0)                      (jump variation)
  RJ = (RV-BV)/RV                         (relative jump)
  ZJ = sqrt(M)*RJ / sqrt(((pi/2)^2+pi-5)*max(1,TQ/BV^2)) ~ N(0,1) under no-jump

    python scripts/exp014_jump_detection.py

Read-only research. No trades.
"""

from __future__ import annotations

import glob
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

ANNUALISE = math.sqrt(252)
MU1 = math.sqrt(2.0 / math.pi)                       # E|Z|, Z~N(0,1)
MU43 = (2.0 ** (2.0 / 3.0)) * math.gamma(7.0 / 6.0) / math.gamma(0.5)  # E|Z|^(4/3)
_THETA = (math.pi / 2.0) ** 2 + math.pi - 5.0        # BNS test constant


def _day_estimators(logret: np.ndarray) -> dict | None:
    """RV/BV/JV/RJ/ZJ for one day's array of intraday log returns."""
    r = logret[~np.isnan(logret)]
    M = len(r)
    if M < 20:
        return None
    ar = np.abs(r)
    RV = float(np.sum(r ** 2))
    BV = (MU1 ** -2) * (M / (M - 1)) * float(np.sum(ar[1:] * ar[:-1]))
    # tripower quarticity (uses i, i-1, i-2)
    if M >= 3:
        tp = np.sum((ar[2:] ** (4/3)) * (ar[1:-1] ** (4/3)) * (ar[:-2] ** (4/3)))
        TQ = M * (MU43 ** -3) * (M / (M - 2)) * float(tp)
    else:
        TQ = float("nan")
    JV = max(RV - BV, 0.0)
    RJ = (RV - BV) / RV if RV > 0 else 0.0
    denom = _THETA * max(1.0, TQ / (BV ** 2)) if BV > 0 else float("nan")
    ZJ = math.sqrt(M) * RJ / math.sqrt(denom) if denom and denom > 0 else float("nan")
    return {"M": M, "RV": RV, "BV": BV, "JV": JV, "RJ": RJ, "ZJ": ZJ}


def daily_table() -> pd.DataFrame:
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
        if len(g) < 12:
            continue
        closes = g["close"].to_numpy()
        logret = np.log(closes[1:] / closes[:-1])
        est = _day_estimators(logret)
        if est is None:
            continue
        est.update({"date": d, "year": d.year})
        rows.append(est)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["rv_daily"] = np.sqrt(out["RV"].clip(lower=1e-18))   # daily std
    out["log_rv"] = np.log(out["rv_daily"].clip(lower=1e-9))
    return out


def hac_ols(y, X, lags=10):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def oos_r2(actual, pred, bench):
    actual, pred, bench = map(np.asarray, (actual, pred, bench))
    sse = np.sum((actual - pred) ** 2)
    sst = np.sum((actual - bench) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def main() -> int:
    t = daily_table()
    n = len(t)
    print("=" * 92)
    print("EXP 014  NIFTY JUMP DETECTION  (continuous vs jump decomposition)")
    print(f"Trading days: {n}  | {t['date'].min().date()} -> {t['date'].max().date()}")
    print(f"Median intraday returns/day M = {int(t['M'].median())} "
          f"(small-M: BNS test is approximate, may slightly over-reject)")
    print(f"Mean annualised RV: {(t['rv_daily'].mean()*ANNUALISE)*100:.1f}%")

    # thresholds
    z95, z99, z999 = ss.norm.ppf(0.95), ss.norm.ppf(0.99), ss.norm.ppf(0.999)

    # --- Q1: jump contribution ----------------------------------------------
    print("\n--- Q1: how much volatility is jumps? ---")
    rj = t["RJ"].clip(lower=0)
    print(f"  Relative jump RJ=(RV-BV)/RV:  mean {rj.mean()*100:5.1f}%   "
          f"median {rj.median()*100:5.1f}%")
    jv_share = (t["JV"] / t["RV"]).clip(lower=0)
    print(f"  Jump-variation share JV/RV :  mean {jv_share.mean()*100:5.1f}%   "
          f"median {jv_share.median()*100:5.1f}%")
    for lab, z in [("alpha=0.05", z95), ("alpha=0.01", z99), ("alpha=0.001", z999)]:
        sig = t["ZJ"] > z
        frac = sig.mean() * 100
        share_on = (t.loc[sig, "JV"] / t.loc[sig, "RV"]).mean() * 100 if sig.any() else float("nan")
        print(f"  significant-jump days {lab:>11}: {sig.sum():3d}/{n} "
              f"({frac:4.1f}%)   mean jump share on those days {share_on:4.1f}%")

    # --- Q2: clustering (alpha=0.01 indicator) ------------------------------
    print("\n--- Q2: are jumps clustered? (alpha=0.01 indicator) ---")
    jump = (t["ZJ"] > z99).astype(int).to_numpy()
    base_rate = jump.mean()
    print(f"  base jump rate: {base_rate*100:.1f}% of days")
    lb = acorr_ljungbox(jump, lags=[1, 5, 10], return_df=True)
    lb_reject = False
    for lag, row in lb.iterrows():
        rej = row["lb_pvalue"] < 0.05
        lb_reject = lb_reject or rej
        print(f"    Ljung-Box lag {lag:>2}: stat={row['lb_stat']:6.2f}  "
              f"p={row['lb_pvalue']:.3f}{'  *' if rej else ''}")
    # 2x2 contingency: jump_t vs jump_{t-1}
    prev, cur = jump[:-1], jump[1:]
    a = int(np.sum((prev == 1) & (cur == 1)))  # prior jump -> jump
    b = int(np.sum((prev == 1) & (cur == 0)))
    c = int(np.sum((prev == 0) & (cur == 1)))
    d = int(np.sum((prev == 0) & (cur == 0)))
    p_given_prior = a / (a + b) if (a + b) else float("nan")
    p_given_none = c / (c + d) if (c + d) else float("nan")
    fisher_p = ss.fisher_exact([[a, b], [c, d]], alternative="greater")[1] \
        if (a + b) and (c + d) else float("nan")
    print(f"  P(jump_t | jump_(t-1))   = {p_given_prior*100:4.1f}%  (n={a+b})")
    print(f"  P(jump_t | no jump_(t-1))= {p_given_none*100:4.1f}%  (n={c+d})")
    print(f"  Fisher exact (one-sided, positive dependence): p={fisher_p:.3f}")
    contingency_pos = (not math.isnan(fisher_p) and fisher_p < 0.05
                       and p_given_prior > p_given_none)
    if lb_reject and contingency_pos:
        q2 = "CLUSTERED"
    elif not lb_reject and not contingency_pos:
        q2 = "NOT CLUSTERED (serially independent)"
    else:
        q2 = "MIXED (only one test rejects; do not over-claim)"
    print(f"  --> Q2 verdict: {q2}")

    # --- Q3: HAR-CJ increment -----------------------------------------------
    print("\n--- Q3: does the jump split improve next-day RV forecasting? ---")
    s = t["log_rv"].reset_index(drop=True)
    har = pd.DataFrame({
        "y": s.shift(-1),
        "d": s.shift(0),
        "w": s.rolling(5).mean(),
        "m": s.rolling(22).mean(),
        "rj": t["RJ"].clip(lower=0).reset_index(drop=True),   # lagged (aligned to t)
    }).dropna().reset_index(drop=True)
    base = hac_ols(har["y"].to_numpy(), har[["d", "w", "m"]].to_numpy())
    aug = hac_ols(har["y"].to_numpy(), har[["d", "w", "m", "rj"]].to_numpy())
    rj_t = aug.tvalues[-1]
    print(f"  augmented HAR: RJ coef={aug.params[-1]:+.3f}  HAC t={rj_t:+.2f}  "
          f"(in-sample R2 {base.rsquared:.3f} -> {aug.rsquared:.3f})")
    # walk-forward incremental OOS R^2
    N = len(har)
    start = N // 2
    bounds = np.linspace(start, N, 5, dtype=int)
    incr = []
    for i in range(4):
        lo, hi = bounds[i], bounds[i + 1]
        tr, te = har.iloc[:lo], har.iloc[lo:hi]
        if len(te) < 20:
            print(f"  fold {i+1}: insufficient (n_test={len(te)})")
            continue
        Xb_tr = sm.add_constant(tr[["d", "w", "m"]].to_numpy())
        Xa_tr = sm.add_constant(tr[["d", "w", "m", "rj"]].to_numpy())
        fb = sm.OLS(tr["y"].to_numpy(), Xb_tr).fit()
        fa = sm.OLS(tr["y"].to_numpy(), Xa_tr).fit()
        Xb_te = sm.add_constant(te[["d", "w", "m"]].to_numpy(), has_constant="add")
        Xa_te = sm.add_constant(te[["d", "w", "m", "rj"]].to_numpy(), has_constant="add")
        pb, pa = fb.predict(Xb_te), fa.predict(Xa_te)
        r2 = oos_r2(te["y"], pa, pb)  # augmented vs baseline as benchmark
        incr.append(r2)
        print(f"  fold {i+1}: n_test={len(te):3d}  incremental OOS R2 (aug vs base) = {r2:+.4f}")
    n_pos = sum(1 for x in incr if x > 0)
    mean_incr = float(np.mean(incr)) if incr else float("nan")
    q3 = ("JUMP DECOMPOSITION HELPS" if abs(rj_t) >= 2.5 and mean_incr > 0 and n_pos >= 3
          else "NO INCREMENTAL VALUE")
    print(f"  mean incremental OOS R2 = {mean_incr:+.4f}  ({n_pos}/{len(incr)} folds positive)")
    print(f"  --> Q3 verdict: {q3}")

    print("\n" + "=" * 92)
    print("SUMMARY (pre-registered):")
    print(f"  Q1 jump contribution: mean RJ {rj.mean()*100:.1f}%, "
          f"median {rj.median()*100:.1f}% (indicative; small M).")
    print(f"  Q2 clustering: {q2}")
    print(f"  Q3 HAR-CJ increment: {q3} (RJ t={rj_t:+.2f}, mean incr OOS R2 {mean_incr:+.4f}).")
    print("  Scope: characterisation only. No trading rule implied.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
