"""Experiment 007: Overnight vs Intraday volatility decomposition. Read-only.

ONE new variable vs 006A: the OVERNIGHT return (today's open / yesterday's
close - 1). We split each day's total variance into an overnight component and
an intraday component, measure where variance is created, and test which
component carries more information about tomorrow's realized volatility.

Definitions:
  overnight_ret = open_t / close_{t-1} - 1
  intraday_var  = sum of squared 5m open->close log returns (= Exp 003 RV^2)
  overnight_var = overnight_ret^2
  total_var     = overnight_var + intraday_var
  RV_next       = sqrt(intraday_var_{t+1})   (same RV object as Exp 003/006A)

No parameters optimised.  python scripts/vol_decomp.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss

RNG = np.random.default_rng(29)
ANN = math.sqrt(252) * 100


def build() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    rows = []
    prev_close = None
    for d, g in df.groupby("date"):
        if len(g) < 12:
            prev_close = float(g["close"].iloc[-1]) if len(g) else prev_close
            continue
        g = g.sort_values("timestamp")
        c = g["close"].to_numpy()
        o = float(g["open"].iloc[0])
        lr = np.log(c[1:] / c[:-1])
        intr_var = float(np.sum(lr ** 2))                 # intraday variance
        if prev_close:
            on_ret = math.log(o / prev_close)             # overnight log return
            on_var = on_ret ** 2                          # overnight variance
            rows.append({
                "date": d, "year": d.year,
                "on_var": on_var, "intr_var": intr_var,
                "tot_var": on_var + intr_var,
            })
        prev_close = float(g["close"].iloc[-1])
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["rv_intr"] = np.sqrt(out["intr_var"])             # = Exp003 RV
    out["log_rv"] = np.log(out["rv_intr"].clip(lower=1e-9))
    out["rv_next"] = out["rv_intr"].shift(-1)
    out["log_rv_next"] = out["log_rv"].shift(-1)
    out["on_share"] = out["on_var"] / out["tot_var"]
    return out.dropna(subset=["rv_next"]).reset_index(drop=True)


def hac(y, X, lags=10):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC",
                                             cov_kwds={"maxlags": lags})


def boot_mean(x, n=5000):
    x = np.asarray(x)
    bs = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    d = build()
    n = len(d)
    print("=" * 92)
    print("EXP 007  OVERNIGHT vs INTRADAY VOLATILITY DECOMPOSITION  NIFTY 5m")
    print(f"Obs: {n}  {d['date'].min().date()} -> {d['date'].max().date()}")

    # --- 1-4) Contribution of each component to total variance ------------
    on_share = d["on_share"].to_numpy()
    print("\n--- Variance decomposition (share of total daily variance) ---")
    lo, hi = boot_mean(on_share)
    print(f"  OVERNIGHT share: mean={on_share.mean()*100:.1f}%  "
          f"median={np.median(on_share)*100:.1f}%  CI95=[{lo*100:.1f},{hi*100:.1f}]%")
    print(f"  INTRADAY  share: mean={(1-on_share).mean()*100:.1f}%  "
          f"median={(1-np.median(on_share))*100:.1f}%")
    # annualised vol of each component (sqrt of mean variance)
    print(f"  Annualised vol  overnight={math.sqrt(d['on_var'].mean())*ANN:.1f}%  "
          f"intraday={math.sqrt(d['intr_var'].mean())*ANN:.1f}%  "
          f"total={math.sqrt(d['tot_var'].mean())*ANN:.1f}%")

    # --- 5-6) Correlations with next-day RV -------------------------------
    print("\n--- Correlation with RV_next (and log-log) ---")
    for nm, col in [("overnight_var", "on_var"), ("intraday_var", "intr_var"),
                    ("total_var", "tot_var")]:
        r, p = ss.pearsonr(d[col], d["rv_next"])
        # log-space (robust to skew)
        rl, pl = ss.pearsonr(np.log(d[col].clip(lower=1e-12)), d["log_rv_next"])
        print(f"  {nm:<14} corr={r:+.3f} (p={p:.1e})   log-log corr={rl:+.3f} (p={pl:.1e})")

    # --- 7) Regression: log RV_next ~ log on_var + log intr_var -----------
    print("\n--- HAC regression  log_RV_next ~ log(overnight_var) + log(intraday_var) ---")
    X = np.column_stack([np.log(d["on_var"].clip(lower=1e-12)),
                         np.log(d["intr_var"].clip(lower=1e-12))])
    base_on = hac(d["log_rv_next"], X[:, :1])     # overnight only
    base_in = hac(d["log_rv_next"], X[:, 1:])     # intraday only
    full = hac(d["log_rv_next"], X)
    for nm, b, t in zip(["const", "log_on_var", "log_intr_var"],
                        full.params, full.tvalues):
        print(f"    {nm:<13} coef={b: .3f}  HAC t={t: .2f}")
    print(f"  R2: overnight-only={base_on.rsquared:.3f}  "
          f"intraday-only={base_in.rsquared:.3f}  full={full.rsquared:.3f}")
    print(f"  partial R2 of overnight (over intraday-only) = "
          f"{full.rsquared-base_in.rsquared:.4f}")
    print(f"  partial R2 of intraday  (over overnight-only) = "
          f"{full.rsquared-base_on.rsquared:.4f}")

    # --- 8) Walk-forward OOS R2: which single component forecasts better? --
    print("\n--- Walk-forward OOS R2 vs unconditional mean (4 folds) ---")
    y = d["log_rv_next"].to_numpy()
    feats = {"overnight": X[:, :1], "intraday": X[:, 1:], "both": X}
    N = len(d); start = N // 2
    edges = np.linspace(start, N, 5, dtype=int)
    print(f"  {'fold':<6}{'overnight':>12}{'intraday':>12}{'both':>12}")
    for i in range(4):
        a, b = edges[i], edges[i + 1]
        ytr, yte = y[:a], y[a:b]
        if len(yte) < 20:
            print(f"  {i+1:<6}Insufficient"); continue
        bench = np.full(len(yte), ytr.mean())
        sst = np.sum((yte - bench) ** 2)
        line = f"  {i+1:<6}"
        for key in ("overnight", "intraday", "both"):
            Xt = feats[key]
            fit = sm.OLS(ytr, sm.add_constant(Xt[:a])).fit()
            pr = fit.predict(sm.add_constant(Xt[a:b], has_constant="add"))
            r2 = 1 - np.sum((yte - pr) ** 2) / sst
            line += f"{r2:>12.3f}"
        print(line)

    # --- 9) Per-year regression coefficients ------------------------------
    print("\n--- Per-year HAC: log_RV_next ~ log_on_var + log_intr_var ---")
    for yv in sorted(d["year"].unique()):
        s = d[d["year"] == yv]
        if len(s) < 40:
            print(f"  {yv}: Insufficient (n={len(s)})"); continue
        Xs = np.column_stack([np.log(s["on_var"].clip(lower=1e-12)),
                              np.log(s["intr_var"].clip(lower=1e-12))])
        m = hac(s["log_rv_next"], Xs)
        pr = np.asarray(m.params); tv = np.asarray(m.tvalues)
        print(f"  {yv}: n={len(s):3d}  on_var coef={pr[1]:+.3f} t={tv[1]:+.2f}"
              f"   intr_var coef={pr[2]:+.3f} t={tv[2]:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
