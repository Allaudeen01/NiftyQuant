"""Bounded, FDR-corrected search for HAR-RV-incremental volatility features.

NOT a loop. Runs ONCE over a pre-declared, modest grid (~15 candidates) of
volatility-STRUCTURE features (never re-testing the already-rejected price-
direction indicators from Exp 001-011). Object: NIFTY next-day realised
volatility -- the one dimension this campaign has repeatedly shown to be
forecastable (Exp 003).

For every candidate:
  1. Fit  log RV_next ~ HAR(d,w,m) + candidate   with HAC errors (full sample).
  2. Walk-forward (4 expanding folds): incremental OOS R^2 over the HAR-only
     baseline, and sign of the candidate coefficient, per fold.
  3. Record the full-sample candidate p-value.
Then Benjamini-Hochberg FDR-correct ALL candidate p-values together (the more
candidates tested, the stricter each must clear) and rank by mean OOS
incremental R^2.

OUTPUT IS HYPOTHESIS-GENERATING, NOT CONFIRMATION. A candidate surviving this
screen must still get its OWN dedicated, pre-registered, one-variable
experiment (as every prior Exp 001-011 did) before being trusted. This script
never declares "found a strategy" -- it flags what's worth a proper follow-up.

    python scripts/volatility_feature_search.py
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

pd.set_option("display.width", 140)


def load_bars() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df = df[(df["timestamp"].dt.strftime("%H:%M") >= "09:15")
            & (df["timestamp"].dt.strftime("%H:%M") <= "15:30")]
    return df


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per day: RV + the ~15 candidate volatility-structure features."""
    rows = []
    prev_close = None
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        c = g["close"].to_numpy()
        o = float(g["open"].iloc[0])
        hi = float(g["high"].max()); lo = float(g["low"].min())
        n = len(c)
        if n < 20:
            prev_close = float(c[-1]) if n else prev_close
            continue
        r = np.log(c[1:] / c[:-1])
        rv = math.sqrt(float(np.sum(r ** 2)))
        row = {"date": d, "year": d.year, "dow": d.dayofweek, "rv": rv}

        # --- candidate features (all computable same-day, no lookahead) ---
        row["intraday_range"] = (hi - lo) / o                       # 1
        row["skew"] = float(ss.skew(r))                              # 2
        row["kurt"] = float(ss.kurtosis(r))                          # 3
        row["max_abs_bar_ret"] = float(np.max(np.abs(r)))            # 4  jump proxy
        row["realized_quarticity"] = float(np.sum(r ** 4)) * n / 3   # 5  vol-of-vol proxy
        sign = np.sign(r); sign[sign == 0] = 1
        row["sign_changes"] = int(np.sum(sign[1:] != sign[:-1]))     # 6  choppiness
        # ARCH-effect strength: lag-1 autocorr of squared returns
        r2 = r ** 2
        if r2.std() > 0 and len(r2) > 2:
            row["arch_lag1"] = float(np.corrcoef(r2[1:], r2[:-1])[0, 1])
        else:
            row["arch_lag1"] = np.nan                                # 7
        down = r[r < 0]; up = r[r > 0]
        dv = float(np.sum(down ** 2)); uv = float(np.sum(up ** 2))
        row["semivar_ratio"] = (dv / uv) if uv > 0 else np.nan       # 8
        # overnight variance share (needs prev close)
        if prev_close:
            on_ret = math.log(o / prev_close)
            on_var = on_ret ** 2
            row["overnight_var_share"] = on_var / (on_var + rv ** 2)  # 9
        else:
            row["overnight_var_share"] = np.nan
        # morning (first third of bars) vs full-day RV share
        third = max(n // 3, 1)
        row["morning_rv_share"] = (
            math.sqrt(float(np.sum(r[:third] ** 2))) / rv if rv > 0 else np.nan
        )                                                             # 10
        # count of "large" bars (>2 std) -> tail activity
        std_r = r.std()
        row["n_large_bars"] = int(np.sum(np.abs(r) > 2 * std_r)) if std_r > 0 else 0  # 11
        # simple Hurst-like slope proxy over the day's cumulative path (cheap)
        path = np.cumsum(r)
        if len(path) > 10:
            lags = np.arange(2, min(20, len(path) // 2))
            taus = [np.std(path[l:] - path[:-l]) for l in lags]
            taus = [t for t in taus if t > 0]
            if len(taus) >= 3:
                row["intraday_hurst"] = float(
                    np.polyfit(np.log(lags[:len(taus)]), np.log(taus), 1)[0]
                )                                                     # 12
            else:
                row["intraday_hurst"] = np.nan
        else:
            row["intraday_hurst"] = np.nan
        row["dow_monday"] = 1.0 if d.dayofweek == 0 else 0.0          # 13
        row["dow_friday"] = 1.0 if d.dayofweek == 4 else 0.0          # 14
        row["n_bars"] = n                                              # 15 (data-density control)

        prev_close = float(c[-1])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    return out


CANDIDATES = [
    "intraday_range", "skew", "kurt", "max_abs_bar_ret", "realized_quarticity",
    "sign_changes", "arch_lag1", "semivar_ratio", "overnight_var_share",
    "morning_rv_share", "n_large_bars", "intraday_hurst", "dow_monday",
    "dow_friday", "n_bars",
]


def har_features(d: pd.DataFrame) -> pd.DataFrame:
    """Build the HAR baseline + LAG every candidate by 1 day.

    CRITICAL: every candidate is computed from day t's own bars, so it must be
    shifted by one day (like the HAR terms) before predicting day t's RV as
    "next-day" RV relative to t-1. Without this shift, features such as
    intraday_range or max_abs_bar_ret are literally components of the same
    sum-of-squared-returns that DEFINES that day's RV -- a tautology, not a
    forecast. This mirrors the exact setup used in Exp 003/006A/007.
    """
    s = d["log_rv"]
    d = d.copy()
    d["har_d"] = s.shift(1)
    d["har_w"] = s.shift(1).rolling(5).mean()
    d["har_m"] = s.shift(1).rolling(22).mean()
    d["y"] = s  # today's log_rv, predicted using ONLY information through t-1
    for cand in CANDIDATES:
        d[cand] = d[cand].shift(1)
    return d


def hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})


def walk_forward_incremental_r2(d: pd.DataFrame, cand: str, n_folds=4):
    """4-fold expanding walk-forward: OOS R2(HAR+cand) vs OOS R2(HAR-only)."""
    cols_base = ["har_d", "har_w", "har_m"]
    cols_full = cols_base + [cand]
    sub = d.dropna(subset=cols_full + ["y"]).reset_index(drop=True)
    n = len(sub)
    if n < 120:
        return [], []
    start = n // 2
    edges = np.linspace(start, n, n_folds + 1, dtype=int)
    incr_r2s, signs = [], []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        train, test = sub.iloc[:a], sub.iloc[a:b]
        if len(test) < 15:
            continue
        yb = test["y"].to_numpy()
        bench = np.full(len(yb), train["y"].mean())
        sst = np.sum((yb - bench) ** 2)
        if sst == 0:
            continue
        fb = sm.OLS(train["y"].to_numpy(),
                    sm.add_constant(train[cols_base].to_numpy())).fit()
        pb = fb.predict(sm.add_constant(test[cols_base].to_numpy(), has_constant="add"))
        r2_base = 1 - np.sum((yb - pb) ** 2) / sst

        ff = sm.OLS(train["y"].to_numpy(),
                    sm.add_constant(train[cols_full].to_numpy())).fit()
        pf = ff.predict(sm.add_constant(test[cols_full].to_numpy(), has_constant="add"))
        r2_full = 1 - np.sum((yb - pf) ** 2) / sst

        incr_r2s.append(r2_full - r2_base)
        signs.append(np.sign(ff.params[-1]))
    return incr_r2s, signs


def main() -> int:
    print("=" * 100)
    print("BOUNDED VOLATILITY-FEATURE SEARCH (HAR-RV incremental, FDR-corrected)")
    print("ONE-SHOT run over a pre-declared grid. Output is HYPOTHESIS-GENERATING")
    print("ONLY -- any survivor needs its own dedicated pre-registered experiment.")
    print("=" * 100)

    daily = build_daily(load_bars())
    d = har_features(daily).dropna(subset=["har_d", "har_w", "har_m", "y"])
    print(f"Daily observations: {len(d)}  candidates tested: {len(CANDIDATES)}")

    rows = []
    for cand in CANDIDATES:
        sub = d.dropna(subset=["har_d", "har_w", "har_m", cand, "y"])
        if len(sub) < 60 or sub[cand].std() == 0:
            rows.append({"candidate": cand, "n": len(sub), "coef": np.nan,
                        "t": np.nan, "p": np.nan, "mean_incr_r2": np.nan,
                        "folds_positive": 0, "n_folds": 0})
            continue
        X = sub[["har_d", "har_w", "har_m", cand]].to_numpy()
        m = hac(sub["y"].to_numpy(), X)
        coef = m.params[-1]; tval = m.tvalues[-1]; pval = m.pvalues[-1]

        incr_r2s, signs = walk_forward_incremental_r2(d, cand)
        mean_incr = float(np.mean(incr_r2s)) if incr_r2s else np.nan
        pos = sum(1 for x in incr_r2s if x > 0)

        rows.append({
            "candidate": cand, "n": len(sub), "coef": coef, "t": tval, "p": pval,
            "mean_incr_r2": mean_incr, "folds_positive": pos,
            "n_folds": len(incr_r2s),
        })

    res = pd.DataFrame(rows)
    valid = res.dropna(subset=["p"]).copy()
    if len(valid):
        _, adj, _, _ = multipletests(valid["p"].to_numpy(), method="fdr_bh")
        valid["p_adj"] = adj
        res = res.merge(valid[["candidate", "p_adj"]], on="candidate", how="left")
    else:
        res["p_adj"] = np.nan

    res["survives"] = (
        (res["p_adj"] < 0.05)
        & (res["mean_incr_r2"] > 0)
        & (res["folds_positive"] >= 3)
    )
    res = res.sort_values("mean_incr_r2", ascending=False, na_position="last")

    print("\n--- Full ranking (by mean walk-forward incremental OOS R2) ---")
    show = res.copy()
    for c in ("coef", "t", "mean_incr_r2"):
        show[c] = show[c].round(4)
    for c in ("p", "p_adj"):
        show[c] = show[c].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "nan")
    print(show.to_string(index=False))

    survivors = res[res["survives"]]
    print("\n" + "=" * 100)
    if len(survivors):
        print(f"SURVIVED SCREEN ({len(survivors)}): significant after FDR, positive "
              f"mean OOS incremental R2, positive in >=3/4 folds:")
        for _, r in survivors.iterrows():
            print(f"  - {r['candidate']}: mean_incr_R2={r['mean_incr_r2']:.4f}  "
                  f"folds+={r['folds_positive']}/{r['n_folds']}  p_adj={r['p_adj']:.2e}")
        print("\n  These are CANDIDATES FOR A DEDICATED FOLLOW-UP EXPERIMENT ONLY.")
        print("  Do NOT treat as a validated strategy or trading signal.")
    else:
        print("NO CANDIDATE SURVIVED the FDR-corrected screen.")
        print("This is a valid, informative result: none of these volatility-structure")
        print("features add incremental, stable, out-of-sample forecasting power over")
        print("the existing HAR-RV baseline (Exp 003).")
    print("=" * 100)

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/volatility_feature_search.csv"
    res.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
