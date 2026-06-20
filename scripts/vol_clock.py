"""Experiment 008: Intraday Volatility Clock. Read-only structural research.

ONE new variable vs Exp 007: time-of-day. For each 5-minute bar we compute its
squared log return, normalise by that day's total intraday realized variance,
and build the average intraday variance profile. We test for the classical
U-shape and whether the profile is stable across years and walk-forward folds.

No parameters optimised.  python scripts/vol_clock.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

RNG = np.random.default_rng(31)


def load_bars() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    df["year"] = df["timestamp"].dt.year
    # Data-quality gate: keep only regular NIFTY session bars (09:15-15:30).
    # A few days carry spurious post-market prints (18:05-18:55) that would
    # otherwise corrupt the intraday profile; exclude them.
    before = len(df)
    df = df[(df["tod"] >= "09:15") & (df["tod"] <= "15:30")]
    excluded = before - len(df)
    if excluded:
        print(f"[quality] excluded {excluded} out-of-session bars "
              f"(outside 09:15-15:30)")
    return df


def per_bar_shares(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, bar): share of that day's intraday variance."""
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        if len(g) < 12:
            continue
        c = g["close"].to_numpy()
        tod = g["tod"].to_numpy()[1:]            # return aligns to bar t (vs t-1)
        yr = int(g["year"].iloc[0])
        r2 = np.log(c[1:] / c[:-1]) ** 2
        tot = r2.sum()
        if tot <= 0:
            continue
        for t, v in zip(tod, r2):
            rows.append({"date": d, "year": yr, "tod": t,
                         "share": v / tot, "r2": v})
    return pd.DataFrame(rows)


def session_of(tod: str) -> str:
    h, m = map(int, tod.split(":"))
    x = h * 60 + m
    if x <= 9 * 60 + 20:      # 09:15, 09:20 prints
        return "Opening"
    if x < 11 * 60 + 30:      # up to 11:25 bar
        return "Morning"
    if x < 13 * 60:           # 11:30-12:55
        return "Lunch"
    if x < 15 * 60 + 5:       # 13:00-15:00
        return "Afternoon"
    return "Closing"          # 15:05, 15:10, 15:15, 15:20, 15:25


def boot_ci(x, n=4000):
    x = np.asarray(x)
    bs = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    df = load_bars()
    pb = per_bar_shares(df)
    ndays = pb["date"].nunique()
    print("=" * 96)
    print("EXP 008  INTRADAY VOLATILITY CLOCK  NIFTY 5m")
    print(f"Days: {ndays}  bars/day~{pb.groupby('date').size().median():.0f}  "
          f"{pb['date'].min().date()} -> {pb['date'].max().date()}")

    # --- Per-bar variance profile (% of daily variance) -------------------
    prof = pb.groupby("tod")["share"].agg(["mean", "median", "std", "count"])
    prof["cv"] = prof["std"] / prof["mean"]
    n_bins = len(prof)
    uniform = 100.0 / n_bins
    print(f"\n--- Variance profile by 5-min bar (% of daily variance; "
          f"uniform={uniform:.2f}%) ---")
    print(f"  {'bar':<7}{'mean%':>8}{'med%':>8}{'CI95%':>16}{'CV':>7}")
    for tod, row in prof.iterrows():
        sub = pb[pb["tod"] == tod]["share"].to_numpy()
        lo, hi = boot_ci(sub)
        print(f"  {tod:<7}{row['mean']*100:>8.2f}{row['median']*100:>8.2f}"
              f"  [{lo*100:>5.2f},{hi*100:>5.2f}]{row['cv']:>7.2f}")

    # --- Session aggregation ---------------------------------------------
    pb["session"] = pb["tod"].map(session_of)
    print("\n--- Session shares (% of daily variance, mean per bar-day) ---")
    sess_order = ["Opening", "Morning", "Lunch", "Afternoon", "Closing"]
    # total share per session per day, then average across days
    day_sess = pb.groupby(["date", "session"])["share"].sum().reset_index()
    for s in sess_order:
        sub = day_sess[day_sess["session"] == s]["share"].to_numpy()
        lo, hi = boot_ci(sub)
        nbars = pb[pb["session"] == s]["tod"].nunique()
        print(f"  {s:<10} bars={nbars:>2}  mean={sub.mean()*100:>5.1f}%  "
              f"median={np.median(sub)*100:>5.1f}%  CI95=[{lo*100:.1f},{hi*100:.1f}]%")

    # --- ANOVA / Kruskal across sessions (per-bar share) ------------------
    groups = [pb[pb["session"] == s]["share"].to_numpy() for s in sess_order]
    F, pF = ss.f_oneway(*groups)
    H, pH = ss.kruskal(*groups)
    print(f"\n  ANOVA across sessions F={F:.1f} p={pF:.2e} | "
          f"Kruskal H={H:.1f} p={pH:.2e}")

    # --- U-shape: opening peak / lunch trough / closing peak --------------
    first = "09:20"; last = pb["tod"].max()
    # midday trough = min mean-share bar between 11:30 and 13:30
    midbars = [t for t in prof.index if "11:30" <= t <= "13:30"]
    trough = prof.loc[midbars, "mean"].idxmin()
    open_peak = prof.loc[[first], "mean"].iloc[0] if first in prof.index else np.nan
    close_peak = prof.loc[[last], "mean"].iloc[0]
    trough_v = prof.loc[trough, "mean"]
    print("\n--- U-shape check ---")
    print(f"  opening bar {first}: {open_peak*100:.2f}%   "
          f"lunch trough {trough}: {trough_v*100:.2f}%   "
          f"closing bar {last}: {close_peak*100:.2f}%")
    print(f"  open/trough ratio = {open_peak/trough_v:.2f}   "
          f"close/trough ratio = {close_peak/trough_v:.2f}")

    # --- Per-year stability: correlation of profiles ----------------------
    print("\n--- Per-year profile stability (Spearman corr of bar-profiles) ---")
    yr_prof = pb.pivot_table(index="tod", columns="year", values="share",
                             aggfunc="mean")
    years = sorted(pb["year"].unique())
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            a, b = yr_prof[years[i]], yr_prof[years[j]]
            rho, p = ss.spearmanr(a, b, nan_policy="omit")
            print(f"  {years[i]} vs {years[j]}: Spearman rho={rho:.3f} (p={p:.1e})")

    # --- Walk-forward stability: split days into 4 folds, corr of profiles -
    print("\n--- Walk-forward profile stability (4 folds vs fold-1) ---")
    dts = sorted(pb["date"].unique())
    folds = np.array_split(np.array(dts), 4)
    fold_profiles = []
    for f in folds:
        sub = pb[pb["date"].isin(pd.DatetimeIndex(f))]
        fold_profiles.append(sub.groupby("tod")["share"].mean())
    base = fold_profiles[0]
    common = base.index
    for fp in fold_profiles:
        common = common.intersection(fp.index)
    for i, fp in enumerate(fold_profiles):
        rho, p = ss.spearmanr(base.reindex(common), fp.reindex(common))
        print(f"  fold {i+1} vs fold 1: Spearman rho={rho:.3f} (p={p:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
