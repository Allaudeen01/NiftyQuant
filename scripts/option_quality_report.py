"""Day-level data-quality report for collected option-chain snapshots.

Read-only. Scans data/option_chain/<YYYY>/<MM>/<DD>/*.parquet for a date and
reports coverage, completeness (both expiries present), minute gaps, and basic
field sanity (OI / IV / bid-ask / spot), plus the India VIX series for the day.

    python scripts/option_quality_report.py --date 2026-06-22
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Option-chain data-quality report (read-only).")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="YYYY-MM-DD (default: today)")
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def load_day(data_dir: str, d: date):
    folder = os.path.join(data_dir, "option_chain", f"{d.year:04d}",
                          f"{d.month:02d}", f"{d.day:02d}")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    return folder, files


def main() -> int:
    args = parse_args()
    d = date.fromisoformat(args.date)
    folder, files = load_day(args.data_dir, d)
    print("=" * 84)
    print(f"OPTION-CHAIN DATA-QUALITY REPORT  {d.isoformat()}")
    print(f"Folder: {folder}")
    if not files:
        print("No snapshot files found for this date.")
        return 0

    # --- per-file summary -------------------------------------------------
    rows = []
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
        frames.append(df)
        minute = os.path.splitext(os.path.basename(f))[0]  # HH_MM
        expiries = sorted(df["expiry"].astype(str).unique())
        rows.append({
            "minute": minute,
            "n_quotes": len(df),
            "n_expiries": df["expiry"].nunique(),
            "expiries": ",".join(pd.to_datetime(e).strftime("%m-%d") for e in expiries),
        })
    summ = pd.DataFrame(rows).sort_values("minute").reset_index(drop=True)
    all_df = pd.concat(frames, ignore_index=True)

    n_files = len(summ)
    n_expected_expiries = summ["n_expiries"].max()
    complete = (summ["n_expiries"] == n_expected_expiries).sum()
    partial = n_files - complete
    print(f"\nSnapshots (minute-files): {n_files}")
    print(f"First: {summ['minute'].iloc[0].replace('_',':')}  "
          f"Last: {summ['minute'].iloc[-1].replace('_',':')}")
    print(f"Complete ({n_expected_expiries} expiries): {complete}  "
          f"Partial: {partial}  ({100*complete/n_files:.1f}% complete)")
    if partial:
        bad = summ[summ["n_expiries"] < n_expected_expiries]["minute"].tolist()
        print(f"Partial minutes: {[m.replace('_',':') for m in bad]}")

    # --- minute-gap analysis ---------------------------------------------
    mins = pd.to_datetime(d.isoformat() + " " + summ["minute"].str.replace("_", ":"))
    deltas = mins.diff().dropna().dt.total_seconds() / 60.0
    print(f"\nCadence: median {deltas.median():.1f} min  max gap {deltas.max():.1f} min")
    big = mins[1:][deltas.values > 5.0]
    if len(big):
        print(f"Gaps > 5 min at: {[t.strftime('%H:%M') for t in big]}")
    else:
        print("No gaps > 5 min.")

    # --- field sanity (across all quotes) --------------------------------
    n = len(all_df)
    def pct(mask):
        return 100.0 * float(np.mean(mask))
    has_oi = pct(all_df["open_interest"].fillna(0) > 0)
    has_iv = pct(all_df["implied_volatility"].notna())
    two_sided = pct((all_df["bid"].fillna(0) > 0) & (all_df["ask"].fillna(0) > 0))
    crossed = pct((all_df["bid"].fillna(0) > 0) & (all_df["ask"].fillna(0) > 0)
                  & (all_df["ask"] < all_df["bid"]))
    print(f"\nField sanity over {n} quotes:")
    print(f"  open_interest > 0 : {has_oi:5.1f}%")
    print(f"  IV present        : {has_iv:5.1f}%")
    print(f"  two-sided (bid&ask): {two_sided:5.1f}%")
    print(f"  crossed (ask<bid) : {crossed:5.1f}%  (should be ~0)")
    print(f"  spot range        : {all_df['spot'].min():.1f} - {all_df['spot'].max():.1f}")
    print(f"  strikes/snapshot  : ~{int(all_df.groupby(['snapshot_ts','expiry']).size().median())}")

    # --- VIX series -------------------------------------------------------
    vix_files = glob.glob(os.path.join(args.data_dir, "vix", f"{d.year:04d}",
                                       f"INDIAVIX_{d.isoformat()}.parquet"))
    if vix_files:
        v = pd.read_parquet(vix_files[0])
        print(f"\nIndia VIX: {len(v)} readings  "
              f"range {v['india_vix'].min():.2f} - {v['india_vix'].max():.2f}  "
              f"last {v['india_vix'].iloc[-1]:.2f}")
    else:
        print("\nIndia VIX: no file found.")

    # --- verdict ----------------------------------------------------------
    print("\n" + "-" * 84)
    grade = "GOOD" if (complete / n_files >= 0.95 and crossed < 1.0) else "REVIEW"
    print(f"VERDICT: {grade}  ({complete}/{n_files} complete, "
          f"{partial} partial flagged for exclusion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
