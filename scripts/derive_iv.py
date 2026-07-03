"""Materialise implied volatility into a SEPARATE derived store (non-destructive).

Reads raw option-chain snapshots and writes per-quote Black-Scholes IV to
  data/derived/option_chain_iv/<YYYY>/<MM>/<DD>/<HH_MM>.parquet
The RAW warehouse is never modified. Re-runnable; overwrites only the derived
files. IV is recovered on read for research (VRP, IV surface, skew) without
bloating the collector or touching pristine data.

    python scripts/derive_iv.py --date 2026-07-02
    python scripts/derive_iv.py --all            # every collected day

Read-only w.r.t. raw data. No trades.
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import date
from pathlib import Path

import pandas as pd

from nifty_quant.research.derive_iv import derive_iv_frame

RAW = os.path.join("data", "option_chain")
DERIVED = os.path.join("data", "derived", "option_chain_iv")

_KEEP = ["snapshot_ts", "underlying", "expiry", "strike", "option_type",
         "last_price", "bid", "ask", "spot"]


def derive_day(d: date, r: float, q: float) -> tuple[int, int]:
    folder = os.path.join(RAW, f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        return (0, 0)
    out_dir = Path(DERIVED) / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_iv = 0
    for f in files:
        df = pd.read_parquet(f)
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
        iv = derive_iv_frame(df, r=r, q=q)
        cols = [c for c in _KEEP if c in df.columns]
        out = df[cols].copy()
        out["iv"] = iv.to_numpy()
        tmp = out_dir / (Path(f).stem + ".parquet.tmp")
        final = out_dir / (Path(f).stem + ".parquet")
        out.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(final)
        n_files += 1
        n_iv += int(iv.notna().sum())
    return (n_files, n_iv)


def all_collected_days() -> list[date]:
    days = []
    for f in glob.glob(os.path.join(RAW, "*", "*", "*")):
        parts = Path(f).parts[-3:]
        try:
            days.append(date(int(parts[0]), int(parts[1]), int(parts[2])))
        except (ValueError, IndexError):
            continue
    return sorted(set(days))


def main() -> int:
    p = argparse.ArgumentParser(description="Derive IV into a separate store (non-destructive).")
    p.add_argument("--date", default=None, help="YYYY-MM-DD")
    p.add_argument("--all", action="store_true", help="process every collected day")
    p.add_argument("--r", type=float, default=0.065, help="risk-free rate")
    p.add_argument("--q", type=float, default=0.012, help="dividend yield")
    args = p.parse_args()

    if args.all:
        days = all_collected_days()
    elif args.date:
        days = [date.fromisoformat(args.date)]
    else:
        days = all_collected_days()[-1:]  # most recent by default
    if not days:
        print("No collected days found.")
        return 0

    print(f"Deriving IV -> {DERIVED}  (raw data untouched)")
    for d in days:
        nf, niv = derive_day(d, args.r, args.q)
        if nf:
            print(f"  {d.isoformat()}: {nf} snapshots  ({niv} quotes with solvable IV)")
        else:
            print(f"  {d.isoformat()}: no raw snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
