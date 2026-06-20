"""Overnight-gap event study on NIFTY 5m candles (read-only research).

Tests one question: after a significant overnight gap (open vs prior close),
does NIFTY partially FADE the gap during the first 15/30/45/60 minutes, and
does the fade scale with gap size?

This is an EVENT STUDY, not a trading optimiser. No parameters are tuned; gap
buckets and horizons are fixed in advance. Primary declared test: 30-min
horizon, |gap| > 0.5 sigma, both directions.

    python scripts/gap_study.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd

SYMBOL = "NIFTY"
TF = "5m"
DATA_GLOB = os.path.join("data", "candles", TF, "*", f"{SYMBOL}_*.parquet")

HORIZON_BARS = {15: 3, 30: 6, 45: 9, 60: 12}  # 5m bars per horizon
VOL_LOOKBACK = 20  # trading days for daily-return sigma
# Round-trip cost in return space: 2 x slippage(3bps) + ~fee. Open is the
# worst-liquidity entry of the day, so this is optimistic if anything.
ROUND_TRIP_COST = 0.0006


def load_days() -> dict:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise SystemExit(f"No parquet found at {DATA_GLOB}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    return {d: g.reset_index(drop=True) for d, g in df.groupby("date")}


def build_events() -> pd.DataFrame:
    days = load_days()
    dates = sorted(days.keys())
    rows = []
    prev_close = None
    for d in dates:
        g = days[d]
        if len(g) < max(HORIZON_BARS.values()) + 1:
            prev_close = float(g["close"].iloc[-1]) if len(g) else prev_close
            continue
        day_open = float(g["open"].iloc[0])
        day_close = float(g["close"].iloc[-1])
        rec = {
            "date": d,
            "year": d.year,
            "dow": d.dayofweek,  # 0=Mon
            "open": day_open,
            "prev_close": prev_close,
        }
        if prev_close:
            rec["gap"] = day_open / prev_close - 1.0
            for mins, bars in HORIZON_BARS.items():
                fwd_close = float(g["close"].iloc[bars])
                rec[f"ret_{mins}"] = fwd_close / day_open - 1.0
        rows.append(rec)
        prev_close = day_close

    ev = pd.DataFrame(rows).dropna(subset=["gap"]).reset_index(drop=True)
    # Daily close-to-close returns -> rolling sigma for gap normalisation.
    ev["c2c"] = ev["prev_close"].pct_change()
    ev["sigma"] = ev["c2c"].rolling(VOL_LOOKBACK).std()
    ev = ev.dropna(subset=["sigma"]).reset_index(drop=True)
    ev["gap_sigma"] = ev["gap"] / ev["sigma"]
    # Fade return per horizon: short up-gaps, long down-gaps.
    for mins in HORIZON_BARS:
        ev[f"fade_{mins}"] = -np.sign(ev["gap"]) * ev[f"ret_{mins}"]
    return ev


def tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    s = x.std(ddof=1)
    if s == 0:
        return float("nan")
    return float(x.mean() / (s / math.sqrt(len(x))))


def summarize(ev: pd.DataFrame, mask, label: str, mins: int) -> dict:
    sub = ev[mask]
    fade = sub[f"fade_{mins}"].to_numpy()
    fade = fade[~np.isnan(fade)]
    n = len(fade)
    if n == 0:
        return {"bucket": label, "n": 0}
    return {
        "bucket": label,
        "n": n,
        "mean_fade_bps": round(fade.mean() * 1e4, 2),
        "median_fade_bps": round(float(np.median(fade)) * 1e4, 2),
        "hit_rate": round(float((fade > 0).mean()), 3),
        "t_stat": round(tstat(fade), 2),
        "net_fade_bps": round((fade.mean() - ROUND_TRIP_COST) * 1e4, 2),
    }


def buckets(ev: pd.DataFrame):
    a = ev["gap_sigma"].abs()
    return {
        "noise |g|<0.25": a < 0.25,
        "small 0.25-0.5": (a >= 0.25) & (a < 0.5),
        "medium 0.5-1.0": (a >= 0.5) & (a < 1.0),
        "large >1.0": a >= 1.0,
        "PRIMARY |g|>0.5": a >= 0.5,
    }


def show(title, rows):
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    cols = ["bucket", "n", "mean_fade_bps", "median_fade_bps", "hit_rate",
            "t_stat", "net_fade_bps"]
    print("  " + "".join(str(c).rjust(16) for c in cols))
    for r in rows:
        print("  " + "".join(str(r.get(c, "")).rjust(16) for c in cols))


def main() -> int:
    ev = build_events()
    print("=" * 96)
    print(f"OVERNIGHT GAP EVENT STUDY  {SYMBOL} {TF}")
    print(f"Events: {len(ev)}  | date range {ev['date'].min().date()} -> "
          f"{ev['date'].max().date()}")
    up = (ev["gap"] > 0).sum()
    print(f"Gap ups: {up}  gap downs: {len(ev) - up}  "
          f"mean |gap|: {ev['gap'].abs().mean()*100:.3f}%  "
          f"mean |gap_sigma|: {ev['gap_sigma'].abs().mean():.2f}")

    bk = buckets(ev)
    for mins in (15, 30, 45, 60):
        rows = [summarize(ev, m, lbl, mins) for lbl, m in bk.items()]
        show(f"--- Horizon {mins} min (fade return, bps) ---", rows)

    # Direction split at primary bucket / primary horizon.
    print("\n=== Direction split (PRIMARY |g|>0.5, 30 min) ===")
    prim = ev["gap_sigma"].abs() >= 0.5
    for lbl, m in {
        "gap UP": prim & (ev["gap"] > 0),
        "gap DOWN": prim & (ev["gap"] < 0),
    }.items():
        show(lbl, [summarize(ev, m, lbl, 30)])

    # Per-year (primary bucket, 30 min).
    print("\n=== Per-year (PRIMARY |g|>0.5, 30 min) ===")
    yr_rows = []
    for y in sorted(ev["year"].unique()):
        yr_rows.append(summarize(ev, prim & (ev["year"] == y), str(y), 30))
    show("by year", yr_rows)

    # Day-of-week (primary bucket, 30 min).
    print("\n=== Day-of-week (PRIMARY |g|>0.5, 30 min) ===")
    dows = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_rows = []
    for i, name in enumerate(dows):
        dow_rows.append(summarize(ev, prim & (ev["dow"] == i), name, 30))
    show("by dow", dow_rows)

    # Walk-forward: sign stability across 4 equal time folds (primary, 30 min).
    print("\n=== Walk-forward sign stability (PRIMARY |g|>0.5, 30 min) ===")
    prim_ev = ev[prim].reset_index(drop=True)
    fold_idx = np.array_split(np.arange(len(prim_ev)), 4)
    for i, idx in enumerate(fold_idx):
        f = prim_ev.iloc[idx]
        fade = f["fade_30"].to_numpy()
        fade = fade[~np.isnan(fade)]
        if len(fade):
            print(f"  fold {i+1}: n={len(fade):4d}  mean_fade_bps="
                  f"{fade.mean()*1e4:7.2f}  t={tstat(fade):5.2f}  "
                  f"hit={ (fade>0).mean():.3f}")

    # Simple fade-sim equity (primary, 30 min, net of cost), for drawdown.
    print("\n=== Fade sim (PRIMARY |g|>0.5, 30 min, net of cost) ===")
    s = ev[prim].copy().sort_values("date")
    pnl = s["fade_30"].to_numpy() - ROUND_TRIP_COST
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    print(f"  trades={len(pnl)}  total_net_return={eq[-1]*100:.2f}%  "
          f"avg/trade={pnl.mean()*1e4:.2f}bps  "
          f"max_drawdown={dd.min()*100:.2f}%  "
          f"win_rate={(pnl>0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
