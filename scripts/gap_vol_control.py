"""Experiment 002: isolate gap size from volatility regime (read-only).

Question: is the gap-fade effect caused by the gap itself, or is it a
manifestation of volatility regimes? We hold the independent variable (gap
size > 0.5 sigma) and split by a FIXED volatility control (terciles of prior-
day realized volatility -> LOW/MED/HIGH). Nothing is optimized.

Fixed design (identical to Exp 001 cost model):
  entry  = open of the 09:20 bar
  exit   = open of the 09:50 bar   (30-minute hold)
  fade   = -sign(gap) * (exit/entry - 1)   [short up-gaps, long down-gaps]
  cost   = 6 bps round-trip (2 x 3bps slippage + fee)

    python scripts/gap_vol_control.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd

SYMBOL = "NIFTY"
DATA_GLOB = os.path.join("data", "candles", "5m", "*", f"{SYMBOL}_*.parquet")
VOL_LOOKBACK = 20          # days, for gap sigma-normalisation (as Exp 001)
ROUND_TRIP_COST = 0.0006   # identical to Exp 001
ENTRY_T, EXIT_T = "09:20", "09:50"
RNG = np.random.default_rng(7)


def load_days() -> dict:
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise SystemExit(f"No parquet at {DATA_GLOB}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df["t"] = df["timestamp"].dt.strftime("%H:%M")
    return {d: g.reset_index(drop=True) for d, g in df.groupby("date")}


def price_at(g: pd.DataFrame, t: str):
    row = g[g["t"] == t]
    return float(row["open"].iloc[0]) if len(row) else None


def build_events() -> pd.DataFrame:
    days = load_days()
    dates = sorted(days)
    rows = []
    prev_close = None
    prev_rv = None
    for d in dates:
        g = days[d]
        entry = price_at(g, ENTRY_T)
        exit_ = price_at(g, EXIT_T)
        day_open = float(g["open"].iloc[0]) if len(g) else None
        # this day's realized vol = std of its 5m pct returns
        rv = float(g["close"].pct_change().std()) if len(g) > 2 else np.nan
        day_close = float(g["close"].iloc[-1]) if len(g) else prev_close
        if prev_close and prev_rv is not None and entry and exit_ and day_open:
            gap = day_open / prev_close - 1.0
            ret = exit_ / entry - 1.0               # 09:20 -> 09:50
            ret_orig = exit_ / day_open - 1.0        # 09:15 -> 09:50 (Exp 001)
            ret_5 = entry / day_open - 1.0           # 09:15 -> 09:20 (first bar)
            rows.append({
                "date": d, "year": d.year,
                "prev_close": prev_close,
                "gap": gap,
                "fade": -np.sign(gap) * ret,
                "fade_orig": -np.sign(gap) * ret_orig,
                "fade_5": -np.sign(gap) * ret_5,
                "prev_rv": prev_rv,   # control variable = PRIOR day's realized vol
            })
        prev_rv = rv
        prev_close = day_close

    ev = pd.DataFrame(rows).reset_index(drop=True)
    # gap sigma-normalisation via rolling std of daily close-to-close returns
    # (same approach as Exp 001).
    ev["c2c"] = ev["prev_close"].pct_change()
    ev["sigma"] = ev["c2c"].rolling(VOL_LOOKBACK).std()
    ev = ev.dropna(subset=["sigma", "prev_rv", "gap"]).reset_index(drop=True)
    ev["gap_sigma"] = ev["gap"] / ev["sigma"]
    return ev


def boot_ci(x: np.ndarray, n: int = 5000) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    means = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)) * 1e4,
            float(np.percentile(means, 97.5)) * 1e4)


def stats(fade: np.ndarray) -> dict:
    fade = fade[~np.isnan(fade)]
    n = len(fade)
    if n < 2:
        return {"n": n}
    m = fade.mean(); sd = fade.std(ddof=1); se = sd / math.sqrt(n)
    lo, hi = boot_ci(fade)
    pnl = fade - ROUND_TRIP_COST
    eq = np.cumsum(pnl); dd = (eq - np.maximum.accumulate(eq)).min()
    return {
        "n": n,
        "mean_bps": round(m * 1e4, 2),
        "median_bps": round(float(np.median(fade)) * 1e4, 2),
        "hit": round(float((fade > 0).mean()), 3),
        "std_bps": round(sd * 1e4, 1),
        "sharpe_tr": round(m / sd, 3) if sd else float("nan"),
        "sharpe_ann": round(m / sd * math.sqrt(252), 2) if sd else float("nan"),
        "t": round(m / se, 2),
        "ci95_bps": (round((m - 1.96 * se) * 1e4, 2),
                     round((m + 1.96 * se) * 1e4, 2)),
        "boot_bps": (round(lo, 2), round(hi, 2)),
        "net_bps": round((m - ROUND_TRIP_COST) * 1e4, 2),
        "maxdd_pct": round(dd * 100, 2),
    }


def vol_buckets(ev: pd.DataFrame) -> pd.Series:
    q1, q2 = ev["prev_rv"].quantile([1/3, 2/3])
    lab = pd.Series(index=ev.index, dtype=object)
    lab[ev["prev_rv"] < q1] = "LOW"
    lab[(ev["prev_rv"] >= q1) & (ev["prev_rv"] < q2)] = "MED"
    lab[ev["prev_rv"] >= q2] = "HIGH"
    return lab, float(q1), float(q2)


def ols(y: np.ndarray, X: np.ndarray, names) -> None:
    """OLS with t-stats. X already includes intercept column."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * XtX_inv))
    for nm, b, s in zip(names, beta, se):
        print(f"    {nm:<16} coef={b: .6e}  t={b/s: .2f}")


def line(d: dict) -> str:
    keys = ["n", "mean_bps", "median_bps", "hit", "std_bps", "sharpe_ann",
            "t", "ci95_bps", "boot_bps", "net_bps", "maxdd_pct"]
    return "  ".join(f"{k}={d.get(k)}" for k in keys)


def main() -> int:
    ev = build_events()
    lab, q1, q2 = vol_buckets(ev)
    ev["vol"] = lab
    prim = ev["gap_sigma"].abs() >= 0.5
    print("=" * 96)
    print("EXP 002  GAP-FADE vs VOLATILITY CONTROL  NIFTY 5m  entry 09:20 exit 09:50")
    print(f"All events: {len(ev)}  | prim(|g|>0.5sigma): {int(prim.sum())}  | "
          f"date {ev['date'].min().date()} -> {ev['date'].max().date()}")
    print(f"Vol terciles (prev-day realized vol): LOW<{q1:.5f}<=MED<{q2:.5f}<=HIGH")

    print("\n=== PRIMARY (|g|>0.5sigma) by VOLATILITY bucket ===")
    for b in ("LOW", "MED", "HIGH"):
        m = prim & (ev["vol"] == b)
        print(f"  [{b}] " + line(stats(ev[m]["fade"].to_numpy())))

    print("\n=== CONTROL: within each vol bucket, LARGE gap vs SMALL gap (mean fade) ===")
    for b in ("LOW", "MED", "HIGH"):
        big = (ev["vol"] == b) & (ev["gap_sigma"].abs() >= 0.5)
        sml = (ev["vol"] == b) & (ev["gap_sigma"].abs() < 0.5)
        sb = stats(ev[big]["fade"].to_numpy())
        ss = stats(ev[sml]["fade"].to_numpy())
        print(f"  [{b}] large: n={sb.get('n')} mean={sb.get('mean_bps')}bps "
              f"t={sb.get('t')}  |  small: n={ss.get('n')} "
              f"mean={ss.get('mean_bps')}bps t={ss.get('t')}")

    print("\n=== TEST 5/6: OLS  fade ~ |gap_sigma| + prev_rv (standardized) ===")
    sub = ev.dropna(subset=["gap_sigma", "prev_rv", "fade"])
    g = (sub["gap_sigma"].abs() - sub["gap_sigma"].abs().mean()) / sub["gap_sigma"].abs().std()
    v = (sub["prev_rv"] - sub["prev_rv"].mean()) / sub["prev_rv"].std()
    X = np.column_stack([np.ones(len(sub)), g.to_numpy(), v.to_numpy()])
    ols(sub["fade"].to_numpy(), X, ["intercept", "abs_gap_sigma", "prev_rv"])
    print("    (H1 supported if abs_gap_sigma t stays significant controlling for vol)")

    print("\n=== ROBUSTNESS: per-year mean fade (PRIMARY) by vol bucket ===")
    for y in sorted(ev["year"].unique()):
        parts = []
        for b in ("LOW", "MED", "HIGH"):
            m = prim & (ev["vol"] == b) & (ev["year"] == y)
            s = stats(ev[m]["fade"].to_numpy())
            parts.append(f"{b}:n={s.get('n')},{s.get('mean_bps')}bps,t={s.get('t')}")
        print(f"  {y}: " + "  ".join(parts))

    print("\n=== ROBUSTNESS: 4-fold walk-forward (PRIMARY, all vol) ===")
    pe = ev[prim].reset_index(drop=True)
    for i, idx in enumerate(np.array_split(np.arange(len(pe)), 4)):
        f = pe.iloc[idx]["fade"].to_numpy(); f = f[~np.isnan(f)]
        s = stats(f)
        print(f"  fold {i+1}: n={s.get('n')} mean={s.get('mean_bps')}bps "
              f"t={s.get('t')} hit={s.get('hit')}")

    # --- Decompose the ORIGINAL Exp 001 signal (09:15 open entry) by vol -----
    print("\n" + "=" * 96)
    print("CROSS-CHECK: ORIGINAL Exp 001 signal (entry 09:15 OPEN -> 09:50) "
          "decomposed by vol")
    agg = stats(ev[prim]["fade_orig"].to_numpy())
    print(f"  PRIMARY aggregate (09:15 entry): mean={agg.get('mean_bps')}bps "
          f"t={agg.get('t')} hit={agg.get('hit')} n={agg.get('n')}  "
          f"[Exp 001 reproduced]")
    for b in ("LOW", "MED", "HIGH"):
        m = prim & (ev["vol"] == b)
        s = stats(ev[m]["fade_orig"].to_numpy())
        print(f"  [{b}] n={s.get('n')} mean={s.get('mean_bps')}bps "
              f"median={s.get('median_bps')}bps hit={s.get('hit')} t={s.get('t')} "
              f"net={s.get('net_bps')}bps")
    print("\n  OLS fade_orig ~ |gap_sigma| + prev_rv (standardized):")
    sub = ev.dropna(subset=["gap_sigma", "prev_rv", "fade_orig"])
    g = (sub["gap_sigma"].abs() - sub["gap_sigma"].abs().mean()) / sub["gap_sigma"].abs().std()
    v = (sub["prev_rv"] - sub["prev_rv"].mean()) / sub["prev_rv"].std()
    X = np.column_stack([np.ones(len(sub)), g.to_numpy(), v.to_numpy()])
    ols(sub["fade_orig"].to_numpy(), X, ["intercept", "abs_gap_sigma", "prev_rv"])

    # First-5-min contribution (09:15 -> 09:20): fade_orig - fade is NOT exact
    # (different bases), so measure the 09:15->09:20 fade directly.
    print("\n  First-5-min fade (09:15 OPEN -> 09:20), PRIMARY by vol:")
    agg5 = stats(ev[prim]["fade_5"].to_numpy())
    print(f"  PRIMARY aggregate: mean={agg5.get('mean_bps')}bps "
          f"t={agg5.get('t')} hit={agg5.get('hit')} n={agg5.get('n')}")
    for b in ("LOW", "MED", "HIGH"):
        m = prim & (ev["vol"] == b)
        s = stats(ev[m]["fade_5"].to_numpy())
        print(f"  [{b}] n={s.get('n')} mean={s.get('mean_bps')}bps "
              f"hit={s.get('hit')} t={s.get('t')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
