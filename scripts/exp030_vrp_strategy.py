"""Experiment 030: VRP Harvesting Strategy v1.0 -- 10-year macro backtest (PAPER).

Pre-registered in docs/preregistrations/exp030_vrp_strategy_v1.md.

Rules-based short-volatility harvesting of the premium validated in Exp 028.
Non-overlapping ~monthly (21-trading-day) blocks; per-trade P&L (vol points) =
VIX_start - realized_vol - cost. Two fixed sizing rules: constant and
vol-targeted (weight ~ 1/VIX). PAPER ONLY -- no live orders, VRP-proxy P&L.

    python scripts/exp030_vrp_strategy.py

Read-only research. No trades.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

BLOCK = 21                     # trading days per position (~1 month)
COST_VOL_PTS = 1.5             # round-trip transaction cost (vol points)
CACHE = Path("data") / "external" / "macro_vrp_daily.parquet"
ANN_BLOCKS = 252 / BLOCK       # ~12 blocks/year, for annualizing


def load() -> pd.DataFrame:
    if not CACHE.exists():
        raise SystemExit("Run scripts/exp028_macro_vrp.py first to cache the data.")
    df = pd.read_parquet(CACHE)
    df["ret"] = np.log(df["nifty"] / df["nifty"].shift(1))
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().reset_index(drop=True)


def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping 21-day blocks: VIX at start vs realized over the block."""
    r = df["ret"].to_numpy()
    vix = df["vix"].to_numpy()
    dates = df["date"].to_numpy()
    rows = []
    n = len(df)
    for start in range(0, n - BLOCK, BLOCK):
        window = r[start + 1: start + 1 + BLOCK]
        if len(window) < BLOCK:
            break
        rv = math.sqrt(252.0 / BLOCK * np.sum(window ** 2)) * 100.0
        rows.append({
            "date": dates[start],
            "year": pd.Timestamp(dates[start]).year,
            "vix": float(vix[start]),
            "realized": rv,
            "gross_vrp": float(vix[start]) - rv,     # vol points, before cost
        })
    return pd.DataFrame(rows)


def metrics(pnl: np.ndarray, label: str, cap_base: float | None = None) -> dict:
    n = len(pnl)
    mean = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    downside = pnl[pnl < 0]
    dd_sd = math.sqrt(float(np.mean(downside ** 2))) if downside.size else 0.0
    sharpe = (mean / sd * math.sqrt(ANN_BLOCKS)) if sd > 0 else float("nan")
    sortino = (mean / dd_sd * math.sqrt(ANN_BLOCKS)) if dd_sd > 0 else float("nan")
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else float("nan")
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())
    return {
        "label": label, "n": n, "mean": mean, "hit": float(np.mean(pnl > 0)) * 100,
        "sharpe": sharpe, "sortino": sortino, "t": t,
        "worst": float(np.min(pnl)), "total": float(equity[-1]), "max_dd": max_dd,
    }


def main() -> int:
    df = load()
    tr = build_trades(df)
    print("=" * 90)
    print("EXP 030  VRP HARVESTING STRATEGY v1.0  (10y macro backtest, PAPER ONLY)")
    print(f"Trades: {len(tr)} non-overlapping {BLOCK}-day blocks | "
          f"{tr['date'].min().date()} -> {tr['date'].max().date()}  cost={COST_VOL_PTS} vol pts")

    # cost sensitivity on the constant-size book
    print("\n--- cost sensitivity (constant size, mean net VRP capture, vol pts) ---")
    for c in (0.0, 1.5, 3.0):
        net = tr["gross_vrp"].to_numpy() - c
        print(f"  cost {c:>4.1f}: mean {np.mean(net):+.2f}  hit {np.mean(net>0)*100:.0f}%  "
              f"Sharpe {metrics(net,'')['sharpe']:.2f}")

    net = tr["gross_vrp"].to_numpy() - COST_VOL_PTS      # constant size, base cost
    # vol-targeted sizing: weight ~ 1/VIX, normalized to mean weight 1
    w = (1.0 / tr["vix"].to_numpy())
    w = w / w.mean()
    net_vt = w * net

    m_const = metrics(net, "constant")
    m_vt = metrics(net_vt, "vol-target")
    print("\n--- performance after cost (annualized where applicable) ---")
    print(f"  {'sizing':<12}{'mean':>7}{'hit%':>6}{'Sharpe':>8}{'Sortino':>9}"
          f"{'t':>6}{'worst':>8}{'cumVP':>8}{'maxDD':>8}")
    for m in (m_const, m_vt):
        print(f"  {m['label']:<12}{m['mean']:>7.2f}{m['hit']:>6.0f}{m['sharpe']:>8.2f}"
              f"{m['sortino']:>9.2f}{m['t']:>6.1f}{m['worst']:>8.1f}"
              f"{m['total']:>8.1f}{m['max_dd']:>8.1f}")

    # per-year stability (constant, net)
    print("\n--- per-year mean net capture (constant size, vol pts) ---")
    tr = tr.assign(net=net)
    yrs = sorted(tr["year"].unique())
    yr_pos = 0
    for y in yrs:
        sub = tr[tr["year"] == y]["net"].to_numpy()
        if len(sub) == 0:
            continue
        mv = float(np.mean(sub))
        yr_pos += 1 if mv > 0 else 0
        flag = "" if mv > 0 else "   <== negative"
        print(f"  {y}: n={len(sub):2d}  mean {mv:+.2f}{flag}")

    # tail / 2020 focus
    worst_idx = int(np.argmin(net))
    worst_row = tr.iloc[worst_idx]
    equity_const = np.cumsum(net)
    prior_gain = float(equity_const[worst_idx - 1]) if worst_idx > 0 else 0.0
    print("\n--- tail / ruin check ---")
    print(f"  worst trade: {worst_row['date'].date()}  net {net[worst_idx]:+.1f} vol pts "
          f"(VIX {worst_row['vix']:.1f} -> realized {worst_row['realized']:.1f})")
    print(f"  cumulative gain BEFORE that trade: {prior_gain:+.1f} vol pts  "
          f"-> {'WIPES prior gains' if -net[worst_idx] > prior_gain else 'survivable'}")
    # % drawdown under a stated capital base for vol-target sizing:
    # assume 1 vol-pt of P&L = 1% of capital at the mean weight (a modest,
    # explicit leverage choice for interpretation only).
    cap_pct_dd = m_vt["max_dd"]  # in vol pts == % under this 1pt=1% convention
    print(f"  vol-target max drawdown: {m_vt['max_dd']:.1f} vol pts "
          f"(= {cap_pct_dd:.1f}% under the stated 1pt=1% sizing convention)")

    # verdict
    viable = (m_const["t"] >= 2 and m_vt["sharpe"] >= 0.5 and yr_pos >= 7
              and (-net[worst_idx] <= prior_gain) and cap_pct_dd >= -40)
    marginal = (m_const["mean"] > 0 and not viable and m_vt["sharpe"] >= 0.2)
    print("\n" + "=" * 90)
    print("VERDICT (pre-registered, PAPER ONLY):")
    if viable:
        print("  VIABLE (paper): positive after-cost carry, Sharpe>=0.5 (vol-target),")
        print(f"  profitable {yr_pos}/{len(yrs)} yrs, tail survivable. -> eligible for LIVE")
        print("  PAPER trading on collected chains (Exp 027+). NOT a live-order greenlight.")
    elif marginal:
        print("  MARGINAL: positive after cost but risk-adjusted/robustness bar not fully")
        print("  cleared. Keep researching; do NOT paper-trade yet.")
    else:
        print("  NOT VIABLE / REDESIGN under v1.0 rules on this proxy.")
    print(f"  [const t={m_const['t']:.1f} | VT Sharpe={m_vt['sharpe']:.2f} | "
          f"yrs+={yr_pos}/{len(yrs)} | worst={net[worst_idx]:+.0f}vp]")
    print("  Reminder: VRP-proxy P&L; real straddle execution is WORSE. Paper only.")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
