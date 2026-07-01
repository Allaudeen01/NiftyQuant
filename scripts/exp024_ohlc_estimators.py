"""Experiment 024: OHLC volatility estimators as HAR inputs.

Pre-registered in docs/preregistrations/exp024_ohlc_estimators.md.

Which daily volatility estimator, used as the HAR information source, best
forecasts NIFTY next-day (5m) realized volatility?

  CC    close-to-close squared return (naive baseline)
  RV5M  5-minute realized variance (incumbent, most accurate)
  PARK  Parkinson (high/low)
  GK    Garman-Klass (OHLC)
  RS    Rogers-Satchell (drift-independent)
  YZ1   single-day Yang-Zhang-style: overnight^2 + Rogers-Satchell (incl. gap)

Fair design (Patton 2011): every estimator forecasts the SAME target -- next-day
RV5M (the most accurate proxy) -- so QLIKE/Diebold-Mariano are comparable. Each
HAR maps the estimator's own lagged 1/5/22-day history to log RV5M_{t+1}.

    python scripts/exp024_ohlc_estimators.py

Read-only research. No trades.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp013_rv_forecast_comparison import qlike, diebold_mariano, benjamini_hochberg

from nifty_quant.analytics.vol_estimators import (
    garman_klass, parkinson, rogers_satchell, yang_zhang_daily,
)

ANNUALISE = math.sqrt(252)
FLOOR = 1e-10


def rv5m_by_day() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    rows = []
    for d, g in df.groupby("date"):
        c = g.sort_values("timestamp")["close"].to_numpy()
        if len(c) < 12:
            continue
        lr = np.log(c[1:] / c[:-1])
        rows.append({"date": d, "rv5m_var": float(np.sum(lr ** 2))})
    return pd.DataFrame(rows)


def daily_ohlc() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "1d", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df.rename(columns={"timestamp": "date"})
    return df[["date", "open", "high", "low", "close"]]


ESTIMATORS = ["CC", "RV5M", "PARK", "GK", "RS", "YZ1"]


def build_estimators() -> pd.DataFrame:
    ohlc = daily_ohlc()
    rv5 = rv5m_by_day()
    m = ohlc.merge(rv5, on="date", how="inner").sort_values("date").reset_index(drop=True)
    o, h, l, c = (m["open"].to_numpy(), m["high"].to_numpy(),
                  m["low"].to_numpy(), m["close"].to_numpy())
    pc = m["close"].shift(1).to_numpy()
    r_cc = np.log(m["close"] / m["close"].shift(1)).to_numpy()

    m["CC"] = r_cc ** 2
    m["RV5M"] = m["rv5m_var"].to_numpy()
    m["PARK"] = parkinson(h, l)
    m["GK"] = garman_klass(o, h, l, c)
    m["RS"] = rogers_satchell(o, h, l, c)
    m["YZ1"] = yang_zhang_daily(o, h, l, c, pc)
    # per-day variance -> daily std (clip to floor), then log for HAR
    for e in ESTIMATORS:
        std = np.sqrt(np.clip(m[e].to_numpy(), FLOOR, None))
        m[f"log_{e}"] = np.log(np.clip(std, 1e-9, None))
    return m.dropna(subset=["log_CC"]).reset_index(drop=True)


def har_forecast_series(m: pd.DataFrame, est: str):
    """Expanding one-step-ahead: map estimator's log-history -> next-day RV5M.

    Returns aligned arrays (actual_rv5m_std, pred_rv5m_std) on the OOS set.
    """
    log_est = m[f"log_{est}"]
    d = log_est.shift(1)
    w = log_est.shift(1).rolling(5).mean()
    mo = log_est.shift(1).rolling(22).mean()
    y = m["log_RV5M"]                              # target: log RV5M (std) today
    frame = pd.DataFrame({"y": y, "d": d, "w": w, "m": mo}).dropna().reset_index(drop=True)
    n = len(frame)
    start = n // 2
    preds, acts = [], []
    Xall = frame[["d", "w", "m"]].to_numpy()
    yall = frame["y"].to_numpy()
    for t in range(start, n):
        Xtr = np.column_stack([np.ones(t), Xall[:t]])
        beta = sm.OLS(yall[:t], Xtr).fit().params
        xte = np.r_[1.0, Xall[t]]
        preds.append(math.exp(float(xte @ beta)))
        acts.append(math.exp(yall[t]))
    return np.array(acts), np.array(preds)


def main() -> int:
    m = build_estimators()
    print("=" * 92)
    print("EXP 024  OHLC VOLATILITY ESTIMATORS as HAR INPUTS  (target: next-day RV5M)")
    print(f"Days: {len(m)}  | {m['date'].min().date()} -> {m['date'].max().date()}")
    print("  mean annualised vol by estimator:")
    for e in ESTIMATORS:
        mv = np.sqrt(np.clip(m[e].to_numpy(), FLOOR, None)).mean() * ANNUALISE * 100
        print(f"    {e:<6} {mv:5.1f}%")

    # forecasts on a COMMON OOS set (align by taking the min length across all)
    results = {e: har_forecast_series(m, e) for e in ESTIMATORS}
    L = min(len(v[0]) for v in results.values())
    actual = results["RV5M"][0][-L:]               # same target for all
    losses = {}
    print(f"\n--- OOS loss (common {L} days; lower is better) ---")
    print(f"  {'estimator':<8}{'RMSE(bp)':>12}{'MAE(bp)':>12}{'QLIKE':>12}")
    stats = {}
    for e in ESTIMATORS:
        pred = results[e][1][-L:]
        rmse = math.sqrt(np.mean((actual - pred) ** 2)) * 1e4
        mae = np.mean(np.abs(actual - pred)) * 1e4
        ql = qlike(actual, pred)
        losses[e] = ql
        stats[e] = (rmse, mae, float(np.mean(ql)))
        print(f"  {e:<8}{rmse:>12.2f}{mae:>12.2f}{np.mean(ql):>12.4f}")

    ranked = sorted(ESTIMATORS, key=lambda e: stats[e][2])
    best = ranked[0]
    print(f"\nLowest QLIKE: {best}")

    # Diebold-Mariano vs RV5M (incumbent), BH-corrected
    print("\n--- Diebold-Mariano vs RV5M (QLIKE, HAC). DM<0 => estimator BEATS RV5M ---")
    comp = [e for e in ESTIMATORS if e != "RV5M"]
    dm_stats, dm_ps = {}, []
    for e in comp:
        dm, p = diebold_mariano(losses[e], losses["RV5M"])
        dm_stats[e] = (dm, p)
        dm_ps.append(p)
    bh = benjamini_hochberg(dm_ps)
    print(f"  {'estimator':<8}{'DM':>10}{'p':>12}{'BH p':>12}   verdict vs RV5M")
    for e, bhp in zip(comp, bh):
        dm, p = dm_stats[e]
        if math.isnan(dm):
            verd = "n/a"
        elif bhp < 0.05 and dm < 0:
            verd = "BEATS RV5M"
        elif bhp < 0.05 and dm > 0:
            verd = "worse than RV5M"
        else:
            verd = "no sig. difference"
        print(f"  {e:<8}{dm:>10.2f}{p:>12.4f}{bhp:>12.4f}   {verd}")

    # CC -> best improvement (what the naive baseline leaves on the table)
    cc_q, best_q = stats["CC"][2], stats[best][2]
    impr = (cc_q - best_q) / cc_q * 100 if cc_q else float("nan")

    print("\n" + "=" * 92)
    print("VERDICT (pre-registered):")
    if best == "RV5M":
        print("  RV5M REMAINS BEST (lowest QLIKE). Keep the 5m realized-variance input;")
        print("  range estimators are confirmed fallbacks for no-intraday-data periods.")
    else:
        dm, p = dm_stats[best]; bhp = dict(zip(comp, bh))[best]
        if not math.isnan(dm) and bhp < 0.05 and dm < 0:
            print(f"  UPGRADE ADOPTED: {best} beats RV5M (BH p={bhp:.4f}). Switch HAR input")
            print(f"  to {best}" + ("  (gain from the overnight term)" if best == "YZ1" else "")
                  + ". NOTE: a 6-way compare -> confirm with an Exp 023 snooping check.")
        else:
            print(f"  INCONCLUSIVE: {best} has lowest QLIKE but does not clear DM vs RV5M.")
            print("  Keep RV5M; ranking is descriptive.")
    print(f"  (naive CC QLIKE {cc_q:.4f} -> best {best} {best_q:.4f}: {impr:+.1f}% vs CC)")
    print("  Scope: forecasting-input comparison. No trading rule implied.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
