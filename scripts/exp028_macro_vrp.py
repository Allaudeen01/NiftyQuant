"""Experiment 028: Macro Volatility Risk Premium (VRP) -- the 10-year Go/No-Go.

Pre-registered in docs/preregistrations/exp028_macro_vrp.md.

THE gatekeeper: does India VIX (implied) systematically exceed the NIFTY realized
volatility that follows over the next ~month, over 10 years? If not, the entire
option-chain-collection thesis is in question.

  VRP_t = VIX_t - RV_fwd_t   (vol points; positive => short-vol would have paid)
  RV_fwd_t = annualized realized vol of NIFTY over the next 21 trading days.

    python scripts/exp028_macro_vrp.py

Data: 10y of ^NSEI + ^INDIAVIX daily closes (Yahoo, cached). Read-only. No trades.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

from nifty_quant.analytics.stat_tests import adf_test, half_life

HORIZON = 21                       # trading days ~ 30 calendar (VIX horizon)
CACHE = Path("data") / "external" / "macro_vrp_daily.parquet"


def load_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    nifty = yf.download("^NSEI", period="10y", progress=False, auto_adjust=True)["Close"]
    vix = yf.download("^INDIAVIX", period="10y", progress=False, auto_adjust=True)["Close"]
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.iloc[:, 0]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    df = pd.DataFrame({"nifty": nifty, "vix": vix}).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index().reset_index().rename(columns={"index": "date", "Date": "date"})
    df.columns = ["date", "nifty", "vix"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def main() -> int:
    df = load_data()
    df["ret"] = np.log(df["nifty"] / df["nifty"].shift(1))
    n = len(df)

    # forward realized vol over the next HORIZON trading days, annualized %
    r = df["ret"].to_numpy()
    rv_fwd = np.full(n, np.nan)
    for t in range(n - HORIZON):
        window = r[t + 1: t + 1 + HORIZON]
        rv_fwd[t] = math.sqrt(252.0 / HORIZON * np.nansum(window ** 2)) * 100.0
    df["rv_fwd"] = rv_fwd
    df["vrp"] = df["vix"] - df["rv_fwd"]
    df["ratio"] = df["vix"] / df["rv_fwd"]
    df["year"] = df["date"].dt.year
    d = df.dropna(subset=["vrp"]).reset_index(drop=True)
    m = len(d)

    print("=" * 92)
    print("EXP 028  MACRO VOLATILITY RISK PREMIUM (VRP)  -- 10-year Go/No-Go")
    print(f"Days: {m}  | {d['date'].min().date()} -> {d['date'].max().date()}  "
          f"(VIX horizon = {HORIZON} trading days)")

    vrp = d["vrp"].to_numpy()
    pos_frac = float(np.mean(vrp > 0)) * 100
    mean_vrp = float(np.mean(vrp))
    med_vrp = float(np.median(vrp))
    mean_ratio = float(np.mean(d["ratio"]))
    print("\n--- overall premium ---")
    print(f"  VRP > 0 on {pos_frac:.1f}% of days")
    print(f"  mean VRP  = {mean_vrp:+.2f} vol pts   median = {med_vrp:+.2f}")
    print(f"  mean VIX/RV ratio = {mean_ratio:.2f}x  "
          f"(mean VIX {d['vix'].mean():.1f}  vs mean fwd-RV {d['rv_fwd'].mean():.1f})")

    # significance
    t_stat, t_p_two = ss.ttest_1samp(vrp, 0.0)
    t_p_one = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
    n_pos = int(np.sum(vrp > 0))
    sign_p = ss.binomtest(n_pos, m, 0.5, alternative="greater").pvalue
    print(f"  one-sided t-test mean>0: t={t_stat:.1f}  p={t_p_one:.2e}")
    print(f"  sign test (robust):      {n_pos}/{m} positive  p={sign_p:.2e}")

    # per-year
    print("\n--- per-year (structural persistence; stress years exposed) ---")
    year_pos = 0
    years = sorted(d["year"].unique())
    for y in years:
        sub = d[d["year"] == y]["vrp"].to_numpy()
        if len(sub) < 20:
            continue
        mv = float(np.mean(sub)); pf = float(np.mean(sub > 0)) * 100
        year_pos += 1 if mv > 0 else 0
        flag = "" if mv > 0 else "   <== NEGATIVE (realized > implied)"
        print(f"  {y}: n={len(sub):3d}  mean VRP {mv:+6.2f}  pos {pf:4.1f}%{flag}")

    # tail honesty
    print("\n--- tail honesty (the short-vol blow-up risk) ---")
    p1, p5 = np.percentile(vrp, [1, 5])
    worst = float(np.min(vrp))
    frac_bad = float(np.mean(vrp < -5)) * 100
    worst_date = d.loc[int(np.argmin(vrp)), "date"].date()
    print(f"  most negative VRP = {worst:+.1f} vol pts  (on {worst_date})")
    print(f"  1st pct {p1:+.1f}   5th pct {p5:+.1f}   "
          f"days with VRP < -5: {frac_bad:.1f}%")

    # mean-reversion / harvestability
    print("\n--- mean-reversion / harvestability ---")
    adf_vrp = adf_test(vrp)
    hl_vrp = half_life(vrp)
    adf_vix = adf_test(d["vix"].to_numpy())
    print(f"  VRP: ADF p={adf_vrp['pvalue']:.3f} "
          f"({'stationary' if adf_vrp['stationary_5pct'] else 'non-stationary'})  "
          f"OU half-life={hl_vrp:.1f} days")
    print(f"  VIX: ADF p={adf_vix['pvalue']:.3f} "
          f"({'stationary' if adf_vix['stationary_5pct'] else 'non-stationary'})")

    # verdict
    go = (sign_p < 0.01) and (pos_frac > 60) and (year_pos >= 7)
    severe_tail = (frac_bad > 5) or (worst < -20)
    print("\n" + "=" * 92)
    print("VERDICT (pre-registered Go/No-Go):")
    if go and severe_tail:
        print("  CONDITIONAL GO: VRP is structurally positive (thesis validated) BUT")
        print(f"  the negative tail is severe (worst {worst:+.0f}, {frac_bad:.0f}% of days < -5).")
        print("  -> KEEP collecting option data; ANY short-vol strategy MUST be")
        print("     risk-managed & fractionally sized (Exp 016 tail work). Never naive.")
    elif go:
        print("  GO: VRP is structurally positive and the tail is manageable.")
        print("  -> The volatility-selling thesis is validated; keep building.")
    else:
        print("  NO-GO / RETHINK: the premium is not robustly positive on this sample.")
        print(f"  (pos {pos_frac:.0f}%, sign p={sign_p:.1e}, positive in {year_pos}/{len(years)} yrs)")
        print("  -> Reconsider before investing more in the options pipeline.")
    print(f"  [pos-frac {pos_frac:.0f}% | mean {mean_vrp:+.1f} | years+ {year_pos}/{len(years)}"
          f" | half-life {hl_vrp:.0f}d]")
    print("  Scope: macro validation only. No trading rule implied.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
