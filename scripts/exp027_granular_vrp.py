"""Experiment 027: Granular VRP -- implied vs term-matched realized vol.

Pre-registered in docs/preregistrations/exp027_granular_vrp.md.

Fixes the interim readout's flagged flaw (vrp_derived_check.py): realized vol is
now TERM-MATCHED to each option's actual remaining life (entry -> expiry date),
using close-to-close returns across the real calendar -- capturing weekend/
overnight gaps, the same convention as the Exp 028 macro VIX comparison. This is
the decisive gate-check for the intraday VRP, run at the 20-collected-day gate.

    python scripts/exp027_granular_vrp.py

Read-only. No trades. Directional confirmation only -- NOT a significance claim
(sample is intentionally small; see prereg).
"""

from __future__ import annotations

import glob
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

from nifty_quant.analytics import options as opt
from nifty_quant.research import live_lens as lens

ANN = math.sqrt(252)
ENTRY_WINDOW_END = "09:30"   # first 15 minutes for the entry-IV average


def collected_days(data_dir="data") -> list[date]:
    days = []
    for f in glob.glob(os.path.join(data_dir, "option_chain", "*", "*", "*")):
        parts = Path(f).parts[-3:]
        try:
            days.append(date(int(parts[0]), int(parts[1]), int(parts[2])))
        except (ValueError, IndexError):
            continue
    return sorted(set(days))


def day_summary(d: date):
    """Return (closing_spot, near_expiry, entry_atm_iv_pct, intraday_realized_pct)
    or None if the day has no usable near-expiry data."""
    chains = lens.read_day("data", d)
    if not chains:
        return None
    near = min(c.expiry for c in chains)
    nc = sorted((c for c in chains if c.expiry == near), key=lambda c: c.timestamp)
    if len(nc) < 3:
        return None
    closing_spot = nc[-1].spot

    entry = [c for c in nc if c.timestamp.strftime("%H:%M") <= ENTRY_WINDOW_END]
    if not entry:
        entry = nc[:3]
    ivs = [opt.atm_iv(c) for c in entry]
    ivs = [v * 100 for v in ivs if v is not None]
    entry_iv = float(np.mean(ivs)) if ivs else float("nan")

    # intraday realized (for the 0-DTE case only)
    s = pd.Series([c.spot for c in nc],
                  index=pd.to_datetime([c.timestamp for c in nc])).sort_index()
    s5 = s.resample("5min").last().dropna()
    if len(s5) >= 5:
        r = np.diff(np.log(s5.to_numpy()))
        intraday_rv = math.sqrt(float(np.sum(r ** 2))) * ANN * 100.0
    else:
        intraday_rv = float("nan")
    return closing_spot, near, entry_iv, intraday_rv


def main() -> int:
    days = collected_days()
    today_incomplete = days[-1] if days else None  # exclude the in-progress day
    rows = []
    for d in days:
        res = day_summary(d)
        if res is None:
            continue
        closing_spot, near, entry_iv, intraday_rv = res
        rows.append({"date": d, "spot": closing_spot, "expiry": near,
                     "entry_iv": entry_iv, "intraday_rv": intraday_rv})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # exclude the final, possibly-incomplete day (today, if session not closed)
    complete_dates = set(df["date"])
    if today_incomplete in complete_dates:
        chains_today = lens.read_day("data", today_incomplete)
        if chains_today:
            near_today = min(c.expiry for c in chains_today)
            n_today = sum(1 for c in chains_today if c.expiry == near_today)
            if n_today < 150:   # heuristic: a full day has ~165-172 snapshots
                df = df[df["date"] != today_incomplete].reset_index(drop=True)
                print(f"(excluding {today_incomplete} -- session appears in-progress, "
                      f"only {n_today} near-expiry snapshots)")

    trading_days = list(df["date"])
    n_days = len(trading_days)
    print("=" * 92)
    print("EXP 027  GRANULAR VRP  (implied vs TERM-MATCHED realized, our own chains)")
    print(f"Complete collected trading days: {n_days}  "
          f"({trading_days[0]} -> {trading_days[-1]})" if n_days else "No data.")

    spot_by_date = dict(zip(df["date"], df["spot"]))

    print(f"\n{'date':<12}{'DTE':>4}{'expiry':<12}{'entryIV%':>9}{'realzd%':>9}"
          f"{'VRP':>8}  status")
    eligible = []
    excluded_window = []
    dte0 = []
    for _, row in df.iterrows():
        d, near, entry_iv = row["date"], row["expiry"], row["entry_iv"]
        if d not in trading_days:
            continue
        dte = trading_days.index(near) - trading_days.index(d) if near in trading_days else None

        if near == d:
            # 0-DTE: realized is that day's own intraday move
            vrp0 = entry_iv - row["intraday_rv"]
            dte0.append({"date": d, "entry_iv": entry_iv,
                        "realized": row["intraday_rv"], "vrp": vrp0})
            print(f"{str(d):<12}{'0':>4}{str(near):<12}{entry_iv:>9.1f}"
                  f"{row['intraday_rv']:>9.1f}{vrp0:>+8.1f}  0-DTE (intraday, separate)")
            continue

        if near not in spot_by_date:
            excluded_window.append(d)
            print(f"{str(d):<12}{'?':>4}{str(near):<12}{entry_iv:>9.1f}{'--':>9}"
                  f"{'--':>8}  window insufficient (expiry beyond collected data)")
            continue

        # term-matched realized vol: close-to-close from entry date to expiry date
        span = df[(df["date"] >= d) & (df["date"] <= near)]["spot"].to_numpy()
        if len(span) < 2:
            excluded_window.append(d)
            continue
        r = np.diff(np.log(span))
        # real calendar days spanned (captures weekend gap sizing correctly enough
        # for an annualization denominator at this scale)
        cal_days = (near - d).days
        ann_factor = math.sqrt(365.0 / max(cal_days, 1))
        realized = math.sqrt(float(np.sum(r ** 2))) * ann_factor * 100.0
        vrp = entry_iv - realized
        eligible.append({"date": d, "expiry": near, "dte_days": cal_days,
                        "entry_iv": entry_iv, "realized": realized, "vrp": vrp})
        print(f"{str(d):<12}{cal_days:>4}{str(near):<12}{entry_iv:>9.1f}"
              f"{realized:>9.1f}{vrp:>+8.1f}  eligible")

    elig_df = pd.DataFrame(eligible)
    print("\n" + "-" * 92)
    print(f"Eligible (term-matched, full window observed): {len(elig_df)}")
    print(f"Excluded (expiry beyond collected window):      {len(excluded_window)}")
    print(f"0-DTE (reported separately, intraday only):     {len(dte0)}")

    if len(elig_df) >= 1:
        vrp = elig_df["vrp"].to_numpy()
        pos = float(np.mean(vrp > 0)) * 100
        n_pos = int(np.sum(vrp > 0))
        print(f"\nTerm-matched VRP: mean {vrp.mean():+.1f}  median "
              f"{np.median(vrp):+.1f}  positive on {pos:.0f}% ({n_pos}/{len(vrp)})")
        if len(vrp) >= 3:
            sp = ss.binomtest(n_pos, len(vrp), 0.5, alternative="greater").pvalue
            print(f"one-sided sign test (positive dependence): p={sp:.3f}  "
                  f"(n too small for a real significance claim)")
    else:
        print("\nNo eligible term-matched trades (every expiry falls beyond the "
              "collected window). This is a real INSUFFICIENT DATA outcome.")

    if len(dte0):
        d0 = pd.DataFrame(dte0)
        v0 = d0["vrp"].to_numpy()
        print(f"\n0-DTE (intraday-only, separate bucket): mean VRP {v0.mean():+.1f}  "
              f"positive on {np.mean(v0>0)*100:.0f}% ({len(v0)} days)")

    print("\n" + "=" * 92)
    print("VERDICT (pre-registered):")
    if len(elig_df) < 8:
        print(f"  INCONCLUSIVE -- only {len(elig_df)} eligible term-matched days "
              f"(< 8 minimum). Descriptive only, no directional verdict forced.")
    else:
        pos_frac = float(np.mean(elig_df['vrp'].to_numpy() > 0))
        if pos_frac > 0.60:
            print(f"  CONFIRMS MACRO DIRECTION: VRP positive on {pos_frac*100:.0f}% of "
                  f"{len(elig_df)} eligible days (> 60% threshold).")
            print("  Consistent with Exp 028. NOT a significance claim (n too small);")
            print("  a real estimate needs ~60-100+ days. No trading rule implied.")
        else:
            print(f"  CONTRADICTS MACRO DIRECTION: VRP positive on only {pos_frac*100:.0f}% "
                  f"of {len(elig_df)} eligible days. Report honestly, do not explain away.")
    print("  Scope: directional gate-check on live data. No strategy, no trades.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
