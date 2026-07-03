"""Interim VRP readout on our own collected option chains (DESCRIPTIVE, pre-gate).

NOT the pre-registered Exp 027. With only a handful of collected days this is a
descriptive sanity check -- does the volatility risk premium that Exp 028 proved
at the macro level (India VIX > realized 80% of the time over 10y) show up in our
own intraday ATM implied vol vs the realized vol actually delivered? No
significance is claimed until the 20-day gate (Exp 027) is met.

For each collected day:
  implied  = mean ATM implied vol (annualised %), derived via Black-Scholes from
             the stored chain (options.atm_iv -- the ATM slice of the derived IV)
  realized = annualised realized vol from that day's intraday spot path (5-min)
  VRP      = implied - realized   (positive => options richer than delivered)

    python scripts/vrp_derived_check.py

Read-only. No trades.
"""

from __future__ import annotations

import glob
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_quant.analytics import options as opt
from nifty_quant.research import live_lens as lens

ANN = math.sqrt(252)
GATE_DAYS = 20        # live-lens VRP significance gate (Exp 027)


def collected_days(data_dir="data") -> list[date]:
    days = []
    for f in glob.glob(os.path.join(data_dir, "option_chain", "*", "*", "*")):
        parts = Path(f).parts[-3:]
        try:
            days.append(date(int(parts[0]), int(parts[1]), int(parts[2])))
        except (ValueError, IndexError):
            continue
    return sorted(set(days))


def realized_vol_annual(near_chains) -> float:
    """Annualised realized vol from the day's intraday spot path (5-min resample)."""
    s = pd.Series(
        [c.spot for c in near_chains],
        index=pd.to_datetime([c.timestamp for c in near_chains]),
    ).sort_index()
    s5 = s.resample("5min").last().dropna()      # reduce 2-min microstructure noise
    if len(s5) < 5:
        return float("nan")
    r = np.diff(np.log(s5.to_numpy()))
    return math.sqrt(float(np.sum(r ** 2))) * ANN * 100.0


def main() -> int:
    days = collected_days()
    print("=" * 88)
    print("INTERIM VRP READOUT (derived ATM IV vs realized) -- DESCRIPTIVE, PRE-GATE")
    print(f"Collected days: {len(days)}  (significance gate = {GATE_DAYS}; "
          f"status: {'READY' if len(days) >= GATE_DAYS else f'ACCUMULATING {len(days)}/{GATE_DAYS}'})")
    print("-" * 88)
    print(f"  {'date':<12}{'DTE':>4}{'ATM IV%':>9}{'realzd%':>9}{'VRP':>8}"
          f"{'VIX':>7}  note")

    rows = []
    for d in days:
        chains = lens.read_day("data", d)
        if not chains:
            continue
        near = min(c.expiry for c in chains)
        nc = sorted((c for c in chains if c.expiry == near), key=lambda c: c.timestamp)
        ivs = [opt.atm_iv(c) for c in nc]
        ivs = [v * 100 for v in ivs if v is not None]
        if not ivs:
            continue
        implied = float(np.mean(ivs))
        realized = realized_vol_annual(nc)
        dte = (near - d).days
        vix = np.nanmean([c.context.get("india_vix", np.nan) for c in nc])
        vrp = implied - realized
        expiry_flag = "expiry day" if near == d else ""
        rows.append({"date": d, "dte": dte, "implied": implied,
                     "realized": realized, "vrp": vrp, "vix": vix})
        print(f"  {d.isoformat():<12}{dte:>4}{implied:>9.1f}{realized:>9.1f}"
              f"{vrp:>+8.1f}{vix:>7.1f}  {expiry_flag}")

    if not rows:
        print("No collected data.")
        return 0

    df = pd.DataFrame(rows)
    vrp = df["vrp"].to_numpy()
    pos = float(np.mean(vrp > 0)) * 100
    print("-" * 88)
    print(f"  mean ATM IV {df['implied'].mean():.1f}%  vs mean realized "
          f"{df['realized'].mean():.1f}%  ->  mean VRP {vrp.mean():+.1f} vol pts")
    print(f"  VRP > 0 on {pos:.0f}% of {len(df)} days  "
          f"(mean India VIX {df['vix'].mean():.1f})")
    print("-" * 88)
    print("READING (honest):")
    if len(days) < GATE_DAYS:
        print(f"  n={len(days)} is BELOW the {GATE_DAYS}-day gate -> DESCRIPTIVE ONLY, no")
        print("  significance/verdict. This only shows whether the macro VRP (Exp 028,")
        print("  GO: implied>realized ~80% over 10y) is directionally visible early.")
    sign = "consistent with" if vrp.mean() > 0 else "NOT yet consistent with"
    print(f"  Early signal is {sign} the macro VRP (positive premium expected).")
    print("  Caveat: implied is ~to-expiry annualised; realized here is intraday-only")
    print("  (no overnight). A true term-matched VRP arrives with Exp 027 at the gate.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
