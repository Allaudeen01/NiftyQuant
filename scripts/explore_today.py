"""One-day EXPLORATORY analysis of collected option-chain data (read-only).

NOT a strategy. With a single session you cannot validate anything -- this only
describes today's structure (ATM straddle decay, PCR, realized vs implied move)
and frames observations as hypotheses to test once history accumulates.

    python scripts/explore_today.py --date 2026-06-22
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import date

import numpy as np

# load the collector module to reuse read_snapshots_for_day
spec = importlib.util.spec_from_file_location("cmd", "scripts/collect_market_data.py")
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)


def mid(q):
    if q.bid > 0 and q.ask > 0:
        return (q.bid + q.ask) / 2.0
    return q.last_price


def atm_straddle(chain):
    """ATM strike + call/put mid + straddle premium for one chain."""
    strikes = chain.strikes()
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - chain.spot))
    call = next((q for q in chain.calls() if q.strike == atm), None)
    put = next((q for q in chain.puts() if q.strike == atm), None)
    if not call or not put:
        return None
    return atm, mid(call), mid(put), mid(call) + mid(put)


def pcr_oi(chain):
    coi = sum(q.open_interest for q in chain.calls())
    poi = sum(q.open_interest for q in chain.puts())
    return (poi / coi) if coi > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-22")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    d = date.fromisoformat(args.date)

    chains = cmd.read_snapshots_for_day(args.data_dir, d)
    if not chains:
        print("No snapshots for", d)
        return 0

    # nearest expiry only
    expiries = sorted({c.expiry for c in chains})
    near = expiries[0]
    near_chains = sorted((c for c in chains if c.expiry == near),
                         key=lambda c: c.timestamp)
    print("=" * 78)
    print(f"ONE-DAY EXPLORATION  {d}  (nearest expiry {near}, {len(near_chains)} snapshots)")
    print("NOTE: descriptive only; n=1 day. NOT a validated strategy.")

    # spot path
    spots = [c.spot for c in near_chains]
    print(f"\nSpot: open {spots[0]:.1f}  close {spots[-1]:.1f}  "
          f"range {min(spots):.1f}-{max(spots):.1f}  "
          f"net {(spots[-1]/spots[0]-1)*100:+.2f}%  "
          f"intraday range {(max(spots)-min(spots))/spots[0]*100:.2f}%")

    # ATM straddle path
    first = atm_straddle(near_chains[0])
    last = atm_straddle(near_chains[-1])
    if first and last:
        print(f"\nATM straddle (expected move to {near}):")
        print(f"  open  : strike {first[0]:.0f}  call {first[1]:.1f} + put {first[2]:.1f} "
              f"= {first[3]:.1f}")
        print(f"  close : strike {last[0]:.0f}  call {last[1]:.1f} + put {last[2]:.1f} "
              f"= {last[3]:.1f}")
        decay = last[3] - first[3]
        print(f"  straddle change over session: {decay:+.1f} pts "
              f"({decay/first[3]*100:+.1f}%)")
        realized = abs(spots[-1] - spots[0])
        print(f"  realized spot move today: {realized:.1f} pts  vs  open straddle "
              f"{first[3]:.1f} pts priced")

    # PCR path
    pcrs = [pcr_oi(c) for c in near_chains]
    pcrs = [p for p in pcrs if not np.isnan(p)]
    if pcrs:
        print(f"\nPCR(OI): open {pcrs[0]:.2f}  close {pcrs[-1]:.2f}  "
              f"range {min(pcrs):.2f}-{max(pcrs):.2f}")
        # correlation of PCR change vs spot change (contemporaneous, illustrative)
        if len(pcrs) == len(spots) and len(pcrs) > 5:
            dp = np.diff(pcrs); ds = np.diff(spots[:len(pcrs)])
            if dp.std() > 0 and ds.std() > 0:
                r = np.corrcoef(dp, ds)[0, 1]
                print(f"  contemporaneous corr(ΔPCR, Δspot) = {r:+.2f}  (illustrative, n=1 day)")

    # VIX
    import glob, pandas as pd
    vf = glob.glob(f"{args.data_dir}/vix/{d.year}/INDIAVIX_{d.isoformat()}.parquet")
    if vf:
        v = pd.read_parquet(vf[0])
        print(f"\nIndia VIX: {v['india_vix'].iloc[0]:.2f} -> {v['india_vix'].iloc[-1]:.2f} "
              f"(range {v['india_vix'].min():.2f}-{v['india_vix'].max():.2f})")

    print("\n" + "-" * 78)
    print("These are OBSERVATIONS, not signals. Each must be tested across many")
    print("expiry cycles with walk-forward + OOS before it could become a strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
