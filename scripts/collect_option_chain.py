"""Collect Angel option-chain snapshots into the Parquet warehouse.

IMPORTANT: option-chain data (OI / IV / Greeks) CANNOT be backfilled like price
candles -- Angel exposes only the *current* chain. So this script snapshots the
chain forward in time; run it repeatedly during market hours to accumulate the
history that Phase-5 (OI/PCR/gamma/IV) research will need.

    # one snapshot
    python scripts/collect_option_chain.py --underlying NIFTY --expiry 2026-06-26

    # poll every 5 minutes, 60 times (~5 hours of a session)
    python scripts/collect_option_chain.py --underlying NIFTY --expiry 2026-06-26 \
        --count 60 --interval 300

If the expiry isn't found, the script prints the available expiries.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime

from nifty_quant.data.providers.angel_instruments import InstrumentMaster
from nifty_quant.data.providers.angelone import AngelOneProvider
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.dotenv import load_dotenv
from nifty_quant.log import get_logger

_log = get_logger("scripts.collect_option_chain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot Angel option chains.")
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    p.add_argument("--count", type=int, default=1, help="number of snapshots")
    p.add_argument("--interval", type=float, default=300.0,
                   help="seconds between snapshots")
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    expiry = date.fromisoformat(args.expiry)

    master = InstrumentMaster(cache_path=f"{args.data_dir}/angel_scrip_master.json")
    provider = AngelOneProvider.from_env(instrument_master=master)
    storage = ParquetStorage(args.data_dir)

    # Validate the expiry early with a helpful message.
    if not master.option_instruments(args.underlying, expiry):
        avail = master.available_expiries(args.underlying)
        print(f"No {args.underlying} contracts for {args.expiry}.")
        print("Available expiries:", ", ".join(d.isoformat() for d in avail[:20]))
        return

    for i in range(args.count):
        try:
            chain = provider.get_option_chain(args.underlying, expiry)
            rows = storage.write_option_chain(chain)
            print(f"[{datetime.now():%H:%M:%S}] snapshot {i + 1}/{args.count}: "
                  f"{rows} quotes, spot={chain.spot:.2f}, "
                  f"ATM={chain.atm_strike():.0f}")
            _log.event("option_chain_snapshot", underlying=args.underlying,
                       expiry=args.expiry, rows=rows, spot=chain.spot)
        except Exception as exc:  # keep collecting on transient errors
            _log.event("option_chain_snapshot_error", level=40, error=str(exc))
            print(f"snapshot {i + 1} failed: {exc}")

        if i + 1 < args.count:
            time.sleep(args.interval)

    print(f"Done. Snapshots stored under {args.data_dir}/option_chain/")


if __name__ == "__main__":
    main()
