"""Manually-launched historical data collector for Angel One SmartAPI.

Run it yourself in a terminal (NOT started by the agent tooling):

    python scripts/collect_angel_history.py --symbol NIFTY --timeframe 5m \
        --start 2024-01-01 --end 2024-12-31

Pulls historical candles via the Angel provider and writes them to the Parquet
store under data/, ready for replay/backtesting. Long ranges are auto-chunked
to respect SmartAPI's per-request limits.

Auth comes from environment variables (never hard-code keys):
    ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_MPIN (or ANGEL_PASSWORD),
    ANGEL_TOTP_SECRET
"""

from __future__ import annotations

import argparse
from datetime import date

from nifty_quant.data.collectors.historical import HistoricalCollector
from nifty_quant.data.providers.angelone import AngelOneProvider
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.dotenv import load_dotenv
from nifty_quant.log import get_logger

_log = get_logger("scripts.collect_angel_history")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Angel One historical candles.")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m",
                   help="1m, 3m, 5m, 10m, 15m, 30m, 1h, 1d")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--token", default=None,
                   help="Explicit symbol token (overrides built-in lookup)")
    p.add_argument("--exchange", default="NSE")
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()  # load .env if present (inline comments / quotes handled)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    symbol_tokens = {args.symbol.upper(): args.token} if args.token else None
    provider = AngelOneProvider.from_env(
        exchange=args.exchange, symbol_tokens=symbol_tokens
    )
    storage = ParquetStorage(args.data_dir)
    collector = HistoricalCollector(provider, storage)

    count = collector.collect_candles(args.symbol, args.timeframe, start, end)
    _log.event(
        "angel_history_collected",
        symbol=args.symbol, timeframe=args.timeframe,
        start=args.start, end=args.end, candles=count,
    )
    print(
        f"Collected {count} {args.timeframe} candles for {args.symbol} "
        f"({args.start} -> {args.end}) into {args.data_dir}/"
    )


if __name__ == "__main__":
    main()
