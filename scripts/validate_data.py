"""Validate stored OHLCV history and write a data-quality report.

Reads candles from the Parquet warehouse and runs the quality battery. Writes
a machine-readable JSON report and a human-readable text report under
reports/, and exits non-zero if any CRITICAL check fails -- so it can gate
downstream benchmarking/backtesting.

    python scripts/validate_data.py --symbol NIFTY --timeframe 5m \
        --start 2024-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from nifty_quant.data.quality import validate_ohlcv
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.log import get_logger

_log = get_logger("scripts.validate_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate stored OHLCV data quality.")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    storage = ParquetStorage(args.data_dir)
    series = storage.read_candles(
        args.symbol,
        args.timeframe,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.max.time()),
    )

    report = validate_ohlcv(
        series, expected_start=start, expected_end=end
    )

    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = out / f"quality_{args.symbol}_{args.timeframe}_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    base.with_suffix(".txt").write_text(report.to_text(), encoding="utf-8")

    print(report.to_text())
    print(f"\nReports written to {base}.json / .txt")
    _log.event(
        "data_quality_report",
        symbol=args.symbol, timeframe=args.timeframe,
        passed=report.passed, num_bars=report.num_bars,
        failed=report.summary.get("failed_checks", []),
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
