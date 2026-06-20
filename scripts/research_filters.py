"""Research Project 001+: EMA entry with layered filters, year-by-year CV.

Keeps the EMA-cross entry fixed and tests whether entry FILTERS (ADX, time,
volatility) improve risk-adjusted return after costs versus the unfiltered
baseline and Buy & Hold. Reports the full window plus a per-year breakdown so
an improvement has to hold across periods, not just fit one.

    python scripts/research_filters.py --symbol NIFTY --timeframe 5m \
        --start 2024-06-21 --end 2026-06-19
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime

import pandas as pd

from nifty_quant.backtest.broker import PercentSlippage, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.filters import (
    AdxFilter,
    AtrPercentileFilter,
    TimeWindowFilter,
)
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.benchmarks import BuyAndHold
from nifty_quant.backtest.strategies.filtered_ema import FilteredEmaStrategy
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.log import get_logger

_log = get_logger("scripts.research_filters")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMA filter research with yearly CV.")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--quantity", type=int, default=10)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--fee", type=float, default=20.0)
    p.add_argument("--slippage-bps", type=float, default=3.0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args()


def _variants(instrument, fast, slow, quantity):
    """label -> factory() returning a fresh strategy."""
    def ema(filters=None):
        return lambda: FilteredEmaStrategy(
            instrument, fast=fast, slow=slow, quantity=quantity, filters=filters or []
        )
    return {
        "BuyAndHold": lambda: BuyAndHold(instrument, quantity=quantity),
        "EMA(baseline)": ema(),
        "EMA+ADX>=20": ema([AdxFilter(20)]),
        "EMA+ADX>=25": ema([AdxFilter(25)]),
        "EMA+ADX>=30": ema([AdxFilter(30)]),
        "EMA+Time[10:00-14:30]": ema([TimeWindowFilter("10:00", "14:30")]),
        "EMA+ATRpct>=0.5": ema([AtrPercentileFilter(0.5)]),
        "EMA+ADX>=25+Time": ema([AdxFilter(25), TimeWindowFilter("10:00", "14:30")]),
    }


def _year_segments(start: date, end: date):
    """[(label, seg_start, seg_end)] split at calendar-year boundaries."""
    segs = []
    for year in range(start.year, end.year + 1):
        s = max(start, date(year, 1, 1))
        e = min(end, date(year, 12, 31))
        if s <= e:
            segs.append((str(year), s, e))
    return segs


def _run(factory, storage, symbol, timeframe, s: date, e: date, args):
    start_dt = datetime.combine(s, datetime.min.time())
    end_dt = datetime.combine(e, datetime.max.time())
    engine = BacktestEngine(
        factory(),
        portfolio=Portfolio(starting_cash=args.cash),
        risk_engine=BasicRiskEngine(default_quantity=args.quantity),
        broker=SimulatedBroker(
            fill_model=PercentSlippage(args.slippage_bps / 10_000.0),
            fee_per_order=args.fee,
        ),
    )
    feed = ReplayFeed(storage, start_dt, end_dt, candle_specs=[(symbol, timeframe)])
    return engine.run(feed).metrics


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    storage = ParquetStorage(args.data_dir)
    instrument = Instrument(args.symbol, InstrumentType.INDEX)
    variants = _variants(instrument, args.fast, args.slow, args.quantity)
    segments = [("FULL", start, end)] + _year_segments(start, end)

    full_rows = []
    matrix = {}  # variant -> {segment_label: total_return}
    for label, factory in variants.items():
        matrix[label] = {}
        for seg_label, s, e in segments:
            m = _run(factory, storage, args.symbol, args.timeframe, s, e, args)
            matrix[label][seg_label] = m.total_return
            if seg_label == "FULL":
                full_rows.append({
                    "variant": label,
                    "total_return": m.total_return,
                    "sharpe": m.sharpe,
                    "max_drawdown": m.max_drawdown,
                    "profit_factor": m.profit_factor,
                    "win_rate": m.win_rate,
                    "trades": m.num_trades,
                })
            _log.event("filter_research_run", variant=label, segment=seg_label,
                       total_return=m.total_return, sharpe=m.sharpe,
                       trades=m.num_trades)

    full_df = pd.DataFrame(full_rows).sort_values(
        "sharpe", ascending=False, na_position="last").reset_index(drop=True)

    os.makedirs(args.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = f"{args.report_dir}/filters_{args.symbol}_{args.timeframe}_{stamp}"
    full_df.to_csv(f"{base}.csv", index=False)
    with open(f"{base}.json", "w", encoding="utf-8") as fh:
        json.dump({"full": full_rows, "by_segment": matrix}, fh, indent=2, default=str)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=" * 88)
    print(f"FILTER RESEARCH - EMA {args.fast}/{args.slow}  {args.symbol} "
          f"{args.timeframe}  ({args.start} -> {args.end})")
    print(f"costs: {args.slippage_bps}bps + {args.fee}/order")
    print("=" * 88)
    show = full_df.copy()
    for col in ("total_return", "max_drawdown", "win_rate"):
        show[col] = (show[col] * 100).round(2).astype(str) + "%"
    for col in ("sharpe", "profit_factor"):
        show[col] = show[col].round(2)
    print(show.to_string(index=False))

    print("\nPer-year total return (consistency check):")
    seg_df = pd.DataFrame(matrix).T  # variants x segments
    seg_df = (seg_df * 100).round(2).astype(str) + "%"
    print(seg_df.to_string())
    print(f"\nSaved: {base}.csv / .json")


if __name__ == "__main__":
    main()
