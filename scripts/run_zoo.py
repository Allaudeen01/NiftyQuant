"""Disciplined strategy comparison with year-by-year cross-validation.

Runs a small, curated set of candidate strategies (NOT 180) over the full
window and each calendar year, so any apparent edge must hold across periods.
Treat the output as hypothesis-generating, not confirmation: with a flat
2-year dataset, full-window winners that don't persist year-over-year are
almost certainly noise.

    python scripts/run_zoo.py --symbol NIFTY --timeframe 5m \
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
from nifty_quant.backtest.filters import TimeWindowFilter
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.benchmarks import BuyAndHold
from nifty_quant.backtest.strategies.filtered_ema import FilteredEmaStrategy
from nifty_quant.backtest.strategies.mean_reversion import (
    BollingerReversion,
    RsiReversion,
)
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.log import get_logger

_log = get_logger("scripts.run_zoo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disciplined strategy zoo with year CV.")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--quantity", type=int, default=10)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--fee", type=float, default=20.0)
    p.add_argument("--slippage-bps", type=float, default=3.0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args()


def _candidates(inst, qty):
    return {
        "BuyAndHold": lambda: BuyAndHold(inst, quantity=qty),
        "EMA(baseline)": lambda: FilteredEmaStrategy(inst, fast=20, slow=50, quantity=qty),
        "EMA+Time[10:00-14:30]": lambda: FilteredEmaStrategy(
            inst, fast=20, slow=50, quantity=qty,
            filters=[TimeWindowFilter("10:00", "14:30")]),
        "RSI(14,30/50) MeanRev": lambda: RsiReversion(inst, quantity=qty),
        "RSI(2,10/50) MeanRev": lambda: RsiReversion(
            inst, period=2, oversold=10, exit_level=50, quantity=qty),
        "Bollinger(20,2) Rev": lambda: BollingerReversion(inst, quantity=qty),
    }


def _year_segments(start: date, end: date):
    segs = []
    for year in range(start.year, end.year + 1):
        s = max(start, date(year, 1, 1))
        e = min(end, date(year, 12, 31))
        if s <= e:
            segs.append((str(year), s, e))
    return segs


def _run(factory, storage, symbol, timeframe, s, e, args):
    engine = BacktestEngine(
        factory(),
        portfolio=Portfolio(starting_cash=args.cash),
        risk_engine=BasicRiskEngine(default_quantity=args.quantity),
        broker=SimulatedBroker(
            fill_model=PercentSlippage(args.slippage_bps / 10_000.0),
            fee_per_order=args.fee),
    )
    feed = ReplayFeed(
        storage,
        datetime.combine(s, datetime.min.time()),
        datetime.combine(e, datetime.max.time()),
        candle_specs=[(symbol, timeframe)],
    )
    return engine.run(feed).metrics


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    storage = ParquetStorage(args.data_dir)
    inst = Instrument(args.symbol, InstrumentType.INDEX)
    candidates = _candidates(inst, args.quantity)
    segments = [("FULL", start, end)] + _year_segments(start, end)

    full_rows, matrix = [], {}
    for label, factory in candidates.items():
        matrix[label] = {}
        for seg_label, s, e in segments:
            m = _run(factory, storage, args.symbol, args.timeframe, s, e, args)
            matrix[label][seg_label] = m.total_return
            if seg_label == "FULL":
                full_rows.append({
                    "strategy": label, "total_return": m.total_return,
                    "sharpe": m.sharpe, "max_drawdown": m.max_drawdown,
                    "profit_factor": m.profit_factor, "win_rate": m.win_rate,
                    "trades": m.num_trades})
            _log.event("zoo_run", strategy=label, segment=seg_label,
                       total_return=m.total_return, trades=m.num_trades)

    full_df = pd.DataFrame(full_rows).sort_values(
        "sharpe", ascending=False, na_position="last").reset_index(drop=True)

    os.makedirs(args.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = f"{args.report_dir}/zoo_{args.symbol}_{args.timeframe}_{stamp}"
    full_df.to_csv(f"{base}.csv", index=False)
    with open(f"{base}.json", "w", encoding="utf-8") as fh:
        json.dump({"full": full_rows, "by_segment": matrix}, fh, indent=2, default=str)

    pd.set_option("display.width", 200)
    print("=" * 84)
    print(f"STRATEGY ZOO - {args.symbol} {args.timeframe} ({args.start}->{args.end}) "
          f"| {args.slippage_bps}bps + {args.fee}/order")
    print("=" * 84)
    show = full_df.copy()
    for c in ("total_return", "max_drawdown", "win_rate"):
        show[c] = (show[c] * 100).round(2).astype(str) + "%"
    for c in ("sharpe", "profit_factor"):
        show[c] = show[c].round(2)
    print(show.to_string(index=False))
    print("\nPer-year total return (consistency check):")
    seg_df = (pd.DataFrame(matrix).T * 100).round(2).astype(str) + "%"
    print(seg_df.to_string())
    print(f"\nSaved: {base}.csv / .json")


if __name__ == "__main__":
    main()
