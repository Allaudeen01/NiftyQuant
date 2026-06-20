"""Run the full benchmark suite over stored data and rank the results.

Backtests all 10 baseline strategies with identical costs, prints a ranked
comparison table, and saves it (CSV + JSON) under reports/. Any candidate
strategy you build later must beat these after costs to be worth pursuing.

    python scripts/run_benchmarks.py --symbol NIFTY --timeframe 5m \
        --start 2024-06-21 --end 2026-06-19
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime

import pandas as pd

from nifty_quant.backtest.broker import PercentSlippage, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.benchmarks import BENCHMARKS
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.log import get_logger

_log = get_logger("scripts.run_benchmarks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the benchmark strategy suite.")
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


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    storage = ParquetStorage(args.data_dir)
    instrument = Instrument(args.symbol, InstrumentType.INDEX)
    slippage = PercentSlippage(args.slippage_bps / 10_000.0)

    rows = []
    for name, factory in BENCHMARKS.items():
        strategy = factory(instrument, quantity=args.quantity)
        engine = BacktestEngine(
            strategy,
            portfolio=Portfolio(starting_cash=args.cash),
            risk_engine=BasicRiskEngine(default_quantity=args.quantity,
                                        allow_short=True),
            broker=SimulatedBroker(fill_model=slippage, fee_per_order=args.fee),
        )
        feed = ReplayFeed(storage, start_dt, end_dt,
                          candle_specs=[(args.symbol, args.timeframe)])
        result = engine.run(feed)
        m = result.metrics
        rows.append({
            "strategy": strategy.name,
            "total_return": m.total_return,
            "cagr": m.cagr,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "profit_factor": m.profit_factor,
            "expectancy": m.expectancy,
            "win_rate": m.win_rate,
            "trades": m.num_trades,
        })
        _log.event("benchmark_done", strategy=strategy.name,
                   total_return=m.total_return, sharpe=m.sharpe,
                   trades=m.num_trades)

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False,
                                        na_position="last").reset_index(drop=True)

    out_dir = args.report_dir
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = f"{out_dir}/benchmarks_{args.symbol}_{args.timeframe}_{stamp}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{base}.csv", index=False)
    with open(f"{base}.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=" * 90)
    print(f"BENCHMARK SUITE - {args.symbol} {args.timeframe}  "
          f"({args.start} -> {args.end}), {args.slippage_bps}bps + {args.fee}/order")
    print("=" * 90)
    show = df.copy()
    for col in ("total_return", "cagr", "max_drawdown", "win_rate"):
        show[col] = (show[col] * 100).round(2).astype(str) + "%"
    for col in ("sharpe", "sortino", "profit_factor", "expectancy"):
        show[col] = show[col].round(2)
    print(show.to_string(index=False))
    print(f"\nSaved: {base}.csv / .json")


if __name__ == "__main__":
    main()
