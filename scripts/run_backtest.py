"""Backtest a strategy over stored historical candles.

Replays the Parquet warehouse through the same engine used everywhere else,
with realistic costs (slippage + per-order fee), and writes a full report
(equity curve, drawdown, trades, metrics, journal) under reports/.

    python scripts/run_backtest.py --symbol NIFTY --timeframe 5m \
        --start 2024-06-21 --end 2026-06-19 --strategy ema --fast 20 --slow 50

NOTE: the bundled strategies are REFERENCE implementations to exercise the
framework, not trading recommendations. Index spot is used as a simple proxy
instrument; real option costs/structure are not modelled here.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from nifty_quant.backtest.broker import PercentSlippage, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.ema_cross import EmaCrossStrategy
from nifty_quant.backtest.strategies.feature_momentum import FeatureMomentumStrategy
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.features.engine import FeatureConfig, FeatureEngine
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.log import get_logger

_log = get_logger("scripts.run_backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest a strategy over stored data.")
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--strategy", choices=["ema", "feature"], default="ema")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--quantity", type=int, default=10)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--fee", type=float, default=20.0, help="Per-order fee")
    p.add_argument("--slippage-bps", type=float, default=3.0,
                   help="Adverse slippage in basis points")
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

    feature_engine = None
    if args.strategy == "ema":
        strategy = EmaCrossStrategy(
            instrument, fast=args.fast, slow=args.slow, quantity=args.quantity
        )
    else:
        strategy = FeatureMomentumStrategy(
            instrument, fast=args.fast, slow=args.slow, quantity=args.quantity
        )
        feature_engine = FeatureEngine(
            FeatureConfig(ema_periods=(args.fast, args.slow))
        )

    engine = BacktestEngine(
        strategy,
        portfolio=Portfolio(starting_cash=args.cash),
        risk_engine=BasicRiskEngine(default_quantity=args.quantity),
        broker=SimulatedBroker(
            fill_model=PercentSlippage(args.slippage_bps / 10_000.0),
            fee_per_order=args.fee,
        ),
        feature_engine=feature_engine,
    )

    feed = ReplayFeed(
        storage, start_dt, end_dt, candle_specs=[(args.symbol, args.timeframe)]
    )
    result = engine.run(feed)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = f"{args.report_dir}/backtest_{strategy.name}_{stamp}"
    result.save(out)

    m = result.metrics
    print("=" * 56)
    print(f"BACKTEST  {strategy.name}  {args.symbol} {args.timeframe}")
    print(f"Window: {args.start} -> {args.end}")
    print("-" * 56)
    print(f"Total return    : {m.total_return:+.2%}")
    print(f"CAGR            : {m.cagr:+.2%}")
    print(f"Sharpe          : {m.sharpe:.2f}")
    print(f"Sortino         : {m.sortino:.2f}")
    print(f"Calmar          : {m.calmar:.2f}")
    print(f"Max drawdown    : {m.max_drawdown:.2%}")
    print(f"Profit factor   : {m.profit_factor:.2f}")
    print(f"Expectancy/trade: {m.expectancy:.2f}")
    print(f"Win rate        : {m.win_rate:.1%}" if m.num_trades else "Win rate        : n/a")
    print(f"Trades          : {m.num_trades}")
    print("=" * 56)
    print(f"Full report saved to {out}")


if __name__ == "__main__":
    main()
