"""Manually-launched paper-trading runner.

NOT started by the agent tooling -- run it yourself in a terminal:

    python scripts/run_paper.py --underlying NIFTY --expiry 2026-01-27 \
        --candle-symbol NIFTY --timeframe 5m --poll 60

It wires the live Groww provider to a SimulatedBroker via the same engine used
for backtests. NO LIVE ORDERS are placed -- paper trading fills against polled
quotes, and the real broker's order gate stays shut.

Auth comes from environment variables (never hard-code keys):
    GROWW_API_KEY + GROWW_TOTP_SECRET   (preferred)
    or GROWW_API_KEY + GROWW_API_SECRET

Press Ctrl+C to stop gracefully; a report is written under reports/.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from nifty_quant.backtest.broker import BidAskFill, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies.feature_momentum import FeatureMomentumStrategy
from nifty_quant.data.providers.groww import GrowwProvider
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.features.engine import FeatureConfig, FeatureEngine
from nifty_quant.features.store import ParquetFeatureStore
from nifty_quant.feed.clock import RealClock
from nifty_quant.feed.paper import PaperFeed
from nifty_quant.log import get_logger

_log = get_logger("scripts.run_paper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-trade with the live Groww feed.")
    p.add_argument("--candle-symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--expiry", required=True, help="Option expiry YYYY-MM-DD")
    p.add_argument("--poll", type=float, default=60.0, help="Poll interval (s)")
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--max-polls", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    expiry = date.fromisoformat(args.expiry)

    provider = GrowwProvider.from_env()  # live_trading_enabled stays False
    storage = ParquetStorage(args.data_dir)

    instrument = Instrument(args.candle_symbol, InstrumentType.INDEX)
    strategy = FeatureMomentumStrategy(instrument, fast=20, slow=50, quantity=75)

    engine = BacktestEngine(
        strategy,
        portfolio=Portfolio(starting_cash=args.cash),
        risk_engine=BasicRiskEngine(default_quantity=75),
        broker=SimulatedBroker(fill_model=BidAskFill()),
        feature_engine=FeatureEngine(FeatureConfig()),
        feature_store=ParquetFeatureStore(args.data_dir),
    )

    feed = PaperFeed(
        provider,
        candle_specs=[(args.candle_symbol, args.timeframe)],
        chain_specs=[(args.underlying, expiry)],
        poll_interval_seconds=args.poll,
        clock=RealClock(),
        storage=storage,            # persistence ON by default
    )
    feed.subscribe(engine)

    _log.event("paper_session_starting", expiry=args.expiry, poll=args.poll)
    try:
        feed.run(max_polls=args.max_polls)
    except KeyboardInterrupt:
        feed.stop()
        _log.event("paper_session_interrupted")

    result = engine.build_result()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = f"{args.report_dir}/paper_{stamp}"
    result.save(out)
    _log.event(
        "paper_session_report_saved",
        path=out,
        trades=result.metrics.num_trades,
        total_return=result.metrics.total_return,
    )
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
