"""Backtest EVERY current strategy over the stored dataset in a single pass.

Builds one :class:`BacktestEngine` per strategy (reference strategies +
benchmarks), subscribes them all to a single :class:`ReplayFeed`, and replays
the warehouse once -- so the data is read a single time and every strategy sees
the identical event stream. Prints a ranked comparison table and saves each
strategy's full report under ``reports/all_strategies_<stamp>/``.

    python scripts/run_all_strategies.py \
        --symbol NIFTY --timeframe 5m \
        --start 2024-06-20 --end 2026-06-19

All bundled strategies are REFERENCE/benchmark implementations used to exercise
the framework, NOT trading recommendations. Index spot is used as a simple proxy
instrument; real option costs/structure are not modelled here.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from datetime import date, datetime

from nifty_quant.backtest.broker import PercentSlippage, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies import benchmarks as bench
from nifty_quant.backtest.strategies.ema_cross import EmaCrossStrategy
from nifty_quant.backtest.strategies.feature_momentum import FeatureMomentumStrategy
from nifty_quant.backtest.strategies.filtered_ema import FilteredEmaStrategy
from nifty_quant.backtest.strategies.mean_reversion import (
    BollingerReversion,
    RsiReversion,
)
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.features.engine import FeatureConfig
from nifty_quant.features.vector import FeatureVector
from nifty_quant.analytics import indicators as ind
import pandas as pd
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.log import configure as configure_logging
from nifty_quant.log import get_logger

_log = get_logger("scripts.run_all_strategies")


class PrecomputedFeatureEngine:
    """Drop-in ``feature_engine`` serving full-series indicators per bar.

    The production :class:`~nifty_quant.features.engine.FeatureEngine` rebuilds
    a DataFrame and recomputes every indicator over a rolling 300-bar buffer on
    *each* candle. That is correct for a live feed (one bar every few minutes)
    but costs ~47 ms/bar -- roughly 29 minutes over 37k historical bars. This
    shim computes the SAME indicators (identical ``ind.*`` functions) ONCE over
    the full close series in vectorised pandas, then serves the row matching
    each candle's timestamp in O(1). EMA/RSI use continuous history rather than
    a truncated window, which is at least as faithful for their consumers.

    It implements the minimal surface ``BacktestEngine`` calls on a feature
    engine: ``on_option_chain(event)`` and ``on_candle(event) -> FeatureVector``.
    """

    version = "precomputed-v1"

    def __init__(self, series, config: FeatureConfig | None = None) -> None:
        cfg = config or FeatureConfig()
        self._symbol = series.symbol
        candles = series.candles
        df = pd.DataFrame(
            {
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
            },
            index=pd.DatetimeIndex([c.timestamp for c in candles]),
        )
        close = df["close"]
        feat = pd.DataFrame(index=df.index)
        feat["close"] = close
        for p in cfg.ema_periods:
            feat[f"ema_{p}"] = ind.ema(close, p)
        feat[f"rsi_{cfg.rsi_period}"] = ind.rsi(close, cfg.rsi_period)
        macd = ind.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        feat["macd_line"] = macd["macd"]
        feat["macd_signal"] = macd["signal"]
        feat["macd_hist"] = macd["hist"]
        feat[f"atr_{cfg.atr_period}"] = ind.atr(df, cfg.atr_period)
        adx = ind.adx(df, cfg.adx_period)
        feat[f"adx_{cfg.adx_period}"] = adx["adx"]
        feat["plus_di"] = adx["plus_di"]
        feat["minus_di"] = adx["minus_di"]
        bb = ind.bollinger_bands(close, cfg.bb_period)
        feat["bb_bandwidth"] = bb["bandwidth"]

        # Map each bar timestamp -> {feature: value}. NaN warm-up values are
        # preserved so FeatureVector.is_ready() gates correctly.
        self._rows: dict = {}
        cols = list(feat.columns)
        for ts, *vals in feat.itertuples(name=None):
            self._rows[ts.to_pydatetime()] = {
                k: float(v) for k, v in zip(cols, vals)
            }

    def on_option_chain(self, event) -> None:  # no option features here
        return None

    def on_candle(self, event) -> FeatureVector:
        values = self._rows.get(event.timestamp, {})
        return FeatureVector(
            timestamp=event.timestamp,
            symbol=event.symbol,
            version=self.version,
            values=values,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest every current strategy over the stored dataset."
    )
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", default="2024-06-20", help="YYYY-MM-DD")
    p.add_argument("--end", default="2026-06-19", help="YYYY-MM-DD")
    p.add_argument("--quantity", type=int, default=10)
    p.add_argument("--cash", type=float, default=1_000_000.0)
    p.add_argument("--fee", type=float, default=20.0, help="Per-order fee")
    p.add_argument("--slippage-bps", type=float, default=3.0,
                   help="Adverse slippage in basis points")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing per-strategy reports to disk.")
    return p.parse_args()


def build_strategies(instrument: Instrument, quantity: int):
    """Instantiate every current strategy with its default parameters.

    Returns a list of ``(strategy, needs_features)`` pairs. ``needs_features``
    is True for strategies that consume FeatureVectors (their engine is wired
    with a FeatureEngine).
    """
    pairs: list[tuple[object, bool]] = [
        # --- Reference strategies ---------------------------------------
        (EmaCrossStrategy(instrument, quantity=quantity), False),
        (FilteredEmaStrategy(instrument, quantity=quantity), False),
        (RsiReversion(instrument, quantity=quantity), False),
        (BollingerReversion(instrument, quantity=quantity), False),
        (FeatureMomentumStrategy(instrument, quantity=quantity), True),
    ]
    # --- Benchmarks (baselines any real strategy must beat) -------------
    for factory in bench.BENCHMARKS.values():
        pairs.append((factory(instrument, quantity=quantity), False))
    return pairs


def make_engine(strategy, needs_features: bool, args, feature_engine=None) -> BacktestEngine:
    return BacktestEngine(
        strategy,
        portfolio=Portfolio(starting_cash=args.cash),
        risk_engine=BasicRiskEngine(default_quantity=args.quantity),
        broker=SimulatedBroker(
            fill_model=PercentSlippage(args.slippage_bps / 10_000.0),
            fee_per_order=args.fee,
        ),
        feature_engine=feature_engine if needs_features else None,
    )


def _fmt(value: float, *, pct: bool = False, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value))):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if pct:
        return f"{value * 100:+.{digits}f}%"
    return f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    # Quiet the per-event INFO journal stream: each engine journals every
    # signal/order/fill, which at thousands of trades x many engines turns
    # console I/O into the dominant cost. WARNING keeps real problems visible.
    configure_logging(level=logging.WARNING)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    storage = ParquetStorage(args.data_dir)
    instrument = Instrument(args.symbol, InstrumentType.INDEX)

    # Precompute features once for the feature-consuming strategies (see
    # PrecomputedFeatureEngine for why the live engine is impractical in bulk).
    series = storage.read_candles(args.symbol, args.timeframe, start_dt, end_dt)
    feature_engine = PrecomputedFeatureEngine(series)

    pairs = build_strategies(instrument, args.quantity)
    engines = [make_engine(s, nf, args, feature_engine) for s, nf in pairs]

    # One feed, one pass: every engine sees the identical event stream.
    feed = ReplayFeed(
        storage, start_dt, end_dt, candle_specs=[(args.symbol, args.timeframe)]
    )
    for engine in engines:
        feed.subscribe(engine)

    print("=" * 100)
    print(f"RUN ALL STRATEGIES  {args.symbol} {args.timeframe}  "
          f"{args.start} -> {args.end}")
    print(f"Strategies: {len(engines)}  |  cash={args.cash:,.0f}  "
          f"fee={args.fee}  slippage={args.slippage_bps}bps  qty={args.quantity}")
    print("=" * 100)

    n_events = feed.run()
    print(f"Replayed {n_events} events.\n")

    results = [engine.build_result() for engine in engines]
    # Rank by total return (desc); NaN sorts last.
    results.sort(
        key=lambda r: (r.metrics.total_return
                       if not math.isnan(r.metrics.total_return) else -math.inf),
        reverse=True,
    )

    header = (f"{'Strategy':<28} {'Trades':>7} {'TotRet':>9} {'CAGR':>9} "
              f"{'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'PF':>6} {'Win':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.metrics
        print(
            f"{r.strategy_name:<28} {m.num_trades:>7d} "
            f"{_fmt(m.total_return, pct=True):>9} {_fmt(m.cagr, pct=True):>9} "
            f"{_fmt(m.sharpe):>7} {_fmt(m.sortino):>8} "
            f"{_fmt(m.max_drawdown, pct=True):>8} {_fmt(m.profit_factor):>6} "
            f"{_fmt(m.win_rate, pct=True, digits=1):>7}"
        )
    print("=" * 100)

    if not args.no_save:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_root = f"{args.report_dir}/all_strategies_{stamp}"
        summary_rows = []
        for r in results:
            m = r.metrics
            safe = r.strategy_name.replace("/", "_").replace(" ", "_")
            r.save(f"{out_root}/{safe}", plots=False)
            summary_rows.append({
                "strategy": r.strategy_name,
                "num_trades": m.num_trades,
                "total_return": m.total_return,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "calmar": m.calmar,
                "max_drawdown": m.max_drawdown,
                "profit_factor": m.profit_factor,
                "expectancy": m.expectancy,
                "win_rate": m.win_rate,
            })
        summary_path = f"{out_root}/summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Per-strategy reports + summary.csv saved under {out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
