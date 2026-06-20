"""Parquet storage backend.

Layout (day-partitioned, matching the agreed research-friendly structure)::

    <root>/candles/<TIMEFRAME>/<YEAR>/<SYMBOL>_<YYYY-MM-DD>.parquet
    <root>/option_chain/<YEAR>/<UNDERLYING>_<YYYY-MM-DD>.parquet

One file per (symbol, timeframe, day) for candles, and one file per
(underlying, day) for option-chain snapshots. Day partitioning keeps individual
files small, makes range reads cheap (only touch relevant days), and converts
cleanly to TimescaleDB hypertables later.

All writes are idempotent: existing rows for the same keys are merged and
de-duplicated rather than appended blindly.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from nifty_quant.data.models import (
    Candle,
    OHLCVSeries,
    OptionChain,
    OptionQuote,
    OptionType,
)
from nifty_quant.data.storage.base import Storage
from nifty_quant.log import get_logger

_log = get_logger("storage.parquet")

_CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_CHAIN_COLUMNS = [
    "snapshot_ts",
    "underlying",
    "spot",
    "expiry",
    "strike",
    "option_type",
    "last_price",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "oi_change",
    "implied_volatility",
    "context",
]


class ParquetStorage(Storage):
    """File-based Parquet implementation of :class:`Storage`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- candles ------------------------------------------------------------

    def write_candles(self, series: OHLCVSeries) -> int:
        if len(series) == 0:
            return 0
        df = series.to_frame().reset_index()  # timestamp column + ohlcv
        written = 0
        for day, day_df in df.groupby(df["timestamp"].dt.date):
            path = self._candle_path(series.symbol, series.timeframe, day)
            merged = self._merge_parquet(
                path, day_df, key_cols=["timestamp"]
            )
            self._atomic_write(path, merged)
            written += len(day_df)
        _log.event(
            "candles_written",
            symbol=series.symbol,
            timeframe=series.timeframe,
            count=written,
        )
        return written

    def read_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> OHLCVSeries:
        frames: list[pd.DataFrame] = []
        for day in _days_between(start.date(), end.date()):
            path = self._candle_path(symbol, timeframe, day)
            if path.exists():
                frames.append(pd.read_parquet(path))
        candles: list[Candle] = []
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
            df = df.sort_values("timestamp").drop_duplicates("timestamp")
            for row in df.itertuples(index=False):
                candles.append(
                    Candle(
                        timestamp=row.timestamp.to_pydatetime(),
                        open=float(row.open),
                        high=float(row.high),
                        low=float(row.low),
                        close=float(row.close),
                        volume=float(row.volume),
                    )
                )
        return OHLCVSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    # --- option chains ------------------------------------------------------

    def write_option_chain(self, chain: OptionChain) -> int:
        if not chain.quotes:
            return 0
        context_json = json.dumps(chain.context, default=str)
        rows = [
            {
                "snapshot_ts": chain.timestamp,
                "underlying": chain.underlying,
                "spot": chain.spot,
                "expiry": pd.Timestamp(chain.expiry),
                "strike": q.strike,
                "option_type": q.option_type.value,
                "last_price": q.last_price,
                "bid": q.bid,
                "ask": q.ask,
                "volume": q.volume,
                "open_interest": q.open_interest,
                "oi_change": q.oi_change,
                "implied_volatility": q.implied_volatility,
                "context": context_json,
            }
            for q in chain.quotes
        ]
        df = pd.DataFrame(rows, columns=_CHAIN_COLUMNS)
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
        path = self._chain_path(chain.underlying, chain.timestamp.date())
        merged = self._merge_parquet(
            path, df, key_cols=["snapshot_ts", "strike", "option_type"]
        )
        self._atomic_write(path, merged)
        _log.event(
            "option_chain_written",
            underlying=chain.underlying,
            expiry=str(chain.expiry),
            rows=len(df),
        )
        return len(df)

    def read_option_chains(
        self,
        underlying: str,
        start: datetime,
        end: datetime,
    ) -> list[OptionChain]:
        frames: list[pd.DataFrame] = []
        for day in _days_between(start.date(), end.date()):
            path = self._chain_path(underlying, day)
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True)
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
        df = df[(df["snapshot_ts"] >= start) & (df["snapshot_ts"] <= end)]
        df = df.sort_values("snapshot_ts")

        chains: list[OptionChain] = []
        for snap_ts, snap in df.groupby("snapshot_ts", sort=True):
            quotes = [
                OptionQuote(
                    strike=float(r.strike),
                    option_type=OptionType(r.option_type),
                    expiry=pd.Timestamp(r.expiry).date(),
                    last_price=float(r.last_price),
                    bid=float(r.bid),
                    ask=float(r.ask),
                    volume=float(r.volume),
                    open_interest=float(r.open_interest),
                    oi_change=float(r.oi_change),
                    implied_volatility=(
                        None
                        if pd.isna(r.implied_volatility)
                        else float(r.implied_volatility)
                    ),
                )
                for r in snap.itertuples(index=False)
            ]
            first = snap.iloc[0]
            context = {}
            if "context" in snap.columns and pd.notna(first.get("context")):
                try:
                    context = json.loads(first["context"])
                except (ValueError, TypeError):
                    context = {}
            chains.append(
                OptionChain(
                    underlying=str(first["underlying"]),
                    spot=float(first["spot"]),
                    expiry=pd.Timestamp(first["expiry"]).date(),
                    timestamp=pd.Timestamp(snap_ts).to_pydatetime(),
                    quotes=tuple(quotes),
                    context=context,
                )
            )
        return chains

    # --- internals ----------------------------------------------------------

    def _candle_path(self, symbol: str, timeframe: str, day: date) -> Path:
        return (
            self.root
            / "candles"
            / timeframe
            / str(day.year)
            / f"{symbol}_{day.isoformat()}.parquet"
        )

    def _chain_path(self, underlying: str, day: date) -> Path:
        return (
            self.root
            / "option_chain"
            / str(day.year)
            / f"{underlying}_{day.isoformat()}.parquet"
        )

    @staticmethod
    def _merge_parquet(
        path: Path, new_df: pd.DataFrame, key_cols: list[str]
    ) -> pd.DataFrame:
        """Merge new rows with any existing file, de-duplicating on keys."""
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df.copy()
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined = combined.sort_values(key_cols).reset_index(drop=True)
        return combined

    @staticmethod
    def _atomic_write(path: Path, df: pd.DataFrame) -> None:
        """Write via a temp file then replace, so readers never see a partial file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(path)


def _days_between(start: date, end: date) -> list[date]:
    if end < start:
        return []
    span = (end - start).days
    return [date.fromordinal(start.toordinal() + i) for i in range(span + 1)]
