"""Versioned feature storage.

Features are keyed by ``(version, symbol)`` so changing the feature definition
(and thus :data:`FEATURE_VERSION`) never silently mixes incompatible feature
sets. Two backends:

- :class:`InMemoryFeatureStore` -- fast, ephemeral; ideal for a single backtest.
- :class:`ParquetFeatureStore`  -- persistent, partitioned by version/symbol/day.

Both implement :class:`FeatureStore`, so research code is backend-agnostic.
"""

from __future__ import annotations

import abc
from datetime import datetime
from pathlib import Path

import pandas as pd

from nifty_quant.features.vector import FeatureVector


class FeatureStore(abc.ABC):
    """Persistence for computed feature vectors."""

    @abc.abstractmethod
    def put(self, fv: FeatureVector) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get(
        self,
        symbol: str,
        version: str,
        start: datetime,
        end: datetime,
    ) -> list[FeatureVector]:
        raise NotImplementedError

    @abc.abstractmethod
    def to_frame(self, symbol: str, version: str) -> pd.DataFrame:
        """All stored features for (symbol, version) as a timestamp-indexed frame."""
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], dict[datetime, FeatureVector]] = {}

    def put(self, fv: FeatureVector) -> None:
        key = (fv.version, fv.symbol)
        self._data.setdefault(key, {})[fv.timestamp] = fv

    def get(
        self,
        symbol: str,
        version: str,
        start: datetime,
        end: datetime,
    ) -> list[FeatureVector]:
        bucket = self._data.get((version, symbol), {})
        out = [
            fv for ts, fv in bucket.items() if start <= ts <= end
        ]
        return sorted(out, key=lambda f: f.timestamp)

    def to_frame(self, symbol: str, version: str) -> pd.DataFrame:
        bucket = self._data.get((version, symbol), {})
        if not bucket:
            return pd.DataFrame()
        rows = [fv.as_row() for fv in sorted(bucket.values(), key=lambda f: f.timestamp)]
        return pd.DataFrame(rows).set_index("timestamp")


class ParquetFeatureStore(FeatureStore):
    """Parquet-backed store partitioned ``features/<version>/<symbol>/<day>.parquet``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, version: str, symbol: str, day) -> Path:
        return (
            self.root / "features" / version / symbol
            / f"{day.isoformat()}.parquet"
        )

    def put(self, fv: FeatureVector) -> None:
        row = fv.as_row()
        row["symbol"] = fv.symbol
        df = pd.DataFrame([row])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        path = self._path(fv.version, fv.symbol, fv.timestamp.date())
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(path)

    def _read_range(
        self, symbol: str, version: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        base = self.root / "features" / version / symbol
        if not base.exists():
            return pd.DataFrame()
        frames = [pd.read_parquet(p) for p in sorted(base.glob("*.parquet"))]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        return df.sort_values("timestamp").reset_index(drop=True)

    def get(
        self,
        symbol: str,
        version: str,
        start: datetime,
        end: datetime,
    ) -> list[FeatureVector]:
        df = self._read_range(symbol, version, start, end)
        out: list[FeatureVector] = []
        for row in df.to_dict("records"):
            ts = pd.Timestamp(row.pop("timestamp")).to_pydatetime()
            row.pop("symbol", None)
            values = {k: v for k, v in row.items() if pd.notna(v)}
            out.append(FeatureVector(ts, symbol, version, values))
        return out

    def to_frame(self, symbol: str, version: str) -> pd.DataFrame:
        df = self._read_range(
            symbol, version, datetime.min, datetime.max
        )
        if df.empty:
            return df
        return df.drop(columns=["symbol"], errors="ignore").set_index("timestamp")
