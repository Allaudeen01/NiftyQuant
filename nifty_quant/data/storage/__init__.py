"""Pluggable persistence for candles and option-chain snapshots.

The default backend is Parquet (columnar, compressed, fast analytical reads,
trivially convertible to PostgreSQL/TimescaleDB later). Everything depends only
on the :class:`Storage` interface, so the backend can change without touching
collectors, the replay engine, or the backtester.
"""

from nifty_quant.data.storage.base import Storage
from nifty_quant.data.storage.parquet import ParquetStorage

__all__ = ["Storage", "ParquetStorage"]
