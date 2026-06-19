"""Feature subsystem: deterministic feature engineering + a versioned store.

    market events -> FeatureEngine -> FeatureVector -> FeatureStore -> Strategy

Strategies consume :class:`FeatureVector` objects rather than raw candles, so
every strategy sees identical, reproducible, versioned inputs. The store caches
computed features so backtests are repeatable and the (later) AI layer receives
structured features instead of recomputing them.
"""

from nifty_quant.features.vector import FeatureVector
from nifty_quant.features.engine import FeatureEngine, FeatureConfig, FEATURE_VERSION
from nifty_quant.features.store import (
    FeatureStore,
    InMemoryFeatureStore,
    ParquetFeatureStore,
)

__all__ = [
    "FeatureVector",
    "FeatureEngine",
    "FeatureConfig",
    "FEATURE_VERSION",
    "FeatureStore",
    "InMemoryFeatureStore",
    "ParquetFeatureStore",
]
