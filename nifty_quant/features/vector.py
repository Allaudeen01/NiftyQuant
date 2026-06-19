"""FeatureVector: a versioned, timestamped bundle of computed features.

Foundational type with no internal dependencies, so it can be imported by both
the feature engine and strategy code without creating cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class FeatureVector:
    """Named feature values for one symbol at one instant.

    ``version`` tags the feature-set definition that produced these values, so
    results can always be traced back to how features were computed.
    """

    timestamp: datetime
    symbol: str
    version: str
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = math.nan) -> float:
        return self.values.get(name, default)

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def is_ready(self, *required: str) -> bool:
        """True if all ``required`` features are present and non-NaN."""
        for name in required:
            v = self.values.get(name)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
        return True

    def as_row(self) -> dict:
        """Flat dict suitable for a DataFrame row (timestamp + symbol + values)."""
        return {"timestamp": self.timestamp, "symbol": self.symbol, **self.values}
