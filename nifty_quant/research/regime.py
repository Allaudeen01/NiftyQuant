"""Market-regime classification.

Tags a price series with a trend label (UP/DOWN/SIDEWAYS), a volatility label
(HIGH/LOW), and event tags (e.g. GAP). Attaching the regime to every backtest
result lets you later learn statements like "this strategy only works in
trending, low-volatility markets".

Heuristics are intentionally simple and transparent; they are descriptive
labels, not predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    flat_return_threshold: float = 0.03   # |total return| below => SIDEWAYS
    trend_r2_threshold: float = 0.3       # linear fit R^2 below => SIDEWAYS
    high_vol_annual: float = 0.20         # annualised vol above => HIGH
    gap_threshold: float = 0.02           # any daily move above => GAP tag
    trading_days: int = 252


@dataclass(frozen=True)
class Regime:
    trend: str            # "UP" | "DOWN" | "SIDEWAYS"
    volatility: str       # "HIGH" | "LOW"
    tags: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "trend": self.trend,
            "volatility": self.volatility,
            "tags": list(self.tags),
            "stats": dict(self.stats),
        }


def classify_regime(
    prices: pd.Series,
    config: RegimeConfig | None = None,
) -> Regime:
    """Classify the regime of a (time-indexed) price series."""
    cfg = config or RegimeConfig()
    prices = prices.dropna()
    if len(prices) < 3:
        return Regime("SIDEWAYS", "LOW", [], {"n": int(len(prices))})

    values = prices.to_numpy(dtype=float)
    n = len(values)
    total_return = values[-1] / values[0] - 1.0

    # Trend via linear fit on a normalised index.
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if abs(total_return) < cfg.flat_return_threshold or r2 < cfg.trend_r2_threshold:
        trend = "SIDEWAYS"
    elif slope > 0:
        trend = "UP"
    else:
        trend = "DOWN"

    # Volatility from daily returns (annualised).
    returns = pd.Series(values).pct_change().dropna()
    daily_vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(cfg.trading_days)
    volatility = "HIGH" if annual_vol >= cfg.high_vol_annual else "LOW"

    tags: list[str] = []
    if len(returns) and returns.abs().max() >= cfg.gap_threshold:
        tags.append("GAP")

    stats = {
        "total_return": total_return,
        "slope": float(slope),
        "r2": float(r2),
        "annual_vol": float(annual_vol),
        "n": int(n),
    }
    return Regime(trend=trend, volatility=volatility, tags=tags, stats=stats)
