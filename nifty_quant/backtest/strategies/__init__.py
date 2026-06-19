"""Concrete strategy implementations.

Reference implementations only, used to exercise the framework. They are NOT
trading recommendations.
"""

from nifty_quant.backtest.strategies.ema_cross import EmaCrossStrategy
from nifty_quant.backtest.strategies.feature_momentum import FeatureMomentumStrategy

__all__ = ["EmaCrossStrategy", "FeatureMomentumStrategy"]
