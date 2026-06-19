"""Analytics layer: pure-function technical indicators and options math.

Nothing here makes a trading decision or calls an LLM. These are deterministic
transforms over price/option data whose outputs feed the strategy engine and
the (later) LLM explanation layer.
"""

from nifty_quant.analytics import indicators, options, black_scholes

__all__ = ["indicators", "options", "black_scholes"]
