"""nifty_quant: research and decision-support toolkit for Nifty options.

Deterministic, testable building blocks:
- data:      pluggable market-data provider interfaces and models
- analytics: technical indicators and options math (pure functions, no LLM)

The reasoning/LLM layer (added in a later phase) consumes the *outputs* of
these modules. It explains and synthesises; it never makes trading decisions
on its own. Trading decisions come from deterministic, backtested rules.
"""

__version__ = "0.1.0"
