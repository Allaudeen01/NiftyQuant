"""Option-chain analytics: PCR, max pain, IV rank, gamma exposure.

All functions operate on the provider-agnostic :class:`OptionChain` model and
return plain numbers/dicts. No trading decisions, no LLM. These outputs become
features for the strategy engine and inputs to the LLM explanation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nifty_quant.analytics import black_scholes as bs
from nifty_quant.data.models import OptionChain, OptionType


def put_call_ratio(chain: OptionChain, by: str = "oi") -> float:
    """Put/Call ratio by open interest (``by="oi"``) or volume (``by="volume"``).

    PCR > 1 conventionally reads as more put activity (often contrarian
    bullish); < 1 the reverse. It is a *feature*, not a signal on its own.
    """
    if by == "oi":
        call_total = sum(q.open_interest for q in chain.calls())
        put_total = sum(q.open_interest for q in chain.puts())
    elif by == "volume":
        call_total = sum(q.volume for q in chain.calls())
        put_total = sum(q.volume for q in chain.puts())
    else:
        raise ValueError("by must be 'oi' or 'volume'")
    if call_total == 0:
        return float("nan")
    return put_total / call_total


def max_pain(chain: OptionChain) -> float:
    """Strike at which total option-holder payout is minimised (max pain).

    For each candidate expiry settlement price (each listed strike) we compute
    the aggregate intrinsic payout to all open call+put holders, weighted by
    open interest, and return the strike that minimises it.
    """
    strikes = chain.strikes()
    if not strikes:
        raise ValueError("chain has no strikes")

    calls = chain.calls()
    puts = chain.puts()

    best_strike = strikes[0]
    best_pain = float("inf")
    for settle in strikes:
        call_pain = sum(
            max(settle - q.strike, 0.0) * q.open_interest for q in calls
        )
        put_pain = sum(
            max(q.strike - settle, 0.0) * q.open_interest for q in puts
        )
        total = call_pain + put_pain
        if total < best_pain:
            best_pain = total
            best_strike = settle
    return best_strike


def iv_rank(current_iv: float, history: list[float]) -> float:
    """IV Rank in [0, 1]: where current IV sits within its historical range.

    rank = (current - min) / (max - min). Returns NaN if history is empty or
    flat. This needs a rolling window of past ATM IV readings supplied by the
    caller (e.g. last 252 trading days of India VIX or ATM IV).
    """
    if not history:
        return float("nan")
    lo = min(history)
    hi = max(history)
    if hi == lo:
        return float("nan")
    return (current_iv - lo) / (hi - lo)


def iv_percentile(current_iv: float, history: list[float]) -> float:
    """Fraction of historical readings below current IV, in [0, 1]."""
    if not history:
        return float("nan")
    below = sum(1 for v in history if v < current_iv)
    return below / len(history)


@dataclass(frozen=True)
class GammaExposure:
    """Dealer gamma exposure summary.

    total_gex : signed aggregate gamma exposure (sign convention below)
    flip_strike : strike nearest where cumulative GEX crosses zero, if any

    Convention used here: call gamma counted positive, put gamma negative,
    each scaled by open interest and spot^2. Positive total GEX is associated
    with mean-reverting (vol-dampening) dealer hedging; negative with
    trend-amplifying hedging. This is a heuristic feature, not a guarantee.
    """

    total_gex: float
    flip_strike: float | None
    per_strike: dict[float, float]


def gamma_exposure(
    chain: OptionChain,
    r: float = 0.065,
    t: float | None = None,
    q: float = 0.0,
) -> GammaExposure:
    """Compute aggregate dealer gamma exposure across the chain.

    Requires per-contract implied volatility (uses the contract IV if present,
    otherwise solves it from the mid price). ``t`` is time to expiry in years;
    if None it is derived from the chain timestamp and expiry date.
    """
    if t is None:
        t = _years_to_expiry(chain)
    spot = chain.spot

    per_strike: dict[float, float] = {}
    for quote in chain.quotes:
        iv = quote.implied_volatility
        if iv is None:
            iv = bs.implied_volatility(
                quote.mid,
                spot,
                quote.strike,
                t,
                r,
                quote.option_type.value,
                q,
            )
        if iv is None or iv <= 0 or t <= 0:
            continue
        g = bs.greeks(spot, quote.strike, t, r, iv, quote.option_type.value, q)
        sign = 1.0 if quote.option_type is OptionType.CALL else -1.0
        # Scale gamma to spot notional and open interest.
        gex = sign * g.gamma * quote.open_interest * (spot**2) * 0.01
        per_strike[quote.strike] = per_strike.get(quote.strike, 0.0) + gex

    total = float(sum(per_strike.values()))
    flip = _gamma_flip_strike(per_strike)
    return GammaExposure(total_gex=total, flip_strike=flip, per_strike=per_strike)


def atm_iv(chain: OptionChain, r: float = 0.065, q: float = 0.0) -> float | None:
    """Average implied vol of the ATM call and put (solved if not supplied)."""
    t = _years_to_expiry(chain)
    if t <= 0:
        return None
    atm = chain.atm_strike()
    ivs: list[float] = []
    for quote in chain.quotes:
        if quote.strike != atm:
            continue
        iv = quote.implied_volatility
        if iv is None:
            iv = bs.implied_volatility(
                quote.mid, chain.spot, atm, t, r, quote.option_type.value, q
            )
        if iv is not None and iv > 0:
            ivs.append(iv)
    if not ivs:
        return None
    return float(np.mean(ivs))


# --- helpers ---------------------------------------------------------------


def _years_to_expiry(chain: OptionChain) -> float:
    """Calendar-day year fraction from chain timestamp to expiry close."""
    seconds = (
        chain.expiry.toordinal() - chain.timestamp.date().toordinal()
    ) * 86400.0
    # Add intraday remainder to end-of-day expiry (~15:30 IST close approx).
    return max(seconds, 0.0) / (365.0 * 86400.0)


def _gamma_flip_strike(per_strike: dict[float, float]) -> float | None:
    """Strike where cumulative gamma exposure flips sign (zero-gamma level)."""
    if not per_strike:
        return None
    strikes = sorted(per_strike)
    cumulative = 0.0
    prev_strike = strikes[0]
    prev_cum = 0.0
    for k in strikes:
        cumulative += per_strike[k]
        if prev_cum != 0.0 and np.sign(cumulative) != np.sign(prev_cum):
            return (prev_strike + k) / 2.0
        prev_cum = cumulative
        prev_strike = k
    return None
