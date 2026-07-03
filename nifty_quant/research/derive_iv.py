"""Derive implied volatility from stored option-chain snapshots (read-only).

IV is intentionally NOT stored at collection time (raw quotes only). This module
recovers it on demand from the stored fields (price, spot, strike, expiry,
snapshot_ts) using the existing Black-Scholes solver -- non-destructively. It
never mutates the raw warehouse; callers that want IV materialised write it to a
SEPARATE derived store.

Nifty index options are European and cash-settled, so Black-Scholes applies.
For an index we use a small default dividend yield; ATM IV (call/put averaged)
is robust to forward mis-specification, which is what the VRP work relies on.
"""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from nifty_quant.analytics import black_scholes as bs

SESSION_CLOSE = time(15, 30)          # NSE regular-session close (IST)
DEFAULT_R = 0.065                     # ~India risk-free
DEFAULT_Q = 0.012                     # ~NIFTY dividend yield (small, index proxy)


def years_to_expiry(snapshot_ts, expiry) -> float:
    """Year-fraction from a snapshot instant to the expiry's 15:30 close.

    Uses intraday time remaining (not whole calendar days), so it stays positive
    and meaningful on expiry day (0-DTE) instead of collapsing to zero.
    """
    ts = pd.Timestamp(snapshot_ts).to_pydatetime()
    exp = pd.Timestamp(expiry).date()
    close_dt = datetime.combine(exp, SESSION_CLOSE)
    seconds = (close_dt - ts).total_seconds()
    return max(seconds, 0.0) / (365.0 * 86400.0)


def _row_price(row) -> float:
    bid = row.get("bid", 0.0) or 0.0
    ask = row.get("ask", 0.0) or 0.0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return row.get("last_price", 0.0) or 0.0


def derive_iv_frame(df: pd.DataFrame, r: float = DEFAULT_R, q: float = DEFAULT_Q
                    ) -> pd.Series:
    """Return a Series of implied vols (fraction) aligned to ``df``'s rows.

    NaN where IV cannot be solved (price <= 0, below intrinsic, or T <= 0).
    Expects the stored snapshot schema: snapshot_ts, spot, expiry, strike,
    option_type ('CE'/'PE'), last_price, bid, ask.
    """
    ivs = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        d = row._asdict()
        price = _row_price(d)
        spot = d.get("spot", 0.0) or 0.0
        strike = d.get("strike", 0.0) or 0.0
        if price <= 0 or spot <= 0 or strike <= 0:
            continue
        t = years_to_expiry(d["snapshot_ts"], d["expiry"])
        if t <= 0:
            continue
        iv = bs.implied_volatility(price, spot, strike, t, r,
                                   str(d["option_type"]), q)
        if iv is not None and iv > 0:
            ivs[i] = iv
    return pd.Series(ivs, index=df.index, name="iv")


def coverage(df: pd.DataFrame, sample: int | None = 4000,
             r: float = DEFAULT_R, q: float = DEFAULT_Q) -> dict:
    """Fast derivable-IV coverage stats for a day's quotes (optionally sampled).

    Returns {n_evaluated, coverage_pct, median_iv, atm_note}. Sampling keeps the
    quality report snappy; coverage is representative across the full frame.
    """
    work = df
    if sample is not None and len(df) > sample:
        work = df.sample(sample, random_state=0)
    iv = derive_iv_frame(work, r=r, q=q)
    n = len(iv)
    valid = iv.notna().sum()
    return {
        "n_evaluated": int(n),
        "coverage_pct": (100.0 * valid / n) if n else float("nan"),
        "median_iv": float(iv.dropna().median()) if valid else float("nan"),
    }
