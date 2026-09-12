"""EXP033 -- normalized relative-strike option-chain features (pure, causal).

Per `docs/preregistrations/exp033_option_event_predictability.md`. Every
function here operates on a single day's already-loaded chain data (a pandas
DataFrame in the raw `_CHAIN_COLUMNS` schema) and produces values that only
use information at or before the row's own timestamp. No cross-day baselines
are used (see the preregistration's rationale: too few days per
strike x type x time-of-day cell to be a reliable baseline yet).

These are pure numpy/pandas helpers, not I/O -- callers (the panel builder,
and later the live forward-testing loop) own reading/writing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RELATIVE_STRIKES = (-2, -1, 0, 1, 2)
OPTION_TYPES = ("CE", "PE")


def pivot_field(df: pd.DataFrame, opt_type: str, value: str,
                 strikes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(n_times, n_strikes) matrix of `value` for one option type, NaN where absent.

    Same pattern as scripts/strategy_test_framework.py::_pivot -- kept as a
    module-level function here (not copy-pasted from there) since EXP033 needs
    it independently for the relative-strike band, not just the ATM strike.
    """
    sub = df[df["option_type"] == opt_type]
    p = sub.pivot_table(index="snapshot_ts", columns="strike", values=value,
                         aggfunc="last")
    p = p.reindex(index=times, columns=strikes)
    return p.to_numpy(dtype=float)


def relative_strike_indices(strikes: np.ndarray, atm_idx: np.ndarray,
                             offset: int) -> np.ndarray:
    """Index into `strikes` for ATM+offset at each snapshot, clipped at the band edge.

    Clipping (rather than NaN) at the edges of the available strike ladder is
    a deliberate, documented approximation: it only bites on days with an
    unusually narrow collected strike band, and is recorded via the
    `<level>_clipped` flag the caller can derive from `idx == 0` or
    `idx == len(strikes) - 1`.
    """
    idx = atm_idx + offset
    return np.clip(idx, 0, len(strikes) - 1)


def same_day_causal_zscore(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Rolling z-score of `x` using only strictly prior values (shift(1)).

    NaN until `min_periods` prior observations exist -- this is what makes an
    event family return "cannot fire yet" rather than firing on a fabricated
    baseline early in the day.
    """
    s = pd.Series(x)
    prior = s.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.to_numpy(dtype=float)


def same_day_causal_percentile(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Rolling percentile rank (0-100) of the current value vs strictly prior values."""
    out = np.full(len(x), np.nan)
    vals = np.asarray(x, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i - window)
        hist = vals[lo:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_periods or np.isnan(vals[i]):
            continue
        out[i] = 100.0 * float((hist <= vals[i]).mean())
    return out


def derived_oi_change(oi: np.ndarray) -> np.ndarray:
    """OI change per snapshot, derived by same-day differencing.

    The stored `oi_change` column is unpopulated (0% nonzero, see the Phase-2
    data audit) -- this is the only reliable source.
    """
    out = np.diff(oi, prepend=oi[0] if len(oi) else 0.0)
    if len(out):
        out[0] = 0.0
    return out


def nearest_oi_wall(strikes: np.ndarray, call_oi_col: np.ndarray, put_oi_col: np.ndarray,
                     spot: float) -> tuple[float, float, float, float]:
    """Nearest resistance (max call-OI strike above spot) and support (max
    put-OI strike below spot) at one snapshot, plus their distances from spot.
    Returns (resistance_strike, resistance_dist, support_strike, support_dist);
    NaN where no strike qualifies (e.g. spot above/below the whole band).
    """
    above = strikes > spot
    below = strikes < spot
    resistance_strike = resistance_dist = np.nan
    support_strike = support_dist = np.nan
    if above.any():
        idx = np.nanargmax(np.where(above, call_oi_col, -np.inf))
        resistance_strike = float(strikes[idx])
        resistance_dist = resistance_strike - spot
    if below.any():
        idx = np.nanargmax(np.where(below, put_oi_col, -np.inf))
        support_strike = float(strikes[idx])
        support_dist = spot - support_strike
    return resistance_strike, resistance_dist, support_strike, support_dist


def opening_range_state(spot: np.ndarray, times: np.ndarray,
                         or_end="09:30") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal opening-range (OR) high/low and state ('ABOVE'/'INSIDE'/'BELOW').

    OR is only finalized once the OR window has fully elapsed; before that,
    OR high/low are NaN and state is 'FORMING' (never a fabricated value).
    """
    tod = pd.DatetimeIndex(times).strftime("%H:%M")
    or_mask = tod <= or_end
    or_high = np.full(len(spot), np.nan)
    or_low = np.full(len(spot), np.nan)
    if or_mask.any():
        h = float(np.nanmax(spot[or_mask]))
        l = float(np.nanmin(spot[or_mask]))
        first_after = np.argmax(~or_mask) if (~or_mask).any() else len(spot)
        or_high[first_after:] = h
        or_low[first_after:] = l
    state = np.full(len(spot), "FORMING", dtype=object)
    have_or = ~np.isnan(or_high)
    state[have_or & (spot > or_high)] = "ABOVE"
    state[have_or & (spot < or_low)] = "BELOW"
    state[have_or & (spot >= or_low) & (spot <= or_high)] = "INSIDE"
    return or_high, or_low, state
