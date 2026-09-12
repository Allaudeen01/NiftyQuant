"""EXP033 Phase 6 -- causal triple-barrier labeling.

Implements the LOCKED barrier-resolution algorithm and non-overlapping event
selection rule from
`docs/preregistrations/exp033_option_event_predictability.md`. No barrier
order is ever invented: if both target and stop levels fall within the
observed range spanned by two consecutive snapshots, the outcome is
AMBIGUOUS, full stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BarrierOutcome:
    outcome: str  # "TARGET_FIRST" | "STOP_FIRST" | "TIMEOUT" | "AMBIGUOUS"
    barrier_hit_time: pd.Timestamp | None
    mfe: float  # maximum favorable excursion, in points, signed by direction
    mae: float  # maximum adverse excursion, in points, signed by direction
    hit_index: int | None  # index into `times`/`spot` where resolution occurred
    final_excursion: float = 0.0
    """Realized excursion (points, signed by direction) at the exit snapshot
    -- the actual observed spot move, NOT the nominal target/stop level. A
    resolved barrier can overshoot its level between two 2-min snapshots;
    booking P&L at the nominal level would be an optimistic-fill assumption
    the preregistration explicitly forbids (Part 10: never assume a
    favorable fill)."""


def resolve_barrier(times: np.ndarray, spot: np.ndarray, entry_idx: int,
                     target_level: float, stop_level: float,
                     max_hold_min: float, direction: str) -> BarrierOutcome:
    """Walk forward from `entry_idx`, conservative barrier resolution.

    `target_level`/`stop_level` are already direction-adjusted absolute spot
    levels (target above entry for BULLISH, below for BEARISH; vice versa for
    stop). `direction` only affects the sign convention of MFE/MAE.

    Two independent AMBIGUOUS triggers, both required because they catch
    different failure modes with only two known endpoints per interval:

    1. **Endpoint-range test**: if both target and stop levels lie within
       [min, max] of the two endpoints, the path (by the intermediate value
       theorem) crossed both -- order unknowable. Given target/stop start
       symmetric around the entry, this is structurally almost unreachable on
       a walk's first live step (the walk always begins strictly *inside*
       the band and stops at the first single-sided crossing) -- included
       for completeness/edge cases, not as the primary safeguard.
    2. **Jump-magnitude test** (the one that actually matters): if the move
       from the previous to the current snapshot is at least as large as the
       full target-to-stop band width, a single endpoint crossing cannot be
       trusted -- the true intra-interval path may have touched the *other*
       barrier first and reversed, entirely invisible between two 2-minute
       snapshots. This is caught and disclosed here, before any labeling ran
       (matching Exp032's precedent of finding and fixing this class of bug
       up front, not after peeking at outcomes).
    """
    entry_ts = pd.Timestamp(times[entry_idx])
    entry_spot = spot[entry_idx]
    deadline = entry_ts + pd.Timedelta(minutes=max_hold_min)
    sign = 1.0 if direction == "BULLISH" else -1.0
    band_width = abs(target_level - stop_level)

    mfe = 0.0
    mae = 0.0
    last_excursion = 0.0
    prev_spot = entry_spot
    i = entry_idx
    n = len(times)
    while True:
        i += 1
        if i >= n or pd.Timestamp(times[i]) > deadline:
            return BarrierOutcome("TIMEOUT", None, mfe, mae, None, last_excursion)

        cur_spot = spot[i]
        lo, hi = (prev_spot, cur_spot) if prev_spot <= cur_spot else (cur_spot, prev_spot)

        excursion = sign * (cur_spot - entry_spot)
        mfe = max(mfe, excursion)
        mae = min(mae, excursion)
        last_excursion = excursion

        target_in_range = lo <= target_level <= hi
        stop_in_range = lo <= stop_level <= hi
        jump_too_large = abs(cur_spot - prev_spot) >= band_width

        if (target_in_range and stop_in_range) or (jump_too_large and (target_in_range or stop_in_range)):
            return BarrierOutcome("AMBIGUOUS", pd.Timestamp(times[i]), mfe, mae, i, excursion)
        if target_in_range:
            return BarrierOutcome("TARGET_FIRST", pd.Timestamp(times[i]), mfe, mae, i, excursion)
        if stop_in_range:
            return BarrierOutcome("STOP_FIRST", pd.Timestamp(times[i]), mfe, mae, i, excursion)

        prev_spot = cur_spot


def label_series_non_overlapping(event_times: pd.Series, day_times: np.ndarray,
                                  day_spot: np.ndarray, target_pts: float,
                                  stop_pts: float, max_hold_min: float,
                                  direction: str) -> pd.DataFrame:
    """Label one (event_type, relative_strike, option_type, trading_date) series.

    `event_times` are the raw firing timestamps (may be many, consecutive).
    Applies the non-overlap rule: a candidate firing is skipped if it falls
    before the previous KEPT instance's resolution time (or its timeout
    deadline). Returns one row per KEPT, labeled instance.
    """
    day_times_ts = pd.DatetimeIndex(day_times)
    time_to_idx = {t: i for i, t in enumerate(day_times_ts)}

    rows = []
    busy_until = None
    for ts in sorted(event_times):
        ts = pd.Timestamp(ts)
        if busy_until is not None and ts < busy_until:
            continue  # overlapping with an already-open instance -- skip
        if ts not in time_to_idx:
            continue
        entry_idx = time_to_idx[ts]
        entry_spot = day_spot[entry_idx]
        if direction == "BULLISH":
            target_level = entry_spot + target_pts
            stop_level = entry_spot - stop_pts
        else:
            target_level = entry_spot - target_pts
            stop_level = entry_spot + stop_pts

        outcome = resolve_barrier(day_times_ts.to_numpy(), day_spot, entry_idx,
                                   target_level, stop_level, max_hold_min, direction)
        rows.append({
            "event_time": ts,
            "entry_spot": entry_spot,
            "target_level": target_level,
            "stop_level": stop_level,
            "first_barrier": outcome.outcome,
            "barrier_hit_time": outcome.barrier_hit_time,
            "maximum_favorable_excursion": outcome.mfe,
            "maximum_adverse_excursion": outcome.mae,
            "realized_excursion_pts": outcome.final_excursion,
        })
        busy_until = (outcome.barrier_hit_time if outcome.barrier_hit_time is not None
                      else ts + pd.Timedelta(minutes=max_hold_min))
    return pd.DataFrame(rows)


DIRECTIONAL_LABELS = {
    ("BULLISH", "TARGET_FIRST"): "UP_TARGET_FIRST",
    ("BULLISH", "STOP_FIRST"): "DOWN_STOP_FIRST",
    ("BEARISH", "TARGET_FIRST"): "DOWN_TARGET_FIRST",
    ("BEARISH", "STOP_FIRST"): "UP_STOP_FIRST",
    ("BULLISH", "TIMEOUT"): "TIMEOUT",
    ("BEARISH", "TIMEOUT"): "TIMEOUT",
    ("BULLISH", "AMBIGUOUS"): "AMBIGUOUS",
    ("BEARISH", "AMBIGUOUS"): "AMBIGUOUS",
}
