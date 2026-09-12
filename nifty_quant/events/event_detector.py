"""EXP033 Phase 5 -- event detector.

Applies the locked triggers (`event_definitions.py`) to the causal feature
panel (`scripts/exp033_build_panel.py` output) to produce structured event
records: base families E1-E5 per (relative_strike, option_type, snapshot),
and combined E6 events wherever two base legs fire together at the same
(relative_strike, option_type, snapshot).

Pure function of an already-built panel -- no I/O, no new lookahead: every
trigger reads only columns the panel already computed causally.
"""

from __future__ import annotations

import pandas as pd

from nifty_quant.events.event_definitions import (
    base_family_triggers,
    combined_event_definitions,
    load_locked,
)


def _fires(series: pd.Series, condition: str, threshold: float) -> pd.Series:
    if condition == "abs_gte":
        return series.abs() >= threshold
    if condition == "gte":
        return series >= threshold
    if condition == "lte":
        return series <= threshold
    raise ValueError(f"unknown condition {condition!r}")


def detect_base_events(panel: pd.DataFrame, locked: dict | None = None) -> pd.DataFrame:
    """One row per (event_type, relative_strike, option_type, snapshot) firing."""
    locked = locked or load_locked()
    triggers = base_family_triggers(locked)
    levels = locked["relative_strikes"]
    otypes = locked["option_types"]

    rows = []
    for level in levels:
        for otype in otypes:
            prefix = f"L{level:+d}_{otype}"
            strike_col = f"{prefix}_strike"
            if strike_col not in panel.columns:
                continue
            for trig in triggers:
                col = f"{prefix}_{trig.feature_suffix}"
                if col not in panel.columns:
                    continue
                mask = _fires(panel[col], trig.condition, trig.threshold).fillna(False)
                if not mask.any():
                    continue
                sub = panel.loc[mask, ["date", "ts", "spot", strike_col]].copy()
                sub = sub.rename(columns={strike_col: "absolute_strike"})
                sub["relative_strike"] = level
                sub["option_type"] = otype
                sub["event_type"] = trig.event_type
                fam_key = _family_key(trig.event_type)
                sub["event_family_version"] = locked["families"].get(fam_key, {}).get(
                    "version", locked["event_universe_version"]
                )
                sub["event_strength"] = panel.loc[mask, col].to_numpy()
                rows.append(sub)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "event_id", [f"E-{i:07d}" for i in range(len(out))])
    out = out.rename(columns={"date": "trading_date", "ts": "timestamp"})
    return out


def _family_key(event_type: str) -> str:
    if event_type.startswith("E4_PREMIUM_MOMENTUM"):
        return "E4_PREMIUM_MOMENTUM"
    return event_type


def detect_combined_events(base_events: pd.DataFrame, locked: dict | None = None) -> pd.DataFrame:
    """AND two base-family legs at the same (relative_strike, option_type, snapshot)."""
    if base_events.empty:
        return pd.DataFrame()
    locked = locked or load_locked()
    combos = combined_event_definitions(locked)

    key_cols = ["trading_date", "timestamp", "relative_strike", "option_type"]
    by_type = {
        et: g.set_index(key_cols)
        for et, g in base_events.groupby("event_type")
    }

    rows = []
    for combo in combos:
        leg_types = []
        for leg in combo["legs"]:
            fam, otype = leg.split(":")
            event_type = {
                "E1": "E1_VOLUME_SPIKE",
                "E2": "E2_OI_EXPANSION",
                "E3": "E3_OI_UNWINDING",
                "E4_up": "E4_PREMIUM_MOMENTUM_UP",
                "E4_down": "E4_PREMIUM_MOMENTUM_DOWN",
                "E5": "E5_IV_SHOCK",
            }[fam]
            leg_types.append((event_type, otype))

        (et1, ot1), (et2, ot2) = leg_types
        if et1 not in by_type or et2 not in by_type:
            continue
        g1 = by_type[et1]
        g2 = by_type[et2]
        g1 = g1[g1.index.get_level_values("option_type") == ot1]
        g2 = g2[g2.index.get_level_values("option_type") == ot2]
        common = g1.index.intersection(g2.index)
        if len(common) == 0:
            continue
        hit1 = g1.loc[common]
        hit2 = g2.loc[common]
        combined = pd.DataFrame({
            "trading_date": [k[0] for k in common],
            "timestamp": [k[1] for k in common],
            "relative_strike": [k[2] for k in common],
            "option_type": ot1,
            "spot": hit1["spot"].to_numpy(),
            "absolute_strike": hit1["absolute_strike"].to_numpy(),
            "event_type": combo["code"] + "_" + combo["name"],
            "event_family_version": locked["combined_events"]["version"],
            "direction": combo["direction"],
            "leg1_strength": hit1["event_strength"].to_numpy(),
            "leg2_strength": hit2["event_strength"].to_numpy(),
        })
        rows.append(combined)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "event_id", [f"C-{i:06d}" for i in range(len(out))])
    return out


def attach_context(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join full causal panel context onto event rows by (trading_date, timestamp)."""
    if events.empty:
        return events
    ctx = panel.rename(columns={"date": "trading_date", "ts": "timestamp"})
    merged = events.merge(ctx, on=["trading_date", "timestamp"], how="left", suffixes=("", "_ctx"))
    return merged
