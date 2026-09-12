"""EXP033 Phase 7 -- MODEL_A / MODEL_B / MODEL_C definitions.

Per the preregistration: MODEL_A is price+time only, MODEL_B adds India VIX,
MODEL_C adds the option-event feature set. All three are the same LightGBM
multiclass classifier -- only the feature set differs, so any performance
gap is attributable to information content, not model capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

OR_STATE_MAP = {"FORMING": 0, "BELOW": -1, "INSIDE": 0, "ABOVE": 1}

PRICE_TIME_FEATURES = [
    "spot_ret_2m", "spot_ret_4m", "spot_ret_10m", "spot_rv_intraday",
    "twap_dist_pct", "dist_from_high_pct", "dist_from_low_pct", "or_state_num",
    "minute_of_day", "time_bucket_30m", "day_of_week", "dte_cal", "is_expiry_day_num",
]
VIX_FEATURES = ["vix", "vix_chg", "vix_mom", "vix_accel", "vix_pctile_so_far"]
OPTION_EVENT_FEATURES = [
    "pcr_oi", "pcr_oi_change", "oi_concentration", "resistance_dist", "support_dist",
    "event_vol_z", "event_oi_chg_z", "event_mom_pctile", "event_iv_z", "event_iv",
]

FEATURE_SETS = {
    "MODEL_A": PRICE_TIME_FEATURES,
    "MODEL_B": PRICE_TIME_FEATURES + VIX_FEATURES,
    "MODEL_C": PRICE_TIME_FEATURES + VIX_FEATURES + OPTION_EVENT_FEATURES,
}


def add_event_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Populate generic `event_*` columns from the row's own (relative_strike,
    option_type) L{level}_{type}_* panel columns -- so MODEL_C sees "this
    event's own trigger strength" regardless of which of the 10 tracked
    series it fired on.
    """
    df = df.copy()
    for col in ("event_vol_z", "event_oi_chg_z", "event_mom_pctile", "event_iv_z", "event_iv"):
        df[col] = np.nan
    for level in sorted(df["relative_strike"].unique()):
        for otype in ("CE", "PE"):
            mask = (df["relative_strike"] == level) & (df["option_type"] == otype)
            if not mask.any():
                continue
            prefix = f"L{int(level):+d}_{otype}"
            for src, dst in (
                ("vol_z", "event_vol_z"), ("oi_chg_z", "event_oi_chg_z"),
                ("mom_pctile", "event_mom_pctile"), ("iv_z", "event_iv_z"), ("iv", "event_iv"),
            ):
                col = f"{prefix}_{src}"
                if col in df.columns:
                    df.loc[mask, dst] = df.loc[mask, col].to_numpy()
    df["or_state_num"] = df["or_state"].map(OR_STATE_MAP).astype(float)
    df["is_expiry_day_num"] = df["is_expiry_day"].astype(float)
    return df


def build_classifier(random_state: int = 33) -> LGBMClassifier:
    """Fixed hyperparameters -- not tuned per fold (that would be a hidden
    multiple-testing exposure). Kept small/regularized given fold sizes of a
    few thousand rows at most.
    """
    return LGBMClassifier(
        # objective/num_class deliberately omitted -- LGBMClassifier infers
        # binary vs multiclass and the class count from y at fit time, which
        # matters if a small fold's fit set doesn't see all 3 outcomes.
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=-1,
    )
