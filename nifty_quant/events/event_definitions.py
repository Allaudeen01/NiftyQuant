"""Typed access to the locked EXP033 event universe.

Loads `docs/preregistrations/exp033_event_definitions.json` -- the single
source of truth for event thresholds, so detector code and the
pre-registration cannot drift apart. Nothing here re-declares a threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

PREREG_JSON = Path("docs") / "preregistrations" / "exp033_event_definitions.json"


def load_locked() -> dict:
    with open(PREREG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


class BaseFamilyTrigger(NamedTuple):
    event_type: str
    feature_suffix: str  # column suffix on the panel, e.g. "vol_z"
    condition: str       # one of "abs_gte", "gte", "lte"
    threshold: float


def base_family_triggers(locked: dict) -> list[BaseFamilyTrigger]:
    fam = locked["families"]
    return [
        BaseFamilyTrigger("E1_VOLUME_SPIKE", "vol_z", "abs_gte", fam["E1_VOLUME_SPIKE"]["trigger_abs_z"]),
        BaseFamilyTrigger("E2_OI_EXPANSION", "oi_chg_z", "gte", fam["E2_OI_EXPANSION"]["trigger_z_gte"]),
        BaseFamilyTrigger("E3_OI_UNWINDING", "oi_chg_z", "lte", fam["E3_OI_UNWINDING"]["trigger_z_lte"]),
        BaseFamilyTrigger("E4_PREMIUM_MOMENTUM_UP", "mom_pctile", "gte", fam["E4_PREMIUM_MOMENTUM"]["trigger_pctile_up_gte"]),
        BaseFamilyTrigger("E4_PREMIUM_MOMENTUM_DOWN", "mom_pctile", "lte", fam["E4_PREMIUM_MOMENTUM"]["trigger_pctile_down_lte"]),
        BaseFamilyTrigger("E5_IV_SHOCK", "iv_z", "abs_gte", fam["E5_IV_SHOCK"]["trigger_abs_z"]),
    ]


def combined_event_definitions(locked: dict) -> list[dict]:
    return locked["combined_events"]["definitions"]
