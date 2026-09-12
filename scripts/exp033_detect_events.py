"""EXP033 Phase 5 -- run the event detector over the cached feature panel.

    python scripts/exp033_detect_events.py

Reads data/derived/exp033_feature_panel.parquet (Phase 4 output). Writes:
    data/derived/exp033_event_panel.parquet   -- base E1-E5 events + context
    data/derived/exp033_combined_events.parquet -- E6 combined events + context
    reports/exp033/event_detection_summary.md  -- counts per event type/day
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from nifty_quant.events import event_detector as ed
from nifty_quant.events.event_definitions import load_locked

PANEL_PATH = Path("data") / "derived" / "exp033_feature_panel.parquet"
EVENT_PANEL_PATH = Path("data") / "derived" / "exp033_event_panel.parquet"
COMBINED_PATH = Path("data") / "derived" / "exp033_combined_events.parquet"
OUT_DIR = Path("reports") / "exp033"


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH)
    locked = load_locked()
    print(f"Loaded panel: {len(panel)} rows, {panel['date'].nunique()} days.")

    base = ed.detect_base_events(panel, locked)
    print(f"Base (E1-E5) events detected: {len(base)}")

    combined = ed.detect_combined_events(base, locked)
    print(f"Combined (E6) events detected: {len(combined)}")

    base_with_ctx = ed.attach_context(base, panel)
    combined_with_ctx = ed.attach_context(combined, panel)

    EVENT_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    base_with_ctx.to_parquet(EVENT_PANEL_PATH, engine="pyarrow", index=False)
    combined_with_ctx.to_parquet(COMBINED_PATH, engine="pyarrow", index=False)
    print(f"Wrote {EVENT_PANEL_PATH} ({len(base_with_ctx)} rows, {len(base_with_ctx.columns)} cols)")
    print(f"Wrote {COMBINED_PATH} ({len(combined_with_ctx)} rows, {len(combined_with_ctx.columns)} cols)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_type = base.groupby("event_type").size().sort_values(ascending=False) if len(base) else pd.Series(dtype=int)
    by_day = base.groupby("trading_date").size() if len(base) else pd.Series(dtype=int)
    combo_counts = combined.groupby("event_type").size().sort_values(ascending=False) if len(combined) else pd.Series(dtype=int)

    lines = [
        "# EXP033 Phase 5 -- Event Detection Summary",
        "",
        f"Base events (E1-E5): **{len(base)}** across {panel['date'].nunique()} days.",
        f"Combined events (E6): **{len(combined)}**.",
        "",
        "## Base event counts by type",
        by_type.to_string() if len(by_type) else "(none)",
        "",
        "## Combined (E6) event counts by type",
        combo_counts.to_string() if len(combo_counts) else "(none)",
        "",
        "## Base events per day (min/median/max)",
        (f"min={by_day.min()} median={by_day.median()} max={by_day.max()}"
         if len(by_day) else "(none)"),
    ]
    (OUT_DIR / "event_detection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
