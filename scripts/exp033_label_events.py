"""EXP033 Phase 6 -- causal triple-barrier labeling of combined (E6) events.

Applies CONFIG_A/B/C (locked in the preregistration) to every combined event,
using the conservative barrier-resolution algorithm and the non-overlapping
event selection rule (`nifty_quant.labels.triple_barrier`). Labeling is done
per (event_type, relative_strike, option_type, trading_date, config) series,
walking the full intraday spot path from the cached feature panel -- never
just the event's own snapshot.

    python scripts/exp033_label_events.py

Output: data/derived/exp033_labeled_events.parquet
        reports/exp033/labeling_summary.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nifty_quant.events.event_definitions import load_locked
from nifty_quant.labels.triple_barrier import (
    DIRECTIONAL_LABELS,
    label_series_non_overlapping,
)

PANEL_PATH = Path("data") / "derived" / "exp033_feature_panel.parquet"
COMBINED_PATH = Path("data") / "derived" / "exp033_combined_events.parquet"
OUT_PATH = Path("data") / "derived" / "exp033_labeled_events.parquet"
OUT_DIR = Path("reports") / "exp033"


def main() -> int:
    locked = load_locked()
    configs = locked["triple_barrier_configs"]

    panel = pd.read_parquet(PANEL_PATH, columns=["date", "ts", "spot"])
    combined = pd.read_parquet(
        COMBINED_PATH,
        columns=["event_id", "trading_date", "timestamp", "relative_strike",
                  "option_type", "event_type", "direction"],
    )
    print(f"Loaded {len(combined)} combined events across "
          f"{combined['trading_date'].nunique()} days.")

    day_series = {
        d: (g["ts"].to_numpy(), g["spot"].to_numpy())
        for d, g in panel.groupby("date")
    }

    all_rows = []
    for config_name, cfg in configs.items():
        target_pts, stop_pts, max_hold = cfg["target_pts"], cfg["stop_pts"], cfg["max_hold_min"]
        group_cols = ["event_type", "relative_strike", "option_type", "trading_date"]
        for (etype, level, otype, day), g in combined.groupby(group_cols):
            if day not in day_series:
                continue
            times, spot = day_series[day]
            direction = g["direction"].iloc[0]
            labeled = label_series_non_overlapping(
                g["timestamp"], times, spot, target_pts, stop_pts, max_hold, direction
            )
            if labeled.empty:
                continue
            labeled["config"] = config_name
            labeled["event_type"] = etype
            labeled["relative_strike"] = level
            labeled["option_type"] = otype
            labeled["trading_date"] = day
            labeled["direction"] = direction
            labeled["directional_label"] = labeled["first_barrier"].map(
                lambda o, d=direction: DIRECTIONAL_LABELS[(d, o)]
            )
            all_rows.append(labeled)

    result = pd.concat(all_rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"Wrote {OUT_PATH}: {len(result)} labeled (non-overlapping) instances.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# EXP033 Phase 6 -- Labeling Summary", ""]
    for config_name in configs:
        sub = result[result["config"] == config_name]
        n = len(sub)
        lines.append(f"## {config_name} (n={n})")
        if n:
            counts = sub["first_barrier"].value_counts()
            pct = (100 * counts / n).round(1)
            for outcome in ["TARGET_FIRST", "STOP_FIRST", "TIMEOUT", "AMBIGUOUS"]:
                c = int(counts.get(outcome, 0))
                p = float(pct.get(outcome, 0.0))
                lines.append(f"- {outcome}: {c} ({p}%)")
            lines.append("")
            lines.append("By event type:")
            by_type = sub.groupby("event_type")["first_barrier"].value_counts(normalize=True).unstack().round(3)
            lines.append(by_type.to_string())
        lines.append("")

    (OUT_DIR / "labeling_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
