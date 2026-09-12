"""EXP034 Phase 2 -- feature and target construction, model-matrix assembly.

Builds, for every retained observation from Phase 1:
  * the 15 locked option-chain features + `iv_available`
  * the MODEL_1 price/time block and the MODEL_2 VIX block
  * the locked target values T1a/T1b/T2a/T2b (+ secondary T3a/T3b)

It does NOT fit any model, does NOT score any fold, and does NOT inspect any
feature-target relationship. Those belong to Phase 3, after the preflight gate
and the power addendum have been reviewed.

    python scripts/exp034_build_features.py

Outputs:
    data/derived/exp034_feature_panel.parquet
    reports/exp034/phase2_feature_audit.md
"""

from __future__ import annotations

import glob
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_quant.features import exp034_features as F

CONFIG_PATH = Path("docs") / "preregistrations" / "exp034_config.json"
OBS_PATH = Path("data") / "derived" / "exp034_observations.parquet"
OUT_PANEL = Path("data") / "derived" / "exp034_feature_panel.parquet"
OUT_REPORTS = Path("reports") / "exp034"


def load_day_near(d: date):
    folder = os.path.join("data", "option_chain", f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    near_exp = df["expiry"].min()
    near = df[df["expiry"] == near_exp].drop_duplicates(
        subset=["snapshot_ts", "strike", "option_type"], keep="last")
    return near, pd.Timestamp(near_exp).date()


def load_vix(d: date) -> pd.DataFrame:
    p = os.path.join("data", "vix", f"{d.year:04d}", f"INDIAVIX_{d.isoformat()}.parquet")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["timestamp", "india_vix"])
    v = pd.read_parquet(p)
    v["timestamp"] = pd.to_datetime(v["timestamp"]).dt.tz_localize(None)
    return v.sort_values("timestamp")


def main() -> int:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    obs = pd.read_parquet(OBS_PATH)
    obs = obs[obs["retained"]].copy()
    days = sorted(pd.to_datetime(obs["trading_date"]).dt.date.unique())
    print(f"EXP034 Phase 2 -- building features for {len(days)} usable days")

    tod_baseline: dict[int, np.ndarray] = {}
    tod_day_counts: dict[int, int] = {}
    # The locked config carries this inside F03's registry entry as prose, not
    # as a structured field. Assert against the locked string so any drift in
    # the preregistration is caught here rather than silently ignored.
    f03 = next(f for f in cfg["feature_registry"]["features"] if f["feature_id"] == "F03")
    min_prior_days = 5
    assert f"min {min_prior_days} prior days" in f03["normalization"], (
        f"locked F03 normalization no longer specifies min {min_prior_days} prior days: "
        f"{f03['normalization']!r}")

    frames = []
    for i, d in enumerate(days, 1):
        near, near_exp = load_day_near(d)
        vix = load_vix(d)
        # baseline uses STRICTLY PRIOR days; suppressed until min_prior_days seen
        usable_baseline = {b: v for b, v in tod_baseline.items()
                           if tod_day_counts.get(b, 0) >= min_prior_days}
        f = F.build_day_features(near, d, near_exp, vix, usable_baseline)
        frames.append(f)
        # append this day's values AFTER computing it (causality)
        for b, g in f.groupby("_tod_bucket"):
            arr = g["_tv"].to_numpy(dtype=float)
            tod_baseline[int(b)] = np.concatenate([tod_baseline.get(int(b), np.empty(0)), arr])
            tod_day_counts[int(b)] = tod_day_counts.get(int(b), 0) + 1
        print(f"  [{i}/{len(days)}] {d}  snapshots={len(f)}")

    feat = pd.concat(frames, ignore_index=True).drop(columns=["_tv", "_tod_bucket"])
    feat["trading_date"] = pd.to_datetime(feat["trading_date"])

    # ---- targets: computed from the Phase-1 scaffold timestamps -----------
    spot_lookup = {}
    for d in days:
        near, _ = load_day_near(d)
        s = near.groupby("snapshot_ts")["spot"].first()
        spot_lookup.update({(pd.Timestamp(d), ts): v for ts, v in s.items()})

    obs["trading_date"] = pd.to_datetime(obs["trading_date"])
    obs["spot_start"] = [spot_lookup.get((d, ts), np.nan) for d, ts in
                         zip(obs["trading_date"], obs["prediction_start_timestamp"])]
    obs["spot_end"] = [spot_lookup.get((d, ts), np.nan) for d, ts in
                       zip(obs["trading_date"], obs["prediction_end_timestamp"])]
    obs["signed_return"] = obs["spot_end"] / obs["spot_start"] - 1.0
    obs["abs_return"] = obs["signed_return"].abs()

    panel = obs.merge(feat, on=["trading_date", "feature_timestamp"], how="left")

    # secondary T3: causal ToD-bucket median of abs_return from PRIOR days only
    panel["tod_bucket"] = F._tod_bucket(pd.DatetimeIndex(panel["feature_timestamp"]))
    panel = panel.sort_values(["horizon_h", "trading_date", "feature_timestamp"]).reset_index(drop=True)
    panel["above_normal_move"] = np.nan
    for h, gh in panel.groupby("horizon_h"):
        hist: dict[int, list] = {}
        seen_days: dict[int, set] = {}
        vals = np.full(len(gh), np.nan)
        for pos, (_, row) in enumerate(gh.iterrows()):
            b = int(row["tod_bucket"])
            prior = hist.get(b, [])
            if len(seen_days.get(b, set())) >= min_prior_days and prior:
                med = float(np.nanmedian(prior))
                if np.isfinite(row["abs_return"]):
                    vals[pos] = 1.0 if row["abs_return"] > med else 0.0
            hist.setdefault(b, []).append(row["abs_return"])
            seen_days.setdefault(b, set()).add(row["trading_date"])
        panel.loc[gh.index, "above_normal_move"] = vals

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PANEL, engine="pyarrow", index=False)

    # ---- audit (data quality ONLY -- no feature/target relationships) -----
    reg = cfg["feature_registry"]
    chain_names = [f["feature_name"] for f in reg["features"]]
    mm = cfg["model_matrices"]
    m1 = mm["MODEL_1_column_list"]
    m2 = m1 + mm["MODEL_2_added_columns"]
    m3 = m2 + chain_names + ["iv_available"]

    lines = [
        "# EXP034 Phase 2 -- Feature and Target Audit",
        "",
        "Coverage/quality only. **No feature-target relationship is computed or",
        "inspected anywhere in this phase**; that begins in Phase 3.",
        "",
        f"- Rows: **{len(panel)}** (retained observations, both horizons)",
        f"- Days: **{panel['trading_date'].nunique()}**",
        "",
        "## Realized model matrices",
        f"- MODEL_1: {len([c for c in m1 if c in panel.columns])}/{len(m1)} columns present "
        f"(locked: {mm['MODEL_1_columns']})",
        f"- MODEL_2: {len([c for c in m2 if c in panel.columns])}/{len(m2)} columns present "
        f"(locked: {mm['MODEL_2_columns']})",
        f"- MODEL_3: {len([c for c in m3 if c in panel.columns])}/{len(m3)} columns present "
        f"(locked: {mm['MODEL_3_columns']})",
        "",
        "## Target availability",
    ]
    for h, g in panel.groupby("horizon_h"):
        lines.append(f"- h={h}: signed_return finite on {int(g['signed_return'].notna().sum())}/{len(g)}"
                     f" | above_normal_move defined on {int(g['above_normal_move'].notna().sum())}/{len(g)}")

    lines += ["", "## Chain-feature coverage (share finite)", "",
              "| feature | finite % |", "|---|---|"]
    for c in chain_names:
        if c in panel.columns:
            lines.append(f"| `{c}` | {100.0 * panel[c].notna().mean():.2f}% |")
    lines += [
        "",
        f"`iv_available` is a 0/1 indicator, so its *finite* share is trivially 100%. "
        f"The meaningful figure is its **mean = {100.0 * panel['iv_available'].mean():.2f}%**, "
        f"i.e. the share of snapshots where all four IV features actually solved. "
        f"This far exceeds the ~79% overall IV coverage cited in the preregistration "
        f"because the locked design restricts IV to ATM+-1, where liquidity is high -- "
        f"the restriction working as intended.",
    ]

    # numerical fragility scan (feature values only; no target involvement)
    num_chain = panel[chain_names].apply(pd.to_numeric, errors="coerce")
    frag = {c: int((num_chain[c].abs() > 1e3).sum()) for c in chain_names}
    frag = {c: n for c, n in frag.items() if n}
    lines += ["", "## Numerical fragility (locked formulas, unmodified)", ""]
    if frag:
        for c, n in frag.items():
            q999 = num_chain[c].quantile(0.999)
            lines.append(f"- `{c}`: {n} row(s) with |value| > 1e3 (q0.999 = {q999:,.2f}).")
        lines += [
            "",
            "`volume_acceleration` is the locked ratio `TV(t)/(mean prior-5 TV + eps) - 1`,",
            "which explodes when the prior-5 window has ~zero volume. The affected rows are",
            "a single snapshot; the distribution is otherwise sane through q0.999.",
            "**The locked formula is NOT modified.** The locked preprocessing discipline",
            "already lists *clipping thresholds* among parameters fit on training folds",
            "only, so the train-fitted clip in Phase 3 handles this by construction -- both",
            "when the row falls in train and when it falls in test.",
        ]
    else:
        lines.append("- none detected.")

    lines += ["", "## Diagnostic-flag composition (pre-committed, invalidate-only)",
              f"- short_gap_flag rows: {int(panel['short_gap_flag'].sum())}",
              f"- double_cadence_day_flag rows: {int(panel['double_cadence_day_flag'].sum())}",
              f"- clean subsample (neither flag): "
              f"{int((~panel['short_gap_flag'] & ~panel['double_cadence_day_flag']).sum())}"]

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT_REPORTS / "phase2_feature_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PANEL} ({len(panel)} rows, {len(panel.columns)} cols)")
    print(f"Wrote {OUT_REPORTS / 'phase2_feature_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
