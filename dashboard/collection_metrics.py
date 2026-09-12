"""Collection-diagnostics metrics for the dashboard. READ-ONLY.

Reads only raw collected data (`data/option_chain`, `data/vix`). Never reads,
writes or touches any frozen experiment artifact.

IMPORTANT HONESTY CONSTRAINT
----------------------------
Under the current data-generating process (`dgp-v1`) every row of a poll shares
one `snapshot_ts`. **Per-contract observation timestamps do not exist.** This
module therefore distinguishes:

  * MEASURED  -- computed from recorded data.
  * MODELLED  -- derived from the known collector algorithm (batch size, pause,
                 strike-sorted token order), NOT observed.
  * UNAVAILABLE -- requires `dgp-v2` (see docs/data_collection_redesign.md).

Nothing modelled is ever presented as if it were measured.
"""

from __future__ import annotations

import glob
import math
import os
from datetime import date

import numpy as np
import pandas as pd

# Collector/provider constants that define dgp-v1 timing behaviour.
BATCH_SIZE = 50           # angelone._fetch_market_data
REQUEST_PAUSE_S = 1.2     # collector --request-pause
EXPIRY_GAP_S = 1.5        # collect_market_data.poll_once expiry_gap
SESSION_OPEN, SESSION_CLOSE = "09:15", "15:30"
NOMINAL_POLL_S = 120.0

# Warning thresholds (configurable limits for sequential skew).
SKEW_WARN_S = 2.5
SKEW_CRIT_S = 4.0


V1_ROOT = "option_chain"
V2_ROOT = "option_chain_v2"


def available_days(data_dir: str = "data", root: str = V1_ROOT) -> list[date]:
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, root, "*", "*", "*"))):
        if os.path.isdir(p):
            y, m, d = p.replace("\\", "/").split("/")[-3:]
            try:
                out.append(date(int(y), int(m), int(d)))
            except ValueError:
                continue
    return sorted(out)


def available_roots(data_dir: str = "data") -> dict[str, int]:
    """Which data-generating processes have data on disk.

    v1 and v2 are NEVER merged here. They are different processes and mixing
    them in one view would be the same error the storage separation exists to
    prevent.
    """
    return {r: len(available_days(data_dir, r))
            for r in (V1_ROOT, V2_ROOT)
            if os.path.isdir(os.path.join(data_dir, r))}


def load_day(d: date, data_dir: str = "data", root: str = V1_ROOT) -> pd.DataFrame:
    folder = os.path.join(data_dir, root, f"{d.year:04d}",
                          f"{d.month:02d}", f"{d.day:02d}")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # dgp-v2 has no `snapshot_ts`; its poll anchor is `poll_started_ts` and its
    # real observation times are per-contract in `observed_ts`.
    if "snapshot_ts" not in df.columns and "poll_started_ts" in df.columns:
        df["snapshot_ts"] = pd.to_datetime(df["poll_started_ts"])
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    if "observed_ts" in df.columns:
        df["observed_ts"] = pd.to_datetime(df["observed_ts"])
    return df


def has_per_contract_timing(df: pd.DataFrame) -> bool:
    """True only once dgp-v2 records per-contract observation times."""
    return "observed_ts" in df.columns


def poll_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-poll MEASURED facts, plus MODELLED batch/skew estimates."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("snapshot_ts")
    out = pd.DataFrame({
        "snapshot_ts": list(g.groups.keys()),
        "n_contracts": g.size().to_numpy(),
        "n_strikes": g["strike"].nunique().to_numpy(),
        "n_expiries": g["expiry"].nunique().to_numpy(),
        "spot": g["spot"].first().to_numpy(),
    }).sort_values("snapshot_ts").reset_index(drop=True)

    out["gap_s"] = out["snapshot_ts"].diff().dt.total_seconds()

    # MODELLED, matching the collector's ACTUAL control flow: poll_once() loops
    # over expiries, sleeping `expiry_gap` between them, and each expiry's
    # get_option_chain() runs its own independent batch sequence. Modelling all
    # contracts as one batch sequence would understate the spread.
    per_exp = (df.groupby(["snapshot_ts", "expiry"]).size()
                 .rename("n").reset_index())
    rows = []
    for ts, g in per_exp.groupby("snapshot_ts"):
        g = g.sort_values("expiry")
        batches = np.ceil(g["n"].to_numpy() / BATCH_SIZE).astype(int)
        # cumulative elapsed time up to the LAST batch of the LAST expiry
        elapsed = 0.0
        for r, b in enumerate(batches):
            if r > 0:
                elapsed += EXPIRY_GAP_S
            elapsed += (b - 1) * REQUEST_PAUSE_S
        # Two distinct quantities, deliberately NOT conflated:
        #  * within-expiry skew  -- what single-expiry analysis actually faces
        #    (EXP034/EXP035 both used the near expiry only)
        #  * full-poll skew      -- spans expiries, larger, only relevant if an
        #    analysis compares across expiries
        within = (batches[0] - 1) * REQUEST_PAUSE_S if len(batches) else 0.0
        rows.append({"snapshot_ts": ts, "modelled_batches": int(batches.sum()),
                     "modelled_batches_per_expiry": ",".join(map(str, batches)),
                     "modelled_skew_within_expiry_s": round(float(within), 2),
                     "modelled_skew_s": round(float(elapsed), 2)})
    out = out.merge(pd.DataFrame(rows), on="snapshot_ts", how="left")
    # The collector fetches the FULL ladder then trims to the stored band, so
    # contracts-on-disk understates contracts actually fetched: this is a BOUND.
    out["modelled_skew_is_lower_bound"] = True

    if has_per_contract_timing(df):
        t = df.groupby("snapshot_ts")["observed_ts"]
        out["measured_first_contact_ts"] = t.min().to_numpy()
        out["measured_last_contact_ts"] = t.max().to_numpy()
        out["measured_skew_s"] = (t.max() - t.min()).dt.total_seconds().to_numpy()
    return out


def modelled_contract_offsets(df: pd.DataFrame, snapshot_ts) -> pd.DataFrame:
    """MODELLED per-contract offset for one poll.

    Reconstructs the collector's ordering: `option_instruments()` returns
    contracts sorted by (strike, option_type), `_fetch_market_data()` chunks
    that list into batches of 50 with a pause between batches, and each expiry
    is fetched in turn separated by `expiry_gap`. The resulting offset is
    therefore MONOTONIC IN STRIKE within an expiry.

    This is a model of the algorithm, not an observation. Real offsets are
    unavailable until dgp-v2.
    """
    snap = df[df["snapshot_ts"] == snapshot_ts].copy()
    if snap.empty:
        return snap
    frames = []
    elapsed = 0.0   # running wall-clock offset, sequential across expiries
    for rank, exp in enumerate(sorted(snap["expiry"].unique())):
        sub = snap[snap["expiry"] == exp].copy()
        # collector sort order: (strike, option_type) with "CE" < "PE"
        sub = sub.sort_values(["strike", "option_type"], kind="mergesort").reset_index(drop=True)
        sub["fetch_rank"] = np.arange(len(sub))
        sub["batch_index"] = sub["fetch_rank"] // BATCH_SIZE
        sub["expiry_rank"] = rank
        if rank > 0:
            elapsed += EXPIRY_GAP_S          # poll_once sleeps between expiries
        sub["modelled_offset_s"] = elapsed + sub["batch_index"] * REQUEST_PAUSE_S
        elapsed += float(sub["batch_index"].max()) * REQUEST_PAUSE_S
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def day_quality(d: date, df: pd.DataFrame) -> dict:
    """Daily collection-quality score from MEASURED quantities only (0-100)."""
    if df.empty:
        return {"day": d.isoformat(), "score": 0.0, "verdict": "NO DATA"}

    ps = poll_summary(df)
    n_snap = len(ps)
    expected = (pd.Timestamp(f"{d} {SESSION_CLOSE}")
                - pd.Timestamp(f"{d} {SESSION_OPEN}")).total_seconds() / NOMINAL_POLL_S
    coverage = min(1.0, n_snap / expected) if expected else 0.0

    gaps = ps["gap_s"].dropna()
    cadence_ok = float((gaps <= 300).mean()) if len(gaps) else 0.0
    max_gap = float(gaps.max()) if len(gaps) else np.nan

    spot_by_snap = ps["spot"]
    spot_std = float(spot_by_snap.std()) if len(spot_by_snap) > 1 else 0.0
    spot_alive = 1.0 if spot_std > 0.5 else 0.0        # frozen-feed detector

    two_sided = float(((df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)).mean())
    crossed = float(((df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)
                     & (df["ask"] < df["bid"])).mean())

    skew = float(ps["modelled_skew_within_expiry_s"].max()) if len(ps) else np.nan
    skew_full = float(ps["modelled_skew_s"].max()) if len(ps) else np.nan
    skew_ok = 1.0 if (np.isfinite(skew) and skew <= SKEW_WARN_S) else (
        0.5 if (np.isfinite(skew) and skew <= SKEW_CRIT_S) else 0.0)

    components = {
        "coverage": coverage,
        "cadence": cadence_ok,
        "spot_alive": spot_alive,
        "two_sided": two_sided,
        "not_crossed": 1.0 - min(1.0, crossed * 100),
        "sync": skew_ok,
    }
    weights = {"coverage": 0.25, "cadence": 0.15, "spot_alive": 0.25,
               "two_sided": 0.15, "not_crossed": 0.05, "sync": 0.15}
    # Coverage is also a GATE, not merely a weighted term. Purely additive
    # weighting let a day with 22 of 188 expected polls still score 76/100,
    # which is misleading: a day that was barely collected cannot be a good
    # day regardless of how clean the few polls it does have happen to be.
    coverage_gate = min(1.0, coverage / 0.9)
    score = 100.0 * sum(components[k] * weights[k] for k in weights) * coverage_gate
    components["coverage_gate"] = coverage_gate

    warnings = []
    if spot_alive == 0.0:
        warnings.append("SPOT FEED FROZEN - spot std <= 0.5 pts across the session")
    if coverage < 0.9:
        warnings.append(f"LOW COVERAGE - {n_snap} polls vs ~{expected:.0f} expected")
    if np.isfinite(max_gap) and max_gap > 300:
        warnings.append(f"CADENCE GAP - max {max_gap:.0f}s exceeds 300s")
    if np.isfinite(skew) and skew > SKEW_CRIT_S:
        warnings.append(f"SEQUENTIAL SKEW CRITICAL - modelled within-expiry {skew:.1f}s > {SKEW_CRIT_S}s")
    elif np.isfinite(skew) and skew > SKEW_WARN_S:
        warnings.append(f"SEQUENTIAL SKEW HIGH - modelled within-expiry {skew:.1f}s > {SKEW_WARN_S}s")
    if crossed > 0:
        warnings.append(f"CROSSED QUOTES - {100*crossed:.3f}% of rows")

    verdict = "GOOD" if score >= 85 else ("REVIEW" if score >= 60 else "POOR")
    return {"day": d.isoformat(), "score": round(score, 1), "verdict": verdict,
            "n_snapshots": n_snap, "expected_snapshots": round(expected),
            "median_gap_s": float(gaps.median()) if len(gaps) else np.nan,
            "max_gap_s": max_gap, "spot_std": round(spot_std, 3),
            "two_sided_pct": round(100 * two_sided, 2),
            "crossed_pct": round(100 * crossed, 4),
            "modelled_max_skew_within_expiry_s": skew,
            "modelled_max_skew_full_poll_s": skew_full,
            "n_contracts_median": int(ps["n_contracts"].median()),
            "modelled_batches_max": int(ps["modelled_batches"].max()),
            "components": components, "warnings": warnings}
