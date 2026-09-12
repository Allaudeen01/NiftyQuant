"""EXP034 Phase 1 -- deterministic data pipeline (NO models, NO targets, NO features).

Implements exactly the six Phase-1 items authorised after the preregistration
was locked at commit 18a178e:

  1. deterministic data loading
  2. dgp-v1 certification
  3. per-day data-quality screen
  4. near-expiry selection
  5. t+1 snapshot pairing
  6. mandatory audit columns

It deliberately does NOT compute features, does NOT compute target/label
values, does NOT train anything, and does NOT inspect any predictive
relationship. Those belong to later, separately authorised phases.

    python scripts/exp034_build_observations.py

Outputs (EXP034 namespace only -- EXP033 is never touched):
    data/derived/exp034_day_certification.parquet
    data/derived/exp034_observations.parquet
    reports/exp034/phase1_data_audit.md
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path("docs") / "preregistrations" / "exp034_config.json"
DATA_DIR = "data"
OUT_DERIVED = Path("data") / "derived"
OUT_REPORTS = Path("reports") / "exp034"

# Source files whose semantics define dgp-v1. Fingerprinted so that any future
# collector/provider change is detectable when certifying forward-collected days.
DGP_SOURCE_FILES = [
    "scripts/collect_market_data.py",
    "nifty_quant/data/providers/angelone.py",
]

SNAP_COLUMNS = [
    "snapshot_ts", "underlying", "spot", "expiry", "strike", "option_type",
    "last_price", "bid", "ask", "volume", "open_interest", "oi_change",
    "implied_volatility", "context",
]

SESSION_CLOSE = "15:29"


def _norm_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    b = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(b).hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def collected_days(data_dir: str = DATA_DIR) -> list[date]:
    days = []
    for d in sorted(glob.glob(os.path.join(data_dir, "option_chain", "*", "*", "*"))):
        if not os.path.isdir(d):
            continue
        y, m, dd = d.replace("\\", "/").split("/")[-3:]
        try:
            days.append(date(int(y), int(m), int(dd)))
        except ValueError:
            continue
    return sorted(days)


def load_day_raw(d: date, data_dir: str = DATA_DIR) -> pd.DataFrame:
    folder = os.path.join(data_dir, "option_chain", f"{d.year:04d}",
                           f"{d.month:02d}", f"{d.day:02d}")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


# A cumulative counter yields ~0% negative first differences; a per-interval
# field would yield ~50%. Observed near-expiry rates across all 59 days: median
# 0.01%, worst day 0.50% (sporadic missing readings, 47% of which drop to
# exactly 0). Any threshold in roughly [1%, 45%] classifies every day
# identically, so this discriminator is insensitive to its exact value -- it
# tests the SEMANTIC claim "volume is cumulative", not data perfection.
VOLUME_NEG_DIFF_MAX_PCT = 1.0


def certify_dgp_v1(near: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Observable-property certification of the dgp-v1 semantics.

    Certification is scoped to the NEAR EXPIRY -- the data EXP034 actually
    consumes under the locked preregistration. Far-expiry anomalies are
    recorded as diagnostics rather than gates, because they never enter the
    analysis universe.

    Honest limitation, recorded in the output: properties that live in the
    COLLECTOR (quote-before-spot ordering, pre-stamped timestamp, 30s spot
    cache, VIX-fetched-first) are NOT observable from stored rows. For
    historical days they cannot be verified retrospectively; for forward days
    the source fingerprint is the operative control.
    """
    out: dict = {}
    out["schema_ok"] = all(c in near.columns for c in SNAP_COLUMNS)

    def _diff_stats(frame: pd.DataFrame, col: str) -> tuple[int, int]:
        neg = total = 0
        for _, g in frame.sort_values("snapshot_ts").groupby(
                ["strike", "option_type"], sort=False):
            v = g[col].to_numpy(dtype=float)
            if len(v) > 1:
                d = np.diff(v)
                neg += int((d < 0).sum())
                total += len(d)
        return neg, total

    # volume must behave as CUMULATIVE daily traded volume
    neg, total = _diff_stats(near, "volume")
    rate = (100.0 * neg / total) if total else 0.0
    out["volume_negative_diffs"] = neg
    out["volume_total_diffs"] = total
    out["volume_negative_diff_pct"] = round(rate, 4)
    out["volume_cumulative_ok"] = rate < VOLUME_NEG_DIFF_MAX_PCT

    # open_interest must behave as a LEVEL (falls as well as rises) -- diagnostic
    oi_neg, _ = _diff_stats(near, "open_interest")
    out["oi_negative_diffs"] = oi_neg
    out["oi_is_level_ok"] = oi_neg > 0

    out["iv_all_null_ok"] = bool(near["implied_volatility"].isna().all())
    out["oi_change_all_zero_ok"] = bool((near["oi_change"].fillna(0) == 0).all())
    out["one_spot_per_snapshot_ok"] = bool(
        near.groupby("snapshot_ts")["spot"].nunique().max() == 1)

    # Diagnostic only: cross-expiry spot divergence within one poll is the
    # documented 30s spot-cache TTL expiring between the two expiry fetches.
    # It cannot affect EXP034, which reads the near expiry only.
    out["cross_expiry_spot_divergence_snapshots"] = int(
        (raw.groupby("snapshot_ts")["spot"].nunique() > 1).sum())

    hard = ["schema_ok", "volume_cumulative_ok", "iv_all_null_ok",
            "oi_change_all_zero_ok", "one_spot_per_snapshot_ok"]
    out["dgp_v1_certified"] = all(bool(out[k]) for k in hard)
    out["dgp_v1_failed_checks"] = ",".join(k for k in hard if not out[k])
    return out


def select_near_expiry(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Near expiry = the minimum expiry present. Verified constant intraday."""
    per_snap_min = raw.groupby("snapshot_ts")["expiry"].min()
    constant = bool(per_snap_min.nunique() == 1)
    near = raw["expiry"].min()
    sub = raw[raw["expiry"] == near].copy()
    sub = sub.drop_duplicates(subset=["snapshot_ts", "strike", "option_type"], keep="last")
    return sub, {"near_expiry": pd.Timestamp(near).date().isoformat(),
                 "near_expiry_constant_intraday": constant}


def quality_screen(near: pd.DataFrame, cfg: dict) -> dict:
    """Deterministic per-day screen, thresholds taken from the locked config."""
    scr = cfg["data"]["per_day_quality_screen"]
    snaps = np.sort(near["snapshot_ts"].unique())
    spot = near.groupby("snapshot_ts")["spot"].first().reindex(snaps).to_numpy(dtype=float)

    n_snapshots = len(snaps)
    spot_std = float(np.std(spot)) if n_snapshots > 1 else 0.0
    n_strikes = int(near["strike"].nunique())

    # ATM presence: both CE and PE at the strike nearest spot, per snapshot
    strikes = np.sort(near["strike"].unique())
    have_both = 0
    crossed_at_atm = 0
    by_snap = {ts: g for ts, g in near.groupby("snapshot_ts")}
    for i, ts in enumerate(snaps):
        g = by_snap[ts]
        if not np.isfinite(spot[i]):
            continue
        k = strikes[np.argmin(np.abs(strikes - spot[i]))]
        at_k = g[g["strike"] == k]
        types = set(at_k["option_type"])
        if {"CE", "PE"}.issubset(types):
            have_both += 1
        two_sided = at_k[(at_k["bid"].fillna(0) > 0) & (at_k["ask"].fillna(0) > 0)]
        crossed_at_atm += int((two_sided["ask"] < two_sided["bid"]).sum())

    atm_pct = 100.0 * have_both / n_snapshots if n_snapshots else 0.0

    checks = {
        "n_snapshots": n_snapshots,
        "spot_std": round(spot_std, 4),
        "n_strikes": n_strikes,
        "atm_ce_pe_present_pct": round(atm_pct, 2),
        "crossed_quotes_at_atm": int(crossed_at_atm),
    }
    checks["pass_min_snapshots"] = n_snapshots >= scr["min_distinct_snapshots"]
    checks["pass_spot_std"] = spot_std > scr["min_spot_std_points"]
    checks["pass_min_strikes"] = n_strikes >= scr["min_distinct_strikes"]
    checks["pass_atm_presence"] = atm_pct >= scr["min_atm_ce_pe_present_pct"]
    checks["pass_no_crossed"] = crossed_at_atm <= scr["max_crossed_quotes_at_atm"]
    failed = [k for k in checks if k.startswith("pass_") and not checks[k]]
    checks["quality_screen_passed"] = not failed
    checks["quality_screen_failed_checks"] = ",".join(failed)
    return checks


def build_pairing(near: pd.DataFrame, d: date, horizons: list[int],
                   max_elapsed_s: float) -> pd.DataFrame:
    """t+1 pairing and the mandatory audit columns. No targets, no features."""
    snaps = pd.DatetimeIndex(np.sort(near["snapshot_ts"].unique()))
    n = len(snaps)
    gaps = np.full(n, np.nan)
    if n > 1:
        # Unit-safe: snapshot_ts is microsecond-resolution, so a raw
        # .view("int64")/1e9 would understate every gap by ~1000x and silently
        # neuter the 300s cap. total_seconds() is resolution-independent.
        gaps[1:] = pd.Series(snaps).diff().dt.total_seconds().to_numpy()[1:]

    close_cut = pd.Timestamp(f"{d.isoformat()} {SESSION_CLOSE}")
    rows = []
    for h in horizons:
        for i in range(n):
            t_ts = snaps[i]
            j = i + 1              # t+1 = next valid snapshot (never a nominal +2 min)
            k = i + 1 + h          # label window close
            if j >= n:
                rows.append(dict(trading_date=d, horizon_h=h, t_index=i,
                                 feature_timestamp=t_ts,
                                 prediction_start_timestamp=pd.NaT,
                                 prediction_end_timestamp=pd.NaT,
                                 elapsed_seconds_t_to_t1=np.nan,
                                 label_window_max_gap_seconds=np.nan,
                                 retained=False, exclusion_reason="no_next_snapshot"))
                continue
            elapsed = float(gaps[j])
            if k >= n:
                rows.append(dict(trading_date=d, horizon_h=h, t_index=i,
                                 feature_timestamp=t_ts,
                                 prediction_start_timestamp=snaps[j],
                                 prediction_end_timestamp=pd.NaT,
                                 elapsed_seconds_t_to_t1=elapsed,
                                 label_window_max_gap_seconds=np.nan,
                                 retained=False,
                                 exclusion_reason="label_window_exceeds_session_close"))
                continue

            win_max_gap = float(np.nanmax(gaps[j + 1:k + 1])) if k > j else 0.0
            reason = ""
            if snaps[k] > close_cut:
                reason = "label_window_exceeds_session_close"
            elif not np.isfinite(elapsed) or elapsed > max_elapsed_s:
                reason = "elapsed_t_to_t1_over_cap"
            elif np.isfinite(win_max_gap) and win_max_gap > max_elapsed_s:
                reason = "label_window_gap_over_cap"

            rows.append(dict(trading_date=d, horizon_h=h, t_index=i,
                             feature_timestamp=t_ts,
                             prediction_start_timestamp=snaps[j],
                             prediction_end_timestamp=snaps[k],
                             elapsed_seconds_t_to_t1=elapsed,
                             label_window_max_gap_seconds=win_max_gap,
                             retained=(reason == ""), exclusion_reason=reason))
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    horizons = sorted({v["horizon_snapshots"] for v in cfg["targets"]["primary"].values()})
    max_elapsed = float(cfg["t_plus_1_definition"]["max_elapsed_seconds_t_to_t1"])
    excluded = set(cfg["data"]["excluded_trading_days"].keys())

    print(f"EXP034 Phase 1 -- horizons {horizons}, t->t+1 cap {max_elapsed:.0f}s")
    print(f"Config: {CONFIG_PATH} (locked)")

    fingerprints = {p: _norm_sha256(Path(p)) for p in DGP_SOURCE_FILES}

    days = collected_days()
    cert_rows, obs_frames = [], []
    for i, d in enumerate(days, 1):
        raw = load_day_raw(d)
        if raw.empty:
            continue
        near, near_info = select_near_expiry(raw)
        cert = certify_dgp_v1(near, raw)
        scr = quality_screen(near, cfg)

        preregistered_exclusion = d.isoformat() in excluded
        usable = bool(cert["dgp_v1_certified"] and scr["quality_screen_passed"]
                       and not preregistered_exclusion)

        row = {"trading_date": d.isoformat(), **near_info, **cert, **scr,
               "preregistered_exclusion": preregistered_exclusion, "usable": usable}
        cert_rows.append(row)

        if usable:
            obs_frames.append(build_pairing(near, d, horizons, max_elapsed))
        status = "USABLE" if usable else "EXCLUDED"
        print(f"  [{i}/{len(days)}] {d} {status:8s} snaps={scr['n_snapshots']:4d} "
              f"spot_std={scr['spot_std']:8.2f} "
              f"{'' if usable else '<- ' + (cert['dgp_v1_failed_checks'] or scr['quality_screen_failed_checks'] or 'preregistered_exclusion')}")

    cert_df = pd.DataFrame(cert_rows)
    obs_df = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame()

    OUT_DERIVED.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    cert_df.to_parquet(OUT_DERIVED / "exp034_day_certification.parquet", index=False)
    obs_df.to_parquet(OUT_DERIVED / "exp034_observations.parquet", index=False)

    n_usable = int(cert_df["usable"].sum())
    ret = obs_df[obs_df["retained"]] if len(obs_df) else obs_df

    lines = [
        "# EXP034 Phase 1 -- Data Pipeline Audit",
        "",
        f"Locked config: `{CONFIG_PATH}`. Preflight gate must pass before any later phase.",
        "",
        "**Scope of this phase:** deterministic loading, dgp-v1 certification, per-day",
        "quality screen, near-expiry selection, t+1 pairing, audit columns.",
        "**No features, no target values, no models, no predictive inspection.**",
        "",
        "## Day certification",
        f"- Days on disk: **{len(cert_df)}**",
        f"- dgp-v1 certified: **{int(cert_df['dgp_v1_certified'].sum())}**",
        f"- Quality screen passed: **{int(cert_df['quality_screen_passed'].sum())}**",
        f"- Preregistered exclusions: **{int(cert_df['preregistered_exclusion'].sum())}**",
        f"- **Usable days: {n_usable}**",
        "",
        "### Excluded days",
    ]
    for _, r in cert_df[~cert_df["usable"]].iterrows():
        why = r["dgp_v1_failed_checks"] or r["quality_screen_failed_checks"] or "preregistered_exclusion"
        lines.append(f"- `{r['trading_date']}` -> {why} "
                     f"(snaps={r['n_snapshots']}, spot_std={r['spot_std']}, "
                     f"atm_pct={r['atm_ce_pe_present_pct']})")

    lines += ["", "## Observation scaffold (t+1 pairing)", ""]
    if len(obs_df):
        lines.append(f"- Candidate (day, t, h) rows: **{len(obs_df)}**")
        lines.append(f"- Retained: **{len(ret)}**  |  excluded: **{len(obs_df) - len(ret)}**")
        lines.append("")
        lines.append("### Exclusion reasons")
        vc = obs_df.loc[~obs_df["retained"], "exclusion_reason"].value_counts()
        for reason, cnt in vc.items():
            lines.append(f"- `{reason}`: {cnt}")
        lines.append("")
        lines.append("### Retained rows per horizon")
        for h, g in ret.groupby("horizon_h"):
            lines.append(f"- h={h}: {len(g)} rows across {g['trading_date'].nunique()} days")
        lines.append("")
        lines.append("### elapsed_seconds_t_to_t1 (retained rows)")
        e = ret["elapsed_seconds_t_to_t1"]
        lines.append(f"- min {e.min():.1f}s | median {e.median():.1f}s | "
                     f"p99 {e.quantile(0.99):.1f}s | max {e.max():.1f}s")
        lines.append("")
        lines.append("**Timing invariant:** "
                     f"prediction_start_timestamp > feature_timestamp on "
                     f"{int((pd.to_datetime(ret['prediction_start_timestamp']) > pd.to_datetime(ret['feature_timestamp'])).sum())}"
                     f"/{len(ret)} retained rows.")
    else:
        lines.append("- No observations built.")

    lines += [
        "",
        "## dgp-v1 source fingerprints (for forward-day certification)",
        "",
        "Recorded so any post-lock collector/provider change is detectable. Properties",
        "that live in the collector -- quote-before-spot ordering, the pre-stamped",
        "timestamp, the 30s spot cache, VIX-fetched-first -- are **not observable from",
        "stored rows** and cannot be verified retrospectively for historical days; these",
        "fingerprints are the operative control going forward.",
        "",
    ]
    for p, h in fingerprints.items():
        lines.append(f"- `{p}`: `{h}`")

    (OUT_REPORTS / "phase1_data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DERIVED / 'exp034_day_certification.parquet'} ({len(cert_df)} days)")
    print(f"Wrote {OUT_DERIVED / 'exp034_observations.parquet'} ({len(obs_df)} rows)")
    print(f"Wrote {OUT_REPORTS / 'phase1_data_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
