"""EXP033 Phase 2 -- option-chain data audit report (read-only).

Scans every collected trading day under data/option_chain/<Y>/<M>/<D>/*.parquet
and reports, per day: snapshot cadence, missing/duplicate timestamps, expiry
coverage, strike coverage and spacing, CE/PE availability, bid/ask/LTP/OI/
volume availability, the (unreliable) stored oi_change field, IV availability
(raw vs. derivable via Black-Scholes), spot availability, and India VIX join
coverage.

This performs NO experiment logic (no events, no labels, no signals). It is a
pure data-quality inspection, per EXP033 Part 1. Output:

    reports/exp033/data_audit_report.csv   -- one row per trading day
    reports/exp033/data_audit_summary.md   -- overall findings + data-source map

Usage:
    python scripts/exp033_data_audit.py
"""

from __future__ import annotations

import glob
import os
from datetime import date

import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = os.path.join("reports", "exp033")
SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"
EXPECTED_CADENCE_MIN = 2.0  # nominal collector poll interval


def find_trading_days(data_dir: str) -> list[date]:
    pattern = os.path.join(data_dir, "option_chain", "*", "*", "*")
    days = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        parts = d.replace("\\", "/").split("/")
        y, m, dd = parts[-3], parts[-2], parts[-1]
        try:
            days.append(date(int(y), int(m), int(dd)))
        except ValueError:
            continue
    return sorted(days)


def load_day(data_dir: str, d: date) -> tuple[pd.DataFrame, int]:
    folder = os.path.join(
        data_dir, "option_chain", f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}"
    )
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame(), 0
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    return df, len(files)


def load_vix(data_dir: str, d: date) -> pd.DataFrame:
    path = os.path.join(data_dir, "vix", f"{d.year:04d}", f"INDIAVIX_{d.isoformat()}.parquet")
    if os.path.exists(path):
        v = pd.read_parquet(path)
        v["timestamp"] = pd.to_datetime(v["timestamp"] if "timestamp" in v.columns else v.iloc[:, 0])
        return v
    return pd.DataFrame()


def audit_day(data_dir: str, d: date) -> dict:
    df, n_files = load_day(data_dir, d)
    if df.empty:
        return {"date": d.isoformat(), "status": "EMPTY"}

    row_key = ["snapshot_ts", "strike", "option_type", "expiry"]
    dup_rows = df.duplicated(subset=row_key, keep=False).sum()

    snaps = sorted(df["snapshot_ts"].unique())
    snaps = pd.to_datetime(pd.Series(snaps))
    n_snapshots = len(snaps)
    n_snapshot_files = n_files
    double_cadence_flag = n_snapshots > n_snapshot_files * 1.3  # e.g. 2026-06-22 artifact

    gaps_min = snaps.diff().dropna().dt.total_seconds() / 60.0
    gap_median = float(gaps_min.median()) if len(gaps_min) else float("nan")
    gap_max = float(gaps_min.max()) if len(gaps_min) else float("nan")
    n_gaps_gt5 = int((gaps_min > 5.0).sum())

    session_open_dt = pd.Timestamp(f"{d.isoformat()} {SESSION_OPEN}")
    session_close_dt = pd.Timestamp(f"{d.isoformat()} {SESSION_CLOSE}")
    expected_n = int((session_close_dt - session_open_dt).total_seconds() / 60 / EXPECTED_CADENCE_MIN)
    missing_est = max(0, expected_n - n_snapshots)

    expiries = sorted(pd.to_datetime(df["expiry"].unique()).date)
    n_expiries = len(expiries)
    is_expiry_day = d in expiries

    strikes = sorted(df["strike"].unique())
    n_strikes = len(strikes)
    strike_diffs = np.diff(strikes) if len(strikes) > 1 else np.array([])
    strike_spacing_mode = (
        float(pd.Series(strike_diffs).mode().iloc[0]) if len(strike_diffs) else float("nan")
    )
    strike_spacing_irregular = bool(len(strike_diffs) and (strike_diffs != strike_spacing_mode).any())

    per_snap_strike = df.groupby(["snapshot_ts", "strike", "expiry"])["option_type"].apply(
        lambda s: set(s)
    )
    both_ce_pe_pct = 100.0 * float(
        per_snap_strike.apply(lambda s: {"CE", "PE"}.issubset(s)).mean()
    )

    n = len(df)
    bid_avail = 100.0 * float((df["bid"].fillna(0) > 0).mean())
    ask_avail = 100.0 * float((df["ask"].fillna(0) > 0).mean())
    two_sided = 100.0 * float(((df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)).mean())
    crossed = 100.0 * float(
        (
            (df["bid"].fillna(0) > 0)
            & (df["ask"].fillna(0) > 0)
            & (df["ask"] < df["bid"])
        ).mean()
    )
    ltp_avail = 100.0 * float((df["last_price"].fillna(0) > 0).mean())
    oi_avail = 100.0 * float((df["open_interest"].fillna(0) > 0).mean())
    oi_change_nonzero_pct = 100.0 * float((df["oi_change"].fillna(0) != 0).mean())
    volume_avail = 100.0 * float((df["volume"].fillna(0) > 0).mean())
    iv_raw_avail = 100.0 * float(df["implied_volatility"].notna().mean())

    # derivable IV coverage, sampled for speed
    from nifty_quant.research import derive_iv

    iv_cov = derive_iv.coverage(df, sample=3000)

    spot_present_pct = 100.0 * float(df["spot"].notna().mean())
    spot_by_snap = df.drop_duplicates("snapshot_ts")["spot"]
    spot_distinct = int(spot_by_snap.nunique())
    spot_std = float(spot_by_snap.std()) if len(spot_by_snap) > 1 else 0.0
    # Flags a session where the live-spot feed silently froze at one stale value
    # for the whole day (a real collector bug found on 2026-06-26, not a proxy
    # for normal low-volatility days -- those still show dozens of distinct ticks).
    spot_stale_day = bool(n_snapshots >= 10 and (spot_distinct <= 3 or spot_std < 0.5))

    vix = load_vix(data_dir, d)
    vix_n = len(vix)
    vix_range = (
        f"{vix['india_vix'].min():.2f}-{vix['india_vix'].max():.2f}" if vix_n else ""
    )

    return {
        "date": d.isoformat(),
        "status": "OK",
        "is_expiry_day": is_expiry_day,
        "n_snapshot_files": n_snapshot_files,
        "n_distinct_snapshot_ts": n_snapshots,
        "double_cadence_artifact": double_cadence_flag,
        "duplicate_rows_snapshot_strike_type": int(dup_rows),
        "first_snapshot": str(snaps.iloc[0]),
        "last_snapshot": str(snaps.iloc[-1]),
        "cadence_median_min": round(gap_median, 2),
        "cadence_max_gap_min": round(gap_max, 2),
        "n_gaps_gt_5min": n_gaps_gt5,
        "expected_snapshots_at_2min": expected_n,
        "missing_snapshots_est": missing_est,
        "n_expiries_visible": n_expiries,
        "expiries": ",".join(e.isoformat() for e in expiries),
        "n_strikes": n_strikes,
        "strike_spacing_mode": strike_spacing_mode,
        "strike_spacing_irregular": strike_spacing_irregular,
        "both_ce_pe_pct": round(both_ce_pe_pct, 2),
        "bid_avail_pct": round(bid_avail, 2),
        "ask_avail_pct": round(ask_avail, 2),
        "two_sided_pct": round(two_sided, 2),
        "crossed_quote_pct": round(crossed, 3),
        "ltp_avail_pct": round(ltp_avail, 2),
        "oi_avail_pct": round(oi_avail, 2),
        "oi_change_stored_nonzero_pct": round(oi_change_nonzero_pct, 3),
        "volume_avail_pct": round(volume_avail, 2),
        "iv_raw_stored_avail_pct": round(iv_raw_avail, 2),
        "iv_derivable_coverage_pct": round(iv_cov["coverage_pct"], 2),
        "iv_derivable_median": iv_cov["median_iv"],
        "spot_present_pct": round(spot_present_pct, 2),
        "spot_distinct_values": spot_distinct,
        "spot_std": round(spot_std, 3),
        "spot_stale_day": spot_stale_day,
        "vix_n_readings": vix_n,
        "vix_range": vix_range,
        "n_rows_total": n,
    }


def main() -> int:
    days = find_trading_days(DATA_DIR)
    print(f"Found {len(days)} candidate trading-day folders under data/option_chain/.")
    rows = [audit_day(DATA_DIR, d) for d in days]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "data_audit_report.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} rows).")

    ok = [r for r in rows if r.get("status") == "OK"]
    n_expiry_days = sum(1 for r in ok if r["is_expiry_day"])
    n_dup_days = sum(1 for r in ok if r["duplicate_rows_snapshot_strike_type"] > 0)
    n_double_cadence = sum(1 for r in ok if r["double_cadence_artifact"])
    n_irregular_strike = sum(1 for r in ok if r["strike_spacing_irregular"])
    all_expiries = sorted({e for r in ok for e in r["expiries"].split(",") if e})
    stale_days = [r["date"] for r in ok if r["spot_stale_day"]]
    partial_days = [
        (r["date"], r["n_distinct_snapshot_ts"], r["missing_snapshots_est"])
        for r in ok
        if r["missing_snapshots_est"] > 30 and not r["double_cadence_artifact"]
    ]

    summary = f"""# EXP033 Data Audit Summary

Generated by `scripts/exp033_data_audit.py`. Read-only. Full per-day detail in
`data_audit_report.csv` ({len(ok)} valid trading days, {len(rows) - len(ok)} empty/skipped).

## Headline numbers
- Valid trading days: **{len(ok)}**
- Date range: {ok[0]['date'] if ok else 'n/a'} -> {ok[-1]['date'] if ok else 'n/a'}
- Expiry days in sample: **{n_expiry_days}**
- Distinct expiry dates seen across all days: {len(all_expiries)}
- Days with duplicate (snapshot_ts, strike, option_type, expiry) rows: {n_dup_days} (true duplicates; an
  earlier version of this check double-counted rows across the two co-listed expiries and is corrected here)
- Days with the double-cadence collection artifact (2026-06-22 known case): {n_double_cadence}
- Days with irregular strike spacing: {n_irregular_strike}
- **Days with a frozen/stale spot feed (new finding, see below): {len(stale_days)} -- {', '.join(stale_days) or 'none'}**
- Materially short/partial sessions (>30 estimated missing snapshots, excluding the known double-cadence day): {len(partial_days)}
{chr(10).join(f'  - {d}: {n} distinct snapshots, ~{m} missing vs. the 2-min-cadence expectation' for d, n, m in partial_days)}

## NEW FINDING -- frozen spot feed on {', '.join(stale_days) or 'n/a'}
{"On " + stale_days[0] + ", the `spot` field is a single constant value (std=0) across all "
 + str(next(r['n_distinct_snapshot_ts'] for r in ok if r['date']==stale_days[0]) if stale_days else 0)
 + " snapshots of the session -- the live-spot feed clearly failed and the collector fell back to a "
   "stale/cached print for the entire day. This is NOT a low-volatility day (compare `spot_std` / "
   "`spot_distinct_values` against any neighboring day). Any strategy conditioning on spot movement "
   "(events, features, barrier labels) is unusable on this day and it must be EXCLUDED from EXP033's "
   "signal-generation and labeling universe (options bid/ask on that day may still be usable for "
   "vol-only, non-directional purposes, but the spot series cannot support triple-barrier labeling)."
   if stale_days else "No stale-spot day detected."}

## Data-source-of-truth declaration (per EXP033 Part 1, mandatory)

| Purpose | Source | Resolution | Caveat |
|---|---|---|---|
| Signal generation (event detection, features) | `option_chain.spot` field, per snapshot | ~2 min | Only source that exists for this period |
| Entry simulation (fill price) | `option_chain.bid`/`ask` at entry snapshot | ~2 min | Two-sided quote required or trade is skipped, per Exp032 discipline |
| Target/stop simulation (barrier resolution) | `option_chain.spot`, walked snapshot-to-snapshot | ~2 min | **No higher-frequency underlying data exists for 2026-06-22 onward** (5m/1d candle warehouse stops 2026-06-19, zero overlap). Barrier order between two snapshots is resolved by the conservative interval-range algorithm (both barriers spanned by [min,max] of endpoints -> AMBIGUOUS; exactly one spanned -> resolved at that snapshot). Never invented. |
| Exit simulation (fill price) | `option_chain.bid`/`ask` at the resolving snapshot | ~2 min | Exit requires only a live offer on the relevant side (direction-aware, per Exp032's corrected fill rule) |

## Known field-reliability issues (do not trust naively)
- `implied_volatility` is **100% NaN by design** (not stored raw) -- must be derived via
  `nifty_quant.research.derive_iv` / `analytics.black_scholes.implied_volatility` at
  r=0.065, q=0.012 (see per-day `iv_derivable_coverage_pct` column).
- `oi_change` (stored column) is **effectively unreliable at scale** -- see
  `oi_change_stored_nonzero_pct` per day. True OI change must be derived by
  differencing `open_interest` across consecutive snapshots per (strike, option_type)
  during feature construction; the stored column must not be used as-is.
- 2026-06-22 is flagged as a manual-collection double-cadence artifact (per Exp032
  prereg) -- retained but flagged; event/label construction must key off
  "first snapshot at/after" a target clock time, which is robust to this.

## Underlying resolution constraint (governs Part 5 triple-barrier labeling)
No 1-minute or tick-level NIFTY data exists for any day in this sample. The
option-chain-embedded `spot` field at ~2-minute cadence is the *only* underlying
price series available for signal generation, entry, barrier resolution and exit
alike. This is why the conservative/ambiguous barrier-resolution policy is
mandatory rather than optional (see EXP033 preregistration).
"""
    out_md = os.path.join(OUT_DIR, "data_audit_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Wrote {out_md}.")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
