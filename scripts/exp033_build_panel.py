"""EXP033 Phase 4 -- normalized relative-strike feature panel builder.

Builds one row per option-chain snapshot, per trading day, with:
- relative-strike (ATM-2..ATM+2) x {CE,PE} raw + derived quote/OI/volume/IV
  fields and their same-day causal rolling statistics (feeds E1-E5 triggers,
  Part 3 of the EXP033 preregistration),
- spot/VIX/time context features (Part 4),
- nothing beyond the entry timestamp is ever read to compute a row.

Locked parameters (window sizes, min_periods, relative-strike band, IV
pricing constants) are loaded from
`docs/preregistrations/exp033_event_definitions.json`, not re-declared here,
so the code and the pre-registration cannot silently drift apart.

Output: data/derived/exp033_feature_panel.parquet (wide, one row/snapshot).
This is a cached, rebuildable artifact -- rerun with --rebuild after any
locked-file change (which itself would require a new experiment id per the
project's freeze discipline; this script does not enforce that, the human
process around docs/preregistrations does).

    python scripts/exp033_build_panel.py [--rebuild] [--limit-days N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_quant.analytics import black_scholes as bs
from nifty_quant.features import option_event_features as oef

DATA_DIR = "data"
PREREG_JSON = Path("docs") / "preregistrations" / "exp033_event_definitions.json"
PANEL_PATH = Path("data") / "derived" / "exp033_feature_panel.parquet"

with open(PREREG_JSON, "r", encoding="utf-8") as _f:
    LOCKED = json.load(_f)

RELATIVE_STRIKES = LOCKED["relative_strikes"]
OPTION_TYPES = tuple(LOCKED["option_types"])
MIN_PRIOR = LOCKED["min_prior_snapshots_same_day"]
E1_WINDOW = LOCKED["families"]["E1_VOLUME_SPIKE"]["window_snapshots"]
E2_WINDOW = LOCKED["families"]["E2_OI_EXPANSION"]["window_snapshots"]
E4_WINDOW = LOCKED["families"]["E4_PREMIUM_MOMENTUM"]["window_snapshots"]
E5_WINDOW = LOCKED["families"]["E5_IV_SHOCK"]["window_snapshots"]
R_RATE = LOCKED["families"]["E5_IV_SHOCK"]["iv_defaults"]["r_rate"]
Q_YIELD = LOCKED["families"]["E5_IV_SHOCK"]["iv_defaults"]["q_yield"]
EXCLUDED_DAYS = set(LOCKED["excluded_trading_days"].keys())


def collected_days(data_dir: str = DATA_DIR) -> list[date]:
    days = []
    for d in sorted(glob.glob(os.path.join(data_dir, "option_chain", "*", "*", "*"))):
        if not os.path.isdir(d):
            continue
        y, m, dd = d.replace("\\", "/").split("/")[-3:]
        try:
            iso = date(int(y), int(m), int(dd)).isoformat()
        except ValueError:
            continue
        if iso in EXCLUDED_DAYS:
            continue
        days.append(date(int(y), int(m), int(dd)))
    return sorted(days)


def load_vix_day(d: date, data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, "vix", f"{d.year:04d}", f"INDIAVIX_{d.isoformat()}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "india_vix"])
    v = pd.read_parquet(path)
    v["timestamp"] = pd.to_datetime(v["timestamp"]).dt.tz_localize(None)
    return v.sort_values("timestamp")


def build_day_panel(d: date, data_dir: str = DATA_DIR) -> pd.DataFrame | None:
    folder = os.path.join(data_dir, "option_chain", f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    near = df["expiry"].min()
    df = df[df["expiry"] == near].copy()
    df = df.drop_duplicates(subset=["snapshot_ts", "strike", "option_type"], keep="last")
    if df.empty:
        return None

    times = np.sort(df["snapshot_ts"].unique())
    strikes = np.sort(df["strike"].unique())
    if len(times) < MIN_PRIOR + 2 or len(strikes) < 5:
        return None
    n_t = len(times)

    spot = df.groupby("snapshot_ts")["spot"].last().reindex(times).to_numpy(dtype=float)
    atm_idx = np.argmin(np.abs(strikes[None, :] - spot[:, None]), axis=1)
    atm_strike = strikes[atm_idx]
    rows = np.arange(n_t)

    # full-band OI (for PCR / OI concentration / OI-wall search over ALL collected strikes)
    c_oi_full = np.nan_to_num(oef.pivot_field(df, "CE", "open_interest", strikes, times))
    p_oi_full = np.nan_to_num(oef.pivot_field(df, "PE", "open_interest", strikes, times))
    c_vol_full = np.nan_to_num(oef.pivot_field(df, "CE", "volume", strikes, times))
    p_vol_full = np.nan_to_num(oef.pivot_field(df, "PE", "volume", strikes, times))

    call_oi_tot = c_oi_full.sum(axis=1)
    put_oi_tot = p_oi_full.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pcr_oi = np.where(call_oi_tot > 0, put_oi_tot / call_oi_tot, np.nan)
    pcr_oi_change = np.diff(pcr_oi, prepend=pcr_oi[0] if len(pcr_oi) else np.nan)
    total_oi_full = c_oi_full + p_oi_full
    top3 = -np.sort(-total_oi_full, axis=1)[:, :3].sum(axis=1)
    grand_total = total_oi_full.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        oi_concentration = np.where(grand_total > 0, top3 / grand_total, np.nan)

    resistance_strike = np.full(n_t, np.nan)
    resistance_dist = np.full(n_t, np.nan)
    support_strike = np.full(n_t, np.nan)
    support_dist = np.full(n_t, np.nan)
    for i in range(n_t):
        r_s, r_d, s_s, s_d = oef.nearest_oi_wall(strikes, c_oi_full[i], p_oi_full[i], spot[i])
        resistance_strike[i], resistance_dist[i] = r_s, r_d
        support_strike[i], support_dist[i] = s_s, s_d

    # --- spot/context features ---------------------------------------------
    spot_s = pd.Series(spot)
    spot_ret_2m = spot_s.pct_change(1).to_numpy()
    spot_ret_4m = spot_s.pct_change(2).to_numpy()
    spot_ret_10m = spot_s.pct_change(5).to_numpy()
    spot_rv_intraday = pd.Series(spot_ret_2m).shift(1).expanding(min_periods=MIN_PRIOR).std().to_numpy()
    twap_so_far = spot_s.shift(1).expanding(min_periods=1).mean().to_numpy()
    twap_dist_pct = np.where(twap_so_far > 0, (spot - twap_so_far) / twap_so_far * 100.0, np.nan)
    high_so_far = spot_s.shift(1).expanding(min_periods=1).max().to_numpy()
    low_so_far = spot_s.shift(1).expanding(min_periods=1).min().to_numpy()
    dist_from_high_pct = np.where(high_so_far > 0, (spot - high_so_far) / high_so_far * 100.0, np.nan)
    dist_from_low_pct = np.where(low_so_far > 0, (spot - low_so_far) / low_so_far * 100.0, np.nan)
    or_high, or_low, or_state = oef.opening_range_state(spot, times)

    tod = pd.DatetimeIndex(times).strftime("%H:%M")
    minute_of_day = pd.DatetimeIndex(times).hour * 60 + pd.DatetimeIndex(times).minute
    time_bucket_30m = (minute_of_day // 30) * 30
    day_of_week = pd.Timestamp(d).dayofweek
    exp_date = pd.Timestamp(near).date()
    dte_cal = (exp_date - d).days
    is_expiry_day = exp_date == d

    # --- VIX (asof, backward-only join) -------------------------------------
    vix_df = load_vix_day(d)
    if len(vix_df):
        vix_level = pd.merge_asof(
            pd.DataFrame({"snapshot_ts": times}), vix_df,
            left_on="snapshot_ts", right_on="timestamp", direction="backward",
        )["india_vix"].to_numpy(dtype=float)
    else:
        vix_level = np.full(n_t, np.nan)
    vix_s = pd.Series(vix_level)
    vix_chg = vix_s.diff().to_numpy()
    vix_mom = vix_s.diff().shift(1).rolling(5, min_periods=3).mean().to_numpy()
    vix_accel = pd.Series(vix_chg).diff().to_numpy()
    vix_pctile = oef.same_day_causal_percentile(vix_level, window=n_t, min_periods=5)

    out = {
        "date": d, "ts": pd.DatetimeIndex(times), "tod": tod,
        "expiry": exp_date, "dte_cal": dte_cal, "is_expiry_day": is_expiry_day,
        "minute_of_day": minute_of_day, "time_bucket_30m": time_bucket_30m,
        "day_of_week": day_of_week,
        "spot": spot, "atm_strike": atm_strike.astype(float),
        "spot_ret_2m": spot_ret_2m, "spot_ret_4m": spot_ret_4m, "spot_ret_10m": spot_ret_10m,
        "spot_rv_intraday": spot_rv_intraday,
        "twap_dist_pct": twap_dist_pct,
        "dist_from_high_pct": dist_from_high_pct, "dist_from_low_pct": dist_from_low_pct,
        "or_high": or_high, "or_low": or_low, "or_state": or_state,
        "pcr_oi": pcr_oi, "pcr_oi_change": pcr_oi_change,
        "oi_concentration": oi_concentration,
        "resistance_strike": resistance_strike, "resistance_dist": resistance_dist,
        "support_strike": support_strike, "support_dist": support_dist,
        "vix": vix_level, "vix_chg": vix_chg, "vix_mom": vix_mom,
        "vix_accel": vix_accel, "vix_pctile_so_far": vix_pctile,
        "n_strikes": len(strikes),
    }

    close_dt = datetime.combine(exp_date, pd.Timestamp("15:30").time())
    t_years = np.array([
        max((close_dt - pd.Timestamp(ts).to_pydatetime()).total_seconds(), 0.0) / (365.0 * 86400.0)
        for ts in times
    ])

    for level in RELATIVE_STRIKES:
        idx = oef.relative_strike_indices(strikes, atm_idx, level)
        clipped = (idx == 0) | (idx == len(strikes) - 1)
        level_strike = strikes[idx]
        for otype in OPTION_TYPES:
            bid = oef.pivot_field(df, otype, "bid", strikes, times)[rows, idx]
            ask = oef.pivot_field(df, otype, "ask", strikes, times)[rows, idx]
            last = oef.pivot_field(df, otype, "last_price", strikes, times)[rows, idx]
            oi = np.nan_to_num(oef.pivot_field(df, otype, "open_interest", strikes, times))[rows, idx]
            vol = np.nan_to_num(oef.pivot_field(df, otype, "volume", strikes, times))[rows, idx]
            mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)

            oi_chg = oef.derived_oi_change(oi)
            mid_ret_2m = pd.Series(mid).pct_change(1).to_numpy()

            price_for_iv = np.where((bid > 0) & (ask > 0), mid, last)
            # per-row strike varies with `level` (relative-strike band tracks ATM
            # as spot moves), so IV must be solved row-by-row against level_strike[i]
            # rather than via the fixed-strike _iv_series helper.
            iv = np.full(n_t, np.nan)
            for i in range(n_t):
                p, s, t = price_for_iv[i], spot[i], t_years[i]
                if p <= 0 or s <= 0 or t <= 0:
                    continue
                v = bs.implied_volatility(p, s, float(level_strike[i]), t, R_RATE, otype, Q_YIELD)
                if v is not None and v > 0:
                    iv[i] = v

            vol_z = oef.same_day_causal_zscore(vol, window=E1_WINDOW, min_periods=MIN_PRIOR)
            oi_chg_z = oef.same_day_causal_zscore(oi_chg, window=E2_WINDOW, min_periods=MIN_PRIOR)
            mom_pctile = oef.same_day_causal_percentile(mid_ret_2m, window=E4_WINDOW, min_periods=MIN_PRIOR)
            iv_z = oef.same_day_causal_zscore(iv, window=E5_WINDOW, min_periods=MIN_PRIOR)

            prefix = f"L{level:+d}_{otype}"
            out[f"{prefix}_strike"] = level_strike
            out[f"{prefix}_clipped"] = clipped
            out[f"{prefix}_bid"] = bid
            out[f"{prefix}_ask"] = ask
            out[f"{prefix}_mid"] = mid
            out[f"{prefix}_oi"] = oi
            out[f"{prefix}_oi_chg"] = oi_chg
            out[f"{prefix}_volume"] = vol
            out[f"{prefix}_iv"] = iv
            out[f"{prefix}_vol_z"] = vol_z
            out[f"{prefix}_oi_chg_z"] = oi_chg_z
            out[f"{prefix}_mom_pctile"] = mom_pctile
            out[f"{prefix}_iv_z"] = iv_z

    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=None)
    args = ap.parse_args()

    days = collected_days()
    if args.limit_days:
        days = days[: args.limit_days]
    print(f"Building EXP033 feature panel over {len(days)} days "
          f"({EXCLUDED_DAYS} excluded per preregistration) ...")

    frames = []
    for i, d in enumerate(days, 1):
        panel = build_day_panel(d)
        if panel is None:
            print(f"  [{i}/{len(days)}] {d} -- SKIPPED (insufficient snapshots/strikes)")
            continue
        frames.append(panel)
        print(f"  [{i}/{len(days)}] {d} -- {len(panel)} rows")

    full = pd.concat(frames, ignore_index=True)
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(PANEL_PATH, engine="pyarrow", index=False)
    print(f"\nWrote {PANEL_PATH}: {len(full)} rows, {len(full.columns)} columns, "
          f"{full['date'].nunique()} days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
