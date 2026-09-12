"""EXP034 feature construction -- exact implementations of the locked registry.

Every formula here is transcribed from `docs/preregistrations/exp034_config.json`
(`feature_registry`), locked at commit 18a178e. Nothing is invented here and
nothing may be added: the preflight gate enforces that the realized matrices
match the locked column counts exactly.

All computation is causal: a value at snapshot t uses only information at or
before t. Cross-day time-of-day baselines use strictly PRIOR days.

This module is EXP034-owned. EXP033's `option_event_features.py` is in the
EXP033 freeze manifest and is neither imported nor modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_quant.analytics import black_scholes as bs

EPS = 1e-9
RELATIVE_STRIKES = (-2, -1, 0, 1, 2)
IV_LEVELS = (-1, 0, 1)          # locked: IV restricted to ATM+-1
STRIKE_STEP = 50.0              # verified constant across all collected days
R_RATE = 0.065
Q_YIELD = 0.012
OR_END = "09:30"                # opening-range window close
TOD_BUCKET_MIN = 30
TOD_N_BUCKETS = 13              # 09:15-15:29 in 30-min buckets
SESSION_OPEN = "09:15"


# --------------------------------------------------------------------------
# per-snapshot matrices
# --------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, otype: str, value: str,
           strikes: np.ndarray, times: np.ndarray) -> np.ndarray:
    sub = df[df["option_type"] == otype]
    p = sub.pivot_table(index="snapshot_ts", columns="strike", values=value,
                        aggfunc="last")
    return p.reindex(index=times, columns=strikes).to_numpy(dtype=float)


def _mid(bid: np.ndarray, ask: np.ndarray, last: np.ndarray) -> np.ndarray:
    """(bid+ask)/2 when both positive, else last_price -- the locked rule."""
    return np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)


def _years_to_expiry(times: pd.DatetimeIndex, expiry_date) -> np.ndarray:
    close = pd.Timestamp(f"{expiry_date} 15:30")
    secs = (close - times).total_seconds().to_numpy(dtype=float)
    return np.maximum(secs, 0.0) / (365.0 * 86400.0)


def _solve_iv(price: np.ndarray, spot: np.ndarray, strike: np.ndarray,
              t_years: np.ndarray, otype: str) -> np.ndarray:
    out = np.full(len(price), np.nan)
    for i in range(len(price)):
        p, s, k, t = price[i], spot[i], strike[i], t_years[i]
        if not (np.isfinite(p) and p > 0 and s > 0 and k > 0 and t > 0):
            continue
        v = bs.implied_volatility(p, s, float(k), t, R_RATE, otype, Q_YIELD)
        if v is not None and v > 0:
            out[i] = v
    return out


# --------------------------------------------------------------------------
# main per-day builder
# --------------------------------------------------------------------------

def build_day_features(near: pd.DataFrame, d, expiry_date,
                       vix: pd.DataFrame, tod_baseline: dict) -> pd.DataFrame:
    """All MODEL_1/2/3 feature columns for one trading day's near-expiry chain.

    `tod_baseline` maps bucket index -> array of TV values observed on STRICTLY
    PRIOR days (see build_all in the orchestrator); it is read, never written,
    here.
    """
    times = pd.DatetimeIndex(np.sort(near["snapshot_ts"].unique()))
    strikes = np.sort(near["strike"].unique())
    n = len(times)
    spot = near.groupby("snapshot_ts")["spot"].first().reindex(times).to_numpy(dtype=float)

    atm_idx = np.argmin(np.abs(strikes[None, :] - spot[:, None]), axis=1)
    atm_strike = strikes[atm_idx]
    rows = np.arange(n)

    c_oi = np.nan_to_num(_pivot(near, "CE", "open_interest", strikes, times))
    p_oi = np.nan_to_num(_pivot(near, "PE", "open_interest", strikes, times))
    c_vol = np.nan_to_num(_pivot(near, "CE", "volume", strikes, times))
    p_vol = np.nan_to_num(_pivot(near, "PE", "volume", strikes, times))
    c_bid, c_ask = _pivot(near, "CE", "bid", strikes, times), _pivot(near, "CE", "ask", strikes, times)
    p_bid, p_ask = _pivot(near, "PE", "bid", strikes, times), _pivot(near, "PE", "ask", strikes, times)
    c_last, p_last = _pivot(near, "CE", "last_price", strikes, times), _pivot(near, "PE", "last_price", strikes, times)
    c_mid, p_mid = _mid(c_bid, c_ask, c_last), _mid(p_bid, p_ask, p_last)

    # per-interval volume: first difference of the CUMULATIVE stored field,
    # clipped at 0 to guard a provider counter reset / missing reading.
    dv_c = np.vstack([np.zeros((1, len(strikes))), np.maximum(0.0, np.diff(c_vol, axis=0))])
    dv_p = np.vstack([np.zeros((1, len(strikes))), np.maximum(0.0, np.diff(p_vol, axis=0))])
    doi_c = np.vstack([np.zeros((1, len(strikes))), np.diff(c_oi, axis=0)])
    doi_p = np.vstack([np.zeros((1, len(strikes))), np.diff(p_oi, axis=0)])

    def lvl(offset: int) -> np.ndarray:
        return np.clip(atm_idx + offset, 0, len(strikes) - 1)

    idx = {k: lvl(k) for k in RELATIVE_STRIKES}

    def gather(mat: np.ndarray, ks) -> np.ndarray:
        return np.sum(np.stack([mat[rows, idx[k]] for k in ks], axis=0), axis=0)

    out: dict[str, np.ndarray] = {}

    # ---- G1 flow ---------------------------------------------------------
    ce1, pe1 = gather(dv_c, (-1, 0, 1)), gather(dv_p, (-1, 0, 1))
    out["ce_pe_volume_imbalance_atm1"] = (ce1 - pe1) / (ce1 + pe1 + EPS)

    TV = gather(dv_c, RELATIVE_STRIKES) + gather(dv_p, RELATIVE_STRIKES)
    tv_s = pd.Series(TV)
    prior5 = tv_s.shift(1).rolling(5, min_periods=5).mean().to_numpy()
    out["volume_acceleration"] = TV / (prior5 + EPS) - 1.0

    tod_bucket = _tod_bucket(times)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    for i in range(n):
        hist = tod_baseline.get(int(tod_bucket[i]))
        if hist is not None and len(hist) > 0:
            mu[i], sd[i] = np.mean(hist), np.std(hist)
    out["volume_anomaly_tod"] = (TV - mu) / (sd + EPS)

    # ---- G2 positioning --------------------------------------------------
    dce2, dpe2 = gather(doi_c, RELATIVE_STRIKES), gather(doi_p, RELATIVE_STRIKES)
    abs2 = (np.sum(np.stack([np.abs(doi_c[rows, idx[k]]) for k in RELATIVE_STRIKES]), axis=0)
            + np.sum(np.stack([np.abs(doi_p[rows, idx[k]]) for k in RELATIVE_STRIKES]), axis=0))
    out["net_oi_change_imbalance_atm2"] = (dce2 - dpe2) / (abs2 + EPS)

    total_oi = c_oi + p_oi
    atm1_oi = gather(c_oi, (-1, 0, 1)) + gather(p_oi, (-1, 0, 1))
    grand = total_oi.sum(axis=1)
    out["oi_concentration_atm1"] = np.where(grand > 0, atm1_oi / (grand + EPS), np.nan)

    s_star = strikes[np.argmax(total_oi, axis=1)]
    out["oi_migration"] = np.concatenate([[np.nan], np.diff(s_star) / STRIKE_STEP])

    call_tot, put_tot = c_oi.sum(axis=1), p_oi.sum(axis=1)
    pcr = np.where(call_tot > 0, put_tot / (call_tot + EPS), np.nan)
    out["pcr_oi_change"] = np.concatenate([[np.nan], np.diff(pcr)])

    # ---- G3 premium (FIXED-STRIKE: the strike ATM at t, read at t and t-1) --
    atm_c_now = c_mid[rows, atm_idx]
    atm_p_now = p_mid[rows, atm_idx]
    atm_c_prev = np.full(n, np.nan)
    atm_p_prev = np.full(n, np.nan)
    atm_c_prev[1:] = c_mid[rows[:-1], atm_idx[1:]]   # strike of t, evaluated at t-1
    atm_p_prev[1:] = p_mid[rows[:-1], atm_idx[1:]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_ce = np.where(atm_c_prev > 0, atm_c_now / atm_c_prev - 1.0, np.nan)
        r_pe = np.where(atm_p_prev > 0, atm_p_now / atm_p_prev - 1.0, np.nan)
    out["ce_pe_premium_momentum_diff"] = r_ce - r_pe

    straddle_now = atm_c_now + atm_p_now
    straddle_prev = atm_c_prev + atm_p_prev
    with np.errstate(divide="ignore", invalid="ignore"):
        out["atm_straddle_return"] = np.where(
            straddle_prev > 0, straddle_now / straddle_prev - 1.0, np.nan)

    # ---- G4 vol surface (IV restricted to ATM+-1) ------------------------
    t_years = _years_to_expiry(times, expiry_date)
    iv = {}
    for k in IV_LEVELS:
        ks = strikes[idx[k]]
        iv[("CE", k)] = _solve_iv(c_mid[rows, idx[k]], spot, ks, t_years, "CE")
        iv[("PE", k)] = _solve_iv(p_mid[rows, idx[k]], spot, ks, t_years, "PE")

    atm_iv = np.nanmean(np.vstack([iv[("CE", 0)], iv[("PE", 0)]]), axis=0)
    out["atm_iv"] = atm_iv
    out["atm_iv_change"] = np.concatenate([[np.nan], np.diff(atm_iv)])
    out["iv_skew_ce_minus_pe"] = iv[("CE", 0)] - iv[("PE", 0)]
    up = np.nanmean(np.vstack([iv[("CE", 1)], iv[("PE", 1)]]), axis=0)
    dn = np.nanmean(np.vstack([iv[("CE", -1)], iv[("PE", -1)]]), axis=0)
    out["iv_skew_slope"] = (up - dn) / 2.0

    # ---- G5 structure ----------------------------------------------------
    above = strikes[None, :] > spot[:, None]
    below = strikes[None, :] < spot[:, None]
    k_call = np.where(above, c_oi, -np.inf).argmax(axis=1)
    k_put = np.where(below, p_oi, -np.inf).argmax(axis=1)
    has_above, has_below = above.any(axis=1), below.any(axis=1)
    res_strike = np.where(has_above, strikes[k_call], np.nan)
    sup_strike = np.where(has_below, strikes[k_put], np.nan)
    out["dist_to_max_call_oi_norm"] = (res_strike - spot) / (straddle_now + EPS)
    out["dist_to_max_put_oi_norm"] = (spot - sup_strike) / (straddle_now + EPS)

    # ---- auxiliary indicator --------------------------------------------
    iv_ok = (np.isfinite(out["atm_iv"]) & np.isfinite(out["atm_iv_change"])
             & np.isfinite(out["iv_skew_ce_minus_pe"]) & np.isfinite(out["iv_skew_slope"]))
    out["iv_available"] = iv_ok.astype(float)

    # ---- MODEL_1 price/time ---------------------------------------------
    sp = pd.Series(spot)
    for lag in (1, 2, 5, 10):
        out[f"ret_{lag}"] = sp.pct_change(lag).to_numpy()
    r1 = pd.Series(out["ret_1"])
    out["rv_intraday_expanding"] = r1.shift(1).expanding(min_periods=10).std().to_numpy()
    out["rv_20"] = r1.shift(1).rolling(20, min_periods=10).std().to_numpy()

    twap = sp.shift(1).expanding(min_periods=1).mean().to_numpy()
    out["twap_dist_pct"] = np.where(twap > 0, (spot - twap) / twap * 100.0, np.nan)
    hi = sp.shift(1).expanding(min_periods=1).max().to_numpy()
    lo = sp.shift(1).expanding(min_periods=1).min().to_numpy()
    out["dist_from_high_pct"] = np.where(hi > 0, (spot - hi) / hi * 100.0, np.nan)
    out["dist_from_low_pct"] = np.where(lo > 0, (spot - lo) / lo * 100.0, np.nan)

    or_state, or_forming = _opening_range(spot, times)
    out["or_state"] = or_state
    out["or_forming_flag"] = or_forming

    for b in range(1, TOD_N_BUCKETS):          # bucket 0 is the reference
        out[f"tod_bucket_dummy_{b + 1:02d}"] = (tod_bucket == b).astype(float)
    dow = pd.Timestamp(d).dayofweek
    for i, name in enumerate(["dow_tue", "dow_wed", "dow_thu", "dow_fri"], start=1):
        out[name] = np.full(n, 1.0 if dow == i else 0.0)

    out["dte_cal"] = np.full(n, float((pd.Timestamp(expiry_date).date() - d).days))
    out["is_expiry_day"] = np.full(n, 1.0 if pd.Timestamp(expiry_date).date() == d else 0.0)
    out["atm_rolled"] = np.concatenate([[0.0], (np.diff(atm_strike) != 0).astype(float)])

    # ---- MODEL_2 VIX -----------------------------------------------------
    out.update(_vix_features(times, vix, n))

    frame = pd.DataFrame(out)
    frame.insert(0, "feature_timestamp", times)
    frame.insert(1, "trading_date", d)
    frame["_tv"] = TV
    frame["_tod_bucket"] = tod_bucket
    return frame


def _tod_bucket(times: pd.DatetimeIndex) -> np.ndarray:
    open_min = 9 * 60 + 15
    mins = times.hour * 60 + times.minute - open_min
    return np.clip(mins // TOD_BUCKET_MIN, 0, TOD_N_BUCKETS - 1).to_numpy()


def _opening_range(spot: np.ndarray, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    tod = times.strftime("%H:%M")
    in_or = tod <= OR_END
    state = np.zeros(len(spot))
    forming = np.ones(len(spot))
    if in_or.any() and (~in_or).any():
        hi, lo = np.nanmax(spot[in_or]), np.nanmin(spot[in_or])
        after = np.argmax(~in_or)
        forming[after:] = 0.0
        seg = spot[after:]
        s = np.zeros(len(seg))
        s[seg > hi] = 1.0
        s[seg < lo] = -1.0
        state[after:] = s
    return state, forming


def _vix_features(times: pd.DatetimeIndex, vix: pd.DataFrame, n: int) -> dict:
    if vix is None or len(vix) == 0:
        return {k: np.full(n, np.nan) for k in
                ["vix", "vix_chg", "vix_mom", "vix_accel", "vix_pctile_causal"]}
    merged = pd.merge_asof(
        pd.DataFrame({"feature_timestamp": times}), vix,
        left_on="feature_timestamp", right_on="timestamp", direction="backward")
    lvl = merged["india_vix"].to_numpy(dtype=float)
    s = pd.Series(lvl)
    chg = s.diff()
    pct = np.full(n, np.nan)
    for i in range(n):
        hist = lvl[:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 5 and np.isfinite(lvl[i]):
            pct[i] = 100.0 * float((hist <= lvl[i]).mean())
    return {
        "vix": lvl,
        "vix_chg": chg.to_numpy(),
        "vix_mom": chg.shift(1).rolling(5, min_periods=3).mean().to_numpy(),
        "vix_accel": chg.diff().to_numpy(),
        "vix_pctile_causal": pct,
    }
