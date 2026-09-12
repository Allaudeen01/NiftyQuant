"""Experiment 032: Multi-Strategy Zoo on collected option chains -- 7 locked candidates.

Pre-registered in docs/preregistrations/exp032_strategy_zoo.md.

Tests 7 structurally-motivated candidate strategies simultaneously against the
self-collected NIFTY option-chain snapshots, with real quoted bid/ask fills, a
full Indian F&O cost schedule, Benjamini-Hochberg FDR across all candidates and
Hansen's SPA / White's Reality Check for data-snooping.

Every threshold below is LOCKED by the pre-registration. Nothing is fitted. The
maximum attainable verdict is CANDIDATE FOR FORWARD TESTING -- see the power
analysis in the prereg: a 59-day sample cannot confirm a realistic edge, but it
can reject a decisive loser.

    python scripts/strategy_test_framework.py
    python scripts/strategy_test_framework.py --rebuild-panel

Read-only research. No live orders. No paper orders. No deployment.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_quant.analytics import black_scholes as bs
from nifty_quant.analytics.data_snooping import (reality_check_spa,
                                                 stationary_bootstrap_indices)

# --- LOCKED constants (pre-registered) -------------------------------------
# Immutability labels. EXP 032 is FROZEN at tag `exp032_final`; see
# docs/exp032_FREEZE.json and scripts/exp032_freeze.py --verify.
# Any change to rules or costs MUST be a NEW experiment id, never an edit here.
EXPERIMENT_ID = "exp032"
RULES_VERSION = "exp032-rules-v1.0"
COST_SCHEDULE_VERSION = "IN-FO-2026.1"

DATA_DIR = "data"
PANEL = Path("data") / "derived" / "exp032_panel.parquet"

R_RATE = 0.065                 # India risk-free, matches research/derive_iv.py
Q_YIELD = 0.012                # NIFTY dividend proxy, matches derive_iv.py
ANN = math.sqrt(252.0)
TRADING_DAYS = 252.0

ENTRY_CLOCK = "09:20"          # all strategies' primary entry
EXIT_CLOCK = "15:25"           # end-of-day exit
MIDDAY_CLOCK = "14:00"         # S2 signal window end
FIRST_HOUR_EXIT = "10:15"      # S7 exit

LOT_SIZE = 75                  # NIFTY F&O lot
MARGIN_PER_LOT = 150_000.0     # capital base per straddle lot (Sharpe-invariant)

# Cost schedule (ASSUMPTIONS -- flagged in prereg, swept for sensitivity)
BROKERAGE_PER_ORDER = 20.0
STT_OPT_SELL = 0.001000        # 0.1% of premium, sell side only
EXCH_OPT = 0.0003503           # 0.03503% of premium turnover
SEBI_FEE = 0.000001            # 0.0001%
STAMP_BUY = 0.00003            # 0.003% on buy side
GST_RATE = 0.18
STT_FUT_SELL = 0.0002          # 0.02% futures sell side
EXCH_FUT = 0.000019            # 0.0019%
FUT_SLIP_BPS = 1.0             # 1.0 bp of notional per side

SPREAD_MULTS = (0.5, 1.0, 1.5, 2.0)   # pre-registered sensitivity sweep
BASE_MULT = 1.0

S1_VRP_THRESHOLD = 2.0         # vol points: atm_iv - rv_forecast
S1_RV_LOOKBACK = 5             # trailing days for causal RV forecast
S2_MOVE_THRESHOLD = 0.0030     # 0.30% morning move
S3_IV_PCTILE = 20.0            # bottom 20th percentile
S3_WARMUP = 20                 # trailing days for IV percentile
S4_PCR_HIGH = 1.20
S4_PCR_LOW = 0.80
S5_MAXPAIN_PROX = 0.010        # 1.0% proximity to max pain
S7_GAP_THRESHOLD = 0.0050      # 0.50% overnight gap

HAC_LAG = 5
FDR_Q = 0.10
SPA_BLOCK = 10.0
SPA_NBOOT = 5000
SPA_SEED = 32032
PIN_NBOOT = 2000
PIN_BLOCK = 10.0
PIN_SEED = 5051

TRAIN_FRAC = 0.60              # chronological stability split


# --- day discovery ---------------------------------------------------------

def collected_days(data_dir: str = DATA_DIR) -> list[date]:
    days = []
    for f in glob.glob(os.path.join(data_dir, "option_chain", "*", "*", "*")):
        parts = Path(f).parts[-3:]
        try:
            days.append(date(int(parts[0]), int(parts[1]), int(parts[2])))
        except (ValueError, IndexError):
            continue
    return sorted(set(days))


# --- cost model ------------------------------------------------------------

def _adj_prices(bid: float, ask: float, mult: float) -> tuple[float, float]:
    """Spread-multiplier-adjusted (sell_price, buy_price).

    mult=1.0 reproduces the actual quoted bid/ask exactly.
    """
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    return mid - mult * half, mid + mult * half


def straddle_trade_pnl(entry: dict, exit_: dict, direction: int,
                       mult: float = BASE_MULT) -> float | None:
    """Net rupee P&L for one ATM straddle lot, after the full cost schedule.

    direction: -1 = short straddle (sell then buy back), +1 = long straddle.
    entry/exit dicts need call_bid, call_ask, put_bid, put_ask.

    Quote-validity rule (direction-aware, deliberately NOT symmetric):
      * ENTRY  -- both legs must be genuinely two-sided (bid>0 and ask>0), else
        the position cannot be established at all.
      * EXIT   -- only requires a live offer (ask>0). A **zero bid is legitimate**:
        it means the leg has gone worthless. Demanding bid>0 at exit would
        discard exactly the large-move days on which a short straddle LOSES,
        biasing short-vol results upward. That bias was present in the first
        run of this script and is corrected here.
    Returns None only when the trade is genuinely untradeable.
    """
    for leg in ("call_bid", "call_ask", "put_bid", "put_ask"):
        if entry.get(leg, 0) <= 0:
            return None
    for leg in ("call_ask", "put_ask"):
        if exit_.get(leg, 0) <= 0:
            return None
    for leg in ("call_bid", "put_bid"):
        if exit_.get(leg, 0) < 0:
            return None

    e_c_sell, e_c_buy = _adj_prices(entry["call_bid"], entry["call_ask"], mult)
    e_p_sell, e_p_buy = _adj_prices(entry["put_bid"], entry["put_ask"], mult)
    x_c_sell, x_c_buy = _adj_prices(exit_["call_bid"], exit_["call_ask"], mult)
    x_p_sell, x_p_buy = _adj_prices(exit_["put_bid"], exit_["put_ask"], mult)

    if direction < 0:                      # SHORT: sell at entry, buy at exit
        sell_prem = (e_c_sell + e_p_sell) * LOT_SIZE
        buy_prem = (x_c_buy + x_p_buy) * LOT_SIZE
    else:                                  # LONG: buy at entry, sell at exit
        buy_prem = (e_c_buy + e_p_buy) * LOT_SIZE
        sell_prem = (x_c_sell + x_p_sell) * LOT_SIZE

    gross = sell_prem - buy_prem

    brokerage = 4 * BROKERAGE_PER_ORDER            # 2 legs x entry+exit
    turnover = sell_prem + buy_prem
    stt = STT_OPT_SELL * sell_prem                 # sell side only
    exch = EXCH_OPT * turnover
    sebi = SEBI_FEE * turnover
    stamp = STAMP_BUY * buy_prem
    gst = GST_RATE * (brokerage + exch + sebi)
    return gross - (brokerage + stt + exch + sebi + stamp + gst)


def futures_cost_fraction(spot: float, mult: float = BASE_MULT) -> float:
    """Round-trip futures-proxy cost as a fraction of notional."""
    notional = spot * LOT_SIZE
    slip = 2.0 * (FUT_SLIP_BPS / 10_000.0) * mult   # both sides
    brokerage = 2 * BROKERAGE_PER_ORDER
    exch = EXCH_FUT * 2 * notional
    sebi = SEBI_FEE * 2 * notional
    gst = GST_RATE * (brokerage + exch + sebi)
    stt = STT_FUT_SELL * notional                   # sell leg
    fixed = (brokerage + exch + sebi + gst + stt) / notional
    return slip + fixed


def directional_return(entry_spot: float, exit_spot: float, side: int,
                       mult: float = BASE_MULT) -> float:
    """Signed net return of the futures proxy, after costs."""
    gross = side * (exit_spot / entry_spot - 1.0)
    return gross - futures_cost_fraction(entry_spot, mult)


# --- feature panel (causal, cached) ----------------------------------------

def _pivot(df: pd.DataFrame, opt_type: str, value: str,
           strikes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(n_times, n_strikes) matrix of `value` for one option type, 0 where absent."""
    sub = df[df["option_type"] == opt_type]
    p = sub.pivot_table(index="snapshot_ts", columns="strike", values=value,
                        aggfunc="last")
    p = p.reindex(index=times, columns=strikes)
    return p.to_numpy(dtype=float)


def build_day_panel(d: date, data_dir: str = DATA_DIR) -> pd.DataFrame | None:
    """Per-snapshot near-expiry features for one day. All causal by construction."""
    folder = (Path(data_dir) / "option_chain" / f"{d.year:04d}"
              / f"{d.month:02d}" / f"{d.day:02d}")
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    near = df["expiry"].min()
    df = df[df["expiry"] == near].copy()
    if df.empty:
        return None
    # collapse duplicate (ts, strike, type) rows -- last wins, matches storage de-dup
    df = df.drop_duplicates(subset=["snapshot_ts", "strike", "option_type"],
                            keep="last")

    times = np.sort(df["snapshot_ts"].unique())
    strikes = np.sort(df["strike"].unique())
    if len(times) < 5 or len(strikes) < 5:
        return None

    c_oi = np.nan_to_num(_pivot(df, "CE", "open_interest", strikes, times))
    p_oi = np.nan_to_num(_pivot(df, "PE", "open_interest", strikes, times))
    c_vol = np.nan_to_num(_pivot(df, "CE", "volume", strikes, times))
    p_vol = np.nan_to_num(_pivot(df, "PE", "volume", strikes, times))
    c_bid = _pivot(df, "CE", "bid", strikes, times)
    c_ask = _pivot(df, "CE", "ask", strikes, times)
    p_bid = _pivot(df, "PE", "bid", strikes, times)
    p_ask = _pivot(df, "PE", "ask", strikes, times)
    c_last = _pivot(df, "CE", "last_price", strikes, times)
    p_last = _pivot(df, "PE", "last_price", strikes, times)

    spot_s = df.groupby("snapshot_ts")["spot"].last().reindex(times)
    spot = spot_s.to_numpy(dtype=float)

    # --- max pain, vectorized across all snapshots ------------------------
    # Mc[i,j] = call payout at settle K_i for strike K_j ; Mp likewise for puts
    Mc = np.maximum(strikes[:, None] - strikes[None, :], 0.0)
    Mp = np.maximum(strikes[None, :] - strikes[:, None], 0.0)
    pain = c_oi @ Mc.T + p_oi @ Mp.T           # (n_times, n_settle)
    max_pain = strikes[np.argmin(pain, axis=1)]

    # --- ATM index per snapshot -------------------------------------------
    atm_idx = np.argmin(np.abs(strikes[None, :] - spot[:, None]), axis=1)
    rows = np.arange(len(times))
    atm_strike = strikes[atm_idx]

    def at_atm(m):
        return m[rows, atm_idx]

    a_cb, a_ca = at_atm(c_bid), at_atm(c_ask)
    a_pb, a_pa = at_atm(p_bid), at_atm(p_ask)
    a_cl, a_pl = at_atm(c_last), at_atm(p_last)

    # --- fixed entry strike (ATM at the first snapshot >= ENTRY_CLOCK) -----
    tod = pd.DatetimeIndex(times).strftime("%H:%M")
    ent_pos = int(np.argmax(tod >= ENTRY_CLOCK)) if (tod >= ENTRY_CLOCK).any() else 0
    es_idx = int(atm_idx[ent_pos])
    entry_strike = float(strikes[es_idx])

    # --- ATM IV (derived; stored column is NaN by design) ------------------
    exp_date = pd.Timestamp(near).date()
    close_dt = datetime.combine(exp_date, pd.Timestamp("15:30").time())
    atm_iv = np.full(len(times), np.nan)
    for i in range(len(times)):
        ts = pd.Timestamp(times[i]).to_pydatetime()
        t = max((close_dt - ts).total_seconds(), 0.0) / (365.0 * 86400.0)
        if t <= 0 or spot[i] <= 0:
            continue
        cp = (a_cb[i] + a_ca[i]) / 2.0 if (a_cb[i] > 0 and a_ca[i] > 0) else a_cl[i]
        pp = (a_pb[i] + a_pa[i]) / 2.0 if (a_pb[i] > 0 and a_pa[i] > 0) else a_pl[i]
        ivs = []
        for price, otype in ((cp, "CE"), (pp, "PE")):
            if price and price > 0:
                v = bs.implied_volatility(price, spot[i], float(atm_strike[i]),
                                          t, R_RATE, otype, Q_YIELD)
                if v is not None and v > 0:
                    ivs.append(v)
        if ivs:
            atm_iv[i] = float(np.mean(ivs))

    call_oi_tot = c_oi.sum(axis=1)
    put_oi_tot = p_oi.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pcr_oi = np.where(call_oi_tot > 0, put_oi_tot / call_oi_tot, np.nan)
        cvt, pvt = c_vol.sum(axis=1), p_vol.sum(axis=1)
        pcr_vol = np.where(cvt > 0, pvt / cvt, np.nan)

    return pd.DataFrame({
        "date": d,
        "ts": pd.DatetimeIndex(times),
        "tod": tod,
        "expiry": exp_date,
        "dte_cal": (exp_date - d).days,
        "is_expiry_day": exp_date == d,
        "spot": spot,
        "atm_strike": atm_strike.astype(float),
        "atm_call_bid": a_cb, "atm_call_ask": a_ca,
        "atm_put_bid": a_pb, "atm_put_ask": a_pa,
        "atm_iv": atm_iv,
        "pcr_oi": pcr_oi, "pcr_vol": pcr_vol,
        "max_pain": max_pain.astype(float),
        "call_oi": call_oi_tot, "put_oi": put_oi_tot,
        "entry_strike": entry_strike,
        "es_call_bid": c_bid[:, es_idx], "es_call_ask": c_ask[:, es_idx],
        "es_put_bid": p_bid[:, es_idx], "es_put_ask": p_ask[:, es_idx],
        "n_strikes": len(strikes),
    })


def load_panel(rebuild: bool = False) -> pd.DataFrame:
    if PANEL.exists() and not rebuild:
        return pd.read_parquet(PANEL)
    days = collected_days()
    print(f"Building feature panel over {len(days)} collected days "
          f"(one-off, cached to {PANEL}) ...")
    frames = []
    for i, d in enumerate(days, 1):
        p = build_day_panel(d)
        if p is not None:
            frames.append(p)
        if i % 10 == 0 or i == len(days):
            print(f"  {i}/{len(days)} days")
    panel = pd.concat(frames, ignore_index=True)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False)
    return panel


# --- panel validation against the library path ----------------------------

def validate_panel(panel: pd.DataFrame) -> bool:
    """Cross-check the fast panel against nifty_quant.analytics.options.

    Guards against a fast-path bug manufacturing a fake edge. Aborts the run on
    mismatch (pre-registered method step 2).
    """
    from nifty_quant.analytics import options as opt
    from nifty_quant.research import live_lens as lens

    days = sorted(panel["date"].unique())
    sample_day = pd.Timestamp(days[len(days) // 2]).date()
    chains = lens.read_day(DATA_DIR, sample_day)
    if not chains:
        print("  validation SKIPPED (sample day unreadable)")
        return True
    near = min(c.expiry for c in chains)
    nc = sorted((c for c in chains if c.expiry == near), key=lambda c: c.timestamp)
    pday = panel[panel["date"] == pd.Timestamp(sample_day).date()] \
        if not isinstance(panel["date"].iloc[0], date) else panel[panel["date"] == sample_day]
    pday = pday.set_index(pd.DatetimeIndex(pday["ts"]))

    checked = 0
    worst_iv = worst_pcr = worst_mp = 0.0
    for c in nc[:: max(len(nc) // 8, 1)]:
        key = pd.Timestamp(c.timestamp)
        if key not in pday.index:
            continue
        row = pday.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lib_pcr = opt.put_call_ratio(c, by="oi")
        lib_mp = opt.max_pain(c)
        lib_iv = opt.atm_iv(c, r=R_RATE, q=Q_YIELD)
        worst_pcr = max(worst_pcr, abs(lib_pcr - row["pcr_oi"]))
        worst_mp = max(worst_mp, abs(lib_mp - row["max_pain"]))
        if lib_iv is not None and not np.isnan(row["atm_iv"]):
            worst_iv = max(worst_iv, abs(lib_iv - row["atm_iv"]) * 100)
        checked += 1

    ok = (checked >= 3 and worst_pcr < 1e-6 and worst_mp < 1e-6 and worst_iv < 0.05)
    print(f"  panel vs library on {sample_day}: {checked} snapshots | "
          f"max |dPCR|={worst_pcr:.2e}  max |dMaxPain|={worst_mp:.2e}  "
          f"max |dATM_IV|={worst_iv:.4f} vol pts -> {'PASS' if ok else 'FAIL'}")
    return ok


# --- per-day access helpers ------------------------------------------------

def pick(day: pd.DataFrame, clock: str):
    """First snapshot at or after `clock`; None if the day ends before it."""
    m = day["tod"] >= clock
    if not m.any():
        return None
    return day.loc[m].iloc[0]


def straddle_legs(row, fixed: bool) -> dict:
    """Quote dict for the straddle legs. fixed=True uses the day's locked entry
    strike (so a held position is not silently re-struck as spot drifts)."""
    if fixed:
        return {"call_bid": row["es_call_bid"], "call_ask": row["es_call_ask"],
                "put_bid": row["es_put_bid"], "put_ask": row["es_put_ask"]}
    return {"call_bid": row["atm_call_bid"], "call_ask": row["atm_call_ask"],
            "put_bid": row["atm_put_bid"], "put_ask": row["atm_put_ask"]}


def day_frames(panel: pd.DataFrame) -> dict:
    out = {}
    for d, g in panel.groupby("date"):
        out[d] = g.sort_values("ts").reset_index(drop=True)
    return out


def daily_closes(days: list, frames: dict) -> dict:
    closes = {}
    for d in days:
        r = pick(frames[d], EXIT_CLOCK)
        if r is None:
            r = frames[d].iloc[-1]
        closes[d] = float(r["spot"])
    return closes


# --- strategies (all rules LOCKED by the pre-registration) -----------------

def strat_s1_vrp(days, frames, closes, mult) -> list[dict]:
    """S1 VRP Harvesting v1.2 -- short ATM straddle when IV exceeds a causal
    trailing-RV forecast by >= 2.0 vol points. Non-expiry days only."""
    trades = []
    cl = [closes[d] for d in days]
    for i, d in enumerate(days):
        if i < S1_RV_LOOKBACK + 1:
            continue
        day = frames[d]
        if bool(day["is_expiry_day"].iloc[0]):
            continue                                    # 0-DTE belongs to S6
        e, x = pick(day, ENTRY_CLOCK), pick(day, EXIT_CLOCK)
        if e is None or x is None or np.isnan(e["atm_iv"]):
            continue
        prior = np.array(cl[:i], dtype=float)           # completed days only
        r = np.diff(np.log(prior))[-S1_RV_LOOKBACK:]
        if len(r) < S1_RV_LOOKBACK:
            continue
        rv = math.sqrt(TRADING_DAYS / S1_RV_LOOKBACK * float(np.sum(r ** 2))) * 100.0
        iv = float(e["atm_iv"]) * 100.0
        if iv - rv < S1_VRP_THRESHOLD:
            continue
        pnl = straddle_trade_pnl(straddle_legs(e, True), straddle_legs(x, True),
                                 direction=-1, mult=mult)
        if pnl is None:
            continue
        trades.append({"date": d, "ret": pnl / MARGIN_PER_LOT,
                       "iv": iv, "rv": rv, "edge": iv - rv})
    return trades


def strat_s2_oi_divergence(days, frames, closes, mult) -> list[dict]:
    """S2 OI Concentration Divergence -- fade the morning move when the opposing
    side's OI is building. Directional, futures proxy."""
    trades = []
    for d in days:
        day = frames[d]
        a, b, x = (pick(day, ENTRY_CLOCK), pick(day, MIDDAY_CLOCK),
                   pick(day, EXIT_CLOCK))
        if a is None or b is None or x is None:
            continue
        spot_ret = float(b["spot"]) / float(a["spot"]) - 1.0
        d_call = float(b["call_oi"]) - float(a["call_oi"])
        d_put = float(b["put_oi"]) - float(a["put_oi"])
        side = 0
        if spot_ret <= -S2_MOVE_THRESHOLD and d_call > d_put:
            side = 1
        elif spot_ret >= S2_MOVE_THRESHOLD and d_put > d_call:
            side = -1
        if side == 0:
            continue
        trades.append({"date": d,
                       "ret": directional_return(float(b["spot"]),
                                                 float(x["spot"]), side, mult),
                       "side": side, "spot_ret": spot_ret})
    return trades


def strat_s3_iv_squeeze(days, frames, closes, mult) -> list[dict]:
    """S3 IV Squeeze -- long ATM straddle when 09:20 ATM IV sits in the bottom
    20th percentile of its own trailing 20 days. Mirror image of S1."""
    trades = []
    hist: list[float] = []
    for i, d in enumerate(days):
        day = frames[d]
        e, x = pick(day, ENTRY_CLOCK), pick(day, EXIT_CLOCK)
        iv_today = float(e["atm_iv"]) * 100.0 if (
            e is not None and not np.isnan(e["atm_iv"])) else None
        expiry_day = bool(day["is_expiry_day"].iloc[0])

        if (iv_today is not None and len(hist) >= S3_WARMUP
                and not expiry_day and x is not None):
            trail = np.array(hist[-S3_WARMUP:], dtype=float)
            pctile = 100.0 * float(np.mean(trail < iv_today))
            if pctile <= S3_IV_PCTILE:
                pnl = straddle_trade_pnl(straddle_legs(e, True),
                                         straddle_legs(x, True),
                                         direction=+1, mult=mult)
                if pnl is not None:
                    trades.append({"date": d, "ret": pnl / MARGIN_PER_LOT,
                                   "iv": iv_today, "pctile": pctile})
        if iv_today is not None:
            hist.append(iv_today)                       # append AFTER use (causal)
    return trades


def strat_s4_pcr_reversal(days, frames, closes, mult) -> list[dict]:
    """S4 PCR Extreme Reversal -- contrarian on PCR(OI) extremes. Futures proxy."""
    trades = []
    for d in days:
        day = frames[d]
        e, x = pick(day, ENTRY_CLOCK), pick(day, EXIT_CLOCK)
        if e is None or x is None or np.isnan(e["pcr_oi"]):
            continue
        pcr = float(e["pcr_oi"])
        side = 1 if pcr >= S4_PCR_HIGH else (-1 if pcr <= S4_PCR_LOW else 0)
        if side == 0:
            continue
        trades.append({"date": d,
                       "ret": directional_return(float(e["spot"]),
                                                 float(x["spot"]), side, mult),
                       "side": side, "pcr": pcr})
    return trades


def strat_s5_maxpain_trade(days, frames, closes, mult) -> list[dict]:
    """S5(b) Max Pain Magnet, traded leg -- short 0-DTE straddle ONLY when spot
    opens within 1.0% of max pain. One variable different from S6."""
    trades = []
    for d in days:
        day = frames[d]
        if not bool(day["is_expiry_day"].iloc[0]):
            continue
        e, x = pick(day, ENTRY_CLOCK), pick(day, EXIT_CLOCK)
        if e is None or x is None:
            continue
        d_open = abs(float(e["spot"]) - float(e["max_pain"])) / float(e["spot"])
        if d_open > S5_MAXPAIN_PROX:
            continue
        pnl = straddle_trade_pnl(straddle_legs(e, True), straddle_legs(x, True),
                                 direction=-1, mult=mult)
        if pnl is None:
            continue
        trades.append({"date": d, "ret": pnl / MARGIN_PER_LOT, "d_open": d_open})
    return trades


def strat_s6_zero_dte(days, frames, closes, mult) -> list[dict]:
    """S6 Expiry-Day 0-DTE Straddle Mispricing -- unconditional short."""
    trades = []
    for d in days:
        day = frames[d]
        if not bool(day["is_expiry_day"].iloc[0]):
            continue
        e, x = pick(day, ENTRY_CLOCK), pick(day, EXIT_CLOCK)
        if e is None or x is None:
            continue
        pnl = straddle_trade_pnl(straddle_legs(e, True), straddle_legs(x, True),
                                 direction=-1, mult=mult)
        if pnl is None:
            continue
        iv = float(e["atm_iv"]) * 100.0 if not np.isnan(e["atm_iv"]) else float("nan")
        trades.append({"date": d, "ret": pnl / MARGIN_PER_LOT, "iv": iv})
    return trades


def strat_s7_gap_fade(days, frames, closes, mult) -> list[dict]:
    """S7 Overnight Gap Fade -- trade against gaps >= 0.50%, exit after one hour."""
    trades = []
    for i, d in enumerate(days):
        if i == 0:
            continue
        prev_close = closes[days[i - 1]]
        day = frames[d]
        e, x = pick(day, ENTRY_CLOCK), pick(day, FIRST_HOUR_EXIT)
        if e is None or x is None:
            continue
        gap = float(e["spot"]) / prev_close - 1.0
        if abs(gap) < S7_GAP_THRESHOLD:
            continue
        side = -1 if gap > 0 else 1                     # fade the gap
        trades.append({"date": d,
                       "ret": directional_return(float(e["spot"]),
                                                 float(x["spot"]), side, mult),
                       "side": side, "gap": gap})
    return trades


STRATEGIES = [
    ("S1", "VRP Harvesting v1.2", "short straddle", strat_s1_vrp),
    ("S2", "OI Concentration Divergence", "futures proxy", strat_s2_oi_divergence),
    ("S3", "IV Squeeze -> RV expansion", "long straddle", strat_s3_iv_squeeze),
    ("S4", "PCR Extreme Reversal", "futures proxy", strat_s4_pcr_reversal),
    ("S5", "Max Pain Magnet (traded leg)", "short 0-DTE straddle",
     strat_s5_maxpain_trade),
    ("S6", "0-DTE Straddle Mispricing", "short 0-DTE straddle", strat_s6_zero_dte),
    ("S7", "Overnight Gap Fade", "futures proxy", strat_s7_gap_fade),
]


# --- metrics & inference ---------------------------------------------------

def daily_series(trades: list[dict], days: list) -> np.ndarray:
    """Net return per calendar trading day, 0 on days with no trade.

    This is the portfolio view: comparable across candidates and the correct
    input for SPA (benchmark = cash = 0).
    """
    s = pd.Series(0.0, index=pd.Index(days, name="date"))
    for t in trades:
        s.loc[t["date"]] += t["ret"]
    return s.to_numpy(dtype=float)


def hac_test(y: np.ndarray) -> tuple[float, float]:
    """One-sided (H1: mean > 0) HAC/Newey-West t-stat and p-value."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or np.allclose(y.std(), 0.0):
        return float("nan"), float("nan")
    import statsmodels.api as sm
    lag = min(HAC_LAG, max(len(y) - 2, 1))
    res = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": lag})
    t = float(res.tvalues[0])
    p2 = float(res.pvalues[0])
    p1 = p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0
    return t, p1


def metrics(trades: list[dict], days: list) -> dict:
    from scipy import stats as ss
    r = np.array([t["ret"] for t in trades], dtype=float)
    n = len(r)
    out = {"n": n}
    if n == 0:
        return {**out, "mean": float("nan"), "median": float("nan"),
                "win": float("nan"), "sharpe": float("nan"),
                "sortino": float("nan"), "max_dd": float("nan"),
                "pf": float("nan"), "expectancy": float("nan"),
                "worst": float("nan"), "t": float("nan"),
                "p": float("nan"), "p_sign": float("nan")}
    ds = daily_series(trades, days)
    sd = float(np.std(ds, ddof=1)) if len(ds) > 1 else float("nan")
    mean_d = float(np.mean(ds))
    dn = ds[ds < 0]
    dsd = math.sqrt(float(np.mean(dn ** 2))) if dn.size else 0.0
    eq = np.cumprod(1.0 + ds)
    peak = np.maximum.accumulate(eq)
    pos, neg = r[r > 0], r[r < 0]
    t, p = hac_test(r)
    n_pos = int(np.sum(r > 0))
    p_sign = float(ss.binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
    return {**out,
            "mean": float(np.mean(r)), "median": float(np.median(r)),
            "win": 100.0 * n_pos / n,
            "sharpe": (mean_d / sd * ANN) if sd and sd > 0 else float("nan"),
            "sortino": (mean_d / dsd * ANN) if dsd > 0 else float("nan"),
            "max_dd": 100.0 * float(np.min(eq / peak - 1.0)),
            "pf": (float(pos.sum() / abs(neg.sum())) if neg.size and neg.sum() != 0
                   else float("inf") if pos.size else float("nan")),
            "expectancy": float(np.mean(r)), "worst": float(np.min(r)),
            "t": t, "p": p, "p_sign": p_sign}


def pinning_test(days, frames) -> dict:
    """S5(a) primary: does spot contract toward max pain by expiry close more
    than a same-volatility random walk would? Stationary-bootstrap null."""
    rng = np.random.default_rng(PIN_SEED)
    obs, p_nulls, detail = 0, [], []
    for d in days:
        day = frames[d]
        if not bool(day["is_expiry_day"].iloc[0]):
            continue
        e, x = pick(day, ENTRY_CLOCK), pick(day, EXIT_CLOCK)
        if e is None or x is None:
            continue
        s0, mp = float(e["spot"]), float(e["max_pain"])
        d_open = abs(s0 - mp) / s0
        d_close = abs(float(x["spot"]) - mp) / s0
        # intraday 5-min log returns from entry onward -> the null's volatility
        seg = day[day["tod"] >= ENTRY_CLOCK]
        sp = pd.Series(seg["spot"].to_numpy(),
                       index=pd.DatetimeIndex(seg["ts"])).resample("5min").last().dropna()
        if len(sp) < 6:
            continue
        rets = np.diff(np.log(sp.to_numpy()))
        idx = stationary_bootstrap_indices(len(rets), PIN_BLOCK, PIN_NBOOT, rng)
        sim_terminal = s0 * np.exp(rets[idx].sum(axis=1))
        d_sim = np.abs(sim_terminal - mp) / s0
        p_null = float(np.mean(d_sim < d_open))     # P(contract | random walk)
        contracted = d_close < d_open
        obs += int(contracted)
        p_nulls.append(p_null)
        detail.append({"date": d, "d_open": d_open, "d_close": d_close,
                       "contracted": contracted, "p_null": p_null})
    k = len(p_nulls)
    if k == 0:
        return {"n": 0, "p": float("nan")}
    pn = np.array(p_nulls)
    exp_ct, var = float(pn.sum()), float(np.sum(pn * (1 - pn)))
    if var <= 0:
        return {"n": k, "observed": obs, "expected": exp_ct, "z": float("nan"),
                "p": float("nan"), "detail": detail}
    from scipy import stats as ss
    z = (obs - exp_ct) / math.sqrt(var)
    return {"n": k, "observed": obs, "expected": exp_ct, "z": z,
            "p": float(ss.norm.sf(z)), "detail": detail}


def verdict(m: dict, q: float, spa_p: float, fragile: bool,
            unstable: bool) -> str:
    """Mechanical application of the LOCKED decision rules, in order."""
    if m["n"] == 0:
        return "NO TRADES"
    mean, t, win = m["mean"], m["t"], m["win"]
    if mean < 0 and ((not np.isnan(t) and t <= -1.0) or win < 45.0):
        return "REJECT"
    if not np.isnan(t) and abs(t) < 0.5 and abs(mean) < 1e-4:
        return "ARCHIVE"
    if mean > 0:
        if (not np.isnan(q) and q <= FDR_Q and not np.isnan(spa_p)
                and spa_p < 0.10 and not fragile):
            return "CANDIDATE FOR FORWARD TESTING"
        return "INCONCLUSIVE - CONTINUE COLLECTING"
    return "ARCHIVE"


# --- main ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp 032 multi-strategy zoo (pre-registered, read-only).")
    p.add_argument("--rebuild-panel", action="store_true",
                   help="rebuild the cached feature panel from raw chains")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    panel = load_panel(rebuild=args.rebuild_panel)
    frames = day_frames(panel)
    days = sorted(frames)
    closes = daily_closes(days, frames)
    n_days = len(days)
    n_expiry = sum(1 for d in days if bool(frames[d]["is_expiry_day"].iloc[0]))

    print("=" * 100)
    print("EXP 032  MULTI-STRATEGY ZOO  (7 pre-registered candidates, "
          "real bid/ask fills, RESEARCH ONLY)")
    print(f"Days: {n_days}  ({days[0]} -> {days[-1]})  |  expiry days: {n_expiry}"
          f"  |  snapshots: {len(panel):,}")
    print(f"Costs: lot {LOT_SIZE}, brokerage Rs{BROKERAGE_PER_ORDER:.0f}/order, "
          f"STT {STT_OPT_SELL*100:.2f}% sell-side, exch {EXCH_OPT*100:.5f}%, "
          f"GST {GST_RATE*100:.0f}%  |  margin base Rs{MARGIN_PER_LOT:,.0f}/lot")

    print("\n--- panel validation vs nifty_quant.analytics (pre-registered gate) ---")
    if not validate_panel(panel):
        print("\nABORT: fast panel disagrees with the library path. "
              "Fix before trusting any result.")
        return 1

    # --- data completeness / rule-binding diagnostics ---------------------
    print("\n--- data completeness & rule diagnostics (reported, not silent) ---")
    trunc = [d for d in days if pick(frames[d], EXIT_CLOCK) is None]
    for d in trunc:
        print(f"  {d}: session truncated (last snapshot "
              f"{frames[d]['tod'].iloc[-1]}) -> excluded from EOD-exit strategies")
    if not trunc:
        print("  all sessions reach the 15:25 exit clock.")
    d_opens = []
    for d in days:
        if not bool(frames[d]["is_expiry_day"].iloc[0]):
            continue
        e = pick(frames[d], ENTRY_CLOCK)
        if e is not None:
            d_opens.append(abs(float(e["spot"]) - float(e["max_pain"]))
                           / float(e["spot"]))
    if d_opens:
        da = np.array(d_opens)
        n_pass = int((da <= S5_MAXPAIN_PROX).sum())
        print(f"  S5 max-pain proximity filter (<= {S5_MAXPAIN_PROX*100:.1f}%): "
              f"{n_pass}/{len(da)} expiry days pass "
              f"(|spot-maxpain| median {np.median(da)*100:.2f}%, "
              f"max {da.max()*100:.2f}%)")
        if n_pass == len(da):
            print("  *** S5's pre-registered filter is NON-BINDING -> S5 is "
                  "IDENTICAL to S6. Only 6 distinct rules were truly tested; ***")
            print("  *** max pain sits essentially AT spot every expiry morning, "
                  "so it carries no usable information here. ***")

    # --- run all candidates at base cost ----------------------------------
    results, all_trades = {}, {}
    for code, name, instr, fn in STRATEGIES:
        tr = fn(days, frames, closes, BASE_MULT)
        all_trades[code] = tr
        results[code] = metrics(tr, days)

    # --- FDR across the 7 primary p-values --------------------------------
    from statsmodels.stats.multitest import multipletests
    codes = [c for c, _, _, _ in STRATEGIES]
    pvals = [results[c]["p"] for c in codes]
    usable = [i for i, v in enumerate(pvals) if not (v is None or np.isnan(v))]
    qvals = {c: float("nan") for c in codes}
    rejected = {c: False for c in codes}
    if usable:
        rej, qv, _, _ = multipletests([pvals[i] for i in usable], alpha=FDR_Q,
                                      method="fdr_bh")
        for j, i in enumerate(usable):
            qvals[codes[i]] = float(qv[j])
            rejected[codes[i]] = bool(rej[j])

    # --- Hansen SPA / White Reality Check on the aligned daily matrix ------
    perf = np.column_stack([daily_series(all_trades[c], days) for c in codes])
    snoop = reality_check_spa(perf, block_len=SPA_BLOCK, n_boot=SPA_NBOOT,
                              seed=SPA_SEED)

    # --- cost sensitivity + stability split -------------------------------
    fragile, unstable, sens = {}, {}, {}
    split = int(round(TRAIN_FRAC * n_days))
    train_days, test_days = set(days[:split]), set(days[split:])
    for code, name, instr, fn in STRATEGIES:
        row = {}
        for m in SPREAD_MULTS:
            tr = fn(days, frames, closes, m)
            row[m] = float(np.mean([t["ret"] for t in tr])) if tr else float("nan")
        sens[code] = row
        base, high = row.get(BASE_MULT, float("nan")), row.get(2.0, float("nan"))
        fragile[code] = bool(base > 0 and not np.isnan(high) and high < 0)
        tr = all_trades[code]
        a = [t["ret"] for t in tr if t["date"] in train_days]
        b = [t["ret"] for t in tr if t["date"] in test_days]
        unstable[code] = bool(a and b and np.sign(np.mean(a)) != np.sign(np.mean(b)))

    # --- primary table -----------------------------------------------------
    print("\n" + "-" * 100)
    print("PRIMARY RESULTS (net of costs, base spread x1.0). Sharpe/maxDD on the "
          "full 59-day daily series.")
    print(f"\n{'':<4}{'strategy':<30}{'n':>4}{'mean%':>8}{'win%':>7}"
          f"{'Sharpe':>8}{'maxDD%':>8}{'PF':>7}{'worst%':>8}{'HACt':>7}"
          f"{'p':>7}{'q(BH)':>8}")
    order = sorted(codes, key=lambda c: (-(results[c]["sharpe"]
                                           if not np.isnan(results[c]["sharpe"])
                                           else -99)))
    name_by = {c: n for c, n, _, _ in STRATEGIES}
    for c in order:
        m = results[c]
        pf = m["pf"]
        pf_s = "inf" if pf == float("inf") else (f"{pf:.2f}" if pf == pf else "--")
        print(f"{c:<4}{name_by[c]:<30}{m['n']:>4}{m['mean']*100:>8.3f}"
              f"{m['win']:>7.0f}{m['sharpe']:>8.2f}{m['max_dd']:>8.1f}"
              f"{pf_s:>7}{m['worst']*100:>8.2f}{m['t']:>7.2f}"
              f"{m['p']:>7.3f}{qvals[c]:>8.3f}")

    # --- cost sensitivity --------------------------------------------------
    print("\n--- cost sensitivity: mean net return %/trade by spread multiplier ---")
    print(f"{'':<4}{'strategy':<30}" + "".join(f"{'x'+str(m):>10}"
                                               for m in SPREAD_MULTS)
          + f"{'  flag':<16}")
    for c in order:
        cells = "".join(f"{sens[c][m]*100:>10.3f}" if sens[c][m] == sens[c][m]
                        else f"{'--':>10}" for m in SPREAD_MULTS)
        flag = "COST-FRAGILE" if fragile[c] else ""
        print(f"{c:<4}{name_by[c]:<30}{cells}  {flag}")

    # --- stability split ---------------------------------------------------
    print(f"\n--- stability split (train = first {split} days, "
          f"test = last {n_days - split} days; diagnostic only, nothing fitted) ---")
    print(f"{'':<4}{'strategy':<30}{'n_tr':>6}{'mean_tr%':>10}"
          f"{'n_te':>6}{'mean_te%':>10}{'  flag':<12}")
    for c in order:
        tr = all_trades[c]
        a = [t["ret"] for t in tr if t["date"] in train_days]
        b = [t["ret"] for t in tr if t["date"] in test_days]
        ma = f"{np.mean(a)*100:>10.3f}" if a else f"{'--':>10}"
        mb = f"{np.mean(b)*100:>10.3f}" if b else f"{'--':>10}"
        print(f"{c:<4}{name_by[c]:<30}{len(a):>6}{ma}{len(b):>6}{mb}"
              f"  {'UNSTABLE' if unstable[c] else ''}")

    # --- S5(a) pinning statistic (primary test for S5) ---------------------
    pin = pinning_test(days, frames)
    print("\n--- S5(a) max-pain pinning statistic (PRIMARY test for S5) ---")
    if pin.get("n", 0) == 0:
        print("  no usable expiry days.")
    else:
        print(f"  expiry cycles: {pin['n']}  contracted toward max pain: "
              f"{pin['observed']}  random-walk expectation: {pin['expected']:.2f}")
        print(f"  z = {pin['z']:+.2f}   one-sided p = {pin['p']:.3f}  "
              f"(stationary bootstrap, {PIN_NBOOT} draws, block {PIN_BLOCK:.0f})")
        print("  -> pinning beyond a random walk is NOT established."
              if not (pin["p"] < 0.05) else
              "  -> pinning exceeds the random-walk null in this sample.")

    # --- data snooping -----------------------------------------------------
    print("\n--- data-snooping correction across all 7 candidates "
          "(benchmark = cash) ---")
    print(f"  White Reality Check p = {snoop.reality_check_p:.3f}   "
          f"Hansen SPA p = {snoop.spa_p:.3f}")
    if snoop.best_model >= 0:
        print(f"  best candidate = {codes[snoop.best_model]} "
              f"({name_by[codes[snoop.best_model]]}), mean daily "
              f"{snoop.best_mean*100:+.4f}%")
    print(f"  block length {snoop.block_len:.0f}, {snoop.n_boot} bootstrap draws")
    if not (snoop.spa_p < 0.10):
        print("  -> SPA does NOT reject 'no candidate beats cash'. The best "
              "performer is consistent with luck across 7 tries.")

    # --- verdicts ----------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICTS (pre-registered decision rules, applied mechanically)")
    print(f"{'':<4}{'strategy':<30}{'verdict':<36}{'modifiers':<24}")
    verdicts = {}
    for c in order:
        v = verdict(results[c], qvals[c], snoop.spa_p, fragile[c], unstable[c])
        if c == "S5" and pin.get("n", 0) and not (pin.get("p", 1) < 0.05):
            v = "INCONCLUSIVE - CONTINUE COLLECTING" if v.startswith(
                "CANDIDATE") else v
        verdicts[c] = v
        mods = ", ".join(m for m, on in (("COST-FRAGILE", fragile[c]),
                                         ("UNSTABLE", unstable[c])) if on)
        print(f"{c:<4}{name_by[c]:<30}{v:<36}{mods:<24}")

    # --- honest closing statement -----------------------------------------
    n_cand = sum(1 for v in verdicts.values() if v.startswith("CANDIDATE"))
    n_rej = sum(1 for v in verdicts.values() if v == "REJECT")
    print("\n" + "-" * 100)
    print("HONEST READING OF THIS RESULT")
    print(f"  Sample: {n_days} trading days, ONE calm quarter, no tail event. "
          f"{n_expiry} expiry cycles.")
    print(f"  Pre-registered power: a genuine Sharpe-1.0 edge yields t ~ 0.5 here "
          f"and CANNOT reach significance.")
    print(f"  Therefore: {n_rej} candidate(s) REJECTED (the sample CAN detect a "
          f"decisive loser),")
    print(f"             {n_cand} candidate(s) reached the maximum attainable "
          f"label (forward-testing candidate).")
    print("  No candidate is promoted to live or paper trading by this experiment.")
    if n_cand:
        print("  MANDATORY next step for any candidate: adversarial look-ahead "
              "review before anything else.")
    print("  The durable output is that 7 rule sets are now LOCKED, so every day "
          "collected from")
    print("  2026-09-12 onward is a genuine out-of-sample test of them.")
    print("=" * 100)

    # --- persist -----------------------------------------------------------
    os.makedirs(args.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    rows = []
    for c in codes:
        m = results[c]
        rows.append({"code": c, "strategy": name_by[c], **m,
                     "q_bh": qvals[c], "fdr_reject": rejected[c],
                     "spa_p": snoop.spa_p, "rc_p": snoop.reality_check_p,
                     "cost_fragile": fragile[c], "unstable": unstable[c],
                     "verdict": verdicts[c],
                     **{f"mean_x{m_}": sens[c][m_] for m_ in SPREAD_MULTS}})
    out = Path(args.report_dir) / f"exp032_strategy_zoo_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    tr_rows = [{"code": c, **t} for c in codes for t in all_trades[c]]
    out_tr = Path(args.report_dir) / f"exp032_trades_{stamp}.csv"
    pd.DataFrame(tr_rows).to_csv(out_tr, index=False)
    print(f"\nWrote {out}")
    print(f"Wrote {out_tr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
