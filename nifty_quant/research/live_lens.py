"""Live market lens: situational awareness + sample-gated pattern checks.

Read-only. Computes descriptive structural metrics from the option chain we are
already collecting (PCR, ATM straddle / implied move, ATM IV, max pain, spot
context, India VIX) and places today's live readings inside the GROWING history
of collected days.

CRITICAL DISCIPLINE -- this module does NOT find tradeable patterns on tiny
samples and does NOT emit signals. Every inferential check declares a minimum
sample size and reports ``ACCUMULATING (n/need)`` until it is met. Only once the
data is sufficient does a check report a confidence-qualified result. This is
how the lens "grows into" pattern discovery without manufacturing false
positives. It produces observations that feed the offline research engine; it
never tells you to trade.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_quant.analytics import options as opt
from nifty_quant.data.models import OptionChain, OptionQuote, OptionType

# Minimum samples before a check reports a result instead of ACCUMULATING.
MIN_DAYS_DISTRIB = 20    # distributional context (percentile/z) meaningful
MIN_VRP_DAYS = 20        # implied (VIX) vs realised daily move
MIN_EXPIRY_DAYS = 8      # 0-DTE straddle implied-vs-realised (expiry days only)
ANN = math.sqrt(252)


def _mid(q: OptionQuote) -> float:
    return (q.bid + q.ask) / 2.0 if q.bid > 0 and q.ask > 0 else q.last_price


def atm_straddle(chain: OptionChain):
    """(atm_strike, call_mid, put_mid, straddle_premium) or None."""
    strikes = chain.strikes()
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - chain.spot))
    call = next((q for q in chain.calls() if q.strike == atm), None)
    put = next((q for q in chain.puts() if q.strike == atm), None)
    if not call or not put:
        return None
    return atm, _mid(call), _mid(put), _mid(call) + _mid(put)


# --- partitioned snapshot reader (research-side) ---------------------------

def read_day(data_dir: str, day: date) -> list[OptionChain]:
    """Load all per-minute snapshots for a day into OptionChain objects."""
    folder = (Path(data_dir) / "option_chain" / f"{day.year:04d}"
              / f"{day.month:02d}" / f"{day.day:02d}")
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    chains: list[OptionChain] = []
    for (ts, und, exp), g in df.groupby(["snapshot_ts", "underlying", "expiry"]):
        quotes = [
            OptionQuote(
                strike=float(r.strike), option_type=OptionType(r.option_type),
                expiry=pd.Timestamp(r.expiry).date(), last_price=float(r.last_price),
                bid=float(r.bid), ask=float(r.ask), volume=float(r.volume),
                open_interest=float(r.open_interest), oi_change=float(r.oi_change),
                implied_volatility=(None if pd.isna(r.implied_volatility)
                                    else float(r.implied_volatility)),
            )
            for r in g.itertuples(index=False)
        ]
        ctx = {}
        first = g.iloc[0].get("context")
        if isinstance(first, str):
            try:
                ctx = json.loads(first)
            except (ValueError, TypeError):
                ctx = {}
        chains.append(OptionChain(
            underlying=str(und), spot=float(g.iloc[0]["spot"]),
            expiry=pd.Timestamp(exp).date(), timestamp=pd.Timestamp(ts).to_pydatetime(),
            quotes=tuple(quotes), context=ctx,
        ))
    return chains


# --- live (single-snapshot) metrics ----------------------------------------

def snapshot_metrics(chain: OptionChain, india_vix: float | None = None) -> dict:
    """Descriptive metrics for one live chain snapshot. No signals."""
    m = {"timestamp": chain.timestamp, "expiry": chain.expiry, "spot": chain.spot}
    try:
        m["pcr_oi"] = opt.put_call_ratio(chain, by="oi")
    except Exception:
        m["pcr_oi"] = float("nan")
    try:
        m["max_pain"] = opt.max_pain(chain)
    except Exception:
        m["max_pain"] = float("nan")
    st = atm_straddle(chain)
    if st:
        m["atm_strike"], m["straddle"] = st[0], st[3]
        m["implied_move_pct"] = (st[3] / chain.spot * 100) if chain.spot else float("nan")
    try:
        iv = opt.atm_iv(chain)
        m["atm_iv"] = iv * 100 if iv is not None else float("nan")
    except Exception:
        m["atm_iv"] = float("nan")
    if india_vix is not None:
        m["india_vix"] = india_vix
    return m


def status_line(chain: OptionChain, india_vix: float | None,
                prev_vix: float | None = None) -> str:
    """One-line live situational-awareness string (descriptive, not a signal)."""
    m = snapshot_metrics(chain, india_vix)
    vix_part = "VIX --"
    if india_vix is not None:
        chg = f" ({india_vix - prev_vix:+.2f})" if prev_vix is not None else ""
        vix_part = f"VIX {india_vix:.1f}{chg}"
    impl = m.get("implied_move_pct", float("nan"))
    return (f"{chain.timestamp:%H:%M} NIFTY {chain.spot:,.0f} | {vix_part} | "
            f"PCR {m.get('pcr_oi', float('nan')):.2f} | "
            f"straddle {m.get('straddle', float('nan')):.0f} "
            f"(impl +-{impl:.2f}%) | max-pain {m.get('max_pain', float('nan')):,.0f}")


# --- per-day summary --------------------------------------------------------

def compute_day_metrics(data_dir: str, day: date) -> dict | None:
    """One summary row for a collected day (None if no data)."""
    chains = read_day(data_dir, day)
    if not chains:
        return None
    expiries = sorted({c.expiry for c in chains})
    near = expiries[0]
    near_chains = sorted((c for c in chains if c.expiry == near),
                         key=lambda c: c.timestamp)
    spots = [c.spot for c in near_chains]
    if len(spots) < 2:
        return None
    open_spot, close_spot = spots[0], spots[-1]
    realized_move_pct = abs(close_spot / open_spot - 1) * 100

    row = {
        "date": day, "n_snapshots": len(near_chains),
        "near_expiry": near, "is_expiry_day": (near == day),
        "open_spot": open_spot, "close_spot": close_spot,
        "realized_move_pct": realized_move_pct,
        "intraday_range_pct": (max(spots) - min(spots)) / open_spot * 100,
    }
    # PCR / straddle at open and close
    row["pcr_open"] = opt.put_call_ratio(near_chains[0], by="oi")
    row["pcr_close"] = opt.put_call_ratio(near_chains[-1], by="oi")
    s_open = atm_straddle(near_chains[0])
    if s_open:
        row["straddle_open"] = s_open[3]
        row["implied_move_open_pct"] = s_open[3] / open_spot * 100
    # India VIX from the day's vix file
    vf = glob.glob(os.path.join(data_dir, "vix", f"{day.year}",
                                f"INDIAVIX_{day.isoformat()}.parquet"))
    if vf:
        v = pd.read_parquet(vf[0])
        row["vix_open"] = float(v["india_vix"].iloc[0])
        row["vix_close"] = float(v["india_vix"].iloc[-1])
        # implied daily move from VIX (annualised -> daily)
        row["vix_implied_daily_pct"] = row["vix_close"] / ANN
    return row


# --- historical context (grows with collected days) ------------------------

@dataclass
class CheckResult:
    name: str
    ready: bool
    n: int
    need: int
    detail: str


class HistoricalContext:
    """Accumulated per-day metrics; positions live readings in history and runs
    sample-gated pre-registered checks."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    @classmethod
    def load(cls, data_dir: str) -> "HistoricalContext":
        days = []
        for f in sorted(glob.glob(os.path.join(
                data_dir, "option_chain", "*", "*", "*"))):
            # folder .../YYYY/MM/DD
            parts = Path(f).parts[-3:]
            try:
                d = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                continue
            days.append(d)
        rows = []
        for d in sorted(set(days)):
            r = compute_day_metrics(data_dir, d)
            if r:
                rows.append(r)
        return cls(rows)

    def values(self, key: str) -> list[float]:
        return [r[key] for r in self.rows
                if r.get(key) is not None and not _isnan(r.get(key))]

    def percentile(self, key: str, value: float) -> tuple[float, int]:
        vals = self.values(key)
        n = len(vals)
        if n == 0:
            return (float("nan"), 0)
        below = sum(1 for v in vals if v < value)
        return (below / n * 100.0, n)

    def context_line(self, key: str, value: float, label: str) -> str:
        pct, n = self.percentile(key, value)
        conf = "low-confidence" if n < MIN_DAYS_DISTRIB else "ok"
        if n == 0:
            return f"  {label}: no history yet"
        return f"  {label}: {value:.2f} -> {pct:.0f}th pct of {n} days ({conf})"

    def checks(self) -> list[CheckResult]:
        out: list[CheckResult] = []

        # 1) Variance risk premium (lite): VIX-implied daily move vs realised
        impl = self.values("vix_implied_daily_pct")
        real = [r["realized_move_pct"] for r in self.rows
                if r.get("vix_implied_daily_pct") is not None
                and not _isnan(r.get("realized_move_pct"))]
        n = min(len(impl), len(real))
        if n < MIN_VRP_DAYS:
            out.append(CheckResult("VRP (VIX implied vs realised daily move)",
                                   False, n, MIN_VRP_DAYS, "ACCUMULATING"))
        else:
            mi, mr = float(np.mean(impl[:n])), float(np.mean(real[:n]))
            sign = "implied > realised (premium)" if mi > mr else "realised >= implied"
            out.append(CheckResult(
                "VRP (VIX implied vs realised daily move)", True, n, MIN_VRP_DAYS,
                f"mean implied {mi:.2f}% vs realised {mr:.2f}% -> {sign}"))

        # 2) 0-DTE straddle implied-vs-realised on expiry days
        exp_rows = [r for r in self.rows if r.get("is_expiry_day")
                    and r.get("implied_move_open_pct") is not None]
        n2 = len(exp_rows)
        if n2 < MIN_EXPIRY_DAYS:
            out.append(CheckResult("0-DTE straddle implied vs realised (expiry days)",
                                   False, n2, MIN_EXPIRY_DAYS, "ACCUMULATING"))
        else:
            mi = float(np.mean([r["implied_move_open_pct"] for r in exp_rows]))
            mr = float(np.mean([r["realized_move_pct"] for r in exp_rows]))
            out.append(CheckResult(
                "0-DTE straddle implied vs realised (expiry days)", True, n2,
                MIN_EXPIRY_DAYS,
                f"mean implied {mi:.2f}% vs realised {mr:.2f}%"))
        return out


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
