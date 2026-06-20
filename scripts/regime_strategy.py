"""Experiment 004: strategy performance by forecasted volatility regime.

Builds on Exp 003 (RV is forecastable). For each trading day we generate a
ONE-STEP-AHEAD HAR-RV forecast using only past data (expanding window, no
lookahead), bucket days into LOW / NORMAL / HIGH by forecast percentile
(30/40/30, fixed -- not optimised), run every benchmark strategy once over the
5m warehouse (costs included), attribute each day's P&L and each trade to its
regime, and test whether performance differs across regimes.

    python scripts/regime_strategy.py
"""

from __future__ import annotations

import glob
import logging
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss

from nifty_quant.backtest.broker import PercentSlippage, SimulatedBroker
from nifty_quant.backtest.engine import BacktestEngine
from nifty_quant.backtest.instrument import Instrument, InstrumentType
from nifty_quant.backtest.portfolio import Portfolio
from nifty_quant.backtest.risk import BasicRiskEngine
from nifty_quant.backtest.strategies import benchmarks as bench
from nifty_quant.backtest.strategies.mean_reversion import (
    BollingerReversion, RsiReversion,
)
from nifty_quant.data.storage.parquet import ParquetStorage
from nifty_quant.feed.replay import ReplayFeed
from nifty_quant.log import configure as configure_logging

ANNUALISE = math.sqrt(252)
COST = 0.0006  # informational; engine already applies slippage+fee
RNG = np.random.default_rng(11)
START, END = "2024-06-20", "2026-06-19"


# ----------------------------------------------------------------------------
# Stage A: daily RV + walk-forward HAR one-step-ahead forecast -> regimes
# ----------------------------------------------------------------------------
def daily_rv() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    rows = []
    for d, g in df.groupby("date"):
        if len(g) < 12:
            continue
        c = g.sort_values("timestamp")["close"].to_numpy()
        lr = np.log(c[1:] / c[:-1])
        rows.append({"date": d, "rv": math.sqrt(float(np.sum(lr ** 2)))})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    return out


def har_forecasts(rv: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window one-step-ahead HAR forecast of log-RV (no lookahead)."""
    s = rv["log_rv"]
    feat = pd.DataFrame({
        "d": s.shift(1), "w": s.shift(1).rolling(5).mean(),
        "m": s.shift(1).rolling(22).mean(), "y": s,
    })
    feat["date"] = rv["date"]
    feat = feat.dropna().reset_index(drop=True)
    preds = [np.nan] * len(feat)
    min_train = 60
    for i in range(min_train, len(feat)):
        tr = feat.iloc[:i]
        X = sm.add_constant(tr[["d", "w", "m"]].to_numpy())
        fit = sm.OLS(tr["y"].to_numpy(), X).fit()
        xrow = np.r_[1.0, feat.loc[i, ["d", "w", "m"]].to_numpy(dtype=float)]
        preds[i] = float(fit.predict(xrow.reshape(1, -1))[0])
    feat["forecast"] = preds
    return feat[["date", "forecast"]].dropna().reset_index(drop=True)


def label_regimes(fc: pd.DataFrame) -> pd.DataFrame:
    lo, hi = fc["forecast"].quantile([0.30, 0.70])
    lab = np.where(fc["forecast"] < lo, "LOW",
                   np.where(fc["forecast"] >= hi, "HIGH", "NORMAL"))
    fc = fc.copy()
    fc["regime"] = lab
    return fc


# ----------------------------------------------------------------------------
# Stage B: run strategies once, capture daily equity + trades
# ----------------------------------------------------------------------------
def build_engines(inst, qty):
    strategies = {
        "EmaCross(20/50)": bench.BENCHMARKS["EmaCross"](inst, quantity=qty),
        "SmaCross(20/50)": bench.BENCHMARKS["SmaCross"](inst, quantity=qty),
        "DonchianBreakout": bench.BENCHMARKS["DonchianBreakout"](inst, quantity=qty),
        "SuperTrend": bench.BENCHMARKS["SuperTrend"](inst, quantity=qty),
        "VwapTrend": bench.BENCHMARKS["VwapTrend"](inst, quantity=qty),
        "MacdCross": bench.BENCHMARKS["MacdCross"](inst, quantity=qty),
        "RsiReversion": RsiReversion(inst, quantity=qty),
        "BollingerReversion": BollingerReversion(inst, quantity=qty),
        "BuyAndHold": bench.BENCHMARKS["BuyAndHold"](inst, quantity=qty),
    }
    engines = {}
    for name, strat in strategies.items():
        engines[name] = BacktestEngine(
            strat,
            portfolio=Portfolio(starting_cash=1_000_000.0),
            risk_engine=BasicRiskEngine(default_quantity=qty),
            broker=SimulatedBroker(fill_model=PercentSlippage(0.0003),
                                   fee_per_order=20.0),
        )
    return engines


def daily_returns(equity: pd.Series) -> pd.Series:
    d = equity.resample("1D").last().dropna()
    return d.pct_change().dropna()


# ----------------------------------------------------------------------------
# Stage C: per-regime metrics + statistics
# ----------------------------------------------------------------------------
def regime_metrics(r: np.ndarray) -> dict:
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 5:
        return {"n": n}
    mean = r.mean(); sd = r.std(ddof=1)
    downside = r[r < 0]
    dd_std = math.sqrt((downside ** 2).mean()) if downside.size else 0.0
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1).min())
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    ann_ret = mean * 252
    return {
        "n": n,
        "mean_bps": round(mean * 1e4, 2),
        "ann_ret_%": round(ann_ret * 100, 1),
        "sharpe": round(mean / sd * ANNUALISE, 2) if sd else float("nan"),
        "sortino": round(mean / dd_std * ANNUALISE, 2) if dd_std else float("nan"),
        "calmar": round(ann_ret / abs(maxdd), 2) if maxdd < 0 else float("nan"),
        "pf": round(pos / neg, 2) if neg > 0 else float("nan"),
        "win%": round(float((r > 0).mean()) * 100, 1),
        "maxdd_%": round(maxdd * 100, 1),
    }


def boot_ci(r: np.ndarray, n=4000):
    r = r[~np.isnan(r)]
    if len(r) < 5:
        return (float("nan"), float("nan"))
    m = r[RNG.integers(0, len(r), size=(n, len(r)))].mean(axis=1)
    return (round(np.percentile(m, 2.5) * 1e4, 2),
            round(np.percentile(m, 97.5) * 1e4, 2))


def main() -> int:
    configure_logging(level=logging.WARNING)
    storage = ParquetStorage("data")
    inst = Instrument("NIFTY", InstrumentType.INDEX)
    start_dt = pd.Timestamp(START).to_pydatetime()
    end_dt = (pd.Timestamp(END) + pd.Timedelta(hours=23, minutes=59)).to_pydatetime()

    print("=" * 100)
    print("EXP 004  STRATEGY PERFORMANCE BY FORECASTED VOLATILITY REGIME  NIFTY 5m")

    rv = daily_rv()
    fc = label_regimes(har_forecasts(rv))
    reg = dict(zip(fc["date"], fc["regime"]))
    counts = fc["regime"].value_counts().to_dict()
    print(f"Forecast-labelled days: {len(fc)}  regimes={counts}")
    print(f"(LOW=lowest 30% forecast RV, HIGH=highest 30%, NORMAL=middle 40%)")

    engines = build_engines(inst, qty=10)
    feed = ReplayFeed(storage, start_dt, end_dt, candle_specs=[("NIFTY", "5m")])
    for e in engines.values():
        feed.subscribe(e)
    n_ev = feed.run()
    print(f"Replayed {n_ev} events across {len(engines)} strategies "
          f"(slippage 3bps + 20/order applied).\n")

    regimes = ["LOW", "NORMAL", "HIGH"]
    summary = {}
    for name, eng in engines.items():
        res = eng.build_result()
        dr = daily_returns(res.equity_curve)
        # map each daily return date -> regime
        ddf = pd.DataFrame({"ret": dr.values},
                           index=pd.DatetimeIndex(dr.index).normalize())
        ddf["regime"] = [reg.get(d) for d in ddf.index]
        ddf = ddf.dropna(subset=["regime"])
        # trades by entry-day regime (holding time, count)
        by_reg_trades = {g: [] for g in regimes}
        for t in res.trades:
            g = reg.get(pd.Timestamp(t.entry_time).normalize())
            if g:
                by_reg_trades[g].append(t)
        summary[name] = {"ddf": ddf, "trades": by_reg_trades}

    # --- Per-strategy per-regime table ------------------------------------
    for name, blob in summary.items():
        ddf = blob["ddf"]
        print(f"\n### {name}")
        hdr = (f"  {'regime':<7}{'n':>4}{'mean_bps':>10}{'annRet%':>9}"
               f"{'Sharpe':>8}{'Sortino':>8}{'Calmar':>8}{'PF':>6}{'win%':>7}"
               f"{'maxDD%':>8}{'boot95_bps':>18}{'trades':>7}{'avgHoldMin':>11}")
        print(hdr)
        groups = []
        for g in regimes:
            r = ddf[ddf["regime"] == g]["ret"].to_numpy()
            groups.append(r)
            mt = regime_metrics(r)
            ci = boot_ci(r)
            tr = blob["trades"][g]
            avg_hold = (np.mean([x.hold_seconds for x in tr]) / 60.0
                        if tr else float("nan"))
            print(f"  {g:<7}{mt.get('n',0):>4}{mt.get('mean_bps','-'):>10}"
                  f"{mt.get('ann_ret_%','-'):>9}{mt.get('sharpe','-'):>8}"
                  f"{mt.get('sortino','-'):>8}{mt.get('calmar','-'):>8}"
                  f"{mt.get('pf','-'):>6}{mt.get('win%','-'):>7}"
                  f"{mt.get('maxdd_%','-'):>8}{str(ci):>18}{len(tr):>7}"
                  f"{avg_hold if isinstance(avg_hold,float) else avg_hold:>11.1f}")
        # statistical tests across regimes (daily returns)
        groups_nz = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 5]
        if len(groups_nz) == 3:
            f_stat, f_p = ss.f_oneway(*groups_nz)
            h_stat, h_p = ss.kruskal(*groups_nz)
            lo, hi = groups[0], groups[2]
            mw = ss.mannwhitneyu(hi[~np.isnan(hi)], lo[~np.isnan(lo)],
                                 alternative="two-sided")
            tt = ss.ttest_ind(hi[~np.isnan(hi)], lo[~np.isnan(lo)],
                              equal_var=False)
            print(f"  ANOVA F={f_stat:.2f} p={f_p:.3f} | "
                  f"Kruskal H={h_stat:.2f} p={h_p:.3f} | "
                  f"HIGH-vs-LOW Welch t p={tt.pvalue:.3f} MWU p={mw.pvalue:.3f}")

    # --- Walk-forward: sign consistency of (HIGH-LOW) mean per fold --------
    print("\n=== Walk-forward: sign of (HIGH mean - LOW mean) daily ret, 3 folds ===")
    dates_sorted = sorted(reg.keys())
    fold_edges = np.array_split(np.array(dates_sorted), 3)
    for name, blob in summary.items():
        ddf = blob["ddf"]
        signs = []
        for fold in fold_edges:
            fset = set(pd.DatetimeIndex(fold))
            sub = ddf[ddf.index.isin(fset)]
            hi = sub[sub["regime"] == "HIGH"]["ret"]
            lo = sub[sub["regime"] == "LOW"]["ret"]
            if len(hi) >= 5 and len(lo) >= 5:
                signs.append("+" if hi.mean() > lo.mean() else "-")
            else:
                signs.append("?")
        print(f"  {name:<20} HIGH-LOW sign by fold: {signs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
