"""Experiment 005: regime-conditional ALPHA (strategy return - Buy&Hold return).

Removes the market-direction confound found in Exp 004. For each strategy and
each trading day:  alpha = strategy_daily_return - buyhold_daily_return.
We then repeat the regime analysis on DAILY ALPHA, with full statistics,
Benjamini-Hochberg FDR correction across strategies, and walk-forward sign
checks.

    python scripts/regime_alpha.py
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
from statsmodels.stats.multitest import multipletests

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

ANN = math.sqrt(252)
RNG = np.random.default_rng(17)
START, END = "2024-06-20", "2026-06-19"


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


def regimes(rv: pd.DataFrame) -> dict:
    s = rv["log_rv"]
    feat = pd.DataFrame({"d": s.shift(1), "w": s.shift(1).rolling(5).mean(),
                         "m": s.shift(1).rolling(22).mean(), "y": s,
                         "date": rv["date"]}).dropna().reset_index(drop=True)
    preds = [np.nan] * len(feat)
    for i in range(60, len(feat)):
        tr = feat.iloc[:i]
        fit = sm.OLS(tr["y"].to_numpy(),
                     sm.add_constant(tr[["d", "w", "m"]].to_numpy())).fit()
        xr = np.r_[1.0, feat.loc[i, ["d", "w", "m"]].to_numpy(dtype=float)]
        preds[i] = float(fit.predict(xr.reshape(1, -1))[0])
    feat["forecast"] = preds
    feat = feat.dropna(subset=["forecast"])
    lo, hi = feat["forecast"].quantile([0.30, 0.70])
    lab = np.where(feat["forecast"] < lo, "LOW",
                   np.where(feat["forecast"] >= hi, "HIGH", "NORMAL"))
    return dict(zip(feat["date"], lab))


def build_engines(inst, qty):
    strat = {
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
    eng = {}
    for n, s in strat.items():
        eng[n] = BacktestEngine(
            s, portfolio=Portfolio(starting_cash=1_000_000.0),
            risk_engine=BasicRiskEngine(default_quantity=qty),
            broker=SimulatedBroker(fill_model=PercentSlippage(0.0003),
                                   fee_per_order=20.0))
    return eng


def daily_ret(equity: pd.Series) -> pd.Series:
    d = equity.resample("1D").last().dropna()
    r = d.pct_change().dropna()
    r.index = pd.DatetimeIndex(r.index).normalize()
    return r


def stat_block(a: np.ndarray) -> dict:
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 5:
        return {"n": n}
    mean = a.mean(); sd = a.std(ddof=1)
    dn = a[a < 0]
    dstd = math.sqrt((dn ** 2).mean()) if dn.size else 0.0
    eq = np.cumprod(1 + a); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    pos = a[a > 0].sum(); neg = -a[a < 0].sum()
    bs = a[RNG.integers(0, n, size=(4000, n))].mean(axis=1)
    return {
        "n": n, "mean_bps": mean * 1e4, "ann_%": mean * 252 * 100,
        "sharpe": mean / sd * ANN if sd else float("nan"),
        "sortino": mean / dstd * ANN if dstd else float("nan"),
        "pf": pos / neg if neg > 0 else float("nan"),
        "mdd_%": mdd * 100, "hit_%": float((a > 0).mean()) * 100,
        "median_bps": float(np.median(a)) * 1e4,
        "ci": (float(np.percentile(bs, 2.5)) * 1e4,
               float(np.percentile(bs, 97.5)) * 1e4),
    }


def main() -> int:
    configure_logging(level=logging.WARNING)
    storage = ParquetStorage("data")
    inst = Instrument("NIFTY", InstrumentType.INDEX)
    s_dt = pd.Timestamp(START).to_pydatetime()
    e_dt = (pd.Timestamp(END) + pd.Timedelta(hours=23, minutes=59)).to_pydatetime()

    print("=" * 100)
    print("EXP 005  REGIME-CONDITIONAL ALPHA (strategy - Buy&Hold)  NIFTY 5m")
    reg = regimes(daily_rv())

    eng = build_engines(inst, qty=10)
    feed = ReplayFeed(storage, s_dt, e_dt, candle_specs=[("NIFTY", "5m")])
    for e in eng.values():
        feed.subscribe(e)
    feed.run()

    rets = {n: daily_ret(e.build_result().equity_curve) for n, e in eng.items()}
    bh = rets["BuyAndHold"]
    # align all to BH index
    panel = pd.DataFrame({n: r for n, r in rets.items()}).dropna()
    panel["regime"] = [reg.get(d) for d in panel.index]
    panel = panel.dropna(subset=["regime"])
    print(f"Aligned trading days: {len(panel)}  "
          f"regimes={panel['regime'].value_counts().to_dict()}\n")

    strategies = [n for n in eng if n != "BuyAndHold"]
    order = ["LOW", "NORMAL", "HIGH"]
    raw_high_low_p = {}   # for FDR: HIGH-vs-LOW alpha Welch p
    raw_anova_p = {}
    overall = {}

    for name in strategies:
        alpha = (panel[name] - panel["BuyAndHold"])
        print(f"\n### {name}   (alpha = {name} - BuyAndHold)")
        groups = []
        for g in order:
            a = alpha[panel["regime"] == g].to_numpy()
            groups.append(a)
            b = stat_block(a)
            if b.get("n", 0) >= 5:
                print(f"  {g:<7} n={b['n']:>3} meanA={b['mean_bps']:>7.2f}bps "
                      f"annA={b['ann_%']:>6.1f}% Sh={b['sharpe']:>6.2f} "
                      f"Sor={b['sortino']:>6.2f} PF={b['pf']:>5.2f} "
                      f"hit={b['hit_%']:>5.1f}% mdd={b['mdd_%']:>6.1f}% "
                      f"CI95={b['ci'][0]:>7.2f},{b['ci'][1]:>6.2f}")
        # overall (all regimes) alpha
        ov = stat_block(alpha.to_numpy())
        overall[name] = ov
        gnz = [g[~np.isnan(g)] for g in groups]
        F, pF = ss.f_oneway(*gnz)
        H, pH = ss.kruskal(*gnz)
        tt = ss.ttest_ind(gnz[2], gnz[0], equal_var=False)
        mw = ss.mannwhitneyu(gnz[2], gnz[0], alternative="two-sided")
        # one-sample: is overall mean alpha != 0 ?
        t1 = ss.ttest_1samp(alpha.to_numpy(), 0.0)
        raw_high_low_p[name] = tt.pvalue
        raw_anova_p[name] = pF
        ov["t1_p"] = t1.pvalue
        ov["F_p"] = pF
        print(f"  OVERALL alpha: mean={ov['mean_bps']:.2f}bps ann={ov['ann_%']:.1f}% "
              f"Sharpe={ov['sharpe']:.2f} CI95=({ov['ci'][0]:.2f},{ov['ci'][1]:.2f}) "
              f"1samp p={t1.pvalue:.3f}")
        print(f"  Regime tests: ANOVA p={pF:.3f} Kruskal p={pH:.3f} "
              f"HIGH-vs-LOW Welch p={tt.pvalue:.3f} MWU p={mw.pvalue:.3f}")

    # --- Benjamini-Hochberg FDR across strategies -------------------------
    names = strategies
    hl_p = [raw_high_low_p[n] for n in names]
    an_p = [raw_anova_p[n] for n in names]
    t1_p = [overall[n]["t1_p"] for n in names]
    _, hl_adj, _, _ = multipletests(hl_p, method="fdr_bh")
    _, an_adj, _, _ = multipletests(an_p, method="fdr_bh")
    _, t1_adj, _, _ = multipletests(t1_p, method="fdr_bh")

    print("\n" + "=" * 100)
    print("BENJAMINI-HOCHBERG FDR-ADJUSTED p-values")
    print(f"  {'strategy':<20}{'overallAlpha_ann%':>18}{'1samp_raw':>11}"
          f"{'1samp_adj':>11}{'ANOVA_raw':>11}{'ANOVA_adj':>11}"
          f"{'HL_raw':>9}{'HL_adj':>9}")
    for i, n in enumerate(names):
        print(f"  {n:<20}{overall[n]['ann_%']:>18.1f}{t1_p[i]:>11.3f}"
              f"{t1_adj[i]:>11.3f}{an_p[i]:>11.3f}{an_adj[i]:>11.3f}"
              f"{hl_p[i]:>9.3f}{hl_adj[i]:>9.3f}")

    # --- Walk-forward: mean alpha + sign per fold (overall and HIGH) ------
    print("\n=== Walk-forward (3 folds): overall mean ALPHA sign ===")
    dates = sorted(panel.index)
    folds = np.array_split(np.array(dates), 3)
    for name in names:
        alpha = (panel[name] - panel["BuyAndHold"])
        signs, means = [], []
        for f in folds:
            fa = alpha[alpha.index.isin(pd.DatetimeIndex(f))].to_numpy()
            fa = fa[~np.isnan(fa)]
            if len(fa) >= 5:
                means.append(fa.mean() * 1e4)
                signs.append("+" if fa.mean() > 0 else "-")
            else:
                signs.append("?")
        print(f"  {name:<20} sign={signs}  meanA_bps={[round(m,2) for m in means]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
