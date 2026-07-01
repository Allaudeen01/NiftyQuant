"""Experiment 013: Realized-Volatility forecast comparison (read-only research).

Pre-registered in docs/preregistrations/exp013_rv_forecast_comparison.md.

ONE question: among standard volatility models, which forecasts NIFTY next-day
realized volatility (RV) best OUT-OF-SAMPLE, and is any difference statistically
significant?

Models (all standard, no bespoke tuning):
  RW      -- random walk: RV_{t+1} = RV_t
  EWMA    -- RiskMetrics EWMA variance, lambda = 0.94 (fixed)
  HAR     -- HAR-RV (1/5/22 day) in log space (the incumbent)
  GARCH   -- GARCH(1,1) on daily returns (arch)
  EGARCH  -- EGARCH(1,1) leverage/asymmetry (arch)

LSTM is DELIBERATELY EXCLUDED: ~470 daily obs cannot support it without
overfitting (see roadmap / prereg).

Scoring on the common out-of-sample set: RMSE, MAE, QLIKE (primary), and
Diebold-Mariano tests (HAC) vs HAR under QLIKE, with Benjamini-Hochberg
multiple-testing correction.

    python scripts/exp013_rv_forecast_comparison.py

No parameters are optimised on the test set. No trades. Pure forecasting.
"""

from __future__ import annotations

import glob
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss

warnings.filterwarnings("ignore")  # arch/statsmodels convergence chatter

ANNUALISE = math.sqrt(252)
EWMA_LAMBDA = 0.94          # RiskMetrics daily (fixed, not fitted)
TRAIN_FRAC = 0.50          # first half is the initial training window


# --- data -------------------------------------------------------------------

def daily_rv() -> pd.DataFrame:
    """One row per trading day: intraday RV (5m open-to-close) + daily return.

    rv        = sqrt(sum of squared 5m log returns)  [daily standard deviation]
    ret_cc    = close-to-close daily log return (for GARCH/EGARCH, in %)
    """
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    if not files:
        raise SystemExit("No 5m NIFTY parquet found")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()

    rows = []
    prev_close = None
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        if len(g) < 12:
            continue
        closes = g["close"].to_numpy()
        logret = np.log(closes[1:] / closes[:-1])
        rv = math.sqrt(float(np.sum(logret ** 2)))
        close = float(closes[-1])
        ret_cc = float(np.log(close / prev_close)) if prev_close else float("nan")
        rows.append({"date": d, "year": d.year, "rv": rv, "ret_cc": ret_cc})
        prev_close = close
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    out["ret_pct"] = out["ret_cc"] * 100.0   # arch prefers ~percent-scale returns
    return out


# --- forecasters (each returns a 1-step-ahead RV level forecast) ------------

def har_forecast(train_log_rv: np.ndarray) -> float:
    """Fit HAR on training log-RV, forecast next-day RV level (undo log)."""
    s = pd.Series(train_log_rv)
    d = s.shift(0)
    w = s.rolling(5).mean()
    m = s.rolling(22).mean()
    frame = pd.DataFrame({"y": s.shift(-1), "d": d, "w": w, "m": m}).dropna()
    if len(frame) < 30:
        return float("nan")
    X = sm.add_constant(frame[["d", "w", "m"]].to_numpy())
    fit = sm.OLS(frame["y"].to_numpy(), X).fit()
    # features for the LAST available day -> predict next
    last_d = s.iloc[-1]
    last_w = s.iloc[-5:].mean()
    last_m = s.iloc[-22:].mean()
    log_pred = fit.params[0] + fit.params[1]*last_d + fit.params[2]*last_w + fit.params[3]*last_m
    return float(math.exp(log_pred))


def ewma_forecast(train_rv: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """RiskMetrics EWMA of daily variance; forecast = sqrt(EWMA variance)."""
    var = train_rv[0] ** 2
    for r in train_rv[1:]:
        var = lam * var + (1 - lam) * (r ** 2)
    return float(math.sqrt(max(var, 1e-18)))


def garch_forecast(train_ret_pct: np.ndarray, model: str) -> float:
    """GARCH(1,1) or EGARCH(1,1) 1-step variance forecast -> daily RV level.

    arch works in percent; convert the forecast sigma back to fraction.
    """
    from arch import arch_model
    r = train_ret_pct[~np.isnan(train_ret_pct)]
    if len(r) < 60:
        return float("nan")
    r = r - r.mean()
    try:
        if model == "GARCH":
            am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        else:  # EGARCH
            am = arch_model(r, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=1, reindex=False)
        var_pct = float(fc.variance.values[-1, 0])   # in percent^2
        return math.sqrt(max(var_pct, 1e-12)) / 100.0
    except Exception:
        return float("nan")


# --- losses & Diebold-Mariano ----------------------------------------------

def qlike(rv_actual: np.ndarray, rv_pred: np.ndarray) -> np.ndarray:
    """Per-obs QLIKE loss on variance: RV^2/f^2 - ln(RV^2/f^2) - 1."""
    a2 = rv_actual ** 2
    f2 = np.clip(rv_pred ** 2, 1e-18, None)
    x = a2 / f2
    return x - np.log(x) - 1.0


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """DM stat & two-sided p for H0: equal expected loss (a vs b), HAC-corrected.

    d = loss_a - loss_b. Positive mean(d) => a worse than b.
    Newey-West long-run variance with automatic small lag.
    """
    d = loss_a - loss_b
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 20:
        return (float("nan"), float("nan"))
    dbar = d.mean()
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Newey-West rule of thumb
    gamma0 = np.sum((d - dbar) ** 2) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        cov = np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
        lrv += 2 * w * cov
    if lrv <= 0:
        return (float("nan"), float("nan"))
    dm = dbar / math.sqrt(lrv / n)
    p = 2 * (1 - ss.norm.cdf(abs(dm)))
    return (float(dm), float(p))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH-adjusted p-values (same order as input)."""
    arr = np.array(pvals, dtype=float)
    ok = ~np.isnan(arr)
    m = ok.sum()
    adj = np.full_like(arr, np.nan)
    if m == 0:
        return adj.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(arr[idx])]
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = arr[i] * m / k
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj.tolist()


# --- walk-forward driver ----------------------------------------------------

MODELS = ["RW", "EWMA", "HAR", "GARCH", "EGARCH"]


def walk_forward(rv: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead expanding walk-forward forecasts for all models.

    Returns a frame of OOS rows: date, actual RV, and each model's forecast.
    Fitted models (HAR/GARCH/EGARCH) refit every REFIT_EVERY days; between
    refits their PARAMETERS are frozen but inputs still roll forward
    (EWMA/RW update every day by construction).
    """
    n = len(rv)
    start = int(n * TRAIN_FRAC)
    rv_arr = rv["rv"].to_numpy()
    logrv_arr = rv["log_rv"].to_numpy()
    ret_arr = rv["ret_pct"].to_numpy()
    dates = rv["date"].to_numpy()

    out_rows = []
    for t in range(start, n - 1):  # forecast t+1 using data through t
        row = {"date": dates[t + 1], "actual": rv_arr[t + 1]}
        # All models re-estimated on the expanding window through day t.
        # The sample is small enough (~230 OOS days) to refit every step, which
        # is the correct, look-ahead-free choice; no parameter caching needed.
        row["RW"] = rv_arr[t]                              # tomorrow = today
        row["EWMA"] = ewma_forecast(rv_arr[: t + 1])       # RiskMetrics EWMA
        row["HAR"] = har_forecast(logrv_arr[: t + 1])      # HAR-RV
        row["GARCH"] = garch_forecast(ret_arr[: t + 1], "GARCH")
        row["EGARCH"] = garch_forecast(ret_arr[: t + 1], "EGARCH")
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main() -> int:
    rv = daily_rv()
    n = len(rv)
    print("=" * 92)
    print("EXP 013  RV FORECAST COMPARISON  HAR vs EWMA vs GARCH vs EGARCH (vs RW)")
    print(f"Trading days: {n}  | {rv['date'].min().date()} -> {rv['date'].max().date()}")
    print(f"Mean annualised RV: {(rv['rv'].mean()*ANNUALISE)*100:.1f}%")
    print(f"Walk-forward: expanding, initial train = {int(n*TRAIN_FRAC)} days, "
          f"one-step-ahead, all models refit every step")

    fc = walk_forward(rv)
    # keep only rows where all models produced a finite forecast (fair compare)
    cols = MODELS
    fc_valid = fc.dropna(subset=cols + ["actual"]).reset_index(drop=True)
    m_oos = len(fc_valid)
    print(f"\nCommon out-of-sample days (all models finite): {m_oos}")
    if m_oos < 40:
        print("INSUFFICIENT DATA: common OOS set too small for reliable DM.")
        return 0

    actual = fc_valid["actual"].to_numpy()

    # --- metrics ------------------------------------------------------------
    print("\n--- Out-of-sample loss (lower is better) ---")
    print(f"  {'model':<8}{'RMSE(bp)':>12}{'MAE(bp)':>12}{'QLIKE':>12}")
    losses_q = {}
    stats = {}
    for md in cols:
        pred = fc_valid[md].to_numpy()
        rmse = math.sqrt(np.mean((actual - pred) ** 2)) * 1e4  # in basis points of daily RV
        mae = np.mean(np.abs(actual - pred)) * 1e4
        ql = qlike(actual, pred)
        losses_q[md] = ql
        stats[md] = (rmse, mae, float(np.mean(ql)))
        print(f"  {md:<8}{rmse:>12.2f}{mae:>12.2f}{np.mean(ql):>12.4f}")

    ranked = sorted(cols, key=lambda m: stats[m][2])  # by QLIKE
    best = ranked[0]
    print(f"\nLowest QLIKE: {best}")

    # --- Diebold-Mariano vs HAR (primary) -----------------------------------
    print("\n--- Diebold-Mariano vs HAR (QLIKE loss, HAC). "
          "d=loss_model - loss_HAR; DM<0 => model BEATS HAR ---")
    comp = [m for m in cols if m != "HAR"]
    dm_stats, dm_ps = {}, []
    for md in comp:
        dm, p = diebold_mariano(losses_q[md], losses_q["HAR"])
        dm_stats[md] = (dm, p)
        dm_ps.append(p)
    bh = benjamini_hochberg(dm_ps)
    print(f"  {'model':<8}{'DM':>10}{'p':>12}{'BH p':>12}   verdict vs HAR")
    for md, bhp in zip(comp, bh):
        dm, p = dm_stats[md]
        if math.isnan(dm):
            verd = "n/a"
        elif bhp < 0.05 and dm < 0:
            verd = "BEATS HAR"
        elif bhp < 0.05 and dm > 0:
            verd = "worse than HAR"
        else:
            verd = "no sig. difference"
        print(f"  {md:<8}{dm:>10.2f}{p:>12.4f}{bhp:>12.4f}   {verd}")

    # --- sub-period stability of the best vs HAR ----------------------------
    print("\n--- Sub-period sign stability (best model vs HAR, QLIKE mean diff) ---")
    if best != "HAR":
        d = losses_q[best] - losses_q["HAR"]
        k = 4
        bounds = np.linspace(0, len(d), k + 1, dtype=int)
        signs = []
        for i in range(k):
            seg = d[bounds[i]:bounds[i + 1]]
            md = float(np.mean(seg)) if len(seg) else float("nan")
            signs.append(md)
            tag = "beats HAR" if md < 0 else "worse"
            print(f"  sub-period {i+1}: mean QLIKE diff = {md:+.4f}  ({tag})")
        n_beats = sum(1 for s in signs if s < 0)
    else:
        n_beats = 0
        print("  best model IS HAR.")

    # --- locked verdict -----------------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT (against pre-registered rules):")
    if best == "HAR":
        print("  HAR CONFIRMED BEST (lowest QLIKE). Exp 012's HAR-based conclusion stands.")
    else:
        dm, p = dm_stats[best]
        bhp = dict(zip(comp, bh))[best]
        if not math.isnan(dm) and bhp < 0.05 and dm < 0 and n_beats >= 3:
            print(f"  CHALLENGER WINS: {best} beats HAR (BH p={bhp:.4f}, "
                  f"{n_beats}/4 sub-periods). Adopt/study {best}; re-examine Exp 012.")
        elif not math.isnan(dm) and bhp < 0.05 and dm < 0:
            print(f"  INCONCLUSIVE: {best} has lower QLIKE and beats HAR (BH p={bhp:.4f}) "
                  f"but sign unstable ({n_beats}/4 sub-periods). Descriptive only.")
        else:
            print(f"  INCONCLUSIVE: {best} has lowest QLIKE but does NOT clear the DM "
                  f"significance bar vs HAR. No model change; ranking is descriptive.")
    print("  Scope: forecasting comparison only. No trading rule implied.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
