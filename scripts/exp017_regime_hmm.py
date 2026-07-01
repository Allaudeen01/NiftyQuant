"""Experiment 017: Regime switching via a Gaussian Hidden Markov Model.

Pre-registered in docs/preregistrations/exp017_regime_hmm.md.

Questions:
  Q1  Does an unsupervised Gaussian HMM find distinct, interpretable regimes?
  Q2  Are the regimes persistent (real states, not noise)?
  Q3  Does BIC prefer K>=2 over a single Gaussian (K=1)?
  Q4  Using ONLY causal (expanding-window, filtered) regime info, does the
      regime add forward power for (a) next-day RV beyond HAR, (b) return sign?

Discipline: the full-sample Viterbi path is smoothed (uses future data) and is
used ONLY for characterisation. Every forward-predictive test uses a filtered,
expanding-refit state so there is no look-ahead. States are identified by their
RV level (not index) to survive label-switching across refits.

    python scripts/exp017_regime_hmm.py

Read-only research. No trades.
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

warnings.filterwarnings("ignore")

ANNUALISE = math.sqrt(252)
REFIT_EVERY = 21


def daily_table() -> pd.DataFrame:
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
        c = g["close"].to_numpy()
        if len(c) < 12:
            continue
        lr = np.log(c[1:] / c[:-1])
        rv = math.sqrt(float(np.sum(lr ** 2)))
        close = float(c[-1])
        ret = float(np.log(close / prev_close)) if prev_close else float("nan")
        rows.append({"date": d, "ret": ret, "rv": rv})
        prev_close = close
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    return out.dropna().reset_index(drop=True)


def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    return (X - mu) / sd, mu, sd


def _fit_hmm(X, k, seed=0, n_init=6):
    """Best-of-n_init GaussianHMM by log-likelihood."""
    from hmmlearn.hmm import GaussianHMM
    best, best_ll = None, -np.inf
    for s in range(n_init):
        m = GaussianHMM(n_components=k, covariance_type="full",
                        n_iter=300, random_state=seed + s, tol=1e-4)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll


def _bic(ll, k, d, n):
    n_params = k * (k - 1) + (k - 1) + k * d + k * d * (d + 1) // 2
    return -2 * ll + n_params * math.log(n)


def hac_ols(y, X, lags=10):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def oos_r2(actual, pred, bench):
    actual, pred, bench = map(np.asarray, (actual, pred, bench))
    sse = np.sum((actual - pred) ** 2)
    sst = np.sum((actual - bench) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def main() -> int:
    t = daily_table()
    n = len(t)
    feats = ["ret", "log_rv"]
    Xraw = t[feats].to_numpy()
    X, _, _ = _standardize(Xraw)
    d = X.shape[1]
    print("=" * 92)
    print("EXP 017  REGIME SWITCHING via GAUSSIAN HMM  (features: daily return, log-RV)")
    print(f"Trading days: {n}  | {t['date'].min().date()} -> {t['date'].max().date()}")

    # --- Q1/Q2/Q3: full-sample characterisation ----------------------------
    from hmmlearn.hmm import GaussianHMM
    # K=1 baseline log-likelihood (single Gaussian)
    m1 = GaussianHMM(n_components=1, covariance_type="full", n_iter=50,
                     random_state=0).fit(X)
    bic1 = _bic(m1.score(X), 1, d, n)
    print(f"\n--- model selection (BIC, lower better) ---")
    print(f"  K=1 (single Gaussian): BIC={bic1:.1f}")
    best_k, best_model, best_bic = 1, m1, bic1
    for k in (2, 3):
        mk, llk = _fit_hmm(X, k, seed=0)
        if mk is None:
            print(f"  K={k}: fit failed")
            continue
        bick = _bic(llk, k, d, n)
        print(f"  K={k}: logL={llk:.1f}  BIC={bick:.1f}")
        if bick < best_bic:
            best_k, best_model, best_bic = k, mk, bick
    print(f"  --> BIC-preferred K = {best_k}")

    # characterise the BIC-preferred model (or K=2 if BIC picks 1, for description)
    desc_k = max(best_k, 2)
    model = best_model if best_k >= 2 else _fit_hmm(X, 2, seed=0)[0]
    states = model.predict(X)                       # smoothed Viterbi (DESCRIPTION only)
    print(f"\n--- state characterisation (K={model.n_components}, full-sample Viterbi) ---")
    # unstandardize means for reporting
    order = np.argsort(model.means_[:, feats.index("log_rv")])  # by vol level
    durations = []
    for rank, si in enumerate(order):
        mask = states == si
        n_s = int(mask.sum())
        if n_s == 0:
            continue
        mean_ret = t["ret"].to_numpy()[mask].mean()
        mean_rv_ann = t["rv"].to_numpy()[mask].mean() * ANNUALISE
        p_ii = model.transmat_[si, si]
        dur = 1.0 / (1.0 - p_ii) if p_ii < 1 else float("inf")
        durations.append(dur)
        tag = ["CALM", "TURBULENT", "EXTREME"][min(rank, 2)]
        print(f"  state {si} [{tag:<9}] n={n_s:3d} ({n_s/n*100:4.1f}%)  "
              f"mean ret {mean_ret*100:+.3f}%/day  ann-vol {mean_rv_ann*100:4.1f}%  "
              f"self-persist {p_ii:.2f}  exp-duration {dur:.1f}d")

    rv_by_state = [t["rv"].to_numpy()[states == si].mean() for si in order]
    rv_ratio = (max(rv_by_state) / min(rv_by_state)) if min(rv_by_state) > 0 else float("inf")
    mean_dur = float(np.mean([x for x in durations if np.isfinite(x)]))
    regimes_real = (best_k >= 2) and (rv_ratio >= 1.5) and (mean_dur >= 3.0)
    print(f"  RV ratio (highest/lowest state) = {rv_ratio:.2f}x ; "
          f"mean expected duration = {mean_dur:.1f}d")
    print(f"  --> Q1-Q3 REGIMES REAL: {regimes_real}")

    # --- Q4: causal (filtered, expanding-refit) forward test ----------------
    print("\n--- Q4: causal regime (K=2, filtered, refit every "
          f"{REFIT_EVERY}d, no look-ahead) ---")
    p_high = np.full(n, np.nan)                     # causal P(high-vol state) at t
    start = max(80, n // 4)
    cur_model = None
    for i in range(start, n):
        if cur_model is None or (i - start) % REFIT_EVERY == 0:
            cur_model, _ = _fit_hmm(X[:i], 2, seed=0, n_init=4)
        if cur_model is None:
            continue
        # filtered posterior at the last obs of the sequence [0..i]
        gamma = cur_model.predict_proba(X[:i + 1])
        hi_state = int(np.argmax(cur_model.means_[:, feats.index("log_rv")]))
        p_high[i] = gamma[-1, hi_state]

    df = pd.DataFrame({
        "log_rv": t["log_rv"].to_numpy(),
        "ret": t["ret"].to_numpy(),
        "p_high": p_high,
    })
    s = df["log_rv"]
    har = pd.DataFrame({
        "y": s.shift(-1),
        "d": s.shift(0),
        "w": s.rolling(5).mean(),
        "m": s.rolling(22).mean(),
        "p_high": df["p_high"],
        "ret_next": df["ret"].shift(-1),
    }).dropna().reset_index(drop=True)

    # (a) RV forecast increment
    base = hac_ols(har["y"].to_numpy(), har[["d", "w", "m"]].to_numpy())
    aug = hac_ols(har["y"].to_numpy(), har[["d", "w", "m", "p_high"]].to_numpy())
    ph_t = aug.tvalues[-1]
    print(f"  (a) RV: p_high coef={aug.params[-1]:+.3f}  HAC t={ph_t:+.2f}  "
          f"(in-sample R2 {base.rsquared:.3f} -> {aug.rsquared:.3f})")
    N = len(har); st = N // 2
    bounds = np.linspace(st, N, 5, dtype=int)
    incr = []
    for i in range(4):
        lo, hi = bounds[i], bounds[i + 1]
        tr, te = har.iloc[:lo], har.iloc[lo:hi]
        if len(te) < 20:
            continue
        def fp(cols):
            Xtr = np.column_stack([np.ones(len(tr)), tr[cols].to_numpy()])
            Xte = np.column_stack([np.ones(len(te)), te[cols].to_numpy()])
            b = sm.OLS(tr["y"].to_numpy(), Xtr).fit().params
            return Xte @ b
        r2 = oos_r2(te["y"], fp(["d", "w", "m", "p_high"]), fp(["d", "w", "m"]))
        incr.append(r2)
    mean_incr = float(np.mean(incr)) if incr else float("nan")
    n_pos = sum(1 for x in incr if x > 0)
    q4a = "ADDS RV VALUE" if abs(ph_t) >= 2.5 and mean_incr > 0 and n_pos >= 3 else "NO INCREMENTAL VALUE over HAR"
    print(f"      walk-forward mean incr OOS R2 = {mean_incr:+.4f} ({n_pos}/{len(incr)} folds +)"
          f"  --> {q4a}")

    # (b) direction: next-day return by causal high/low-vol state
    hi_mask = har["p_high"] > 0.5
    ret_hi = har.loc[hi_mask, "ret_next"].to_numpy()
    ret_lo = har.loc[~hi_mask, "ret_next"].to_numpy()
    if len(ret_hi) >= 5 and len(ret_lo) >= 5:
        tt = ss.ttest_ind(ret_hi, ret_lo, equal_var=False)
        print(f"  (b) next-day mean ret: high-vol {ret_hi.mean()*100:+.3f}% "
              f"(n={len(ret_hi)}) vs low-vol {ret_lo.mean()*100:+.3f}% (n={len(ret_lo)})"
              f"  Welch p={tt.pvalue:.3f}")
        q4b = ("PREDICTS DIRECTION" if tt.pvalue < 0.01 else "NO DIRECTIONAL EDGE")
    else:
        q4b = "INSUFFICIENT"
    print(f"      --> {q4b}")

    print("\n" + "=" * 92)
    print("SUMMARY (pre-registered):")
    print(f"  Q1-Q3 regimes real & persistent: {regimes_real} "
          f"(BIC K={best_k}, RV ratio {rv_ratio:.1f}x, dur {mean_dur:.1f}d)")
    print(f"  Q4a regime -> next-day RV beyond HAR: {q4a}")
    print(f"  Q4b regime -> next-day direction: {q4b}")
    print("  Interpretation: HMM regimes (if real) re-describe volatility clustering")
    print("  (Exp 003); useful for risk context/labeling, alpha only if Q4 passes.")
    print("  Scope: characterisation only. No trading rule implied.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
