"""Experiment 016: Tail risk of NIFTY daily returns (read-only research).

Pre-registered in docs/preregistrations/exp016_tail_risk.md.

Questions:
  Q1  How non-normal are NIFTY daily returns (skew, excess kurtosis)?
  Q2  How fat are the tails (Hill tail index, GPD shape, finite moments)?
  Q3  Are left (crash) and right (rally) tails asymmetric?
  Q4  How badly does Gaussian VaR/ES understate true tail risk?

Uses the reusable EVT primitives in nifty_quant.analytics.stat_tests
(hill_estimator, pot_gpd_fit).

    python scripts/exp016_tail_risk.py

Characterisation for position sizing. No trades, no strategy.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

from nifty_quant.analytics.stat_tests import hill_estimator, pot_gpd_fit


def daily_returns() -> pd.DataFrame:
    """NIFTY daily close-to-close log returns from the 5m warehouse."""
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    if not files:
        raise SystemExit("No 5m NIFTY parquet found")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    closes = df.groupby("date")["close"].last()
    closes = closes[df.groupby("date").size() >= 12]     # skip partial days
    ret = np.log(closes / closes.shift(1)).dropna()
    return pd.DataFrame({"date": ret.index, "ret": ret.to_numpy()})


def gaussian_var_es(mu, sigma, q):
    """Gaussian VaR and ES (as positive loss magnitudes) at confidence q."""
    z = ss.norm.ppf(q)
    var = -(mu - sigma * z)                          # loss magnitude
    es = -(mu - sigma * ss.norm.pdf(z) / (1 - q))    # ES (Expected Shortfall)
    return var, es


def empirical_var_es(losses, q):
    """Empirical VaR/ES on loss magnitudes (positive = loss)."""
    v = np.quantile(losses, q)
    tail = losses[losses >= v]
    return v, (tail.mean() if len(tail) else float("nan"))


def main() -> int:
    r = daily_returns()
    x = r["ret"].to_numpy()
    n = len(x)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    print("=" * 92)
    print("EXP 016  NIFTY DAILY-RETURN TAIL RISK  (EVT / fat-tail characterisation)")
    print(f"Trading days: {n}  | {r['date'].min().date()} -> {r['date'].max().date()}")
    print(f"Daily return: mean {mu*100:+.3f}%  std {sd*100:.3f}%  "
          f"(annualised vol {sd*math.sqrt(252)*100:.1f}%)")

    # --- Q1: non-normality --------------------------------------------------
    skew = float(ss.skew(x))
    exk = float(ss.kurtosis(x))               # excess kurtosis (0 = normal)
    jb_stat, jb_p = ss.jarque_bera(x)
    print("\n--- Q1: non-normality ---")
    print(f"  skewness       : {skew:+.3f}  ({'left/negative' if skew < 0 else 'right/positive'})")
    print(f"  excess kurtosis: {exk:+.3f}  ({'fat' if exk > 0 else 'thin'} vs normal=0)")
    print(f"  Jarque-Bera    : stat={jb_stat:.1f}  p={jb_p:.2e}  "
          f"({'REJECT normal' if jb_p < 0.01 else 'cannot reject normal'})")

    # --- Q2/Q3: Hill + GPD, both tails --------------------------------------
    print("\n--- Q2/Q3: tail index (Hill) & GPD shape, both tails ---")
    print("  Hill alpha (larger=thinner; alpha bounds # finite moments), scan over k:")
    print(f"    {'k%':>5} {'left alpha':>12} {'right alpha':>12}")
    for kpct in (0.05, 0.08, 0.10, 0.15, 0.20):
        k = max(10, int(kpct * n))
        hl = hill_estimator(x, k=k, tail="left")
        hr = hill_estimator(x, k=k, tail="right")
        print(f"    {int(kpct*100):>4}% {hl['alpha']:>12.2f} {hr['alpha']:>12.2f}")

    gpd_l = pot_gpd_fit(x, threshold_pct=95.0, tail="left")
    gpd_r = pot_gpd_fit(x, threshold_pct=95.0, tail="right")
    print("\n  GPD (peaks-over-threshold @95th pct of each tail):")
    for name, g in (("LEFT (losses)", gpd_l), ("RIGHT (gains)", gpd_r)):
        df_str = "inf" if math.isinf(g["tail_df"]) else f"{g['tail_df']:.1f}"
        print(f"    {name:<14} xi={g['xi']:+.3f}  beta={g['beta']*100:.3f}%  "
              f"n_exceed={g['n_exceed']}  ~t-df={df_str}  "
              f"({'HEAVY' if g['xi'] > 0.05 else 'thin/exp' if g['xi'] <= 0.05 else ''})")

    # tail asymmetry
    xi_l, xi_r = gpd_l["xi"], gpd_r["xi"]
    asym = abs(xi_l - xi_r) > 0.10
    fatter = "LEFT (crashes)" if xi_l > xi_r else "RIGHT (rallies)"

    # --- σ-event counts -----------------------------------------------------
    print("\n--- sigma-event counts: observed vs Gaussian-expected ---")
    z = (x - mu) / sd
    for s in (3, 4, 5):
        obs = int(np.sum(np.abs(z) > s))
        exp = 2 * (1 - ss.norm.cdf(s)) * n
        print(f"    |ret| > {s}sigma : observed {obs:3d}   Gaussian-expected {exp:5.2f}")

    # --- Q4: VaR / ES, Gaussian vs empirical vs EVT -------------------------
    print("\n--- Q4: 1-day VaR / ES on LOSSES (Gaussian vs empirical vs EVT-GPD) ---")
    losses = -x                                        # positive = loss
    losses_pos = losses[losses > 0]
    for q in (0.99, 0.995):
        gv, ge = gaussian_var_es(mu, sd, q)
        ev, ee = empirical_var_es(losses, q)
        # EVT VaR via GPD: u + (beta/xi)[ (n/n_exceed * (1-q))^(-xi) - 1 ]
        u, xi, beta, ne = gpd_l["u"], gpd_l["xi"], gpd_l["beta"], gpd_l["n_exceed"]
        if ne > 0 and abs(xi) > 1e-6:
            evt_var = u + (beta / xi) * (((n / ne) * (1 - q)) ** (-xi) - 1)
            evt_es = (evt_var + beta - xi * u) / (1 - xi) if xi < 1 else float("nan")
        else:
            evt_var = evt_es = float("nan")
        ratio = ee / ge if ge > 0 else float("nan")
        print(f"  {int(q*1000)/10:.1f}%:")
        print(f"    VaR  Gaussian {gv*100:.2f}%  empirical {ev*100:.2f}%  EVT {evt_var*100:.2f}%")
        print(f"    ES   Gaussian {ge*100:.2f}%  empirical {ee*100:.2f}%  EVT {evt_es*100:.2f}%"
              f"   (empirical/Gaussian ES = {ratio:.2f}x)")

    gv995, ge995 = gaussian_var_es(mu, sd, 0.995)
    _, ee995 = empirical_var_es(losses, 0.995)
    understates = (ee995 / ge995) > 1.20 if ge995 > 0 else False

    # --- verdicts -----------------------------------------------------------
    heavy = (jb_p < 0.01 and xi_l > 0.05
             and (math.isinf(gpd_l["tail_df"]) is False and gpd_l["tail_df"] < 10))
    print("\n" + "=" * 92)
    print("VERDICTS (pre-registered):")
    print(f"  Q1/Q2 tails: {'HEAVY-TAILED' if heavy else 'NOT clearly heavy (see numbers)'}"
          f"  (JB reject={jb_p < 0.01}, left GPD xi={xi_l:+.3f})")
    print(f"  Q3 asymmetry: {'ASYMMETRIC' if asym else 'roughly symmetric'} "
          f"(left xi={xi_l:+.3f} vs right xi={xi_r:+.3f}; fatter tail = {fatter})")
    print(f"  Q4 Gaussian ES understatement @99.5%: "
          f"{'YES (>20%)' if understates else 'no (<20%)'} "
          f"(empirical/Gaussian = {ee995/ge995:.2f}x)")
    print("  Use: feeds position sizing / risk limits. NOT a trading signal.")
    print("  Caveat: ~25 exceedances above the 95th pct -> EVT shape has a wide CI;")
    print("          numbers are indicative, not precise.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
