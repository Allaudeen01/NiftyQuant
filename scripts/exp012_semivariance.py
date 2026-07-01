"""Experiment 012 (LOCKED pre-registration): semivariance-ratio incremental RV.

Executes docs/preregistrations/exp012_semivariance_ratio.md against the fixed
decision rules. The decisive test: does day-t downside/upside semivariance ratio
predict day-t+1 log RV beyond a LEVERAGE-AUGMENTED HAR baseline (HAR + return-
sign dummy), not merely plain HAR? If it only beats plain HAR -> REDUNDANT with
the Exp 006A leverage effect.

    python scripts/exp012_semivariance.py
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

RNG = np.random.default_rng(1012)


def build_daily() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df = df[(df["timestamp"].dt.strftime("%H:%M") >= "09:15")
            & (df["timestamp"].dt.strftime("%H:%M") <= "15:30")]
    rows = []
    for d, g in df.groupby("date"):
        c = g.sort_values("timestamp")["close"].to_numpy()
        o = float(g.sort_values("timestamp")["open"].iloc[0])
        if len(c) < 20:
            continue
        r = np.log(c[1:] / c[:-1])
        rv = math.sqrt(float(np.sum(r ** 2)))
        rs_down = float(np.sum(r[r < 0] ** 2))
        rs_up = float(np.sum(r[r > 0] ** 2))
        semivar_ratio = (rs_down / rs_up) if rs_up > 0 else np.nan
        day_ret = c[-1] / o - 1.0
        rows.append({"date": d, "year": d.year, "rv": rv,
                     "semivar_ratio": semivar_ratio,
                     "neg": 1.0 if day_ret < 0 else 0.0})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    # HAR terms (lagged) + lag the new variable and the leverage dummy
    s = out["log_rv"]
    out["har_d"] = s.shift(1)
    out["har_w"] = s.shift(1).rolling(5).mean()
    out["har_m"] = s.shift(1).rolling(22).mean()
    out["y"] = s
    out["semivar_ratio_lag"] = out["semivar_ratio"].shift(1)
    out["neg_lag"] = out["neg"].shift(1)
    return out.dropna(subset=["har_d", "har_w", "har_m", "y",
                              "semivar_ratio_lag", "neg_lag"]).reset_index(drop=True)


def hac(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 10})


def oos_r2_incremental(d, base_cols, full_cols, n_folds=4):
    n = len(d); start = n // 2
    edges = np.linspace(start, n, n_folds + 1, dtype=int)
    incr = []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        tr, te = d.iloc[:a], d.iloc[a:b]
        if len(te) < 15:
            continue
        y = te["y"].to_numpy()
        bench = np.full(len(y), tr["y"].mean())
        sst = np.sum((y - bench) ** 2)
        if sst == 0:
            continue
        def r2(cols):
            f = sm.OLS(tr["y"].to_numpy(), sm.add_constant(tr[cols].to_numpy())).fit()
            p = f.predict(sm.add_constant(te[cols].to_numpy(), has_constant="add"))
            return 1 - np.sum((y - p) ** 2) / sst
        incr.append(r2(full_cols) - r2(base_cols))
    return incr


def main() -> int:
    d = build_daily()
    print("=" * 92)
    print("EXPERIMENT 012 (LOCKED) — Semivariance-ratio incremental RV predictor")
    print(f"Daily observations: {len(d)}  ({d['date'].min().date()} -> {d['date'].max().date()})")
    print(f"semivar_ratio: mean {d['semivar_ratio_lag'].mean():.2f}  "
          f"median {d['semivar_ratio_lag'].median():.2f}")

    HAR = ["har_d", "har_w", "har_m"]
    LEV = HAR + ["neg_lag"]                       # leverage-augmented baseline
    FULL = LEV + ["semivar_ratio_lag"]            # decisive model

    # --- Full-sample HAC on the decisive (leverage-augmented) model ------
    print("\n--- Decisive model: log_RV_next ~ HAR + neg + semivar_ratio (HAC) ---")
    m = hac(d["y"].to_numpy(), d[FULL].to_numpy())
    names = ["const"] + FULL
    pr = np.asarray(m.params); tv = np.asarray(m.tvalues); pv = np.asarray(m.pvalues)
    for nm, b, t, p in zip(names, pr, tv, pv):
        mark = " <== new var" if nm == "semivar_ratio_lag" else ""
        print(f"    {nm:<18} coef={b:+.4f}  t={t:+.2f}  p={p:.2e}{mark}")
    sv_t = tv[names.index("semivar_ratio_lag")]
    sv_p = pv[names.index("semivar_ratio_lag")]

    # --- Bootstrap CI (day-block) for semivar_ratio coef in FULL ---------
    idx = np.arange(len(d))
    coefs = []
    for _ in range(1000):
        samp = RNG.choice(idx, len(idx), replace=True)
        sub = d.iloc[samp]
        try:
            f = sm.OLS(sub["y"].to_numpy(), sm.add_constant(sub[FULL].to_numpy())).fit()
            coefs.append(f.params[-1])
        except Exception:
            pass
    lo, hi = np.percentile(coefs, [2.5, 97.5])
    print(f"\n  bootstrap 95% CI for semivar_ratio coef: [{lo:+.4f}, {hi:+.4f}]  "
          f"({'excludes 0' if (lo>0 or hi<0) else 'includes 0'})")

    # --- Incremental OOS R2: over LEVERAGE baseline (decisive) & plain HAR
    incr_lev = oos_r2_incremental(d, LEV, FULL)
    incr_har = oos_r2_incremental(d, HAR, HAR + ["semivar_ratio_lag"])
    print("\n--- Walk-forward incremental OOS R2 (4 folds) ---")
    print(f"  over LEVERAGE-augmented baseline (DECISIVE): "
          f"{[round(x,4) for x in incr_lev]}  mean={np.mean(incr_lev):+.4f}  "
          f"folds+={sum(1 for x in incr_lev if x>0)}/{len(incr_lev)}")
    print(f"  over plain HAR baseline (context):           "
          f"{[round(x,4) for x in incr_har]}  mean={np.mean(incr_har):+.4f}  "
          f"folds+={sum(1 for x in incr_har if x>0)}/{len(incr_har)}")

    # --- Per-year sign/significance in FULL model ------------------------
    print("\n--- Per-year semivar_ratio coef (in leverage-augmented model) ---")
    year_signs = []
    for y in sorted(d["year"].unique()):
        sub = d[d["year"] == y]
        if len(sub) < 40:
            print(f"  {y}: insufficient (n={len(sub)})"); continue
        f = hac(sub["y"].to_numpy(), sub[FULL].to_numpy())
        cf = np.asarray(f.params)[-1]; tt = np.asarray(f.tvalues)[-1]
        year_signs.append(np.sign(cf))
        print(f"  {y}: n={len(sub):3d}  coef={cf:+.4f}  t={tt:+.2f}")

    # --- Verdict against LOCKED decision rules ---------------------------
    mean_incr_lev = float(np.mean(incr_lev)) if incr_lev else float("nan")
    folds_pos = sum(1 for x in incr_lev if x > 0)
    ci_excl = (lo > 0 or hi < 0)
    sig_t = abs(sv_t) >= 2.5
    year_stable = len(set(year_signs)) == 1 and len(year_signs) >= 2
    econ = mean_incr_lev > 0.01

    print("\n" + "=" * 92)
    print("VERDICT (against LOCKED pre-registration rules):")
    print(f"  HAC |t|>=2.5 in full model .......... {sig_t}  (t={sv_t:+.2f})")
    print(f"  bootstrap CI excludes 0 ............. {ci_excl}")
    print(f"  incr OOS R2>0 in >=3/4 folds (lev) .. {folds_pos>=3}  ({folds_pos}/4)")
    print(f"  sign stable across years ............ {year_stable}")
    print(f"  mean incr OOS R2 > 0.01 (lev) ....... {econ}  ({mean_incr_lev:+.4f})")

    supported = sig_t and ci_excl and folds_pos >= 3 and year_stable and econ
    # redundant = beats plain HAR but not leverage-augmented
    beats_har = (np.mean(incr_har) > 0) if incr_har else False
    beats_lev = mean_incr_lev > 0
    if supported:
        verdict = "SUPPORTED"
    elif beats_har and not beats_lev:
        verdict = "REDUNDANT with the Exp 006A leverage effect"
    else:
        verdict = "REJECTED"
    print(f"\n  ==> {verdict}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
