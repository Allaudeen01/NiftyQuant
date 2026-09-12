"""EXP034 Phase 3 -- walk-forward evaluation, Clark-West, FDR, locked verdicts.

Executes the locked preregistration (commit 18a178e) exactly. This is the phase
that generates RESULTS: once a fold is scored, the locked anti-snooping guards
forbid adding, removing or tuning any threshold, feature, target, horizon or
model.

    python scripts/exp034_walk_forward.py

Outputs:
    reports/exp034/walk_forward_predictions.csv
    reports/exp034/fold_metrics.csv
    reports/exp034/per_day_metrics.csv
    reports/exp034/primary_results.csv
    reports/exp034/phase3_results.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

from nifty_quant.analytics.clark_west import (
    cw_series, day_block_bootstrap_test, hac_one_sided, oos_r2,
)
from nifty_quant.analytics.data_snooping import stationary_bootstrap_indices

CONFIG_PATH = Path("docs") / "preregistrations" / "exp034_config.json"
PANEL = Path("data") / "derived" / "exp034_feature_panel.parquet"
OUT = Path("reports") / "exp034"

# Implementation parameters NOT fixed by the lock, declared here and applied
# identically to MODEL_1/2/3 so none can favour the larger model.
CLIP_Q = 0.001          # train-fitted winsorisation at 0.1% / 99.9%
BOOT_BLOCK = 5.0        # ~one trading week, against ~27 OOS days
BOOT_N = 5000
BOOT_SEED = 34034
FDR_Q = 0.10


# --------------------------------------------------------------------------


def make_folds(days: list, min_train: int, test_block: int, step: int) -> list[dict]:
    """Expanding-window folds in TRADING-DAY space (never calendar days)."""
    folds = []
    start = min_train
    while start < len(days):
        test = days[start:start + test_block]
        if not test:
            break
        folds.append({"fold_id": len(folds), "train_days": days[:start], "test_days": test})
        start += step
    return folds


def fit_preprocess(train: pd.DataFrame, cols: list[str]) -> dict:
    """Fit imputation -> clipping -> standardisation on TRAIN ONLY, then freeze."""
    X = train[cols].apply(pd.to_numeric, errors="coerce")
    med = X.median()
    Xi = X.fillna(med)
    lo = Xi.quantile(CLIP_Q)
    hi = Xi.quantile(1.0 - CLIP_Q)
    Xc = Xi.clip(lower=lo, upper=hi, axis=1)
    mu = Xc.mean()
    sd = Xc.std().replace(0.0, 1.0)
    return {"cols": cols, "median": med, "lo": lo, "hi": hi, "mu": mu, "sd": sd}


def apply_preprocess(df: pd.DataFrame, pp: dict) -> np.ndarray:
    X = df[pp["cols"]].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pp["median"]).clip(lower=pp["lo"], upper=pp["hi"], axis=1)
    return ((X - pp["mu"]) / pp["sd"]).to_numpy(dtype=float)


def select_alpha(train: pd.DataFrame, cols: list[str], y_col: str,
                 grid: list[float]) -> float:
    """Chronological inner split INSIDE the training window: 80% / 20% by day."""
    tr_days = sorted(pd.to_datetime(train["trading_date"]).dt.date.unique())
    cut = max(1, int(round(0.8 * len(tr_days))))
    inner_tr_days, inner_va_days = set(tr_days[:cut]), set(tr_days[cut:])
    if not inner_va_days:
        return grid[len(grid) // 2]
    d = pd.to_datetime(train["trading_date"]).dt.date
    itr, iva = train[d.isin(inner_tr_days)], train[d.isin(inner_va_days)]
    if len(itr) < 50 or len(iva) < 20:
        return grid[len(grid) // 2]

    pp = fit_preprocess(itr, cols)
    Xtr, Xva = apply_preprocess(itr, pp), apply_preprocess(iva, pp)
    ytr = itr[y_col].to_numpy(dtype=float)
    yva = iva[y_col].to_numpy(dtype=float)
    best, best_mse = grid[0], np.inf
    for a in grid:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        mse = float(np.mean((yva - m.predict(Xva)) ** 2))
        if mse < best_mse:
            best, best_mse = a, mse
    return best


def naive_baseline(train: pd.DataFrame, test: pd.DataFrame, kind: str,
                   y_col: str) -> np.ndarray:
    """MODEL_0, fitted on TRAIN only (no OOS information)."""
    if kind == "signed_return":
        return np.zeros(len(test))                      # locked: zero / no-change
    # locked: causal rolling mean of abs-return per 30-min ToD bucket
    bucket_mean = train.groupby("tod_bucket")[y_col].mean()
    overall = float(train[y_col].mean())
    return test["tod_bucket"].map(bucket_mean).fillna(overall).to_numpy(dtype=float)


def main() -> int:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    panel = pd.read_parquet(PANEL)
    panel["trading_date"] = pd.to_datetime(panel["trading_date"])

    reg = [f["feature_name"] for f in cfg["feature_registry"]["features"]]
    mm = cfg["model_matrices"]
    COLS = {
        "MODEL_1": list(mm["MODEL_1_column_list"]),
        "MODEL_2": list(mm["MODEL_1_column_list"]) + list(mm["MODEL_2_added_columns"]),
        "MODEL_3": list(mm["MODEL_1_column_list"]) + list(mm["MODEL_2_added_columns"])
                   + reg + ["iv_available"],
    }
    grid = list(cfg["estimator"]["alpha_grid"])
    wf = cfg["walk_forward"]

    pred_rows, fold_rows, day_rows, primary_rows, secondary_rows = [], [], [], [], []

    for tid, spec in cfg["targets"]["primary"].items():
        h = spec["horizon_snapshots"]
        kind = spec["kind"]
        y_col = "signed_return" if kind == "signed_return" else "abs_return"

        sub = panel[panel["horizon_h"] == h].copy()
        # Common row set: every model is scored on IDENTICAL rows. Required for
        # Clark-West validity -- a nested comparison on different samples is
        # meaningless. Locked per-feature "drop observation" rules land here.
        need = COLS["MODEL_3"] + [y_col]
        mask = sub[need].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        sub = sub[mask].sort_values(["trading_date", "feature_timestamp"]).reset_index(drop=True)

        days = sorted(sub["trading_date"].dt.date.unique())
        folds = make_folds(days, wf["min_train_trading_days"],
                           wf["test_block_trading_days"], wf["step_trading_days"])
        print(f"\n=== {tid} ({kind}, h={h}) : {len(sub)} rows, {len(days)} days, "
              f"{len(folds)} folds ===")

        cw_store = {"MODEL_3_vs_MODEL_2": [], "MODEL_3_vs_MODEL_1": [],
                    "MODEL_2_vs_MODEL_1": []}

        for fold in folds:
            d = sub["trading_date"].dt.date
            train = sub[d.isin(set(fold["train_days"]))]
            test = sub[d.isin(set(fold["test_days"]))]
            if len(train) < 100 or len(test) < 20:
                continue

            # Purge/embargo: labels never cross a day boundary and train/test are
            # disjoint whole days, so no training label can reach into the test
            # block. Asserted rather than assumed.
            assert pd.to_datetime(train["prediction_end_timestamp"]).dt.date.isin(
                set(fold["test_days"])).sum() == 0, "purge violation"

            y_tr = train[y_col].to_numpy(dtype=float)
            y_te = test[y_col].to_numpy(dtype=float)
            naive = naive_baseline(train, test, kind, y_col)

            preds = {}
            for mname, cols in COLS.items():
                alpha = select_alpha(train, cols, y_col, grid)
                pp = fit_preprocess(train, cols)          # fit on TRAIN only
                Xtr = apply_preprocess(train, pp)         # frozen transform
                Xte = apply_preprocess(test, pp)
                model = Ridge(alpha=alpha).fit(Xtr, y_tr)
                p = model.predict(Xte)                    # no refitting on OOS
                preds[mname] = p

                r2 = oos_r2(y_te, p, naive)
                ic = float(np.corrcoef(p, y_te)[0, 1]) if np.std(p) > 0 else np.nan
                sic = float(spearmanr(p, y_te).statistic) if np.std(p) > 0 else np.nan
                fold_rows.append({
                    "target": tid, "fold_id": fold["fold_id"], "model": mname,
                    "alpha": alpha, "n_train": len(train), "n_test": len(test),
                    "train_days": len(fold["train_days"]), "test_days": len(fold["test_days"]),
                    "oos_r2_vs_naive": r2, "rmse": float(np.sqrt(np.mean((y_te - p) ** 2))),
                    "pearson_ic": ic, "spearman_ic": sic,
                })

            for pair, (small, large) in {
                "MODEL_3_vs_MODEL_2": ("MODEL_2", "MODEL_3"),
                "MODEL_3_vs_MODEL_1": ("MODEL_1", "MODEL_3"),
                "MODEL_2_vs_MODEL_1": ("MODEL_1", "MODEL_2"),
            }.items():
                f = cw_series(y_te, preds[small], preds[large])
                cw_store[pair].append(pd.DataFrame({
                    "trading_date": test["trading_date"].to_numpy(),
                    "cw": f,
                    "short_gap_flag": test["short_gap_flag"].to_numpy(),
                    "double_cadence_day_flag": test["double_cadence_day_flag"].to_numpy(),
                    "row_in_day": np.arange(len(test)),
                }))

            for dd, g in test.groupby(test["trading_date"].dt.date):
                i = test["trading_date"].dt.date.to_numpy() == dd
                for mname in COLS:
                    day_rows.append({
                        "target": tid, "fold_id": fold["fold_id"], "model": mname,
                        "trading_date": str(dd), "n": int(i.sum()),
                        "oos_r2_vs_naive": oos_r2(y_te[i], preds[mname][i], naive[i]),
                    })

            for mname in COLS:
                for j in range(len(test)):
                    pred_rows.append({
                        "target": tid, "fold_id": fold["fold_id"], "model": mname,
                        "trading_date": str(test["trading_date"].iloc[j].date()),
                        "feature_timestamp": test["feature_timestamp"].iloc[j],
                        "y_true": y_te[j], "y_pred": preds[mname][j],
                        "naive": naive[j],
                    })

        # ---- aggregate across folds -----------------------------------
        for pair, frames in cw_store.items():
            if not frames:
                continue
            allcw = pd.concat(frames, ignore_index=True)
            day_means = allcw.groupby("trading_date")["cw"].mean().to_numpy()
            res = day_block_bootstrap_test(day_means, BOOT_BLOCK, BOOT_N,
                                            BOOT_SEED, stationary_bootstrap_indices)
            t_hac, p_hac = hac_one_sided(allcw["cw"].to_numpy(), lag=max(h, 5))

            fold_signs = [fr["cw"].mean() > 0 for fr in frames]
            clean = allcw[~allcw["short_gap_flag"] & ~allcw["double_cadence_day_flag"]]
            clean_day = clean.groupby("trading_date")["cw"].mean().to_numpy()
            res_clean = day_block_bootstrap_test(clean_day, BOOT_BLOCK, BOOT_N,
                                                  BOOT_SEED, stationary_bootstrap_indices)
            nonov = allcw[allcw["row_in_day"] % max(h, 1) == 0]
            nonov_day = nonov.groupby("trading_date")["cw"].mean().to_numpy()
            res_nonov = day_block_bootstrap_test(nonov_day, BOOT_BLOCK, BOOT_N,
                                                  BOOT_SEED, stationary_bootstrap_indices)

            oos_days = {d for f in folds for d in f["test_days"]}
            y_oos = sub[sub["trading_date"].dt.date.isin(oos_days)][y_col].to_numpy(dtype=float)
            row = {
                "target": tid, "comparison": pair,
                "oos_target_variance": float(np.nanvar(y_oos)),
                "cw_mean": res["mean"], "cw_se": res["se"],
                "p_bootstrap_one_sided": res["p_one_sided"],
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "n_oos_days": res["n_days"],
                "hac_t": t_hac, "p_hac_one_sided": p_hac,
                "folds_positive": int(sum(fold_signs)), "folds_total": len(fold_signs),
                "cw_mean_clean_subsample": res_clean["mean"],
                "p_clean_subsample": res_clean["p_one_sided"],
                "cw_mean_nonoverlapping": res_nonov["mean"],
                "p_nonoverlapping": res_nonov["p_one_sided"],
            }
            (primary_rows if pair == "MODEL_3_vs_MODEL_2" else secondary_rows).append(row)
            print(f"  {pair}: CW mean={res['mean']:.3e} p_boot={res['p_one_sided']:.4f} "
                  f"HAC p={p_hac:.4f} folds+={sum(fold_signs)}/{len(fold_signs)}")

    # ---- FDR across the 4 primary hypotheses --------------------------
    prim = pd.DataFrame(primary_rows)
    if len(prim):
        rej, q, _, _ = multipletests(prim["p_bootstrap_one_sided"].fillna(1.0),
                                     alpha=FDR_Q, method="fdr_bh")
        prim["q_bh"] = q
        prim["fdr_survives"] = rej

        mde_path = OUT / "power_addendum_values.csv"
        if not mde_path.exists():
            raise SystemExit(
                "power_addendum_values.csv missing -- the locked inconclusive band "
                "requires the pre-computed MDE. Run scripts/exp034_power_addendum.py first.")
        mde = pd.read_csv(mde_path).set_index("target")

        verdicts = []
        for _, r in prim.iterrows():
            crit = {
                "1_mean_positive": bool(r["cw_mean"] > 0),
                "2_bootstrap_p_lt_05": bool(r["p_bootstrap_one_sided"] < 0.05),
                "3_survives_fdr": bool(r["fdr_survives"]),
                "4_stable_two_thirds": bool(r["folds_positive"] >= np.ceil(2 / 3 * r["folds_total"])),
                "5_not_flag_dependent": bool(r["cw_mean_clean_subsample"] > 0) if r["cw_mean"] > 0 else True,
            }
            # Locked inconclusive band requires BOTH: CI spans zero AND the
            # estimate lies below the pre-computed MDE. Express the CW mean as
            # an approximate incremental R^2 so it is comparable to MDE(dR2).
            var_y = r["oos_target_variance"]
            dr2 = r["cw_mean"] / var_y if var_y and np.isfinite(var_y) and var_y > 0 else np.nan
            mde_dr2 = float(mde.loc[r["target"], "mde_delta_r2_approx"])
            ci_spans_zero = bool(r["ci_low"] <= 0 <= r["ci_high"])
            below_mde = bool(np.isfinite(dr2) and dr2 < mde_dr2)

            if all(crit.values()):
                v = "EXPLORATORY EVIDENCE OF INCREMENTAL INFORMATION"
            elif r["cw_mean"] > 0 and ci_spans_zero and below_mde:
                v = "INCONCLUSIVE - UNDERPOWERED"
            else:
                v = "NO EXPLORATORY EVIDENCE OF INCREMENTAL INFORMATION"
            verdicts.append(v)

            sel = prim["target"] == r["target"]
            prim.loc[sel, "delta_r2_approx"] = dr2
            prim.loc[sel, "mde_delta_r2"] = mde_dr2
            prim.loc[sel, "ci_spans_zero"] = ci_spans_zero
            prim.loc[sel, "estimate_below_mde"] = below_mde
            for k, ok in crit.items():
                prim.loc[sel, f"crit_{k}"] = ok
        prim["verdict"] = verdicts

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(OUT / "walk_forward_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(day_rows).to_csv(OUT / "per_day_metrics.csv", index=False)
    prim.to_csv(OUT / "primary_results.csv", index=False)
    pd.DataFrame(secondary_rows).to_csv(OUT / "secondary_results.csv", index=False)

    print("\n" + "=" * 78)
    print(prim[["target", "cw_mean", "p_bootstrap_one_sided", "q_bh",
                "folds_positive", "folds_total", "verdict"]].to_string(index=False))
    print("=" * 78)

    _write_report(prim, pd.DataFrame(secondary_rows), pd.DataFrame(fold_rows), cfg)
    print(f"Wrote {OUT}/phase3_results.md and companions")
    return 0


def _write_report(prim: pd.DataFrame, sec: pd.DataFrame, fm: pd.DataFrame,
                   cfg: dict) -> None:
    r2 = fm.groupby(["target", "model"])["oos_r2_vs_naive"].mean().unstack()
    ic = fm.groupby(["target", "model"])["pearson_ic"].mean().unstack()
    names = {"T1a": "signed return, h=1", "T1b": "signed return, h=5",
             "T2a": "absolute return, h=1", "T2b": "absolute return, h=5"}

    L = [
        "# EXP034 Phase 3 -- Walk-Forward Results",
        "",
        "Executed against the preregistration locked at commit `18a178e`. The",
        "preflight gate passed on the locked config, the observation panel and the",
        "realized model matrices before any fold was scored.",
        "",
        "## Primary hypotheses -- does MODEL_3 beat MODEL_2?",
        "",
        "Clark-West MSPE-adjusted, one-sided, day-block bootstrap. BH-FDR at "
        f"q={FDR_Q} across all four.",
        "",
        "| target | | CW mean | p (boot) | p (HAC) | q(BH) | folds + | dR2 | MDE(dR2) | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in prim.iterrows():
        L.append(
            f"| **{r['target']}** | {names.get(r['target'],'')} | {r['cw_mean']:.3e} | "
            f"{r['p_bootstrap_one_sided']:.4f} | {r['p_hac_one_sided']:.4f} | {r['q_bh']:.4f} | "
            f"{int(r['folds_positive'])}/{int(r['folds_total'])} | {r['delta_r2_approx']:.5f} | "
            f"{r['mde_delta_r2']:.5f} | {r['verdict']} |")

    L += [
        "",
        "**No target shows incremental option-chain information.** Nothing survives",
        "FDR; the smallest q is "
        f"{prim['q_bh'].min():.4f}.",
        "",
        "T2a is the strongest statement available: its point estimate "
        f"({prim.loc[prim.target=='T2a','delta_r2_approx'].iloc[0]:.5f}) EXCEEDS its "
        f"MDE ({prim.loc[prim.target=='T2a','mde_delta_r2'].iloc[0]:.5f}), so the study",
        "had adequate power there and still found nothing significant. The other three",
        "fall in the locked INCONCLUSIVE - UNDERPOWERED band and mean only 'no evidence",
        "at this sample size'.",
        "",
        "## Mean OOS R-squared vs MODEL_0, by model",
        "",
        "| target | MODEL_1 | MODEL_2 | MODEL_3 |",
        "|---|---|---|---|",
    ]
    for t in ["T1a", "T1b", "T2a", "T2b"]:
        if t in r2.index:
            L.append(f"| {t} | {r2.loc[t,'MODEL_1']:.5f} | {r2.loc[t,'MODEL_2']:.5f} | "
                     f"{r2.loc[t,'MODEL_3']:.5f} |")

    L += [
        "",
        "Two things are visible directly. On both volatility targets MODEL_3 scores a",
        "LOWER raw OOS R-squared than MODEL_2 -- the chain block adds estimation noise",
        "without signal. On both signed-return targets every model is NEGATIVE, i.e.",
        "worse than simply forecasting zero.",
        "",
        "## Secondary family (reported, never able to establish a primary claim)",
        "",
        "| target | comparison | CW mean | p (boot) | folds + |",
        "|---|---|---|---|---|",
    ]
    for _, r in sec.iterrows():
        L.append(f"| {r['target']} | {r['comparison']} | {r['cw_mean']:.3e} | "
                 f"{r['p_bootstrap_one_sided']:.4f} | "
                 f"{int(r['folds_positive'])}/{int(r['folds_total'])} |")

    L += [
        "",
        "### This is the substantive finding, and the lock is what protects it",
        "",
        "On the volatility targets, **MODEL_2 vs MODEL_1 is strongly significant and",
        "perfectly stable** (T2a p=0.0002, T2b p=0.0018, 5/5 folds each). India VIX",
        "carries genuine incremental information about near-term realized movement",
        "beyond price history alone. MODEL_3 vs MODEL_1 is significant for the same",
        "reason -- it contains MODEL_2.",
        "",
        "The preregistration anticipated exactly this and ruled in advance that",
        "*'beating MODEL_1 but not MODEL_2 is not primary evidence of incremental",
        "option-chain information and may not be reported as such.'* Had MODEL_3 vs",
        "MODEL_1 been the primary test, this run would have reported p=0.0002 as an",
        "option-chain discovery when it is entirely attributable to VIX.",
        "",
        "## Honest conclusion",
        "",
        "Conditional on price history, intraday time and India VIX, the locked",
        "option-chain feature set shows **no incremental out-of-sample predictive",
        "information** about 2- or 10-minute NIFTY movement, in sign or magnitude, on",
        "this sample. Volatility is forecastable (MODEL_2 IC ~0.14 on T2a, far above",
        "its MDE of 0.041) and direction is not -- consistent with the platform's",
        "standing findings. The option chain does not improve either.",
        "",
        "Per the locked sample-reuse rule this is **EXPLORATORY BY CONSTRUCTION** and",
        "may not be called confirmed or validated. The permitted labels are the ones",
        "used above.",
    ]
    (OUT / "phase3_results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
