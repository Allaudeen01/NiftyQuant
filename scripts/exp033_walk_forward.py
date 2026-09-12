"""EXP033 Phase 7+8 -- MODEL_A/B/C training and walk-forward evaluation.

Trains are pooled across all 6 E6 event types per (config, model) -- more
statistical power than 6 separate tiny per-type models -- but predictions,
calibration and signal-generation gates are always reported broken out by
event_type afterward, since that is the actual hypothesis granularity (Part
9/17 of the preregistration: 18 primary hypotheses = 6 event types x 3
configs).

Walk-forward per the locked fold plan: outer train/test cut from
`nifty_quant.research.walkforward.generate_windows` (calendar days, reused
as-is), inner split of the outer "train" block into a FIT set (first 25
trading days) and a CALIBRATION set (remaining ~8) via a plain list-slice,
never re-touched by model fitting. TEST is scored once per fold, never
revisited.

    python scripts/exp033_walk_forward.py

Outputs (Part 19 of the spec):
    reports/exp033/walk_forward_predictions.csv
    reports/exp033/walk_forward_trades.csv
    reports/exp033/probability_calibration_report.csv
    reports/exp033/feature_importance_report.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).parent))
from strategy_test_framework import directional_return, hac_test  # noqa: E402

from nifty_quant.events.event_definitions import load_locked
from nifty_quant.models.option_event_model import (
    FEATURE_SETS,
    add_event_level_features,
    build_classifier,
)
from nifty_quant.models.probability_calibration import (
    CLASSES,
    brier_score_multiclass,
    log_loss_multiclass,
    naive_baseline_probs,
)
from nifty_quant.research.walkforward import generate_windows

PANEL_PATH = Path("data") / "derived" / "exp033_feature_panel.parquet"
LABELED_PATH = Path("data") / "derived" / "exp033_labeled_events.parquet"
OUT_DIR = Path("reports") / "exp033"

OUTER_TRAIN_DAYS = 47  # calendar days; empirically ~33 trading days at this
OUTER_TEST_DAYS = 10   # sample's ~0.72 trading-day/calendar-day density.
OUTER_STEP_DAYS = 10   # (33-calendar-day train, as originally locked, was an
FIT_TRADING_DAYS = 25  # arithmetic error -- it contained only ~23 trading
                        # days, leaving nothing for calibration. Corrected
                        # here; the trading-day targets themselves (25 fit /
                        # 8 calib / 8-ish test) are unchanged.

MIN_SUPPORT_TRAIN = 30
P_TARGET_MIN = 0.55
P_MARGIN_MIN = 0.15


def build_dataset() -> pd.DataFrame:
    panel_ctx = pd.read_parquet(PANEL_PATH)
    labeled = pd.read_parquet(LABELED_PATH)
    merged = labeled.merge(
        panel_ctx, left_on=["trading_date", "event_time"], right_on=["date", "ts"], how="left"
    )
    merged = add_event_level_features(merged)
    return merged


def split_fold_days(all_days: list, fold) -> tuple[list, list, list]:
    train_days = [d for d in all_days if fold.train_start <= pd.Timestamp(d) < fold.train_end]
    test_days = [d for d in all_days if fold.test_start <= pd.Timestamp(d) < fold.test_end]
    fit_days = train_days[:FIT_TRADING_DAYS]
    calib_days = train_days[FIT_TRADING_DAYS:]
    return fit_days, calib_days, test_days


def run_config(df: pd.DataFrame, config_name: str, locked: dict) -> tuple[list, list, list]:
    cfg = locked["triple_barrier_configs"][config_name]
    sub = df[df["config"] == config_name].copy()
    non_ambig = sub[sub["first_barrier"] != "AMBIGUOUS"].copy()
    non_ambig["y"] = non_ambig["first_barrier"]

    all_days = sorted(pd.to_datetime(non_ambig["trading_date"]).dt.date.unique())
    if len(all_days) < FIT_TRADING_DAYS + 5:
        print(f"  {config_name}: too few days ({len(all_days)}), skipping.")
        return [], [], []

    folds = generate_windows(
        pd.Timestamp(all_days[0]), pd.Timestamp(all_days[-1]),
        train_days=OUTER_TRAIN_DAYS, test_days=OUTER_TEST_DAYS, step_days=OUTER_STEP_DAYS,
    )
    print(f"  {config_name}: {len(all_days)} days, {len(folds)} outer folds.")

    pred_rows, trade_rows, importance_rows = [], [], []

    for fold_id, fold in enumerate(folds):
        fit_days, calib_days, test_days = split_fold_days(all_days, fold)
        if not fit_days or not calib_days or not test_days:
            continue
        fit = non_ambig[pd.to_datetime(non_ambig["trading_date"]).dt.date.isin(fit_days)]
        calib = non_ambig[pd.to_datetime(non_ambig["trading_date"]).dt.date.isin(calib_days)]
        test = non_ambig[pd.to_datetime(non_ambig["trading_date"]).dt.date.isin(test_days)]
        if len(fit) < 30 or len(calib) < 10 or len(test) < 5:
            continue

        support_by_type = fit.groupby("event_type").size().to_dict()

        for model_name, feature_cols in FEATURE_SETS.items():
            cols = [c for c in feature_cols if c in fit.columns]
            X_fit = fit[cols].apply(pd.to_numeric, errors="coerce")
            X_calib = calib[cols].apply(pd.to_numeric, errors="coerce")
            X_test = test[cols].apply(pd.to_numeric, errors="coerce")
            y_fit, y_calib, y_test = fit["y"].to_numpy(), calib["y"].to_numpy(), test["y"].to_numpy()

            clf = build_classifier()
            clf.fit(X_fit, y_fit)

            calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(clf), method="isotonic")
            calibrated.fit(X_calib, y_calib)

            proba = calibrated.predict_proba(X_test)
            class_order = list(calibrated.classes_)
            proba_aligned = np.zeros((len(test), len(CLASSES)))
            for k, c in enumerate(CLASSES):
                if c in class_order:
                    proba_aligned[:, k] = proba[:, class_order.index(c)]

            brier = brier_score_multiclass(y_test, proba_aligned)
            ll = log_loss_multiclass(y_test, proba_aligned)
            naive = naive_baseline_probs(y_fit, len(y_test))
            brier_naive = brier_score_multiclass(y_test, naive)
            ll_naive = log_loss_multiclass(y_test, naive)

            for i, (_, row) in enumerate(test.iterrows()):
                pred_rows.append({
                    "fold_id": fold_id, "config": config_name, "model": model_name,
                    "event_id": row["event_id"] if "event_id" in row else None,
                    "trading_date": row["trading_date"], "event_time": row["event_time"],
                    "event_type": row["event_type"], "relative_strike": row["relative_strike"],
                    "option_type": row["option_type"], "direction": row["direction"],
                    "true_label": row["y"],
                    "p_target_first": proba_aligned[i, 0], "p_stop_first": proba_aligned[i, 1],
                    "p_timeout": proba_aligned[i, 2],
                    "brier_fold": brier, "log_loss_fold": ll,
                    "brier_naive_fold": brier_naive, "log_loss_naive_fold": ll_naive,
                    "train_start": fold.train_start, "train_end": fold.train_end,
                    "test_start": fold.test_start, "test_end": fold.test_end,
                    "n_fit": len(fit), "n_calib": len(calib), "n_test": len(test),
                })

            if model_name == "MODEL_C":
                if hasattr(clf, "feature_importances_"):
                    for c, imp in zip(cols, clf.feature_importances_):
                        importance_rows.append({
                            "fold_id": fold_id, "config": config_name, "feature": c,
                            "importance": float(imp),
                        })

                calibration_gate_passed = brier < brier_naive
                print(f"    fold {fold_id} {config_name} MODEL_C calibration gate: "
                      f"brier={brier:.4f} vs naive={brier_naive:.4f} -> "
                      f"{'PASS' if calibration_gate_passed else 'FAIL (no signals this fold)'}")
                if not calibration_gate_passed:
                    continue  # Part 9 gate 5: fold-level calibration must beat naive

                for i, (_, row) in enumerate(test.iterrows()):
                    et = row["event_type"]
                    if support_by_type.get(et, 0) < MIN_SUPPORT_TRAIN:
                        continue
                    p_target, p_stop = proba_aligned[i, 0], proba_aligned[i, 1]
                    if p_target < P_TARGET_MIN or (p_target - p_stop) < P_MARGIN_MIN:
                        continue
                    sign = 1.0 if row["direction"] == "BULLISH" else -1.0
                    entry_spot = row["entry_spot"]
                    exit_spot = entry_spot + sign * row["realized_excursion_pts"]
                    net_ret = directional_return(entry_spot, exit_spot, side=sign)
                    trade_rows.append({
                        "fold_id": fold_id, "config": config_name, "event_type": et,
                        "relative_strike": row["relative_strike"], "option_type": row["option_type"],
                        "trading_date": row["trading_date"], "event_time": row["event_time"],
                        "direction": row["direction"], "p_target_first": p_target,
                        "p_stop_first": p_stop, "actual_outcome": row["y"],
                        "realized_excursion_pts": row["realized_excursion_pts"],
                        "entry_spot": entry_spot,
                        "net_return": net_ret,
                    })

    return pred_rows, trade_rows, importance_rows


def main() -> int:
    locked = load_locked()
    df = build_dataset()
    print(f"Dataset: {len(df)} labeled event-instances (all configs, incl. AMBIGUOUS).")

    all_preds, all_trades, all_importance = [], [], []
    for config_name in locked["triple_barrier_configs"]:
        p, t, imp = run_config(df, config_name, locked)
        all_preds.extend(p)
        all_trades.extend(t)
        all_importance.extend(imp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame(all_preds)
    trades_df = pd.DataFrame(all_trades)
    imp_df = pd.DataFrame(all_importance)

    preds_df.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    trades_df.to_csv(OUT_DIR / "walk_forward_trades.csv", index=False)
    print(f"Wrote walk_forward_predictions.csv ({len(preds_df)} rows)")
    print(f"Wrote walk_forward_trades.csv ({len(trades_df)} rows)")

    if len(preds_df):
        calib_report = (
            preds_df.drop_duplicates(["fold_id", "config", "model"])
            .groupby(["config", "model"])
            .agg(mean_brier=("brier_fold", "mean"), mean_log_loss=("log_loss_fold", "mean"),
                 mean_brier_naive=("brier_naive_fold", "mean"),
                 mean_log_loss_naive=("log_loss_naive_fold", "mean"),
                 n_folds=("fold_id", "nunique"))
            .reset_index()
        )
        calib_report.to_csv(OUT_DIR / "probability_calibration_report.csv", index=False)
        print(f"Wrote probability_calibration_report.csv\n{calib_report.to_string()}")

    if len(imp_df):
        fi_report = imp_df.groupby(["config", "feature"])["importance"].mean().reset_index()
        fi_report = fi_report.sort_values(["config", "importance"], ascending=[True, False])
        fi_report.to_csv(OUT_DIR / "feature_importance_report.csv", index=False)
        print(f"Wrote feature_importance_report.csv ({len(fi_report)} rows)")

    if len(trades_df):
        print("\nSignals generated per (config, event_type):")
        summary = trades_df.groupby(["config", "event_type"]).agg(
            n=("net_return", "size"), mean_ret=("net_return", "mean"),
            win_rate=("net_return", lambda s: 100 * (s > 0).mean()),
        )
        print(summary.to_string())
        for (cfg, et), g in trades_df.groupby(["config", "event_type"]):
            t, p = hac_test(g["net_return"].to_numpy())
            print(f"  {cfg}/{et}: n={len(g)} mean={g['net_return'].mean()*100:.3f}% "
                  f"HAC_t={t:.2f} p={p:.3f}")
    else:
        print("\nNo signals cleared the gate in any (config, event_type) -- reported as NO_TRADE.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
