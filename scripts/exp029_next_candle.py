"""Experiment 029: does the last 5-minute candle predict the next candle?

Pre-registered in docs/preregistrations/exp029_next_candle_predictability.md.

Directly answers: "can I see a candle pattern and predict the next candle up/down
by 10-20 points?" Tests DIRECTION (up/down) and MAGNITUDE (size of move) of the
next 5m bar, out-of-sample, walk-forward, AFTER costs.

    python scripts/exp029_next_candle.py

Read-only research. No trades. A positive result earns a follow-up, not a trade.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

SPOT_REF = 24000.0        # for translating returns <-> index points
ROUND_TRIP_COST = 1.5e-4  # 1.5 bps round-trip (spread+slippage+fees), conservative


def load_bars() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    if not files:
        raise SystemExit("No 5m NIFTY parquet found")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df["hm"] = df["timestamp"].dt.strftime("%H:%M")
    df = df[(df["hm"] >= "09:15") & (df["hm"] <= "15:30")]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per bar with causal features and the NEXT bar's return as target.

    Features never see the future; target is next bar's close/close return,
    computed WITHIN the same day (no overnight crossing)."""
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        l = g["low"].to_numpy(); c = g["close"].to_numpy()
        n = len(g)
        if n < 10:
            continue
        ret = np.zeros(n)
        ret[1:] = np.log(c[1:] / c[:-1])
        cum_pv = np.cumsum((h + l + c) / 3.0)      # rough running VWAP proxy
        vwap = cum_pv / (np.arange(n) + 1)
        for i in range(2, n - 1):                  # need lags and a next bar
            rng = (h[i] - l[i]) / o[i] if o[i] else 0.0
            body = (c[i] - o[i]) / o[i] if o[i] else 0.0
            hl = (h[i] - l[i]) or 1e-9
            rows.append({
                "ret": ret[i],
                "body": body,
                "range": rng,
                "body_frac": abs(c[i] - o[i]) / hl,
                "upper_wick": (h[i] - max(o[i], c[i])) / o[i] if o[i] else 0.0,
                "lower_wick": (min(o[i], c[i]) - l[i]) / o[i] if o[i] else 0.0,
                "ret_lag1": ret[i - 1],
                "ret_lag2": ret[i - 2],
                "dist_vwap": (c[i] - vwap[i]) / vwap[i] if vwap[i] else 0.0,
                "minute_of_day": i,
                "y_next": ret[i + 1],              # TARGET: next bar return
            })
    return pd.DataFrame(rows)


FEATURES = ["ret", "body", "range", "body_frac", "upper_wick", "lower_wick",
            "ret_lag1", "ret_lag2", "dist_vwap", "minute_of_day"]


def folds(n, k=4, train_frac=0.5):
    start = int(n * train_frac)
    edges = np.linspace(start, n, k + 1, dtype=int)
    return [(0, edges[i], edges[i], edges[i + 1]) for i in range(k)]


def main() -> int:
    df = build_features(load_bars())
    n = len(df)
    X = df[FEATURES].to_numpy()
    y_ret = df["y_next"].to_numpy()
    y_up = (y_ret > 0).astype(int)

    print("=" * 92)
    print("EXP 029  NEXT-CANDLE PREDICTABILITY (NIFTY 5m)  direction + magnitude")
    print(f"Candles: {n}  | up-rate {y_up.mean()*100:.1f}%  "
          f"mean |move| {np.mean(np.abs(y_ret))*SPOT_REF:.1f} pts "
          f"({np.mean(np.abs(y_ret))*100:.3f}%)")
    print(f"Cost assumption: round-trip {ROUND_TRIP_COST*1e4:.1f} bps "
          f"(~{ROUND_TRIP_COST*SPOT_REF:.1f} index pts). A '10-pt' move = "
          f"{10/SPOT_REF*100:.3f}%.")

    # ---------------- DIRECTION ----------------
    print("\n--- DIRECTION: predict up/down of next candle (walk-forward) ---")
    accs, aucs, pnls, signs = [], [], [], []
    for f, (a, b, ts, te) in enumerate(folds(n)):
        Xtr, ytr = X[a:b], y_up[a:b]
        Xte, yte = X[ts:te], y_up[ts:te]
        rte = y_ret[ts:te]
        mu = Xtr.mean(0); sdv = Xtr.std(0); sdv[sdv == 0] = 1
        Xtr_s, Xte_s = (Xtr - mu) / sdv, (Xte - mu) / sdv
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)[:, 1]
        pred = (proba > 0.5).astype(int)
        acc = float(np.mean(pred == yte))
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
        # trading proxy: go long if p>0.5 else short; pay cost when position flips
        pos = np.where(proba > 0.5, 1, -1)
        gross = pos * rte
        flips = np.abs(np.diff(np.concatenate([[0], pos]))) / 2.0
        net = gross - flips * ROUND_TRIP_COST
        accs.append(acc); aucs.append(auc)
        pnls.append(float(np.sum(net)))
        signs.append(1 if acc > 0.5 else 0)
        print(f"  fold {f+1}: acc={acc*100:5.2f}%  AUC={auc:.3f}  "
              f"net P&L(after cost)={np.sum(net)*SPOT_REF:+.0f} pts over {te-ts} bars")

    mean_acc = float(np.mean(accs))
    # binomial test on pooled test predictions (rerun pooled for a clean p)
    binom_p = ss.binomtest(int(round(mean_acc * (n // 2))), n // 2, 0.5,
                           alternative="greater").pvalue
    n_pnl_pos = sum(1 for p in pnls if p > 0)
    dir_pred = (mean_acc > 0.53 and binom_p < 0.01 and min(aucs) > 0.55
                and all(s == 1 for s in signs) and n_pnl_pos >= 3)
    print(f"  mean OOS accuracy = {mean_acc*100:.2f}%  (baseline ~{max(y_up.mean(),1-y_up.mean())*100:.1f}%)")
    print(f"  mean AUC = {np.mean(aucs):.3f}   after-cost P&L positive in {n_pnl_pos}/4 folds")
    print(f"  --> DIRECTION {'PREDICTABLE' if dir_pred else 'NOT PREDICTABLE'}")

    # ---------------- MAGNITUDE ----------------
    print("\n--- MAGNITUDE: predict size |move| of next candle (walk-forward) ---")
    incr = []
    for f, (a, b, ts, te) in enumerate(folds(n)):
        Xtr = X[a:b]; Xte = X[ts:te]
        ytr = np.abs(y_ret[a:b]); yte = np.abs(y_ret[ts:te])
        # baseline: recent average |move| (trailing 20-bar mean, causal)
        base_pred = pd.Series(np.abs(y_ret)).rolling(20).mean().shift(1).to_numpy()[ts:te]
        ok = ~np.isnan(base_pred)
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=0)
        reg.fit(Xtr, ytr)
        pred = reg.predict(Xte)
        yy, pp, bb = yte[ok], pred[ok], base_pred[ok]
        sse = np.sum((yy - pp) ** 2); sst = np.sum((yy - bb) ** 2)
        r2 = 1 - sse / sst if sst > 0 else float("nan")
        incr.append(r2)
        print(f"  fold {f+1}: OOS R2 vs recent-avg baseline = {r2:+.4f}")
    n_pos = sum(1 for x in incr if x > 0)
    mag_pred = n_pos >= 3
    print(f"  --> MAGNITUDE {'PREDICTABLE (size, not direction)' if mag_pred else 'NOT beyond baseline'}"
          f"  ({n_pos}/4 folds positive)")

    # ---------------- NET VERDICT ----------------
    print("\n" + "=" * 92)
    print("NET VERDICT (pre-registered):")
    if dir_pred:
        print("  CAN PREDICT NEXT CANDLE DIRECTION -> must now face an Exp 023 snooping check.")
    elif mag_pred:
        print("  You can predict the SIZE of the next candle (volatility clusters),")
        print("  NOT its direction. Direction is ~a coin flip and LOSES after costs.")
    else:
        print("  Next candle is not predictable in direction; magnitude ~ baseline.")
    print(f"  [direction acc {mean_acc*100:.1f}% | AUC {np.mean(aucs):.3f} | "
          f"after-cost P&L +folds {n_pnl_pos}/4 | magnitude +folds {n_pos}/4]")
    print("  Scope: falsification test. No trading rule deployed regardless.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
