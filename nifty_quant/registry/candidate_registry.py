"""EXP033 Phase 9 -- candidate registry construction.

Builds one row per primary hypothesis (6 event types x 3 configs = 18,
locked in the preregistration) from the walk-forward trade log, applies the
mechanical decision rules, and returns a DataFrame matching the locked
schema. Hypotheses with too little data to compute a meaningful statistic
are reported as such, never silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from strategy_test_framework import hac_test  # noqa: E402

MIN_N_FOR_STATS = 5


def _sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2 or np.allclose(daily_returns.std(), 0.0):
        return float("nan")
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252.0))


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return float("nan")
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    return float(dd.min())


def build_registry(trades: pd.DataFrame, all_pairs: list[tuple[str, str]],
                    experiment_id: str = "exp033",
                    event_def_version: str = "exp033-events-v1.0") -> pd.DataFrame:
    rows = []
    for config, event_type in all_pairs:
        g = trades[(trades["config"] == config) & (trades["event_type"] == event_type)]
        n = len(g)
        candidate_id = f"{config}__{event_type}"

        if n < MIN_N_FOR_STATS:
            rows.append({
                "candidate_id": candidate_id, "experiment_id": experiment_id,
                "event_definition_version": event_def_version,
                "feature_set_version": "MODEL_C",
                "model_version": "lightgbm-v1.0-isotonic",
                "target_stop_configuration": config,
                "number_of_events": n, "number_of_trades": n,
                "win_rate": np.nan, "mean_return": np.nan, "median_return": np.nan,
                "expected_value": np.nan, "maximum_drawdown": np.nan, "max_loss": np.nan,
                "sharpe": np.nan, "hac_t_stat": np.nan, "hac_p_value": np.nan,
                "confidence_interval_low": np.nan, "confidence_interval_high": np.nan,
                "verdict": "INSUFFICIENT_DATA", "status": "PRE_REGISTERED",
            })
            continue

        ret = g["net_return"].to_numpy()
        t, p = hac_test(ret)
        se = abs(ret.mean() / t) if (t == t and t != 0) else np.nan
        daily = g.groupby("trading_date")["net_return"].sum().to_numpy()

        mean_ret = float(ret.mean())
        win_rate = float(100.0 * (ret > 0).mean())
        verdict = "ARCHIVE"
        if mean_ret < 0 and (t <= -1.0 or win_rate < 45.0):
            verdict = "REJECT"
        elif mean_ret > 0 and abs(t) < 0.5:
            verdict = "ARCHIVE"
        elif mean_ret > 0:
            verdict = "INCONCLUSIVE - CONTINUE COLLECTING"  # FDR/SPA applied at portfolio level after

        status_map = {
            "REJECT": "REJECTED", "ARCHIVE": "ARCHIVED",
            "INCONCLUSIVE - CONTINUE COLLECTING": "PRE_REGISTERED",
        }

        rows.append({
            "candidate_id": candidate_id, "experiment_id": experiment_id,
            "event_definition_version": event_def_version,
            "feature_set_version": "MODEL_C",
            "model_version": "lightgbm-v1.0-isotonic",
            "target_stop_configuration": config,
            "number_of_events": n, "number_of_trades": n,
            "win_rate": win_rate, "mean_return": mean_ret,
            "median_return": float(np.median(ret)),
            "expected_value": mean_ret,
            "maximum_drawdown": _max_drawdown(ret),
            "max_loss": float(ret.min()),
            "sharpe": _sharpe(daily),
            "hac_t_stat": t, "hac_p_value": p,
            "confidence_interval_low": mean_ret - 1.96 * se if se == se else np.nan,
            "confidence_interval_high": mean_ret + 1.96 * se if se == se else np.nan,
            "verdict": verdict, "status": status_map.get(verdict, "PRE_REGISTERED"),
        })
    return pd.DataFrame(rows)
