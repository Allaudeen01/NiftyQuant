"""EXP034 -- locked power addendum (minimum detectable effect).

The locked preregistration requires: "minimum detectable effect computed for
each target from the initial 30-day training block only, and written into a
locked addendum BEFORE any out-of-sample fold is scored."

This script satisfies that requirement. It touches ONLY the first
`min_train_trading_days` usable days, and it never computes the relationship
between a real feature and a target -- the null distribution is estimated with
SIMULATED predictors, so no predictive information is revealed by running it.

Method
------
For each target, the null standard deviation of an information coefficient is
estimated by drawing K random predictors and computing their IC against the
target on the training block. Random predictors are generated as within-day
AR(1) series whose persistence is matched to the median within-day lag-1
autocorrelation of the 15 real chain features (a property of the FEATURES
ALONE -- no target is involved in measuring it). Matching persistence matters:
an iid predictor would understate the null SD, because a persistent predictor
against a day-clustered target has far more effective dependence.

    MDE(IC) at 80% power, one-sided alpha=0.05 = (1.645 + 0.8416) * SD_null

Reported alongside as an approximate incremental R-squared via R2 ~ IC^2.

    python scripts/exp034_power_addendum.py

Output: reports/exp034/power_addendum.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path("docs") / "preregistrations" / "exp034_config.json"
PANEL = Path("data") / "derived" / "exp034_feature_panel.parquet"
OUT = Path("reports") / "exp034" / "power_addendum.md"

K_DRAWS = 500
SEED = 34034
Z_ALPHA = 1.6449   # one-sided 5%
Z_POWER = 0.8416   # 80% power


def within_day_ar1(df: pd.DataFrame, cols: list[str]) -> float:
    """Median within-day lag-1 autocorrelation across the given feature columns.

    Uses features only; no target is read here.
    """
    acs = []
    for c in cols:
        per_day = []
        for _, g in df.groupby("trading_date"):
            v = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) > 20 and np.std(v) > 0:
                per_day.append(float(np.corrcoef(v[:-1], v[1:])[0, 1]))
        if per_day:
            acs.append(np.nanmedian(per_day))
    return float(np.nanmedian(acs)) if acs else 0.0


def simulate_null_ic_sd(y: np.ndarray, day_ids: np.ndarray, phi: float,
                        rng: np.random.Generator, k: int = K_DRAWS) -> float:
    """SD of the IC between the target and K random AR(1)-within-day predictors."""
    ics = []
    uniq = np.unique(day_ids)
    for _ in range(k):
        x = np.empty(len(y))
        for d in uniq:
            m = day_ids == d
            nd = int(m.sum())
            e = rng.standard_normal(nd)
            series = np.empty(nd)
            series[0] = e[0]
            for t in range(1, nd):
                series[t] = phi * series[t - 1] + e[t]
            x[m] = series
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 10 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            ics.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return float(np.std(ics)) if ics else float("nan")


def main() -> int:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    panel = pd.read_parquet(PANEL)
    n_train_days = int(cfg["walk_forward"]["min_train_trading_days"])

    days = sorted(pd.to_datetime(panel["trading_date"]).dt.date.unique())
    train_days = days[:n_train_days]
    tr = panel[pd.to_datetime(panel["trading_date"]).dt.date.isin(train_days)].copy()
    print(f"Training block: first {len(train_days)} usable days "
          f"({train_days[0]} .. {train_days[-1]}), {len(tr)} rows")

    chain = [f["feature_name"] for f in cfg["feature_registry"]["features"]]
    phi = within_day_ar1(tr, chain)
    print(f"Median within-day lag-1 autocorrelation of chain features: {phi:.3f}")

    rng = np.random.default_rng(SEED)
    rows = []
    targets = cfg["targets"]["primary"]
    for tid, spec in targets.items():
        h = spec["horizon_snapshots"]
        sub = tr[tr["horizon_h"] == h]
        col = "signed_return" if spec["kind"] == "signed_return" else "abs_return"
        y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        day_ids = pd.factorize(sub["trading_date"])[0]
        ok = np.isfinite(y)
        sd = simulate_null_ic_sd(y[ok], day_ids[ok], phi, rng)
        mde_ic = (Z_ALPHA + Z_POWER) * sd
        rows.append({
            "target": tid, "kind": spec["kind"], "horizon_snapshots": h,
            "n_train_rows": int(ok.sum()), "n_train_days": len(train_days),
            "null_ic_sd": sd, "mde_ic_80pct": mde_ic,
            "mde_delta_r2_approx": mde_ic ** 2,
        })
        print(f"  {tid}: null IC SD={sd:.4f}  MDE(IC)={mde_ic:.4f}  "
              f"~MDE(dR2)={mde_ic**2:.5f}")

    res = pd.DataFrame(rows)

    lines = [
        "# EXP034 -- Locked Power Addendum (Minimum Detectable Effect)",
        "",
        f"Locked config: `{CONFIG_PATH}` (lock commit 18a178e).",
        "",
        "**Computed from the initial training block ONLY, before any out-of-sample",
        "fold was scored**, as the locked preregistration requires. No relationship",
        "between a real feature and a target is computed anywhere in this file: the",
        "null distribution is estimated with simulated predictors.",
        "",
        "## Method",
        "",
        f"- Training block: first **{len(train_days)}** usable days "
        f"({train_days[0]} to {train_days[-1]}).",
        f"- Null IC standard deviation estimated from **K={K_DRAWS}** random predictors",
        "  per target, generated as within-day AR(1) series.",
        f"- AR(1) persistence phi = **{phi:.3f}**, set to the median within-day lag-1",
        "  autocorrelation of the 15 locked chain features. This is a property of the",
        "  features alone; no target is involved in measuring it. Matching persistence",
        "  matters because an iid predictor would understate the null SD against a",
        "  day-clustered target.",
        f"- MDE(IC) at 80% power, one-sided alpha=0.05 = ({Z_ALPHA} + {Z_POWER}) x SD.",
        "- Approximate incremental R-squared via R2 ~ IC^2.",
        f"- Seed: {SEED}.",
        "",
        "## Results",
        "",
        "| target | kind | h | train rows | null IC SD | MDE(IC) @80% | ~MDE(dR2) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(f"| {r['target']} | {r['kind']} | {r['horizon_snapshots']} | "
                     f"{r['n_train_rows']} | {r['null_ic_sd']:.4f} | "
                     f"{r['mde_ic_80pct']:.4f} | {r['mde_delta_r2_approx']:.5f} |")

    lines += [
        "",
        "## Interpretation rule (locked)",
        "",
        "Per the locked preregistration: *'no evidence of incremental information' and",
        "'evidence of no incremental information' are different conclusions and must",
        "never be conflated.* A null may be reported as evidence of **absence** only if",
        "the MDE above is SMALLER than the smallest effect that would be meaningful.",
        "",
        "Realistic intraday option-flow information coefficients are in the 0.02-0.05",
        "range. Any target whose MDE(IC) above exceeds that range is **underpowered by",
        "construction**, and a null on it means only 'no evidence at this sample size'.",
        "",
        "This file is written once and is not revised after results are seen.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
