"""Experiment 031: VRP Harvesting Strategy v1.1 -- risk overlays (PAPER ONLY).

Pre-registered in docs/preregistrations/exp031_vrp_strategy_v1_1.md.

v1.1 = v1.0 (short-vol, vol-target sizing) + four LOCKED overlays:
  1. regime filter (VIX<22, VIX not rising, trailing RV not spiking) -> stand aside
  2. tail hedge (loss cap -15 vol pts, cost 1.5) when VIX>18
  3. kill-switch (DD>20% -> half size; DD>30% -> halt until recovered)
  4. cost sensitivity (0 / 1.5 / 3.0 vol pts)

Thresholds are a-priori (NOT tuned). In-sample improvement is PROVISIONAL; the
real test is forward (Exp 027). Convention: 1 vol pt = 1% capital.

    python scripts/exp031_vrp_strategy_v1_1.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK = 21
BASE_COST = 1.5
CACHE = Path("data") / "external" / "macro_vrp_daily.parquet"
ANN_BLOCKS = 252 / BLOCK
SCALE = 0.01                      # 1 vol pt = 1% capital

VIX_MAX = 22.0                    # regime: avoid extreme levels
RV_SPIKE_MULT = 1.2              # regime: trailing RV <= 1.2x its 126d average
HEDGE_VIX = 18.0                 # tail hedge active above this
HEDGE_FLOOR = -15.0              # capped block loss (vol pts)
HEDGE_COST = 1.5                 # cost of the hedge (vol pts)
DD_HALF = 0.20                   # kill-switch: halve size beyond 20% DD
DD_HALT = 0.30                   # kill-switch: halt beyond 30% DD


def load() -> pd.DataFrame:
    if not CACHE.exists():
        raise SystemExit("Run scripts/exp028_macro_vrp.py first to cache the data.")
    df = pd.read_parquet(CACHE)
    df["ret"] = np.log(df["nifty"] / df["nifty"].shift(1))
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().reset_index(drop=True)


def trailing_rv(ret: np.ndarray, end: int, window: int = BLOCK) -> float:
    if end < window:
        return float("nan")
    w = ret[end - window:end]
    return math.sqrt(252.0 / window * np.sum(w ** 2)) * 100.0


def build(df: pd.DataFrame):
    r = df["ret"].to_numpy()
    vix = df["vix"].to_numpy()
    dates = df["date"].to_numpy()
    n = len(df)
    # precompute trailing RV series (causal) at each index
    rv_tr = np.array([trailing_rv(r, i) for i in range(n)])
    rv_tr_avg = pd.Series(rv_tr).rolling(126, min_periods=30).mean().to_numpy()

    trades = []
    for s in range(0, n - BLOCK, BLOCK):
        fwd = r[s + 1:s + 1 + BLOCK]
        if len(fwd) < BLOCK:
            break
        realized = math.sqrt(252.0 / BLOCK * np.sum(fwd ** 2)) * 100.0
        vix_s = float(vix[s])
        vix_5 = float(vix[max(s - 5, 0)])
        rv_s, rv_a = rv_tr[s], rv_tr_avg[s]
        # regime filter (all causal)
        not_rising = vix_s <= vix_5
        not_spiking = (not np.isnan(rv_s) and not np.isnan(rv_a)
                       and rv_s <= RV_SPIKE_MULT * rv_a)
        active = (vix_s < VIX_MAX) and not_rising and not_spiking
        trades.append({
            "date": dates[s], "year": pd.Timestamp(dates[s]).year,
            "vix": vix_s, "realized": realized,
            "gross": vix_s - realized, "active": bool(active),
        })
    return pd.DataFrame(trades)


def run_strategy(tr: pd.DataFrame, cost: float, use_overlays: bool):
    """Sequential backtest -> block % returns. If not use_overlays, this is v1.0
    (all trades, vol-target sizing, no hedge/filter/kill-switch)."""
    equity = 1.0
    peak = 1.0
    rets = []
    taken = 0
    mean_w_norm = (1.0 / tr["vix"]).mean()
    for _, row in tr.iterrows():
        w = (1.0 / row["vix"]) / mean_w_norm            # vol-target, mean ~1
        net = row["gross"] - cost
        if use_overlays:
            if not row["active"]:
                rets.append(0.0)
                continue
            if row["vix"] > HEDGE_VIX:                  # tail hedge (loss cap)
                net = max(net, HEDGE_FLOOR) - HEDGE_COST
            dd = equity / peak - 1.0                     # kill-switch
            if dd < -DD_HALT:
                rets.append(0.0)                         # halt
                continue
            if dd < -DD_HALF:
                w *= 0.5
        taken += 1
        block_ret = w * net * SCALE
        equity *= (1.0 + block_ret)
        peak = max(peak, equity)
        rets.append(block_ret)
    return np.array(rets), taken


def stats(rets: np.ndarray, label: str) -> dict:
    nz = rets[rets != 0.0]
    n = len(nz) if len(nz) else 1
    mean = float(np.mean(nz)) if len(nz) else 0.0
    sd = float(np.std(nz, ddof=1)) if len(nz) > 1 else float("nan")
    dn = nz[nz < 0]
    dsd = math.sqrt(float(np.mean(dn ** 2))) if dn.size else 0.0
    sharpe = mean / sd * math.sqrt(ANN_BLOCKS) if sd and sd > 0 else float("nan")
    sortino = mean / dsd * math.sqrt(ANN_BLOCKS) if dsd > 0 else float("nan")
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / peak - 1.0)) * 100
    cagr = (eq[-1] ** (ANN_BLOCKS / len(rets)) - 1.0) * 100 if eq[-1] > 0 else float("nan")
    return {"label": label, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "cagr": cagr, "total_ret": (eq[-1] - 1) * 100,
            "worst": float(np.min(nz)) * 100 if len(nz) else 0.0,
            "hit": float(np.mean(nz > 0)) * 100 if len(nz) else float("nan")}


def main() -> int:
    df = load()
    tr = build(df)
    n_blocks = len(tr)
    n_active = int(tr["active"].sum())
    print("=" * 92)
    print("EXP 031  VRP STRATEGY v1.1  (risk overlays, 10y macro backtest, PAPER ONLY)")
    print(f"Blocks: {n_blocks} | regime-eligible: {n_active} ({n_active/n_blocks*100:.0f}%)"
          f" | {tr['date'].min().date()} -> {tr['date'].max().date()}")

    print("\n--- cost sensitivity (v1.1 Sharpe / maxDD%) ---")
    for c in (0.0, 1.5, 3.0):
        rets, taken = run_strategy(tr, c, use_overlays=True)
        st = stats(rets, "")
        print(f"  cost {c:>4.1f}: Sharpe {st['sharpe']:>5.2f}  maxDD {st['max_dd']:>6.1f}%  "
              f"CAGR {st['cagr']:>5.1f}%  trades {taken}")

    # head-to-head at base cost
    v10, taken10 = run_strategy(tr, BASE_COST, use_overlays=False)
    v11, taken11 = run_strategy(tr, BASE_COST, use_overlays=True)
    s10, s11 = stats(v10, "v1.0 (vol-target)"), stats(v11, "v1.1 (overlays)")
    print(f"\n--- head-to-head @ cost {BASE_COST} (1 vol pt = 1% capital) ---")
    print(f"  {'strategy':<20}{'Sharpe':>8}{'Sortino':>9}{'maxDD%':>9}{'CAGR%':>8}"
          f"{'worst%':>8}{'hit%':>6}{'trades':>8}")
    for st, tk in ((s10, taken10), (s11, taken11)):
        print(f"  {st['label']:<20}{st['sharpe']:>8.2f}{st['sortino']:>9.2f}"
              f"{st['max_dd']:>9.1f}{st['cagr']:>8.1f}{st['worst']:>8.1f}"
              f"{st['hit']:>6.0f}{tk:>8}")

    # per-year (v1.1)
    print("\n--- per-year mean block return, v1.1 (%) ---")
    tr = tr.assign(v11=v11)
    yr_pos = 0
    yrs = sorted(tr["year"].unique())
    for y in yrs:
        sub = tr[tr["year"] == y]["v11"].to_numpy()
        active_sub = sub[sub != 0.0]
        mv = float(np.mean(active_sub)) * 100 if len(active_sub) else 0.0
        yr_pos += 1 if mv > 0 else 0
        tag = "" if mv >= 0 else "  <== neg"
        print(f"  {y}: trades {len(active_sub):2d}  mean {mv:+.2f}%{tag}")

    part = taken11 / n_blocks * 100
    print("\n" + "=" * 92)
    print("VERDICT (pre-registered, PAPER ONLY, IN-SAMPLE/provisional):")
    clears = (s11["sharpe"] >= 0.7 and s11["max_dd"] >= -30 and yr_pos >= 7
              and part >= 50)
    improved = (s11["sharpe"] > s10["sharpe"] and s11["max_dd"] > s10["max_dd"])
    if clears:
        print(f"  CLEARS THE BAR in-sample (Sharpe {s11['sharpe']:.2f}>=0.7, "
              f"maxDD {s11['max_dd']:.0f}%>=-30, {yr_pos}/{len(yrs)} yrs, "
              f"{part:.0f}% participation).")
        print("  -> Eligible to PRE-REGISTER live paper trading, but WAIT for Exp 027.")
    elif improved:
        print(f"  IMPROVED-BUT-SHORT: better than v1.0 (Sharpe {s10['sharpe']:.2f}->"
              f"{s11['sharpe']:.2f}, maxDD {s10['max_dd']:.0f}%->{s11['max_dd']:.0f}%) "
              f"but misses a bar. -> v1.2 iteration; no paper trading.")
    else:
        print(f"  NO IMPROVEMENT over v1.0. Overlays don't help enough; rethink.")
    print("  Provisional: in-sample, thresholds hindsight-chosen. Forward test (Exp 027)")
    print("  is the real judge. Real straddle execution is worse than this proxy.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
