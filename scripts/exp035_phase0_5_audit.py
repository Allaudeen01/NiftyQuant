"""EXP035 Phase 0.5 -- event power and stability audit. OUTCOME BLIND.

Inspects ONLY: event timestamps, frequency, clustering, overlap, event-day
counts, predictor distributions, event-definition stability, collection timing
metadata, contract identity, snapshot gaps.

It does NOT read future spot returns, target/stop hits, P&L, win rates, Sharpe,
IC, outcome correlations or model performance, and labels nothing using future
prices. No model is fitted. No backtest is run.

Contract-identity rule (docs/RESEARCH_POLICY.md section 2) is enforced: every
temporal difference is taken on ABSOLUTE (expiry, strike, option_type) identity
BEFORE any mapping into relative-strike coordinates.

    python scripts/exp035_phase0_5_audit.py

Outputs: reports/exp035/phase0_5_*.csv
"""

from __future__ import annotations

import glob
import math
import os
from datetime import date

import numpy as np
import pandas as pd

EPS = 1e-9
BAND = tuple(range(-5, 6))
SYNC_BAND = (-1, 0, 1)         # narrow band, likely inside one 50-token batch
OUT = "reports/exp035"
BATCH_SIZE = 50                # angelone._fetch_market_data
REQUEST_PAUSE_S = 1.2          # collector --request-pause

CONC_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
MIGR_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50]

Z_ALPHA, Z_POWER = 1.6449, 0.8416   # one-sided 5%, 80% power
# Externally pre-specified plausible effect range, carried from the EXP034
# preregistration where it was stated BEFORE any result was seen. Not derived
# from any observed outcome in this or any other experiment.
PLAUSIBLE_IC = (0.02, 0.05)


def collected_days() -> list[date]:
    out = []
    for p in sorted(glob.glob("data/option_chain/*/*/*")):
        if os.path.isdir(p):
            y, m, d = p.replace("\\", "/").split("/")[-3:]
            out.append(date(int(y), int(m), int(d)))
    return sorted(out)


def load_near(d: date):
    f = sorted(glob.glob(f"data/option_chain/{d.year:04d}/{d.month:02d}/{d.day:02d}/*.parquet"))
    df = pd.concat([pd.read_parquet(x) for x in f], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df = df[df["expiry"] == df["expiry"].min()]
    return df.drop_duplicates(subset=["snapshot_ts", "strike", "option_type"], keep="last")


def day_series(df: pd.DataFrame) -> dict | None:
    times = pd.DatetimeIndex(np.sort(df["snapshot_ts"].unique()))
    strikes = np.sort(df["strike"].unique())
    n = len(times)
    if n < 20 or len(strikes) < 11:
        return None
    spot = df.groupby("snapshot_ts")["spot"].first().reindex(times).to_numpy(dtype=float)
    atm_idx = np.argmin(np.abs(strikes[None, :] - spot[:, None]), axis=1)
    rows = np.arange(n)

    # ABSOLUTE-identity differencing first (RESEARCH_POLICY section 2)
    dvol_abs = {}
    for ot in ("CE", "PE"):
        p = df[df["option_type"] == ot].pivot_table(
            index="snapshot_ts", columns="strike", values="volume", aggfunc="last")
        v = np.nan_to_num(p.reindex(index=times, columns=strikes).to_numpy(dtype=float))
        dvol_abs[ot] = np.vstack([np.full((1, len(strikes)), np.nan), np.diff(v, axis=0)])

    def band_matrix(levels):
        return np.vstack([
            np.nan_to_num(dvol_abs["CE"][rows, np.clip(atm_idx + k, 0, len(strikes) - 1)])
            + np.nan_to_num(dvol_abs["PE"][rows, np.clip(atm_idx + k, 0, len(strikes) - 1)])
            for k in levels])

    M = band_matrix(BAND)
    tot = M.sum(axis=0)
    share = M / (tot + EPS)
    top1 = share.max(axis=0)
    ks = np.array(BAND, dtype=float)[:, None]
    com = np.where(tot > 0, (ks * M).sum(axis=0) / (tot + EPS), np.nan)
    dcom = np.concatenate([[np.nan], np.diff(com)])

    Ms = band_matrix(SYNC_BAND)
    tots = Ms.sum(axis=0)
    kss = np.array(SYNC_BAND, dtype=float)[:, None]
    com_s = np.where(tots > 0, (kss * Ms).sum(axis=0) / (tots + EPS), np.nan)
    dcom_s = np.concatenate([[np.nan], np.diff(com_s)])

    gaps = pd.Series(times).diff().dt.total_seconds().to_numpy()
    n_contracts = int(df.groupby("snapshot_ts").size().median())
    n_batches = math.ceil(n_contracts / BATCH_SIZE)
    return {"times": times, "n": n, "top1": top1, "dcom": dcom, "dcom_sync": dcom_s,
            "gaps": gaps, "n_contracts": n_contracts, "n_batches": n_batches,
            "implied_spread_s": (n_batches - 1) * REQUEST_PAUSE_S,
            "n_strikes": len(strikes)}


def cluster_stats(flags: np.ndarray, day_ids: np.ndarray, times: np.ndarray) -> dict:
    ev = np.where(flags == 1)[0]
    n_ev = len(ev)
    if n_ev == 0:
        return {"n_events": 0, "event_days": 0, "events_per_event_day": np.nan,
                "pct_within_5_snaps": np.nan, "median_gap_snaps": np.nan,
                "days_0": np.nan, "days_1": np.nan, "days_2plus": np.nan, "days_5plus": np.nan,
                "effective_day_clusters": 0}
    ed = pd.Series(day_ids[ev]).value_counts()
    all_days = pd.Series(day_ids).nunique()
    # consecutive-event spacing WITHIN a day (snapshot index distance)
    gaps_sn = []
    for d, idxs in pd.Series(ev).groupby(day_ids[ev]):
        a = np.sort(idxs.to_numpy())
        if len(a) > 1:
            gaps_sn.extend(np.diff(a).tolist())
    gaps_sn = np.array(gaps_sn, dtype=float)
    return {
        "n_events": int(n_ev),
        "event_days": int(ed.size),
        "events_per_event_day": float(n_ev / ed.size),
        "pct_within_5_snaps": (100.0 * float(np.mean(gaps_sn <= 5)) if len(gaps_sn) else 0.0),
        "median_gap_snaps": (float(np.median(gaps_sn)) if len(gaps_sn) else np.nan),
        "days_0": int(all_days - ed.size),
        "days_1": int((ed == 1).sum()),
        "days_2plus": int((ed >= 2).sum()),
        "days_5plus": int((ed >= 5).sum()),
        # same-day events are dependent; the independent unit is the event DAY
        "effective_day_clusters": int(ed.size),
    }


def mde_from_clusters(n_eff: int) -> float:
    """Outcome-blind minimum detectable correlation at 80% power, one-sided 5%.

    Uses ONLY the cluster count and the standard SE of a correlation. No
    outcome data of any kind enters this calculation.
    """
    if n_eff is None or n_eff < 5:
        return float("nan")
    return (Z_ALPHA + Z_POWER) / math.sqrt(max(n_eff - 3, 1))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    days = collected_days()
    print(f"EXP035 Phase 0.5 -- outcome-blind audit over {len(days)} days")

    top1_all, dcom_all, dcom_s_all, day_ids, times_all = [], [], [], [], []
    meta = []
    for i, d in enumerate(days):
        s = day_series(load_near(d))
        if s is None:
            continue
        top1_all.append(s["top1"]); dcom_all.append(s["dcom"]); dcom_s_all.append(s["dcom_sync"])
        day_ids.append(np.full(s["n"], i)); times_all.append(s["times"].to_numpy())
        meta.append({"day": d.isoformat(), "n_snapshots": s["n"], "n_strikes": s["n_strikes"],
                     "n_contracts": s["n_contracts"], "n_batches": s["n_batches"],
                     "implied_spread_s": s["implied_spread_s"],
                     "median_gap_s": float(np.nanmedian(s["gaps"]))})
    top1 = np.concatenate(top1_all); dcom = np.concatenate(dcom_all)
    dcom_s = np.concatenate(dcom_s_all); dids = np.concatenate(day_ids)
    tms = np.concatenate(times_all)
    md = pd.DataFrame(meta)
    md.to_csv(f"{OUT}/phase0_5_collection_meta.csv", index=False)

    # ---- concentration frequency curve --------------------------------
    rows = []
    for thr in CONC_GRID:
        st = cluster_stats((top1 >= thr).astype(int), dids, tms)
        st.update(family="CONCENTRATION", threshold=thr,
                  fire_pct=100.0 * float(np.mean(top1 >= thr)),
                  events_per_day=float((top1 >= thr).sum()) / md.shape[0],
                  mde_ic=mde_from_clusters(st["effective_day_clusters"]))
        rows.append(st)

    # ---- migration frequency curve ------------------------------------
    for thr in MIGR_GRID:
        f = (np.abs(dcom) >= thr).astype(int)
        st = cluster_stats(f, dids, tms)
        pos = dcom[(np.abs(dcom) >= thr) & np.isfinite(dcom)]
        st.update(family="MIGRATION", threshold=thr,
                  fire_pct=100.0 * float(np.nanmean(np.abs(dcom) >= thr)),
                  events_per_day=float(f.sum()) / md.shape[0],
                  mde_ic=mde_from_clusters(st["effective_day_clusters"]),
                  pct_positive_direction=(100.0 * float(np.mean(pos > 0)) if len(pos) else np.nan))
        rows.append(st)
    freq = pd.DataFrame(rows)
    freq.to_csv(f"{OUT}/phase0_5_event_frequency.csv", index=False)

    # ---- migration collection-artifact diagnostics ---------------------
    art = []
    ref = 1.0
    fire = (np.abs(dcom) >= ref)
    fire_s = (np.abs(dcom_s) >= ref)
    # by implied batch count (proxy for within-snapshot timestamp spread)
    per_day_rate = pd.DataFrame({"day": dids, "fire": fire.astype(float)}).groupby("day")["fire"].mean()
    md2 = md.reset_index(drop=True).copy()
    md2["fire_rate"] = per_day_rate.reindex(range(len(md2))).to_numpy()
    for col in ("n_batches", "implied_spread_s", "n_contracts", "median_gap_s"):
        g = md2.groupby(col)["fire_rate"].agg(["mean", "size"])
        for val, r in g.iterrows():
            art.append({"diagnostic": f"migration_rate_by_{col}", "value": val,
                        "mean_fire_rate_pct": 100.0 * float(r["mean"]), "n_days": int(r["size"])})
    rho = md2[["n_batches", "implied_spread_s", "n_contracts", "median_gap_s", "fire_rate"]].corr(
        method="spearman")["fire_rate"]
    for k, v in rho.items():
        if k != "fire_rate":
            art.append({"diagnostic": "spearman_rate_vs_" + k, "value": np.nan,
                        "mean_fire_rate_pct": float(v), "n_days": len(md2)})
    art.append({"diagnostic": "migration_rate_full_band", "value": ref,
                "mean_fire_rate_pct": 100.0 * float(np.nanmean(fire)), "n_days": len(md2)})
    art.append({"diagnostic": "migration_rate_sync_band_ATMpm1", "value": ref,
                "mean_fire_rate_pct": 100.0 * float(np.nanmean(fire_s)), "n_days": len(md2)})
    art.append({"diagnostic": "pct_positive_direction_full_band", "value": ref,
                "mean_fire_rate_pct": 100.0 * float(np.mean(dcom[fire & np.isfinite(dcom)] > 0)),
                "n_days": len(md2)})
    pd.DataFrame(art).to_csv(f"{OUT}/phase0_5_migration_artifact.csv", index=False)

    print(f"  days used: {len(md)} | contracts/day median {int(md['n_contracts'].median())} "
          f"| batches {md['n_batches'].min()}-{md['n_batches'].max()} "
          f"| implied spread {md['implied_spread_s'].min():.1f}-{md['implied_spread_s'].max():.1f}s")
    print(f"Wrote {OUT}/phase0_5_event_frequency.csv, phase0_5_migration_artifact.csv, "
          f"phase0_5_collection_meta.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
