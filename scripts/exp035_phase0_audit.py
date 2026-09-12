"""EXP035 Phase 0 -- READ-ONLY data capability audit. NO models, NO outcomes.

Measures ONLY predictor-side properties: distributions, data quality, causal
timing, event frequency and redundancy among candidate predictor definitions.

It does NOT read future returns, does NOT compute IC or correlation with any
outcome, does NOT fit a model, does NOT search thresholds, and does NOT touch
EXP032/EXP033/EXP034-A artifacts.

    python scripts/exp035_phase0_audit.py

Output: reports/exp035/phase0_capability_audit.csv
        reports/exp035/phase0_redundancy.csv
        reports/exp035/phase0_atm_roll.csv
"""

from __future__ import annotations

import glob
import os
from datetime import date

import numpy as np
import pandas as pd

EPS = 1e-9
BAND = tuple(range(-5, 6))          # ATM-5 .. ATM+5, wider than EXP034's +-2
OUT = "reports/exp035"


def collected_days() -> list[date]:
    days = []
    for p in sorted(glob.glob("data/option_chain/*/*/*")):
        if os.path.isdir(p):
            y, m, d = p.replace("\\", "/").split("/")[-3:]
            days.append(date(int(y), int(m), int(d)))
    return sorted(days)


def load_day(d: date) -> pd.DataFrame:
    f = sorted(glob.glob(f"data/option_chain/{d.year:04d}/{d.month:02d}/{d.day:02d}/*.parquet"))
    df = pd.concat([pd.read_parquet(x) for x in f], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def pivot(df, otype, value, strikes, times):
    sub = df[df["option_type"] == otype]
    p = sub.pivot_table(index="snapshot_ts", columns="strike", values=value, aggfunc="last")
    return p.reindex(index=times, columns=strikes).to_numpy(dtype=float)


def build_band(df: pd.DataFrame) -> dict | None:
    """Relative-strike matrices for one (day, expiry) chain."""
    df = df.drop_duplicates(subset=["snapshot_ts", "strike", "option_type"], keep="last")
    times = pd.DatetimeIndex(np.sort(df["snapshot_ts"].unique()))
    strikes = np.sort(df["strike"].unique())
    n = len(times)
    if n < 20 or len(strikes) < 11:
        return None
    spot = df.groupby("snapshot_ts")["spot"].first().reindex(times).to_numpy(dtype=float)
    atm_idx = np.argmin(np.abs(strikes[None, :] - spot[:, None]), axis=1)
    rows = np.arange(n)

    raw = {}
    for ot in ("CE", "PE"):
        raw[(ot, "vol")] = np.nan_to_num(pivot(df, ot, "volume", strikes, times))
        raw[(ot, "oi")] = np.nan_to_num(pivot(df, ot, "open_interest", strikes, times))
        raw[(ot, "bid")] = pivot(df, ot, "bid", strikes, times)
        raw[(ot, "ask")] = pivot(df, ot, "ask", strikes, times)
        raw[(ot, "last")] = pivot(df, ot, "last_price", strikes, times)

    # CRITICAL ORDER OF OPERATIONS: difference along ABSOLUTE strikes first,
    # then gather at relative-strike indices. Differencing along the relative
    # axis would subtract two DIFFERENT contracts whenever ATM rolls (~10.7% of
    # snapshots here), which manufactures large spurious negatives and breaks
    # every bounded derived measure. EXP034 does it in this correct order.
    dvol_abs, doi_abs, mid_abs, qchg_abs, midret_abs = {}, {}, {}, {}, {}
    for ot in ("CE", "PE"):
        v, oi = raw[(ot, "vol")], raw[(ot, "oi")]
        bid, ask, last = raw[(ot, "bid")], raw[(ot, "ask")], raw[(ot, "last")]
        mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)
        nan_row = np.full((1, len(strikes)), np.nan)
        dvol_abs[ot] = np.vstack([nan_row, np.diff(v, axis=0)])
        doi_abs[ot] = np.vstack([nan_row, np.diff(oi, axis=0)])
        mid_abs[ot] = mid
        qchg_abs[ot] = np.vstack([nan_row, ((np.diff(bid, axis=0) != 0)
                                            | (np.diff(ask, axis=0) != 0)).astype(float)])
        prev = np.vstack([nan_row, mid[:-1]])
        midret_abs[ot] = np.where(prev > 0, mid / prev - 1.0, np.nan)
        raw[(ot, "mid")] = mid

    band = {}
    clipped = np.zeros(n, dtype=bool)
    for k in BAND:
        idx = np.clip(atm_idx + k, 0, len(strikes) - 1)
        clipped |= (idx == 0) | (idx == len(strikes) - 1)
        for ot in ("CE", "PE"):
            bid = raw[(ot, "bid")][rows, idx]
            ask = raw[(ot, "ask")][rows, idx]
            mid = raw[(ot, "mid")][rows, idx]
            band[(k, ot, "dvol")] = dvol_abs[ot][rows, idx]
            band[(k, ot, "doi")] = doi_abs[ot][rows, idx]
            band[(k, ot, "mid")] = mid
            band[(k, ot, "spread")] = np.where((bid > 0) & (ask > 0), ask - bid, np.nan)
            band[(k, ot, "rel_spread")] = np.where(
                (bid > 0) & (ask > 0) & (mid > 0), (ask - bid) / mid, np.nan)
            band[(k, ot, "quote_changed")] = qchg_abs[ot][rows, idx]
            band[(k, ot, "mid_ret")] = midret_abs[ot][rows, idx]

    gaps = np.concatenate([[np.nan], np.diff(times.view("int64")) / 1e9
                           if times.dtype == "datetime64[ns]"
                           else pd.Series(times).diff().dt.total_seconds().to_numpy()[1:]])
    gaps = pd.Series(times).diff().dt.total_seconds().to_numpy()
    atm_rolled = np.concatenate([[0.0], (np.diff(strikes[atm_idx]) != 0).astype(float)])
    return {"times": times, "n": n, "band": band, "gaps": gaps,
            "atm_rolled": atm_rolled, "clipped": clipped, "n_strikes": len(strikes)}


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    if len(fin) == 0:
        return {"n": 0, "missing_pct": 100.0, "zero_pct": np.nan, "neg_pct": np.nan,
                "q01": np.nan, "median": np.nan, "q99": np.nan, "max_abs": np.nan}
    return {
        "n": int(len(x)),
        "missing_pct": 100.0 * float(np.mean(~np.isfinite(x))),
        "zero_pct": 100.0 * float(np.mean(fin == 0)),
        "neg_pct": 100.0 * float(np.mean(fin < 0)),
        "q01": float(np.quantile(fin, 0.01)),
        "median": float(np.median(fin)),
        "q99": float(np.quantile(fin, 0.99)),
        "max_abs": float(np.max(np.abs(fin))),
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    days = collected_days()
    print(f"EXP035 Phase 0 capability audit over {len(days)} collected days")

    rows, red_frames, roll_rows = [], [], []

    for scope in ("near", "next"):
        acc: dict[str, list] = {}
        gap_all, roll_all, clip_all = [], [], []
        for d in days:
            raw = load_day(d)
            exps = sorted(raw["expiry"].unique())
            if scope == "next" and len(exps) < 2:
                continue
            sel = exps[0] if scope == "near" else exps[1]
            b = build_band(raw[raw["expiry"] == sel])
            if b is None:
                continue
            band, n = b["band"], b["n"]
            gap_all.append(b["gaps"])
            roll_all.append(b["atm_rolled"])
            clip_all.append(b["clipped"].astype(float))

            # --- candidate quantities (predictor side only) ---------------
            dv_tot = {k: np.nan_to_num(band[(k, "CE", "dvol")]) + np.nan_to_num(band[(k, "PE", "dvol")])
                      for k in BAND}
            M = np.vstack([dv_tot[k] for k in BAND])              # (11, n)
            tot = M.sum(axis=0)

            def push(name, arr):
                acc.setdefault(name, []).append(np.asarray(arr, dtype=float))

            # C1/C2 raw incremental quantities, at ATM and at ATM+2
            for k in (0, 2):
                push(f"C1_dvol_CE_k{k:+d}", band[(k, "CE", "dvol")])
                push(f"C1_dvol_PE_k{k:+d}", band[(k, "PE", "dvol")])
                push(f"C2_doi_CE_k{k:+d}", band[(k, "CE", "doi")])
                push(f"C2_doi_PE_k{k:+d}", band[(k, "PE", "doi")])

            # C3 strike-localised activity: cross-sectional z of one strike vs band
            mu, sd = M.mean(axis=0), M.std(axis=0)
            for k in (0, 2):
                i = BAND.index(k)
                push(f"C3_xsec_z_k{k:+d}", (M[i] - mu) / (sd + EPS))

            # C3b cross-sectional z of incremental OI (magnitude), same idea
            Moi = np.vstack([np.abs(np.nan_to_num(band[(k, "CE", "doi")]))
                             + np.abs(np.nan_to_num(band[(k, "PE", "doi")])) for k in BAND])
            mu_oi, sd_oi = Moi.mean(axis=0), Moi.std(axis=0)
            for k in (0, 2):
                i = BAND.index(k)
                push(f"C3b_xsec_z_doi_k{k:+d}", (Moi[i] - mu_oi) / (sd_oi + EPS))

            # C4 CE/PE imbalance AT THE SAME strike
            for k in (0, 2):
                c = np.nan_to_num(band[(k, "CE", "dvol")])
                p = np.nan_to_num(band[(k, "PE", "dvol")])
                push(f"C4_cepe_imb_k{k:+d}", (c - p) / (c + p + EPS))

            # C5 concentration of incremental activity across the band
            share = M / (tot + EPS)
            push("C5_top1_share", share.max(axis=0))
            push("C5_hhi", (share ** 2).sum(axis=0))

            # C6 migration of the centre of incremental activity
            ks = np.array(BAND, dtype=float)[:, None]
            com = (ks * M).sum(axis=0) / (tot + EPS)
            com = np.where(tot > 0, com, np.nan)
            push("C6_com", com)
            dcom = np.concatenate([[np.nan], np.diff(com)])
            push("C6_com_delta", dcom)

            # --- candidate EVENT definitions: FIRING FREQUENCY ONLY, never
            # --- linked to any outcome (Phase 0 forbids outcome screening).
            # --- Thresholds are illustrative round numbers for frequency
            # --- characterisation; they are NOT selected and NOT proposed as
            # --- final event definitions.
            zmax = ((M - mu) / (sd + EPS)).max(axis=0)
            push("EVT_A_localised_vol_burst_z2", (zmax >= 2.0).astype(float))
            push("EVT_A_localised_vol_burst_z25", (zmax >= 2.5).astype(float))
            push("EVT_B_concentration_top1_40", (share.max(axis=0) >= 0.40).astype(float))
            push("EVT_B_concentration_top1_50", (share.max(axis=0) >= 0.50).astype(float))
            imb_band = np.vstack([
                (np.nan_to_num(band[(k, "CE", "dvol")]) - np.nan_to_num(band[(k, "PE", "dvol")]))
                / (np.nan_to_num(band[(k, "CE", "dvol")]) + np.nan_to_num(band[(k, "PE", "dvol")]) + EPS)
                for k in BAND])
            push("EVT_C_cepe_imbalance_abs50", (np.abs(imb_band).max(axis=0) >= 0.50).astype(float))
            push("EVT_D_migration_ge1_strike", (np.abs(dcom) >= 1.0).astype(float))
            zoi = ((Moi - mu_oi) / (sd_oi + EPS)).max(axis=0)
            push("EVT_E_localised_oi_burst_z2", (zoi >= 2.0).astype(float))

            # C7 contemporaneous premium response
            for k in (0, 2):
                push(f"C7_mid_ret_CE_k{k:+d}", band[(k, "CE", "mid_ret")])

            # C8 microstructure
            for k in (0, 2):
                push(f"C8_rel_spread_CE_k{k:+d}", band[(k, "CE", "rel_spread")])
                push(f"C8_quote_changed_CE_k{k:+d}", band[(k, "CE", "quote_changed")])

        if not acc:
            continue
        roll_mask = np.concatenate(roll_all).astype(bool)
        for name, chunks in acc.items():
            x = np.concatenate(chunks)
            s = stats(x)
            s.update(variable=name, expiry_scope=scope)
            # ATM-normalisation sensitivity: the same statistic restricted to
            # snapshots where the ATM strike did NOT roll.
            if len(x) == len(roll_mask):
                sn = stats(x[~roll_mask])
                s["neg_pct_no_roll"] = sn["neg_pct"]
                s["median_no_roll"] = sn["median"]
                s["max_abs_no_roll"] = sn["max_abs"]
            rows.append(s)

        if scope == "near":
            L = min(len(np.concatenate(v)) for v in acc.values())
            red = pd.DataFrame({k: np.concatenate(v)[:L] for k, v in acc.items()})
            red_frames.append(red)
            g = np.concatenate(gap_all)
            r = np.concatenate(roll_all)
            c = np.concatenate(clip_all)
            roll_rows.append({
                "scope": scope,
                "snapshot_gap_median_s": float(np.nanmedian(g)),
                "snapshot_gap_p01_s": float(np.nanquantile(g, 0.01)),
                "snapshot_gap_p99_s": float(np.nanquantile(g, 0.99)),
                "snapshot_gap_max_s": float(np.nanmax(g)),
                "atm_roll_pct_of_snapshots": 100.0 * float(np.nanmean(r)),
                "band_clipped_pct_of_snapshots": 100.0 * float(np.nanmean(c)),
            })
        print(f"  scope={scope}: {len(acc)} candidate variables measured")

    pd.DataFrame(rows).to_csv(f"{OUT}/phase0_capability_audit.csv", index=False)
    pd.DataFrame(roll_rows).to_csv(f"{OUT}/phase0_atm_roll.csv", index=False)

    if red_frames:
        red = red_frames[0]
        corr = red.corr(method="spearman")
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if np.isfinite(v):
                    pairs.append({"a": cols[i], "b": cols[j], "spearman": float(v)})
        pd.DataFrame(pairs).sort_values("spearman", key=abs, ascending=False).to_csv(
            f"{OUT}/phase0_redundancy.csv", index=False)

    print(f"Wrote {OUT}/phase0_capability_audit.csv, phase0_redundancy.csv, phase0_atm_roll.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
