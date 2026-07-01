"""Experiment 018: overnight global -> NIFTY volatility & return spillover.

Pre-registered in docs/preregistrations/exp018_volatility_spillover.md.

Questions:
  Q1  Does overnight US explain NIFTY's OPEN GAP? (coupling; NOT tradeable)
  Q2  Does overnight global predict NIFTY's POST-OPEN return? (the tradeable one)
  Q3  Does overnight US vol add forward power for NIFTY daily RV beyond HAR?

Look-ahead safe: for NIFTY day D, only the last US session STRICTLY BEFORE D is
used (US closes ~01:30 IST, before NIFTY's 09:15 open). Alignment via
merge_asof(direction="backward", allow_exact_matches=False).

    python scripts/exp018_spillover.py

Read-only research. No trades. External data cached to data/external/.
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

FIRST_HOUR_END = "10:15"
CACHE = Path("data") / "external" / "overnight_global.parquet"
TICKERS = {"sp": "^GSPC", "ndx": "^IXIC", "vix": "^VIX", "usdinr": "USDINR=X"}


# --- NIFTY daily frame from the 5m warehouse --------------------------------

def nifty_daily() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join("data", "candles", "5m", "*",
                                          "NIFTY_*.parquet")))
    if not files:
        raise SystemExit("No 5m NIFTY parquet found")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.normalize()
    df["hm"] = df["timestamp"].dt.strftime("%H:%M")

    rows = []
    prev_close = None
    for d, g in df.groupby("date"):
        g = g.sort_values("timestamp")
        c = g["close"].to_numpy()
        if len(c) < 12:
            continue
        open_d = float(g["open"].iloc[0])
        close_d = float(c[-1])
        fh = g[g["hm"] <= FIRST_HOUR_END]
        fh_close = float(fh["close"].iloc[-1]) if len(fh) else float("nan")
        lr = np.log(c[1:] / c[:-1])
        rv = math.sqrt(float(np.sum(lr ** 2)))
        row = {
            "date": d, "open": open_d, "close": close_d, "prev_close": prev_close,
            "gap": (math.log(open_d / prev_close) if prev_close else float("nan")),
            "first_hour_ret": (math.log(fh_close / open_d)
                               if fh_close == fh_close and open_d else float("nan")),
            "o2c": math.log(close_d / open_d) if open_d else float("nan"),
            "rv": rv,
        }
        rows.append(row)
        prev_close = close_d
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["log_rv"] = np.log(out["rv"].clip(lower=1e-9))
    return out


# --- overnight external data ------------------------------------------------

def fetch_overnight(start, end) -> pd.DataFrame | None:
    if CACHE.exists():
        cached = pd.read_parquet(CACHE)
        if cached["us_date"].min() <= pd.Timestamp(start) and \
           cached["us_date"].max() >= pd.Timestamp(end) - pd.Timedelta(days=7):
            print(f"  using cached overnight data ({len(cached)} rows)")
            return cached
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed -> cannot fetch. INSUFFICIENT DATA.")
        return None

    frames = {}
    for key, tk in TICKERS.items():
        try:
            d = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if len(d) == 0:
                print(f"  WARN {tk} returned no rows (dropped)")
                continue
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            frames[key] = close.rename(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN fetch failed for {tk}: {exc} (dropped)")
    if "sp" not in frames:
        print("  S&P 500 unavailable -> INSUFFICIENT DATA.")
        return None
    ext = pd.concat(frames.values(), axis=1, sort=True)
    ext.index = pd.to_datetime(ext.index).tz_localize(None).normalize()
    ext = ext.sort_index()
    # overnight predictors (log returns / changes on the US day)
    out = pd.DataFrame({"us_date": ext.index})
    for key in ("sp", "ndx", "usdinr"):
        if key in ext:
            out[f"{key}_ret"] = np.log(ext[key] / ext[key].shift(1)).to_numpy()
    if "vix" in ext:
        out["vix_chg"] = np.log(ext["vix"] / ext["vix"].shift(1)).to_numpy()
    out = out.dropna(subset=["sp_ret"]).reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    print(f"  fetched & cached {len(out)} overnight rows -> {CACHE}")
    return out


# --- stats helpers ----------------------------------------------------------

def hac_ols(y, X, lags=5):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def oos_r2(actual, pred, bench):
    actual, pred, bench = map(np.asarray, (actual, pred, bench))
    sse = np.sum((actual - pred) ** 2)
    sst = np.sum((actual - bench) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def wf_incremental(frame, base_cols, extra_cols, n_folds=4):
    """Walk-forward incremental OOS R2 of (base+extra) vs base. Returns list."""
    sub = frame.dropna(subset=base_cols + extra_cols + ["y"]).reset_index(drop=True)
    n = len(sub)
    if n < 120:
        return []
    start = n // 2
    edges = np.linspace(start, n, n_folds + 1, dtype=int)
    out = []

    def fp(tr, te, cols):
        Xtr = np.column_stack([np.ones(len(tr)), tr[cols].to_numpy()])
        Xte = np.column_stack([np.ones(len(te)), te[cols].to_numpy()])
        b = sm.OLS(tr["y"].to_numpy(), Xtr).fit().params
        return Xte @ b

    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        tr, te = sub.iloc[:a], sub.iloc[a:b]
        if len(te) < 20:
            continue
        if base_cols:
            pbase = fp(tr, te, base_cols)
        else:
            pbase = np.full(len(te), tr["y"].mean())
        pfull = fp(tr, te, base_cols + extra_cols)
        out.append(oos_r2(te["y"], pfull, pbase))
    return out


def _fmt_reg(name, model, cols):
    print(f"  {name}: in-sample R2 = {model.rsquared:.3f}")
    for nm, coef, tv in zip(["const"] + cols, model.params, model.tvalues):
        print(f"     {nm:<12} coef={coef:+.4f}  HAC t={tv:+.2f}")


def main() -> int:
    nd = nifty_daily()
    print("=" * 92)
    print("EXP 018  OVERNIGHT GLOBAL -> NIFTY SPILLOVER")
    print(f"NIFTY days: {len(nd)}  | {nd['date'].min().date()} -> {nd['date'].max().date()}")
    start = (nd["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (nd["date"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    print("Fetching overnight global data (S&P/Nasdaq/VIX/USDINR)...")
    ext = fetch_overnight(start, end)
    if ext is None:
        print("VERDICT: INSUFFICIENT DATA (no external feed).")
        return 0

    # align: for NIFTY day D use the last US date STRICTLY before D
    nd_s = nd.sort_values("date").copy()
    ext_s = ext.sort_values("us_date").copy()
    nd_s["date"] = pd.to_datetime(nd_s["date"]).astype("datetime64[ns]")
    ext_s["us_date"] = pd.to_datetime(ext_s["us_date"]).astype("datetime64[ns]")
    m = pd.merge_asof(nd_s, ext_s, left_on="date", right_on="us_date",
                      direction="backward", allow_exact_matches=False)
    pred_cols = [c for c in ["sp_ret", "ndx_ret", "vix_chg", "usdinr_ret"]
                 if c in m.columns]
    m = m.dropna(subset=pred_cols)
    n_al = len(m)
    print(f"Aligned days (overnight known pre-open): {n_al}  predictors: {pred_cols}")
    if n_al < 200:
        print("VERDICT: INSUFFICIENT DATA (< 200 aligned days).")
        return 0

    # --- Q1: open gap -------------------------------------------------------
    print("\n--- Q1: NIFTY open gap ~ overnight global (COUPLING; not tradeable) ---")
    g = m.dropna(subset=["gap"] + pred_cols)
    mg = hac_ols(g["gap"].to_numpy(), g[pred_cols].to_numpy())
    _fmt_reg("gap", mg, pred_cols)
    sp_t = mg.tvalues[1 + pred_cols.index("sp_ret")]
    q1 = "GAP COUPLING CONFIRMED" if sp_t >= 2.5 else "weak/none"
    print(f"  --> Q1: {q1} (sp_ret t={sp_t:+.2f})  "
          f"[{mg.rsquared*100:.0f}% of gap variance explained]  NOT TRADEABLE")

    # --- Q2: post-open returns ---------------------------------------------
    print("\n--- Q2: POST-OPEN return ~ overnight global (the tradeable test) ---")
    q2_edge = False
    for tgt in ("first_hour_ret", "o2c"):
        sub = m.dropna(subset=[tgt] + pred_cols)
        mm = hac_ols(sub[tgt].to_numpy(), sub[pred_cols].to_numpy())
        _fmt_reg(tgt, mm, pred_cols)
        frame = sub.rename(columns={tgt: "y"})[["y"] + pred_cols]
        incr = wf_incremental(frame, [], pred_cols)
        mean_incr = float(np.mean(incr)) if incr else float("nan")
        n_pos = sum(1 for x in incr if x > 0)
        max_t = max(abs(t) for t in mm.tvalues[1:])
        edge = (max_t >= 2.5 and mean_incr > 0 and n_pos >= 3)
        q2_edge = q2_edge or edge
        print(f"     walk-forward mean incr OOS R2 = {mean_incr:+.4f} "
              f"({n_pos}/{len(incr)} folds+)  max|t|={max_t:.2f}  "
              f"-> {'EDGE?' if edge else 'no edge'}")
    q2 = "POST-OPEN EDGE (needs Exp 023 snooping check!)" if q2_edge else "NO POST-OPEN EDGE"
    print(f"  --> Q2: {q2}")

    # --- Q3: vol spillover into RV -----------------------------------------
    print("\n--- Q3: NIFTY daily RV ~ HAR + overnight US vol ---")
    s = m["log_rv"].reset_index(drop=True)
    vol_extra = [c for c in ["vix_chg"] if c in m.columns]
    m2 = m.reset_index(drop=True).copy()
    m2["abs_sp"] = m2["sp_ret"].abs()
    vol_extra.append("abs_sp")
    har = pd.DataFrame({
        "y": s, "d": s.shift(1), "w": s.shift(1).rolling(5).mean(),
        "m": s.shift(1).rolling(22).mean(),
    })
    for c in vol_extra:
        har[c] = m2[c]                    # overnight terms known pre-open (no lag)
    har = har.dropna().reset_index(drop=True)
    base = hac_ols(har["y"].to_numpy(), har[["d", "w", "m"]].to_numpy())
    aug = hac_ols(har["y"].to_numpy(), har[["d", "w", "m"] + vol_extra].to_numpy())
    print(f"  HAR R2 {base.rsquared:.3f} -> +overnight {aug.rsquared:.3f}")
    for nm, coef, tv in zip(vol_extra, aug.params[-len(vol_extra):],
                            aug.tvalues[-len(vol_extra):]):
        print(f"     {nm:<10} coef={coef:+.4f}  HAC t={tv:+.2f}")
    incr = wf_incremental(har.rename(columns={}), ["d", "w", "m"], vol_extra)
    mean_incr = float(np.mean(incr)) if incr else float("nan")
    n_pos = sum(1 for x in incr if x > 0)
    max_t = max(abs(t) for t in aug.tvalues[-len(vol_extra):])
    q3 = ("VOL SPILLOVER ADDS VALUE" if max_t >= 2.5 and mean_incr > 0 and n_pos >= 3
          else "NO INCREMENTAL VALUE over HAR")
    print(f"     walk-forward mean incr OOS R2 = {mean_incr:+.4f} "
          f"({n_pos}/{len(incr)} folds+)  --> {q3}")

    print("\n" + "=" * 92)
    print("SUMMARY (pre-registered):")
    print(f"  Q1 open-gap coupling: {q1}  (NOT tradeable)")
    print(f"  Q2 post-open edge:    {q2}")
    print(f"  Q3 vol spillover->RV: {q3}")
    print("  Scope: characterisation + tradeability check. No trading rule implied.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
