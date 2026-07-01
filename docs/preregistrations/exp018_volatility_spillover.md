# PRE-REGISTRATION — Experiment 018

**Overnight Global → NIFTY Volatility & Return Spillover**

> Status: **PRE-REGISTERED (locked before running).** Method and decision rules
> fixed in advance.

---

## Scientific Questions
Using overnight global markets that close *before* NIFTY opens:
1. Does overnight US movement explain NIFTY's **open gap**? (coupling, likely
   yes — but *is it tradeable?* No — the gap is realized at the open.)
2. **The tradeable question:** does overnight global movement predict NIFTY's
   **post-open** return (first hour and open-to-close), i.e. drift/continuation
   or reversal *after* the open you could actually trade?
3. **Vol spillover:** does overnight US volatility (VIX change, |US move|) add
   forward power for NIFTY's **daily realized volatility** beyond HAR?

## Data
- NIFTY daily bars from the 5m warehouse (2024-06 → 2026-06): open, close,
  prev-close, first-hour (≤10:15) price, realized vol.
- Overnight external (Yahoo Finance via `yfinance`, cached to
  `data/external/`): **^GSPC** (S&P 500), **^IXIC** (Nasdaq), **^VIX** (US VIX),
  **USDINR=X**. Daily closes.

## Look-ahead alignment (critical)
US markets close ~01:30 IST; NIFTY opens 09:15 IST. So for NIFTY day **D**, the
newest *available* US session is the last US trading date **strictly before D**.
Alignment is `merge_asof(direction="backward", allow_exact_matches=False)` on the
date key — this also maps NIFTY-Monday to US-Friday over weekends/holidays.
Every predictor is therefore known before NIFTY's open on D: **no look-ahead.**

## Variables
- Predictors (overnight, known pre-open): `sp_ret`, `ndx_ret` (log daily
  returns), `vix_chg` (Δ log US VIX), `usdinr_ret` (log daily return).
- Targets on NIFTY day D:
  - `gap = log(open_D / close_{D-1})`
  - `first_hour_ret = log(P_{10:15} / open_D)` (post-open, tradeable window)
  - `o2c = log(close_D / open_D)` (post-open, full day)
  - `log_rv_D` (daily realized vol; with HAR terms as controls)

## Method (fixed)
1. **Q1 gap:** HAC regression `gap ~ sp_ret + ndx_ret + vix_chg + usdinr_ret`;
   report coefficients, t-stats, and in-sample R² (the coupling strength).
2. **Q2 post-open:** same predictors → `first_hour_ret` and `o2c`; HAC t-stats
   **and** 4-fold expanding walk-forward incremental OOS R² over an
   intercept-only baseline.
3. **Q3 vol:** `log_rv_D ~ har_d + har_w + har_m + vix_chg + |sp_ret|`; HAC
   t-stats on the overnight terms and walk-forward incremental OOS R² over HAR.

## Decision Rules (LOCKED)
- **Q1 GAP COUPLING CONFIRMED** if `sp_ret` HAC t ≥ 2.5 and positive. *(Report
  R². Explicitly label NOT TRADEABLE — the gap is priced at the open.)*
- **Q2 POST-OPEN EDGE** only if some overnight predictor has HAC |t| ≥ 2.5 for a
  post-open target **and** mean walk-forward incremental OOS R² > 0 in ≥3/4
  folds. Otherwise **NO POST-OPEN EDGE** (the open is efficient).
- **Q3 VOL SPILLOVER ADDS VALUE** only if an overnight vol term has HAC |t| ≥ 2.5
  **and** mean incremental OOS R² > 0 in ≥3/4 folds over HAR. Otherwise
  **NO INCREMENTAL VALUE over HAR**.
- **INSUFFICIENT DATA** if the fetch fails or the overlap is < 200 aligned days.

## Expected outcome (honest prior)
- Q1: strong positive coupling (India opens where the world points) — high R²,
  but **not a strategy**.
- Q2: most likely **NO post-open edge** — the gap absorbs the overnight news;
  this is consistent with the closed directional branch (Exp 001–009). Any
  survivor must additionally pass an Exp 023-style snooping check before trust.
- Q3: overnight vol *may* add a little to HAR for RV, but is plausibly redundant
  with HAR's own persistence; genuinely uncertain.

## Caveats
Yahoo daily data can have holiday/timezone mismatches and occasional gaps; the
`merge_asof` strictly-backward join is the guard. USDINR=X is a spot FX proxy and
may be sparse; if a ticker returns empty it is dropped and noted.

## Scope
Characterisation + honest tradeability check. **No trading rule, no strategy.**

---

## Scoring (filled AFTER running — 494 aligned days, real Yahoo data 2024-06→2026-06)
Overnight predictors available and aligned look-ahead-safe: `sp_ret`, `ndx_ret`,
`vix_chg`, `usdinr_ret`.

- **Q1 — GAP COUPLING CONFIRMED.** Overnight global explains **~21%** of NIFTY's
  open-gap variance; `sp_ret` HAC t=+2.69 (positive). (`ndx_ret` shows a negative
  joint coefficient — an artifact of S&P/Nasdaq collinearity, not a real inverse
  link; the *net* coupling is clearly positive.) **India opens where the world
  points — but this is NOT tradeable**: the move is realized in the opening gap
  itself, before you can act on it.
- **Q2 — NO POST-OPEN EDGE.** Neither the first-hour return nor open-to-close is
  predictable from overnight global: in-sample R² ≈ 0.01–0.02, all HAC |t| < 1.3,
  walk-forward incremental OOS R² negative (0/4 and 1/4 folds positive). The
  overnight information is **fully absorbed into the gap**; there is no residual
  drift or reversal to trade after the open. Consistent with the closed
  directional branch (Exp 001–009).
- **Q3 — NO INCREMENTAL VALUE over HAR (vol).** Overnight US VIX change
  (t=1.21) and |S&P move| (t=1.39) are insignificant additions to HAR for NIFTY
  daily RV; walk-forward incremental OOS R² +0.011 but only 2/4 folds positive.
  Redundant with HAR's own persistence.

## Interpretation (honest)
A textbook **"coupling exists, edge does not"** result. NIFTY's open is strongly,
demonstrably linked to overnight global markets (~21% of gap variance), but that
link is efficiently priced into the opening auction: there is no tradeable
post-open drift and no material volatility-forecast improvement over HAR. The one
genuinely tradeable question (Q2) is a clean negative — which is the honest,
expected outcome and needed no Exp 023 snooping escalation (nothing survived to
escalate). Overnight global data is useful for **context** (knowing the gap will
track the world) but is not a source of alpha here.

## Note
Real external data (Yahoo Finance via `yfinance`) now cached at
`data/external/overnight_global.parquet` (gitignored). This unblocks any future
global-linkage work without refetching.

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp018_spillover.py`.*
