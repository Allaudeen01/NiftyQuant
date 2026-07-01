# PRE-REGISTRATION — Experiment 024

**OHLC Volatility Estimators as HAR Inputs (Parkinson / Garman-Klass /
Rogers-Satchell / Yang-Zhang)**

> Status: **PRE-REGISTERED (locked before running).** Method and decision rules
> fixed in advance.

---

## Scientific Question
Does building the HAR-RV model on a more efficient **OHLC volatility estimator**
(Parkinson, Garman-Klass, Rogers-Satchell, or a Yang-Zhang-style measure that
includes the overnight gap) forecast NIFTY next-day volatility **better** than
(a) the naive close-to-close estimator and (b) our incumbent 5-minute realized
variance?

## Honest framing (important)
Range/OHLC estimators give their biggest lift when you have **only daily OHLC**.
We already have **5-minute realized variance (RV5M)**, which is typically the
most accurate volatility proxy available. So the fair, non-inflated question is
not "does YZ beat close-to-close" (it almost certainly does) but **"can any OHLC
estimator match or beat our existing 5m RV as a HAR input?"** The honest prior is
that RV5M is hard to beat; the interesting angle is that RV5M *omits the
overnight gap*, which the Yang-Zhang-style measure captures.

## Data
Daily NIFTY OHLC (from the `data/candles/1d` warehouse) + prior close for the
overnight term, plus `data/candles/5m` for the RV5M incumbent and the forecast
target. 2024-06 → 2026-06 (~490 days).

## Estimators compared (all as per-day variance, then HAR d/w/m in log space)
- **CC** — close-to-close squared return (the naive baseline to beat).
- **RV5M** — 5-minute realized variance (incumbent, most accurate).
- **PARK** — Parkinson (high/low range).
- **GK** — Garman-Klass (OHLC).
- **RS** — Rogers-Satchell (drift-independent).
- **YZ1** — single-day Yang-Zhang-style: overnight² + Rogers-Satchell (includes
  the gap RV5M omits).

## Target & metric
- **Target:** next-day RV5M (the most accurate ex-post variance proxy; the
  Patton-2011 recommendation for fair forecast comparison under an imperfect
  proxy). All estimators forecast the **same** target on the **same** OOS dates.
- **Protocol:** expanding one-step-ahead walk-forward (first 50% train), refit
  each step. HAR = intercept + d/w/m of the estimator's own log-variance history.
- **Metrics:** RMSE, MAE, **QLIKE (primary)**; **Diebold-Mariano** (HAC) of each
  estimator vs the incumbent RV5M under QLIKE, with Benjamini-Hochberg
  correction across the pairwise tests.

## Hypotheses
- **H0:** no estimator significantly improves OOS QLIKE over RV5M.
- **H1:** at least one estimator significantly beats RV5M (BH-adjusted DM
  p<0.05, lower QLIKE).

## Decision Rules (LOCKED)
- **UPGRADE ADOPTED** if an estimator has strictly lower OOS QLIKE than RV5M
  **and** beats it at BH-adjusted DM p<0.05. → switch the HAR input to that
  estimator (and, if it is YZ1, note that the gain comes from the overnight
  term).
- **RV5M REMAINS BEST** if RV5M has the lowest QLIKE or no estimator beats it at
  BH p<0.05. → keep RV5M; range estimators are confirmed as *fallbacks* for
  periods without intraday data.
- **INCONCLUSIVE** if the lowest-QLIKE estimator does not clear DM significance.
- Report the CC → best-estimator improvement regardless (quantifies how much the
  naive baseline leaves on the table).

## Expected outcome (honest prior)
RV5M likely wins or ties (high-frequency beats daily-range on an intraday
target); range estimators comfortably beat naive CC. The valuable practical
takeaway is either "RV5M confirmed best, keep it" or "YZ1's overnight term adds a
real, testable increment." Any winner sourced from this 6-way compare would still
face an Exp 023-style snooping caveat before earning high confidence.

## Scope
Forecasting-input comparison. **No trading rule, no strategy.**

---

## Scoring (filled AFTER running — 494 days, common 236-day OOS)
OOS QLIKE (primary, lower better): **RV5M 0.1462** < PARK 0.1506 < GK 0.1544 <
RS 0.1665 < YZ1 0.1730 < CC 0.2133.

Diebold-Mariano vs RV5M (BH-adjusted): **CC worse** (BH p=0.0006), PARK no diff
(0.72), GK no diff (0.53), RS no diff (0.061), **YZ1 worse** (BH p=0.011).

- **VERDICT: RV5M REMAINS BEST.** Our incumbent 5-minute realized variance has
  the lowest QLIKE and nothing beats it. Keep it.
- **The "free 5–10% YZ boost" did NOT materialize** — and honestly, it can't
  here: YZ1 includes the overnight gap, which is a mismatch when forecasting an
  intraday-only target, so it comes out significantly *worse*. The incoming
  assumption was wrong for the case where you already have intraday data.
- **But range estimators are excellent fallbacks:** Parkinson (0.1506) and
  Garman-Klass (0.1544) are **statistically indistinguishable from RV5M**
  (BH p=0.72, 0.53). For a daily-OHLC-only history (e.g. the 10-year Exp 028
  data), Parkinson gives essentially 5m-RV-quality volatility. That is the real,
  usable finding.
- **Avoid naive close-to-close:** CC leaves **+31%** QLIKE on the table vs RV5M.

## Practical consequence
Keep RV5M as the live HAR input. Use **Parkinson** (not close-to-close) whenever
only daily OHLC is available — notably it could sharpen the realized-vol leg of
the Exp 028 macro VRP, which currently uses close-to-close. Reusable estimators
now live in `nifty_quant/analytics/vol_estimators.py` (5 validated tests).

---
*Pre-registered 2026-07-01, executed same day as `scripts/exp024_ohlc_estimators.py`.*
