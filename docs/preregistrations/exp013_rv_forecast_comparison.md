# PRE-REGISTRATION — Experiment 013

**Realized-Volatility Forecast Comparison: HAR vs EWMA vs GARCH vs EGARCH**

> Status: **PRE-REGISTERED (locked before running).** Hypothesis, models,
> metrics, and decision rules are fixed here in advance. Do not change the
> thresholds after seeing results.

---

## Scientific Question
Among standard, well-established volatility models, **which forecasts NIFTY
next-day realized volatility (RV) best out-of-sample**, and is any difference
**statistically significant** (not just a lucky sample)?

This is the horse-race that Exp 003/006A/012 implicitly assumed HAR wins. Here
we test that assumption directly against the classic alternatives.

## Object of study
Next-day intraday RV, `RV_{t+1}`, built from summed squared 5-minute
open-to-close log returns (overnight gap excluded), identical to Exp 003. Daily
frequency, ~470 observations across 2024–2026.

## Models compared (all standard, no bespoke tuning)
1. **RW (random walk):** `RV_{t+1} = RV_t`. Naive baseline.
2. **EWMA (RiskMetrics):** exponentially-weighted variance, λ = 0.94 (the
   published RiskMetrics daily value — fixed, not fitted).
3. **HAR-RV:** daily/weekly/monthly (1/5/22) log-RV regression. The incumbent.
4. **GARCH(1,1):** on daily close-to-close returns (`arch` package).
5. **EGARCH(1,1):** captures leverage/asymmetry (`arch` package).

**Explicitly EXCLUDED: LSTM / neural nets.** ~470 daily observations cannot
support a high-parameter model without overfitting; it would flatter in-sample
and fail out-of-sample. Documented in the roadmap. May revisit only with far
more data.

FIGARCH and HAR-RV-J are noted as future extensions (FIGARCH needs a long-memory
fit that is unstable on this sample; HAR-J depends on Exp 014 jump detection).

## Method (fixed)
- **Expanding walk-forward**, one-step-ahead. First 50% of days is the initial
  training window; then predict each subsequent day using only past data; refit
  periodically (every 21 trading days) to bound cost. No look-ahead.
- All models forecast the SAME target on the SAME out-of-sample dates so metrics
  are directly comparable.
- GARCH/EGARCH forecast next-day variance → converted to the same RV units
  (daily standard deviation) as HAR/EWMA before scoring.

## Metrics (all on the common OOS set)
- **RMSE** and **MAE** on RV (level).
- **QLIKE** (`σ̂²` vs realized `σ²`): the robust, standard volatility loss,
  `QLIKE = RV²/σ̂² − ln(RV²/σ̂²) − 1`. Primary metric (less sensitive to
  vol-level outliers than RMSE).
- **Diebold-Mariano** test, HAC-corrected, for each pair (primary: each model
  vs HAR) under QLIKE loss. Two-sided.

## Multiple-testing control
Five models → up to 10 pairwise DM tests. Report **Benjamini-Hochberg-adjusted**
p-values. Additionally note that a full **Hansen SPA / White Reality Check**
(Exp 023) is the proper family-wise control across models; this experiment
reports DM+BH and flags SPA as the follow-up.

## Hypotheses
- **H0:** HAR is not significantly beaten by any competitor on OOS QLIKE
  (no DM test rejects in HAR's disfavour after BH correction).
- **H1:** At least one model significantly improves on HAR (BH-adjusted DM
  p < 0.05, in the improving direction).

## Decision Rules (LOCKED)
- **HAR CONFIRMED BEST (or tied):** HAR has the lowest QLIKE, OR no competitor
  beats it at BH-adjusted DM p < 0.05. → HAR remains the volatility workhorse;
  Exp 012's HAR-based conclusion stands.
- **CHALLENGER WINS:** a competitor has strictly lower QLIKE **and** beats HAR at
  BH-adjusted DM p < 0.05 **and** the sign is consistent across ≥3 of 4
  walk-forward sub-periods. → Adopt/study that model; re-examine Exp 012 on the
  new base.
- **INCONCLUSIVE:** lowest-QLIKE model does not clear the DM significance bar, or
  DM signs are unstable across sub-periods. → Report ranking as descriptive only;
  no model change.
- **INSUFFICIENT DATA:** any model fails to fit or OOS set too thin. Report,
  don't force.

## Minimum sample
~470 daily obs; OOS set ≈ the second half (~230 days). Adequate for HAR/EWMA/
GARCH; DM on ~230 paired losses has reasonable power for moderate effects.

## Expected failure modes
- **HAR wins or ties everything** (highest likelihood — the literature's usual
  result on daily RV). A clean, publishable-internally negative-for-challengers
  result.
- **GARCH/EGARCH fit instability** on the short-ish sample → report as
  insufficient for those models rather than forcing a number.
- **DM low power** → many pairs land INCONCLUSIVE; that is an honest outcome.

## Scope
Pure forecasting comparison. **No trading rule, no sizing, no strategy.** A
winner means "use this model to forecast RV," nothing more.

---

## Scoring (filled AFTER running — 496 daily obs, 247 common OOS days)
- Status: **HAR CONFIRMED BEST.**
- OOS losses (lower better):

  | model  | RMSE (bp) | MAE (bp) | QLIKE |
  |--------|-----------|----------|-------|
  | **HAR**    | **15.11** | **10.51** | **0.1500** |
  | RW     | 16.86 | 11.42 | 0.1668 |
  | EWMA   | 16.86 | 12.55 | 0.1782 |
  | GARCH  | 37.59 | 30.98 | 0.4247 |
  | EGARCH | 33.72 | 30.13 | 0.3968 |

- Diebold-Mariano vs HAR (QLIKE, HAC, BH-adjusted): EWMA worse (BH p=0.017),
  GARCH worse (BH p<0.001), EGARCH worse (BH p<0.001); RW not significantly
  different (BH p=0.22). No model beats HAR.
- **Verdict:** HAR is the lowest-QLIKE model and no competitor beats it → HAR
  remains the volatility workhorse; Exp 012's HAR-based conclusion stands. This
  is the literature's usual result on daily RV.

## Honest caveat (important)
The comparison is **not perfectly like-for-like** for GARCH/EGARCH. HAR, EWMA
and RW operate directly on the **intraday** RV target (open-to-close, overnight
gap excluded). GARCH/EGARCH are fit on **close-to-close daily returns**, whose
variance *includes* the overnight gap — a different quantity than the target.
That definitional mismatch structurally handicaps the GARCH family here, so
their large losses should be read as "the standard close-to-close GARCH is the
wrong tool for forecasting intraday RV," **not** as "GARCH is a poor volatility
model in general." The clean, like-for-like takeaway is: **among models that
target intraday RV directly, HAR beats EWMA and the RW baseline.** A fairer
GARCH test would target realized-variance directly (e.g. a Realized-GARCH on the
RV series) — logged as future work, not required to answer the primary question.

## Future extensions (not run here)
- Realized-GARCH / HEAVY on the RV series (fair GARCH-family comparison).
- HAR-RV-J (add the Exp 014 jump component) once jumps are estimated.
- Hansen SPA / White Reality Check (Exp 023) as the family-wise control.

---
*Pre-registered 2026-07-01, executed same day as
`scripts/exp013_rv_forecast_comparison.py`.*
